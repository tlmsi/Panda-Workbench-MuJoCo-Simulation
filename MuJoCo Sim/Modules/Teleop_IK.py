import mujoco
import mujoco.viewer
import numpy as np
import os
import time

XML = os.path.join(os.path.dirname(__file__), "..", "scene", "workbench_scene.xml")
model = mujoco.MjModel.from_xml_path(XML)
data  = mujoco.MjData(model)


# ============================================================
# IDs / MODEL HELPERS
# ============================================================

def require_id(obj_type, name):
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id == -1:
        raise ValueError(f"Could not find '{name}' in the XML model.")
    return obj_id


if model.nu < 9:
    raise ValueError(f"This script expects at least 9 actuators, but model.nu = {model.nu}")


HAND_BODY_ID = require_id(mujoco.mjtObj.mjOBJ_BODY, "hand")
BOX_BODY_ID  = require_id(mujoco.mjtObj.mjOBJ_BODY, "target_box")

LEFT_FINGER_BODY_ID  = require_id(mujoco.mjtObj.mjOBJ_BODY, "left_finger")
RIGHT_FINGER_BODY_ID = require_id(mujoco.mjtObj.mjOBJ_BODY, "right_finger")

ARM_JNT_IDS = [
    require_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
    for i in range(1, 8)
]

ARM_QPOS_ADRS = [model.jnt_qposadr[j] for j in ARM_JNT_IDS]
ARM_DOF_ADRS  = [model.jnt_dofadr[j]  for j in ARM_JNT_IDS]

Q_MIN = np.array([
    model.jnt_range[j, 0] if model.jnt_limited[j] else -np.inf
    for j in ARM_JNT_IDS
])

Q_MAX = np.array([
    model.jnt_range[j, 1] if model.jnt_limited[j] else np.inf
    for j in ARM_JNT_IDS
])


def get_box_geom_of_body(body_id):
    geom_start = model.body_geomadr[body_id]
    geom_count = model.body_geomnum[body_id]

    if geom_count == 0:
        raise ValueError("target_box body has no geom attached to it.")

    for geom_id in range(geom_start, geom_start + geom_count):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            return geom_id

    return geom_start


def get_largest_box_geom_on_body(body_id):
    geom_start = model.body_geomadr[body_id]
    geom_count = model.body_geomnum[body_id]

    best_geom = -1
    best_volume = -1.0

    for geom_id in range(geom_start, geom_start + geom_count):
        if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
            size = model.geom_size[geom_id]
            volume = size[0] * size[1] * size[2]

            if volume > best_volume:
                best_volume = volume
                best_geom = geom_id

    if best_geom == -1:
        raise ValueError("Could not find box collision geom on finger body.")

    return best_geom


BOX_GEOM_ID = get_box_geom_of_body(BOX_BODY_ID)

LEFT_FINGER_PAD_GEOM_ID  = get_largest_box_geom_on_body(LEFT_FINGER_BODY_ID)
RIGHT_FINGER_PAD_GEOM_ID = get_largest_box_geom_on_body(RIGHT_FINGER_BODY_ID)


print("Using box geom ID:", BOX_GEOM_ID)
print("Box half-size:", model.geom_size[BOX_GEOM_ID])
print("Box full-size:", 2.0 * model.geom_size[BOX_GEOM_ID])

print("Using left finger pad geom ID :", LEFT_FINGER_PAD_GEOM_ID)
print("Using right finger pad geom ID:", RIGHT_FINGER_PAD_GEOM_ID)
print("Left finger pad half-size :", model.geom_size[LEFT_FINGER_PAD_GEOM_ID])
print("Right finger pad half-size:", model.geom_size[RIGHT_FINGER_PAD_GEOM_ID])


# ============================================================
# CONSTANTS
# ============================================================

DAMP     = 1e-2
IK_STEPS = 300
IK_TOL   = 1e-3

GRIPPER_OPEN   = 0.04
GRIPPER_CLOSED = 0.00

# Hard-coded waypoints relative to the real physical box center.
WP_ABOVE_OFFSET = np.array([0.0, 0.0, 0.22])
WP_GRASP_OFFSET = np.array([0.0, 0.0, 0.00])
WP_LIFT_OFFSET  = np.array([0.0, 0.0, 0.22])

# Desired direction of the line from left fingertip to right fingertip.
# If it rotates the wrong way, try:
# FINGER_AXIS_WORLD = np.array([1.0, 0.0, 0.0])
# or:
# FINGER_AXIS_WORLD = np.array([0.0, -1.0, 0.0])
FINGER_AXIS_WORLD = np.array([0.0, 1.0, 0.0])

FINGER_AXIS_WEIGHT = 3.0
FINGER_AXIS_TOL = 2e-3

STEP    = 0.0008
TIMEOUT = 0.3


# ============================================================
# KEY CODES
# ============================================================

LEFT     = 263
RIGHT    = 262
UP       = 265
DOWN     = 264

KP_4     = 324
KP_6     = 326
KP_8     = 328
KP_2     = 322

KP_7     = 327
KP_1     = 321
KP_9     = 329
KP_3     = 323

KP_0     = 320
KP_DOT   = 330
KP_PLUS  = 334
KP_MINUS = 333
KP_5     = 325

BACKSPACE = 259

ik_flag    = [False]
reset_flag = [False]
key_times  = {}


# ============================================================
# BASIC UTILITIES
# ============================================================

def clamp_ctrl(i):
    if model.actuator_ctrllimited[i]:
        low, high = model.actuator_ctrlrange[i]
        data.ctrl[i] = np.clip(data.ctrl[i], low, high)


def clamp_all_ctrl():
    for i in range(model.nu):
        clamp_ctrl(i)


def add_ctrl(i, delta):
    data.ctrl[i] += delta
    clamp_ctrl(i)


def set_gripper_ctrl(val):
    data.ctrl[7] = val
    data.ctrl[8] = val
    clamp_ctrl(7)
    clamp_ctrl(8)


def get_gripper_ctrl():
    return 0.5 * (data.ctrl[7] + data.ctrl[8])


def reset_to_backspace_pose():
    """
    Reset to the same default model state as MuJoCo Backspace.
    Then make the actuators hold the reset joint positions.
    """

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    for i in range(model.nu):
        trn_type = int(model.actuator_trntype[i])
        trn_id = int(model.actuator_trnid[i, 0])

        if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT) and trn_id >= 0:
            qadr = model.jnt_qposadr[trn_id]
            data.ctrl[i] = data.qpos[qadr]
            clamp_ctrl(i)

    mujoco.mj_forward(model, data)


def step_and_sync(viewer):
    step_start = time.time()

    mujoco.mj_step(model, data)
    viewer.sync()

    dt = model.opt.timestep
    elapsed = time.time() - step_start

    if elapsed < dt:
        time.sleep(dt - elapsed)


def hold_seconds(viewer, seconds):
    start = time.time()

    while viewer.is_running() and time.time() - start < seconds:
        step_and_sync(viewer)


# Start exactly from reset/backspace pose at simulation launch.
reset_to_backspace_pose()


# ============================================================
# TRUE BOX AND TRUE FINGERTIP CENTER
# ============================================================

def get_box_center_and_size():
    mujoco.mj_forward(model, data)

    box_center = data.geom_xpos[BOX_GEOM_ID].copy()
    box_half_size = model.geom_size[BOX_GEOM_ID].copy()

    box_bottom_z = box_center[2] - box_half_size[2]
    box_top_z    = box_center[2] + box_half_size[2]

    return box_center, box_half_size, box_bottom_z, box_top_z


def get_fingertip_center():
    left_pos = data.geom_xpos[LEFT_FINGER_PAD_GEOM_ID].copy()
    right_pos = data.geom_xpos[RIGHT_FINGER_PAD_GEOM_ID].copy()

    return 0.5 * (left_pos + right_pos)


def get_fingertip_center_jacobian():
    left_pos = data.geom_xpos[LEFT_FINGER_PAD_GEOM_ID].copy()
    right_pos = data.geom_xpos[RIGHT_FINGER_PAD_GEOM_ID].copy()

    jacp_left = np.zeros((3, model.nv))
    jacr_left = np.zeros((3, model.nv))

    jacp_right = np.zeros((3, model.nv))
    jacr_right = np.zeros((3, model.nv))

    mujoco.mj_jac(
        model,
        data,
        jacp_left,
        jacr_left,
        left_pos,
        LEFT_FINGER_BODY_ID
    )

    mujoco.mj_jac(
        model,
        data,
        jacp_right,
        jacr_right,
        right_pos,
        RIGHT_FINGER_BODY_ID
    )

    return 0.5 * (jacp_left + jacp_right)


def get_finger_delta_and_jacobian():
    left_pos = data.geom_xpos[LEFT_FINGER_PAD_GEOM_ID].copy()
    right_pos = data.geom_xpos[RIGHT_FINGER_PAD_GEOM_ID].copy()

    jacp_left = np.zeros((3, model.nv))
    jacr_left = np.zeros((3, model.nv))

    jacp_right = np.zeros((3, model.nv))
    jacr_right = np.zeros((3, model.nv))

    mujoco.mj_jac(
        model,
        data,
        jacp_left,
        jacr_left,
        left_pos,
        LEFT_FINGER_BODY_ID
    )

    mujoco.mj_jac(
        model,
        data,
        jacp_right,
        jacr_right,
        right_pos,
        RIGHT_FINGER_BODY_ID
    )

    delta = right_pos - left_pos
    jac_delta = jacp_right - jacp_left

    return delta, jac_delta


# ============================================================
# IK
# ============================================================

def ik_solve(gripper_target, q_init, align_fingers=False):
    q = q_init.copy()

    HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

    qpos_save = data.qpos.copy()

    pos_err = np.zeros(3)
    axis_err = np.zeros(3)

    for _ in range(IK_STEPS):
        for a, v in zip(ARM_QPOS_ADRS, q):
            data.qpos[a] = v

        mujoco.mj_forward(model, data)

        gripper_pos = get_fingertip_center()
        pos_err = gripper_target - gripper_pos

        jacp_center = get_fingertip_center_jacobian()
        J_pos = jacp_center[:, ARM_DOF_ADRS]

        if align_fingers:
            finger_delta, jac_delta = get_finger_delta_and_jacobian()

            current_width = np.linalg.norm(finger_delta)

            if current_width < 1e-6:
                desired_delta = finger_delta
            else:
                desired_axis = FINGER_AXIS_WORLD / np.linalg.norm(FINGER_AXIS_WORLD)
                desired_delta = current_width * desired_axis

            axis_err = desired_delta - finger_delta
            J_axis = jac_delta[:, ARM_DOF_ADRS]

            total_err = np.concatenate([
                pos_err,
                FINGER_AXIS_WEIGHT * axis_err
            ])

            J = np.vstack([
                J_pos,
                FINGER_AXIS_WEIGHT * J_axis
            ])

            JJT = J @ J.T + DAMP * np.eye(6)

        else:
            total_err = pos_err
            J = J_pos
            JJT = J @ J.T + DAMP * np.eye(3)

        if np.linalg.norm(pos_err) < IK_TOL:
            if not align_fingers or np.linalg.norm(axis_err) < FINGER_AXIS_TOL:
                break

        dq_task = J.T @ np.linalg.solve(JJT, total_err)

        N = np.eye(7) - J.T @ np.linalg.solve(JJT, J)

        dq_null = 0.1 * (HOME_Q - q)

        dq = dq_task + N @ dq_null

        q = np.clip(q + dq, Q_MIN, Q_MAX)

    data.qpos[:] = qpos_save
    mujoco.mj_forward(model, data)

    print(f"    IK position residual: {np.linalg.norm(pos_err):.4f} m")

    if align_fingers:
        print(f"    IK finger-axis residual: {np.linalg.norm(axis_err):.4f} m")

    return q


# ============================================================
# MOTION
# ============================================================

def move_to_q(target_q, gripper_val, viewer, steps=700):
    data.ctrl[:7] = [data.qpos[a] for a in ARM_QPOS_ADRS]

    start_q = data.ctrl[:7].copy()
    start_g = get_gripper_ctrl()

    for i in range(steps):
        if not viewer.is_running():
            return

        alpha = (i + 1) / steps

        data.ctrl[:7] = start_q + alpha * (target_q - start_q)

        g = start_g + alpha * (gripper_val - start_g)
        set_gripper_ctrl(g)

        clamp_all_ctrl()
        step_and_sync(viewer)


def set_gripper(val, viewer, steps=400):
    start = get_gripper_ctrl()

    for i in range(steps):
        if not viewer.is_running():
            return

        alpha = (i + 1) / steps

        g = start + alpha * (val - start)
        set_gripper_ctrl(g)

        step_and_sync(viewer)


# ============================================================
# PICK SEQUENCE
# ============================================================

def pick_box(viewer):
    box_center, box_half_size, box_bottom_z, box_top_z = get_box_center_and_size()

    WP_ABOVE_BOX = box_center + WP_ABOVE_OFFSET
    WP_GRASP_BOX = box_center + WP_GRASP_OFFSET
    WP_LIFT_BOX  = box_center + WP_LIFT_OFFSET

    mujoco.mj_forward(model, data)
    fingertip_center_now = get_fingertip_center()

    print("")
    print("========== PICK SEQUENCE ==========")
    print(f"  Box center      : {box_center.round(3)}")
    print(f"  Box half-size   : {box_half_size.round(3)}")
    print(f"  Box full-size   : {(2.0 * box_half_size).round(3)}")
    print(f"  Box bottom z    : {box_bottom_z:.3f}")
    print(f"  Box center z    : {box_center[2]:.3f}")
    print(f"  Box top z       : {box_top_z:.3f}")
    print("")
    print(f"  Fingertip center now: {fingertip_center_now.round(3)}")
    print("")
    print(f"  WP_ABOVE_BOX    : {WP_ABOVE_BOX.round(3)}")
    print(f"  WP_GRASP_BOX    : {WP_GRASP_BOX.round(3)}")
    print(f"  WP_LIFT_BOX     : {WP_LIFT_BOX.round(3)}")
    print("===================================")

    data.ctrl[:7] = [data.qpos[a] for a in ARM_QPOS_ADRS]
    mujoco.mj_forward(model, data)

    q_current = np.array([data.qpos[a] for a in ARM_QPOS_ADRS])

    print("  [1/6] Solving IK for waypoint above box with finger alignment...")
    q_above = ik_solve(WP_ABOVE_BOX, q_current, align_fingers=True)

    print("  [2/6] Moving from current/random pose to waypoint above box...")
    move_to_q(q_above, GRIPPER_OPEN, viewer, steps=900)

    print("  [3/6] Pausing above box for 1 second...")
    hold_seconds(viewer, 1.0)

    print("  [4/6] Solving IK for grasp waypoint with finger alignment...")
    q_grasp = ik_solve(WP_GRASP_BOX, q_above, align_fingers=True)

    print("  Moving down smoothly to grasp waypoint...")
    move_to_q(q_grasp, GRIPPER_OPEN, viewer, steps=700)

    print("  Pausing at grasp waypoint...")
    hold_seconds(viewer, 0.3)

    print("  [5/6] Closing gripper...")
    set_gripper(GRIPPER_CLOSED, viewer, steps=500)

    print("  Holding closed gripper...")
    hold_seconds(viewer, 0.5)

    print("  [6/6] Solving IK for lift waypoint with finger alignment...")
    q_lift = ik_solve(WP_LIFT_BOX, q_grasp, align_fingers=True)

    print("  Moving up to lift waypoint...")
    move_to_q(q_lift, GRIPPER_CLOSED, viewer, steps=700)

    print("  Pausing above box with object...")
    hold_seconds(viewer, 1.0)

    print("  Pick complete!")


# ============================================================
# KEYBOARD CONTROL
# ============================================================

def key_callback(keycode):
    if keycode == BACKSPACE:
        reset_flag[0] = True
        print("Reset requested")

    elif keycode == KP_0:
        set_gripper_ctrl(GRIPPER_CLOSED)
        print("Gripper CLOSED")

    elif keycode == KP_DOT:
        set_gripper_ctrl(GRIPPER_OPEN)
        print("Gripper OPEN")

    elif keycode == KP_5:
        ik_flag[0] = True
        print("Pick sequence triggered!")

    else:
        key_times[keycode] = time.time()


def apply_held_keys():
    now = time.time()

    expired = [k for k, t in key_times.items() if now - t > TIMEOUT]

    for k in expired:
        del key_times[k]

    if KP_4 in key_times or LEFT in key_times:
        add_ctrl(0, -STEP)

    if KP_6 in key_times or RIGHT in key_times:
        add_ctrl(0, STEP)

    if KP_8 in key_times or UP in key_times:
        add_ctrl(1, -STEP)

    if KP_2 in key_times or DOWN in key_times:
        add_ctrl(1, STEP)

    if KP_7 in key_times:
        add_ctrl(3, STEP)

    if KP_1 in key_times:
        add_ctrl(3, -STEP)

    if KP_9 in key_times:
        add_ctrl(5, STEP)

    if KP_3 in key_times:
        add_ctrl(5, -STEP)

    if KP_PLUS in key_times:
        add_ctrl(6, STEP)

    if KP_MINUS in key_times:
        add_ctrl(6, -STEP)


# ============================================================
# MAIN
# ============================================================

print("Controls:")
print("  Numpad 4/6   : base left/right")
print("  Numpad 8/2   : shoulder up/down")
print("  Numpad 7/1   : elbow up/down")
print("  Numpad 9/3   : wrist up/down")
print("  Numpad +/-   : wrist rotation (joint7)")
print("  Numpad 0     : close gripper")
print("  Numpad .     : open gripper")
print("  Numpad 5     : AUTO PICK BOX")
print("  Backspace    : reset to MuJoCo default pose")
print("  Arrows       : base and shoulder")


with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
    while viewer.is_running():

        if reset_flag[0]:
            reset_flag[0] = False
            ik_flag[0] = False
            key_times.clear()

            reset_to_backspace_pose()
            viewer.sync()

            print("Reset to Backspace pose")

        elif ik_flag[0]:
            ik_flag[0] = False
            pick_box(viewer)

        else:
            apply_held_keys()
            step_and_sync(viewer)
