import os
import numpy as np
import mujoco

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as exc:
    raise ImportError(
        "Gymnasium is required for this environment. Install it with:\n"
        "    pip install gymnasium\n"
    ) from exc


class PandaWorkbenchEnv(gym.Env):
    """
    Minimal Gymnasium-style environment for the MuJoCo Panda workbench scene.

    Folder structure expected:

        MuJoCo Sim/
        ├── scene/
        │   └── workbench_scene.xml
        └── Modules/
            └── panda_workbench_env.py

    Action:
        9D normalized actuator command in [-1, 1].

    Observation:
        Dictionary containing robot joint states, box center, fingertip center,
        fingertip-to-box vector, and fingertip-to-box distance.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 60,
    }

    def __init__(
        self,
        xml_path=None,
        frame_skip=10,
        max_episode_steps=300,
        distance_success_threshold=0.035,
        render_mode=None,
    ):
        super().__init__()

        if xml_path is None:
            xml_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "scene",
                "workbench_scene.xml",
            )

        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)

        self.frame_skip = int(frame_skip)
        self.max_episode_steps = int(max_episode_steps)
        self.distance_success_threshold = float(distance_success_threshold)
        self.render_mode = render_mode

        self.step_count = 0
        self.viewer = None
        self.renderer = None

        self._init_ids()
        self._init_spaces()

        self.reset()

    # ------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------

    def _require_id(self, obj_type, name):
        obj_id = mujoco.mj_name2id(self.model, obj_type, name)

        if obj_id == -1:
            raise ValueError(f"Could not find '{name}' in the MuJoCo model.")

        return obj_id

    def _init_ids(self):
        self.box_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "target_box",
        )

        self.left_finger_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "left_finger",
        )

        self.right_finger_body_id = self._require_id(
            mujoco.mjtObj.mjOBJ_BODY,
            "right_finger",
        )

        self.arm_joint_ids = [
            self._require_id(mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ]

        self.arm_qpos_adrs = [
            self.model.jnt_qposadr[joint_id]
            for joint_id in self.arm_joint_ids
        ]

        self.arm_dof_adrs = [
            self.model.jnt_dofadr[joint_id]
            for joint_id in self.arm_joint_ids
        ]

        self.box_geom_id = self._get_box_geom_of_body(self.box_body_id)

        self.left_finger_pad_geom_id = self._get_largest_box_geom_on_body(
            self.left_finger_body_id
        )

        self.right_finger_pad_geom_id = self._get_largest_box_geom_on_body(
            self.right_finger_body_id
        )

        self.gripper_ctrl_ids = [7, 8]

    def _get_box_geom_of_body(self, body_id):
        geom_start = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]

        if geom_count == 0:
            raise ValueError("target_box body has no geom attached to it.")

        for geom_id in range(geom_start, geom_start + geom_count):
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                return geom_id

        return geom_start

    def _get_largest_box_geom_on_body(self, body_id):
        geom_start = self.model.body_geomadr[body_id]
        geom_count = self.model.body_geomnum[body_id]

        best_geom = -1
        best_volume = -1.0

        for geom_id in range(geom_start, geom_start + geom_count):
            if self.model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                size = self.model.geom_size[geom_id]
                volume = size[0] * size[1] * size[2]

                if volume > best_volume:
                    best_volume = volume
                    best_geom = geom_id

        if best_geom == -1:
            raise ValueError("Could not find box collision geom on finger body.")

        return best_geom

    def _init_spaces(self):
        if self.model.nu < 9:
            raise ValueError(
                f"Expected at least 9 actuators, but model.nu = {self.model.nu}"
            )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )

        self.observation_space = spaces.Dict(
            {
                "arm_qpos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
                "arm_qvel": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(7,),
                    dtype=np.float32,
                ),
                "gripper_qpos": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2,),
                    dtype=np.float32,
                ),
                "box_center": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "fingertip_center": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "vector_fingertip_to_box": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(3,),
                    dtype=np.float32,
                ),
                "distance_fingertip_to_box": spaces.Box(
                    low=0.0,
                    high=np.inf,
                    shape=(1,),
                    dtype=np.float32,
                ),
            }
        )

    # ------------------------------------------------------------
    # Core Gymnasium API
    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.step_count = 0

        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

        self._set_actuators_to_current_qpos()
        mujoco.mj_forward(self.model, self.data)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self._apply_action(action)

        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)

        self.step_count += 1

        observation = self._get_obs()
        reward = self._compute_reward(observation)

        distance = float(observation["distance_fingertip_to_box"][0])

        terminated = distance < self.distance_success_threshold
        truncated = self.step_count >= self.max_episode_steps

        info = self._get_info()
        info["is_success"] = terminated

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------
    # Simulation control
    # ------------------------------------------------------------

    def _set_actuators_to_current_qpos(self):
        for actuator_id in range(self.model.nu):
            trn_type = int(self.model.actuator_trntype[actuator_id])
            joint_id = int(self.model.actuator_trnid[actuator_id, 0])

            if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT) and joint_id >= 0:
                qadr = self.model.jnt_qposadr[joint_id]
                self.data.ctrl[actuator_id] = self.data.qpos[qadr]
                self._clamp_ctrl(actuator_id)

    def _apply_action(self, action):
        arm_delta_scale = 0.04
        gripper_delta_scale = 0.01

        self.data.ctrl[:7] += arm_delta_scale * action[:7]
        self.data.ctrl[7:9] += gripper_delta_scale * action[7:9]

        self._clamp_all_ctrl()

    def _clamp_ctrl(self, actuator_id):
        if self.model.actuator_ctrllimited[actuator_id]:
            low, high = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = np.clip(
                self.data.ctrl[actuator_id],
                low,
                high,
            )

    def _clamp_all_ctrl(self):
        for actuator_id in range(self.model.nu):
            self._clamp_ctrl(actuator_id)

    # ------------------------------------------------------------
    # Observations and reward
    # ------------------------------------------------------------

    def _get_obs(self):
        mujoco.mj_forward(self.model, self.data)

        arm_qpos = np.array(
            [self.data.qpos[adr] for adr in self.arm_qpos_adrs],
            dtype=np.float32,
        )

        arm_qvel = np.array(
            [self.data.qvel[adr] for adr in self.arm_dof_adrs],
            dtype=np.float32,
        )

        gripper_qpos = self.data.qpos[-2:].astype(np.float32).copy()

        box_center = self._get_box_center().astype(np.float32)
        fingertip_center = self._get_fingertip_center().astype(np.float32)

        vector_fingertip_to_box = box_center - fingertip_center

        distance = np.array(
            [np.linalg.norm(vector_fingertip_to_box)],
            dtype=np.float32,
        )

        return {
            "arm_qpos": arm_qpos,
            "arm_qvel": arm_qvel,
            "gripper_qpos": gripper_qpos,
            "box_center": box_center,
            "fingertip_center": fingertip_center,
            "vector_fingertip_to_box": vector_fingertip_to_box.astype(np.float32),
            "distance_fingertip_to_box": distance,
        }

    def _get_info(self):
        box_center = self._get_box_center()
        fingertip_center = self._get_fingertip_center()
        distance = np.linalg.norm(box_center - fingertip_center)

        return {
            "step_count": self.step_count,
            "distance_fingertip_to_box": float(distance),
            "box_center": box_center.copy(),
            "fingertip_center": fingertip_center.copy(),
        }

    def _compute_reward(self, observation):
        distance = float(observation["distance_fingertip_to_box"][0])

        reward = -distance

        if distance < self.distance_success_threshold:
            reward += 1.0

        return float(reward)

    def _get_box_center(self):
        return self.data.geom_xpos[self.box_geom_id].copy()

    def _get_fingertip_center(self):
        left_pos = self.data.geom_xpos[self.left_finger_pad_geom_id].copy()
        right_pos = self.data.geom_xpos[self.right_finger_pad_geom_id].copy()

        return 0.5 * (left_pos + right_pos)

    # ------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(
                    self.model,
                    self.data,
                )

            self.viewer.sync()
            return None

        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model)

            self.renderer.update_scene(self.data)
            return self.renderer.render()

        return None

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None


if __name__ == "__main__":
    env = PandaWorkbenchEnv(render_mode=None)

    obs, info = env.reset()

    print("Environment reset successfully.")
    print("Initial distance:", info["distance_fingertip_to_box"])

    for i in range(20):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"step={i:03d} | "
            f"reward={reward:.4f} | "
            f"distance={info['distance_fingertip_to_box']:.4f} | "
            f"terminated={terminated} | "
            f"truncated={truncated}"
        )

        if terminated or truncated:
            break

    env.close()
