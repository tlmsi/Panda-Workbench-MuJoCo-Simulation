Panda Workbench MuJoCo Simulation

GitHub repository:  [Panda Workbench MuJoCo Simulation](https://github.com/tlmsi/Panda-Workbench-MuJoCo-Simulation)

This project implements a tabletop manipulation scene in MuJoCo using a Franka Panda robot. I chose this scene because it represents a realistic data-collection station: a fixed robot arm is positioned in front of a workbench and interacts with a small graspable object. This setup is simple enough to build and debug clearly, but still meaningful for robot simulation because it includes object geometry, collision, grasping, robot placement, manual control, and a minimal environment interface.

The main design decision was to use the Franka Panda as the robot platform and place it at a reasonable distance from the workbench so the box is inside the robot’s reachable workspace. The scene includes a table/workbench, a target box, and simple surrounding props to make the setup closer to a real manipulation station. For the grasping behavior, I used the actual physical center of the box and the midpoint between the two fingertip collision pads instead of relying on a hard-coded joint pose. This made the automatic pick sequence more robust and better connected to the real simulated geometry. I also added fingertip-axis alignment so the gripper rotates before grasping and approaches the box from the correct side.

The project contains an interactive teleoperation script for the MuJoCo passive viewer and a separate minimal Gymnasium-style environment class implementing reset() and step(action). The interactive script demonstrates manual joint control, gripper control, scene reset, and an automatic box-pick sequence.

If I invested more time, I would improve the scene by adding cameras for data collection, randomizing object poses, tuning contact/friction parameters for more reliable grasps, and expanding the environment reward function into a full pick-and-place learning task.
