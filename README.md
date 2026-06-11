## Repository structure

The project is organized as follows:

```text
MuJoCo Sim/
├── README.md
├── requirements.txt
├── Modules/
│   ├── Teleop_IK.py
│   └── panda_workbench_env.py
├── scene/
│   └── workbench_scene.xml
└── licenses/
    └── source and license files
```

## Important note about the virtual environment

The `venv/` folder is not included in the GitHub repository. This is intentional because Python virtual environments are machine-specific, large, and should be recreated locally on each computer.

To run the project on a fresh Ubuntu machine, create a new virtual environment and install the dependencies using `requirements.txt`.

## Setup on a fresh Ubuntu machine

1. Install basic Python tools:

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip git
```

2. Clone the repository:

```bash
git clone https://github.com/tlmsi/Panda-Workbench-MuJoCo-Simulation.git
cd Panda-Workbench-MuJoCo-Simulation
```

3. Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

4. Install the required Python packages:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is not available, install the required packages manually:

```bash
pip install mujoco numpy gymnasium
```

## Run the interactive MuJoCo viewer

From the main project folder, run:

```bash
source venv/bin/activate
cd Modules
python Teleop_IK.py
```

This opens the MuJoCo passive viewer with the Panda workbench scene.

Controls:

```text
Numpad 4/6  : base left/right
Numpad 8/2  : shoulder up/down
Numpad 7/1  : elbow up/down
Numpad 9/3  : wrist up/down
Numpad +/-  : wrist rotation
Numpad 0    : close gripper
Numpad .    : open gripper
Numpad 5    : automatic box pick
Backspace   : reset simulation
```

## Run the minimal environment class

From the main project folder:

```bash
source venv/bin/activate
cd Modules
python panda_workbench_env.py
```

Expected output:

```text
Environment reset successfully.
Initial distance: ...
step=000 | reward=... | distance=... | terminated=False | truncated=False
```

This verifies that the Gymnasium-style environment loads correctly and that `reset()` and `step(action)` are working.

## Opening and editing the files

To open the full project folder in VS Code:

```bash
cd Panda-Workbench-MuJoCo-Simulation
code .
```

To open a specific file in VS Code:

```bash
code README.md
code Modules/Teleop_IK.py
code Modules/panda_workbench_env.py
code scene/workbench_scene.xml
```

To open a file directly in the terminal using nano:

```bash
nano README.md
nano Modules/Teleop_IK.py
nano Modules/panda_workbench_env.py
nano scene/workbench_scene.xml
```

To list files and folders:

```bash
ls
```

To enter a folder:

```bash
cd folder_name
```

To go back one folder:

```bash
cd ..
```

## Asset and mesh notes

The Franka Panda robot model is based on the MuJoCo Menagerie Franka Emika Panda model. The source and license notes are included in the `licenses/` folder.

The workbench, target box, and simple tabletop objects are included directly in the MJCF scene file as simple/procedural assets. Therefore, no large external mesh download is required for the current version.

If future versions use large mesh files that are too large for GitHub, they should not be committed directly. Instead, the README should include a clear download link, the source of the asset, the license, and the exact folder where the reviewer should place the files.
