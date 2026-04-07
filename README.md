# rs_isaac_uav_sim

Isaac Sim 6.0 UAV simulation package for ROS 2. Provides physics-based drone simulation with MAVLink integration.

## Prerequisites

- **Isaac Sim 6.0** installed and functional
- **ISAAC_SIM_PYTHON** environment variable set to the path of Isaac Sim's `python.sh` script

Example:
```bash
export ISAAC_SIM_PYTHON=/home/robotsix-docker/IsaacSim/_build/linux-x86_64/release/python.sh
```

## Installation

### Without ROS

Install the package directly into Isaac Sim's isolated Python environment:

```bash
export ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh
$ISAAC_SIM_PYTHON -m pip install -e /path/to/rs_isaac_uav_sim
```

### With ROS 2 (colcon)

Set the `ISAAC_SIM_PYTHON` environment variable, then use `colcon build`. The build process automatically installs the package into Isaac Sim's Python environment:

```bash
export ISAAC_SIM_PYTHON=/path/to/isaac-sim/python.sh
cd /path/to/ros2_ws
colcon build --packages-select rs_isaac_uav_sim
```

## Usage

Run the simulation script via Isaac Sim's Python:

```bash
$ISAAC_SIM_PYTHON scripts/scene_mavlink_sim.py
```

Ensure `ISAAC_SIM_PYTHON` is set before execution.
