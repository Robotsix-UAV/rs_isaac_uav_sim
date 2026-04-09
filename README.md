# rs_isaac_uav_sim

GPU-accelerated UAV flight simulation built on **NVIDIA Isaac Sim 6.0**, with
**MAVLink HIL** integration so [PX4 SITL](https://docs.px4.io/main/en/sim_hitl/) can
fly the simulated drone in lockstep over TCP. The repository ships a working
quadrotor (a 3DR Iris clone), a self-contained Docker image, and a one-command
demo that boots Isaac Sim + PX4 and lets PX4 take off into a steady hover.

## Documentation

Full docs are published via GitHub Pages at
**<https://robotsix-uav.github.io/rs_isaac_uav_sim/>**, sourced from the
[`docs/`](docs/) directory:

- [Quickstart](docs/quickstart.md)  – run the demo in <5 commands
- [Architecture](docs/architecture.md) – physics loop, MAVLink lockstep, coordinate frames
- [Configuration](docs/configuration.md) – YAML schema for drones, sensors, GPS

## Quickstart

The fastest path to a flying drone is the bundled Docker image:

```bash
git clone https://github.com/Robotsix-UAV/rs_isaac_uav_sim.git
cd rs_isaac_uav_sim

# One-time on the host: allow the container to draw on your X display
xhost +local:

# Build + run (NVIDIA Container Toolkit required)
docker compose up --build
```

The container drops you into a shell with `ISAAC_SIM_PYTHON`, `PX4_SITL_BUILD_DIR`,
and the workspace already sourced. To launch a steady hover demo:

```bash
python3 scripts/live_flight_demo.py            # GUI window opens on the host
python3 scripts/live_flight_demo.py --headless # no window
```

The demo spawns Isaac Sim + PX4 SITL, waits for EKF + arming checks, arms PX4,
switches to AUTO.TAKEOFF, climbs to ~5 m, and sits in `AUTO.LOITER` until you
hit Ctrl-C.

## Without Docker

Requirements:

- ROS 2 Jazzy
- Isaac Sim 6.0 (set `ISAAC_SIM_PYTHON` to its `python` binary)
- PX4-Autopilot v1.16 SITL build (set `PX4_SITL_BUILD_DIR` to the build dir)
- A working Vulkan + OpenGL stack (`libvulkan1`, `libgl1`, `libglu1-mesa`, …)

Build with `colcon`:

```bash
export ISAAC_SIM_PYTHON=/opt/isaacsim/python
mkdir -p ws/src && cd ws/src
git clone https://github.com/Robotsix-UAV/rs_isaac_uav_sim.git
cd .. && colcon build --packages-select rs_isaac_uav_sim
source install/setup.bash
```

Then either invoke the demo directly:

```bash
python3 src/rs_isaac_uav_sim/scripts/live_flight_demo.py
```

or use the bundled ROS 2 launch files for finer control:

```bash
# Sim + N×PX4 instances together
ros2 launch rs_isaac_uav_sim px4_sitl.launch.py num_drones:=1

# Sim only (you bring your own controller)
ros2 launch rs_isaac_uav_sim scene_mavlink_sim.py num_drones:=1
```

## Tests

```bash
colcon test --packages-select rs_isaac_uav_sim
colcon test-result --test-result-base build/rs_isaac_uav_sim --verbose
```

The suite covers Python lint (flake8, pep257, pyright), license/copyright
checks, unit tests (state rotation conventions), and two integration tests
that boot Isaac Sim end-to-end:

- `test_mavlink_scenarios` — talks to the simulator via a MockMavlinkAutopilot
  for fast initial-state and open-loop liftoff verification
- `test_px4_sitl_takeoff` — boots a real PX4 SITL instance, arms it via a GCS
  client, asserts AUTO.TAKEOFF + AUTO.LOITER hover stability (skipped unless
  `PX4_SITL_BUILD_DIR` is set)

## License

Apache License 2.0 — see [LICENSE](LICENSE).

The MAVLink HIL backend is ported from
[PegasusSimulator](https://github.com/PegasusSimulator/PegasusSimulator)
(BSD-3-Clause, © Marcelo Jacinto), credited inline at the top of
`rs_isaac_uav_sim/sim/mavlink_backend.py`.
