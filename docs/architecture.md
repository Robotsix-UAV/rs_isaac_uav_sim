---
title: Architecture
---

# Architecture

This page walks the simulator's components, the per-step pipeline, and the
coordinate frames the codebase juggles between.

## Components

```
   PX4 SITL                    rs_isaac_uav_sim
  ──────────                  ──────────────────
                            ┌──────────────────────┐
   ┌─────────┐   HIL_SENSOR │  scene_mavlink_sim   │
   │  px4    │ <─────────── │  (Isaac Sim app)     │
   │         │   HIL_GPS    │                      │
   │  EKF    │ <─────────── │  ┌──────────────┐    │
   │         │              │  │ DroneSimMgr  │    │
   │  ATT    │   HIL_       │  │              │    │
   │  CTRL   │   ACTUATOR   │  │  vehicle.py  │    │
   │         │   _CONTROLS  │  │              │    │
   │  ALLOC  │ ───────────> │  └──────┬───────┘    │
   └─────────┘              │         │            │
                            │  ┌──────▼───────┐    │
                            │  │ MAVLink HIL  │    │
                            │  │ backend      │    │
                            │  │ (TCP server) │    │
                            │  └──────────────┘    │
                            └──────────────────────┘
```

- **`scene_mavlink_sim.py`** — the Isaac Sim entry point. Boots a
  `SimulationApp`, configures the renderer, instantiates a `World` (PhysX),
  loads the iris USD asset for each drone, and runs the main loop.

- **`DroneSimManager` (`rs_isaac_uav_sim/sim/vehicle.py`)** — the per-step
  orchestrator. Reads the rigid-body state from the GPU view via
  `get_world_poses` + `get_velocities`, runs the sensor sims, talks to the
  MAVLink backend(s), computes per-rotor thrusts and torques, and applies
  forces back to PhysX through `apply_forces_and_torques_at_pos`.

- **`PX4MavlinkHIL` (`rs_isaac_uav_sim/sim/mavlink_backend.py`)** — one
  instance per drone. Owns a TCP server socket, sends `HIL_SENSOR`,
  `HIL_GPS`, `HIL_STATE_QUATERNION` to PX4, and reads back
  `HIL_ACTUATOR_CONTROLS` to drive the rotors. Ported from
  [PegasusSimulator](https://github.com/PegasusSimulator/PegasusSimulator)
  (BSD-3-Clause, © Marcelo Jacinto), credited inline in the source.

- **Sensors (`rs_isaac_uav_sim/sim/sensors.py`)** — IMU, barometer,
  magnetometer, GPS, and visual-odometry sims with PX4-compatible noise
  models, biases, and random walks. Each sensor instance owns its own state.

- **`QuadrotorDynamics` (`rs_isaac_uav_sim/sim/dynamics.py`)** — pure-numpy
  per-rotor thrust + torque + drag math. Decoupled from PhysX; PhysX only
  sees a single 3-vector force and 3-vector torque per drone applied at the
  body center.

- **`px4_iris.yaml` (`config/`)** — physical parameters of the drone (mass,
  inertia, rotor positions, thrust/torque coefficients, sensor noise) plus
  the GPS reference origin. The `default.yaml` file gives you a generic
  configuration to start from.

## Per-step pipeline

The Isaac Sim main loop in `scene_mavlink_sim.py` is just:

```python
while simulation_app.is_running():
    manager.step(physics_dt)        # our work
    world.step(render=False)        # PhysX integration
    if not args.headless:
        if _render_step >= _render_every_n:
            world.render()          # 50 Hz throttled
            _render_step = 0
```

Inside `DroneSimManager.step()`:

1. **GPU readback** — `get_world_poses(usd=False)` + `get_velocities()` are
   batched into a single `torch.cat(...).cpu().numpy()` so only one CUDA
   stream sync happens per step.

2. **Per-drone loop** — for each drone, populate the `VehicleState`, run the
   stability detector (waits for the spawn drop to settle before switching
   sensors from "ideal" to "noisy"), update the IMU/baro/mag sensors, GPS at
   50 Hz, send `HIL_SENSOR` + `HIL_GPS` + `HIL_STATE_QUATERNION` to PX4,
   `recv_match` the latest `HIL_ACTUATOR_CONTROLS`, compute body-frame
   force + torque from the rotor speeds, and rotate to world frame.

3. **GPU upload** — copy the (N×3) world-frame force and torque arrays into
   pre-allocated CUDA tensors with a non-blocking `copy_()` and call
   `apply_forces_and_torques_at_pos`.

4. **PhysX `world.step()`** — runs outside our `step()`, integrates the
   articulation forward by `physics_dt`.

The simulator-PX4 contract is **strict lockstep**: PX4 only advances when it
receives `HIL_SENSOR`, and Isaac Sim only advances when it has a fresh IMU
sample to send. The reported `time_drift` (sim-time growth ÷ PX4-time growth)
stays at exactly 1.000 in healthy operation.

## Coordinate frames

The codebase juggles three frames:

| frame | axes | used by |
|---|---|---|
| **ENU world**  | x=East, y=North, z=Up    | Isaac Sim, USD assets |
| **FLU body**   | x=Forward, y=Left, z=Up  | dynamics, rotor positions |
| **NED / FRD**  | x=North, y=East, z=Down  | MAVLink, PX4, HIL_STATE_QUATERNION |

`VehicleState` stores everything in ENU/FLU and exposes `get_attitude_ned_frd()`
and `get_linear_velocity_ned()` for the MAVLink path. `state.py` is the single
source of truth for the conversion math; `test_state_rotations.py` covers it.
