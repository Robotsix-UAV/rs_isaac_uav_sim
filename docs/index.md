---
title: rs_isaac_uav_sim
---

# rs_isaac_uav_sim

GPU-accelerated UAV flight simulation built on **NVIDIA Isaac Sim 6.0**, with
**MAVLink HIL** integration so [PX4 SITL](https://docs.px4.io/main/en/sim_hitl/)
can fly the simulated drone in lockstep over TCP.

## What you get

- A working **3DR Iris quadrotor** asset (USD) with PhysX articulation, custom
  thrust + drag dynamics, and a sensor suite (IMU / barometer / magnetometer /
  GPS) plumbed through MAVLink.
- A **lockstep PX4 HIL backend** that speaks `HIL_SENSOR`, `HIL_GPS`,
  `HIL_STATE_QUATERNION`, and `HIL_ACTUATOR_CONTROLS` over TCP, with strict
  sim-time / autopilot-time lockstep enforced through `time_drift` monitoring.
- A **one-command live demo** (`scripts/live_flight_demo.py`) that boots Isaac
  Sim + PX4, waits for EKF + arming checks, arms PX4, switches to AUTO.TAKEOFF,
  climbs to ~5 m, and loiters until you Ctrl-C.
- A **self-contained Docker image** with ROS 2 Jazzy + Isaac Sim 6.0 +
  PX4-Autopilot v1.16 prebuilt, so the only host requirement is the NVIDIA
  driver + container toolkit.
- A real **test suite** (`colcon test`) covering lint, type checks, state
  rotation conventions, an end-to-end mock-controller flight, and an
  end-to-end PX4 SITL takeoff.

## Reading order

If you want to **run it**, start at the [Quickstart](quickstart.md).

If you want to **understand how it works**, read the
[Architecture](architecture.md) — it walks the physics loop, the MAVLink
lockstep handshake, and the coordinate frames the codebase juggles between.

If you want to **change a drone's parameters**, see the
[Configuration](configuration.md) page for the YAML schema.

## License

Apache License 2.0. The MAVLink HIL backend is ported from
[PegasusSimulator](https://github.com/PegasusSimulator/PegasusSimulator)
(BSD-3-Clause, © Marcelo Jacinto), credited inline.
