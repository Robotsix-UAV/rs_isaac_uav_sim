---
title: Quickstart
---

# Quickstart

The fastest path to a flying drone is the bundled Docker image. The image
includes ROS 2 Jazzy, Isaac Sim 6.0 (pip-installed in a venv), PX4-Autopilot
v1.16 SITL (prebuilt), and the GL/Vulkan/X11 stack Isaac Sim's renderer needs.

## Prerequisites

- Linux host with an NVIDIA GPU and a recent driver
- Recent NVIDIA driver (≥ 535)
- Docker Engine 20.10+
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
  (so the container can see the GPU)
- An X server on the host (any Linux desktop session) — only needed for the
  GUI demo, not for headless tests

## 1. Clone

```bash
git clone https://github.com/Robotsix-UAV/rs_isaac_uav_sim.git
cd rs_isaac_uav_sim
```

## 2. Allow X11 forwarding (one-time per session)

```bash
xhost +local:
```

This grants the container's user permission to draw on your X display. To
revoke, `xhost -local:` after you're done.

## 3. Build + start the container

```bash
docker compose up --build -d
docker compose exec sim bash
```

You'll be dropped into a shell at `/workspace` with `ISAAC_SIM_PYTHON`,
`PX4_SITL_BUILD_DIR`, ROS 2 Jazzy, and the package's `colcon` install
already sourced via `~/.bashrc`.

The first build takes ~25 minutes (Isaac Sim pip download + PX4 SITL
compile). Subsequent starts are instant because Docker caches both layers.

## 4. Run the live flight demo

Inside the container shell:

```bash
python3 scripts/live_flight_demo.py
```

A window opens on your host display showing the iris quadrotor on a ground
plane. The demo:

1. Starts Isaac Sim and a PX4 SITL instance.
2. Connects a GCS client to PX4's UDP MAVLink endpoint (`udp://0.0.0.0:14550`).
3. Waits ~30 s for the EKF to converge and arming checks to pass.
4. Arms PX4 and switches to `AUTO.TAKEOFF`.
5. PX4 climbs to ~5 m and auto-transitions to `AUTO.LOITER` for steady hover.
6. Prints `z / vx / vy / vz` samples every 5 s.
7. Cleans up everything on Ctrl-C.

Useful flags:

```bash
python3 scripts/live_flight_demo.py --altitude 10.0    # higher takeoff
python3 scripts/live_flight_demo.py --hover-seconds 120
python3 scripts/live_flight_demo.py --headless         # no GUI window
```

## 5. Run the test suite

```bash
colcon test --packages-select rs_isaac_uav_sim
colcon test-result --test-result-base build/rs_isaac_uav_sim --verbose
```

Expected: **75/75 passing** when `PX4_SITL_BUILD_DIR` is unset (the PX4 SITL
takeoff test is skipped). With `PX4_SITL_BUILD_DIR` set inside the container,
the SITL test runs and adds another 1 test.

The integration tests boot a fresh Isaac Sim subprocess for each test method,
so the first test pays a ~30 s cold-start cost while Isaac Sim populates its
shader cache. Subsequent runs are fast.

By default the integration tests launch Isaac Sim headless. To watch them
run in the GUI (useful for local debugging, X11 forwarding already set up
in the Docker image), set `RS_ISAAC_TEST_HEADLESS=0`:

```bash
RS_ISAAC_TEST_HEADLESS=0 colcon test --packages-select rs_isaac_uav_sim \
  --event-handlers console_direct+ \
  --ctest-args -R test_px4_sitl_takeoff
```

Accepted falsy values: `0`, `false`, `no`. Anything else (or unset) keeps
the tests headless — CI behavior is unchanged.

## Troubleshooting

**`vkCreateInstance failed` / GUI window doesn't open** — check `DISPLAY` is
set inside the container (`echo $DISPLAY`). If not, the X11 socket bind mount
isn't taking effect. Verify `/tmp/.X11-unix/X0` exists in the container, and
that you ran `xhost +local:` on the host.

**`no suitable CUDA GPU was found`** — the NVIDIA Container Toolkit isn't
exposing the GPU. Check `nvidia-smi` works inside the container. If it
doesn't, install or repair `nvidia-container-toolkit` on the host.

**`PX4_SITL_BUILD_DIR not set`** — only matters for `test_px4_sitl_takeoff`
and the live demo. The Dockerfile sets it to
`/opt/PX4-Autopilot/build/px4_sitl_default`. If you're running outside Docker,
set it to your own PX4 SITL build directory.

**Slow first launch** — Isaac Sim is populating its shader cache.
Subsequent launches in the same container are much faster.
