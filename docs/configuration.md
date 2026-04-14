---
title: Configuration
---

# Configuration

The simulator's per-drone parameters live in YAML files under
[`config/`](https://github.com/Robotsix-UAV/rs_isaac_uav_sim/tree/main/config).
Two are shipped:

- **`default.yaml`** — generic 1 kg quadrotor matching the dataclass defaults
  in `rs_isaac_uav_sim/sim/config.py`. Useful as a starting point.
- **`px4_iris.yaml`** — physical parameters for the 3DR Iris (PX4 airframe
  10016), sourced from the gazebo-classic SDF and PX4 control allocation.
  Pair this one with PX4 SITL.

Pass a config file at simulator launch time:

```bash
python3 isaac_run/scene_mavlink_sim.py --config config/px4_iris.yaml
```

Any key you omit falls back to the dataclass default. The schema is split
into four top-level sections, all optional:

```yaml
quadrotor:    {...}    # physical body + rotors
sensor:       {...}    # IMU / baro / mag / GPS / VO noise
mavlink:      {...}    # connection metadata (rarely overridden)
gps_origin:   {...}    # WGS84 reference for the local frame
```

Below is the full reference. The dataclass source of truth is
[`rs_isaac_uav_sim/sim/config.py`](https://github.com/Robotsix-UAV/rs_isaac_uav_sim/blob/main/rs_isaac_uav_sim/sim/config.py).

## `quadrotor:`

| key | type | default | meaning |
|---|---|---|---|
| `mass` | float | `1.0` | Total vehicle mass in kg |
| `inertia` | `[Ixx, Iyy, Izz]` | `[0.0052, 0.0052, 0.0082]` | Diagonal inertia tensor in kg·m² |
| `rotor_positions` | list of `[x, y, z]` | `4 × cardinal arms` | Rotor positions in **FLU body frame**, meters |
| `ct` | float | `1.1e-05` | Thrust coefficient: `T = ct * ω²` (N per (rad/s)²) |
| `cq` | float | `1.56e-07` | Torque coefficient: `Q = cq * ω²` |
| `rot_dirs` | list of `±1` | `[-1, 1, -1, 1]` | Spin direction per rotor (-1=CCW, +1=CW from above in FLU) |
| `omega_min` | float | `0.0` | Minimum rotor angular velocity (rad/s) |
| `omega_max` | float | `1000.0` | Maximum rotor angular velocity (rad/s) |
| `drag` | `[dx, dy, dz]` | `[0, 0, 0]` | Linear drag coefficients (kg/s); `F_drag = -diag(drag) * v_body` |
| `thr_mdl_fac` | float | `0.0` | PX4 `THR_MDL_FAC`: 0.0 = linear thrust curve, 1.0 = quadratic |
| `usd_asset` | string | `''` (basic_quadrotor) | Path to a USD asset relative to the package's `assets/` directory |

The number of rotors is **inferred from the length of `rotor_positions`** —
to add or remove motors, just adjust this list and the matching `rot_dirs`.

## `sensor:`

Sensor noise models follow PX4 conventions. The defaults match the iris
PX4 airframe.

### Gyro

| key | unit | default |
|---|---|---|
| `gyro_noise_density` | rad/s/√Hz | `3.39e-04` |
| `gyro_random_walk` | rad/s·√Hz | `3.88e-05` |
| `gyro_bias_correlation_time` | s | `1000.0` |
| `gyro_turn_on_bias_sigma` | rad/s | `8.73e-03` |

### Accelerometer

| key | unit | default |
|---|---|---|
| `accel_noise_density` | m/s²/√Hz | `4.0e-03` |
| `accel_random_walk` | m/s²·√Hz | `6.0e-03` |
| `accel_bias_correlation_time` | s | `300.0` |
| `accel_turn_on_bias_sigma` | m/s² | `0.196` |

### GPS

| key | unit | default |
|---|---|---|
| `gps_xy_random_walk` | (m/s)/√Hz | `2.0` |
| `gps_z_random_walk` | (m/s)/√Hz | `4.0` |
| `gps_xy_noise_density` | m/√Hz | `2.0e-04` |
| `gps_z_noise_density` | m/√Hz | `4.0e-04` |
| `gps_vxy_noise_density` | (m/s)/√Hz | `0.2` |
| `gps_vz_noise_density` | (m/s)/√Hz | `0.4` |
| `gps_correlation_time` | s | `60.0` |

### Barometer

| key | unit | default |
|---|---|---|
| `baro_drift_pa_per_sec` | Pa/s | `0.0` |

### Magnetometer

| key | unit | default |
|---|---|---|
| `mag_noise_density` | gauss/√Hz | `4.0e-04` |
| `mag_random_walk` | gauss·√Hz | `6.4e-06` |
| `mag_bias_correlation_time` | s | `600.0` |

### Localization mode

| key | type | default | meaning |
|---|---|---|---|
| `localization_mode` | str | `gps` | `gps` or `mocap` (mutually exclusive) |

- **`gps`** — Simulated `GPSSensor` + `MagnetometerSensor` are streamed to PX4 over MAVLink (`HIL_GPS` and the magnetometer fields of `HIL_SENSOR`). PX4's EKF runs GPS + baro + mag. This is the iris reference setup.
- **`mocap`** — No `HIL_GPS` and no `VISION_POSITION_ESTIMATE` is sent over MAVLink. Isaac's `ROS2PublishOdometry` OmniGraph node exposes ground-truth odometry on `/<drone>/isaac_odom` (requires `--ros2_namespaces` to be passed to the launch). An external ROS bridge is expected to subscribe to that topic and forward the (frame-converted, noise-injected as the consumer wishes) odometry to PX4 on `/<drone>/fmu/in/vehicle_visual_odometry`. Mirrors a real-hardware setup where motion-capture is the sole position source.

## `gps_origin:`

WGS84 reference frame for the simulator's local-tangent ENU coordinates.
All `HIL_GPS` lat/lon are computed by reprojecting the drone's local position
relative to this origin.

| key | unit | default |
|---|---|---|
| `lat` | deg | `47.24926` (Nantes, France) |
| `lon` | deg | `-1.54844` |
| `alt` | m AMSL | `0.0` |

## `mavlink:`

Rarely overridden — the CLI flags `--base_port`, `--connection_ip`, and
`--num_drones` on `scene_mavlink_sim.py` set these at launch time.

| key | default | notes |
|---|---|---|
| `base_port` | `4560` | Drone *i* uses TCP port `base_port + i + 1` |
| `connection_ip` | `'localhost'` | Where PX4 SITL expects to find the sim |
| `num_rotors` | derived from `quadrotor.rotor_positions` | not user-set |

## Adding a new airframe

The minimum set of changes for a new quadrotor:

1. Drop a USD asset under `assets/<your_drone>/` and reference it via
   `quadrotor.usd_asset` in your YAML.
2. Set `mass`, `inertia`, `rotor_positions`, `ct`, `cq`, `rot_dirs`,
   `omega_max` to match the physical drone.
3. If you're targeting a PX4 airframe, double-check that the rotor index
   order in `rotor_positions` matches the PX4 `CA_ROTOR*_PX/PY` convention
   for that airframe.
4. Update `gps_origin` if you want to fly somewhere other than Nantes.

Then point the simulator at your new YAML:

```bash
python3 isaac_run/scene_mavlink_sim.py --config config/your_drone.yaml
```
