# Copyright 2026 ROBOTSIX
# Author: Damien SIX (damien@robotsix.net)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MAVLink HIL protocol ported from PegasusSimulator (BSD-3-Clause, Marcelo Jacinto).
"""PX4 MAVLink HIL backend: sends sensor data, receives motor commands via TCP."""

import collections
import time

from isaacsim.core.api import SimulationContext
import numpy as np
from pymavlink import mavutil

from .config import MAVLINK_CONNECTION_TYPE, MavlinkParams, QuadrotorParams
from .state import VehicleState


class SensorSource:
    """Binary codes for HIL_SENSOR fields_updated bitmask."""

    ACCEL: int = 7  # 0b0000000000111
    GYRO: int = 56  # 0b0000000111000
    MAG: int = 448  # 0b0000111000000
    BARO: int = 6656  # 0b1101000000000
    DIFF_PRESS: int = 1024


class PX4MavlinkHIL:
    """
    MAVLink HIL interface for a single PX4 SITL instance.

    Handles TCP connection, sensor message transmission, and motor command
    reception with true lockstep synchronization.

    The main thread calls ``update(dt)`` to send HIL_SENSOR and then blocks
    until HIL_ACTUATOR_CONTROLS arrives before returning. Physics advances
    only after the reply is received.
    """

    def __init__(
        self,
        vehicle_id: int,
        params: MavlinkParams = MavlinkParams(),
        quad_params: QuadrotorParams = QuadrotorParams(),
        use_gps: bool = True,
        verbose: bool = False,
    ):
        self._vehicle_id = vehicle_id
        self._verbose = verbose
        self._use_gps = use_gps
        self._params = params
        self._port = params.base_port + vehicle_id
        self._connection_str = (
            f'{MAVLINK_CONNECTION_TYPE}:{params.connection_ip}:{self._port}'
        )
        self._connection = None
        self._is_running = False

        self._received_first_actuator = False
        self._received_first_heartbeat = False
        self._last_heartbeat_time = 0.0

        # Motor command scaling: maps actuator input [0,1] → [omega_min, omega_max]
        # using the PX4 THR_MDL_FAC thrust model.
        self._num_rotors = params.num_rotors
        self._omega_min = quad_params.omega_min
        self._omega_max = quad_params.omega_max
        self._thr_mdl_fac = quad_params.thr_mdl_fac
        self._motor_commands = np.zeros(self._num_rotors)
        self._armed = False

        # Simulation time tracking
        self._current_utime = 0

        # Actuator message timing
        self._last_actuator_wall_time = None
        self._last_actuator_interval_ms = None
        self._last_actuator_time_usec = None
        self._last_actuator_simdt_us = None

        # Last raw actuator message content
        self._raw_controls = np.zeros(self._num_rotors)
        self._raw_mode = 0

        # Sensor data flags
        self._new_imu = False
        self._received_first_imu = False
        self._new_gps = False
        self._new_baro = False
        self._new_mag = False

        # Cached sensor data
        self._imu_data = {}
        self._gps_data = {}
        self._visual_odometry_data = {}
        self._baro_data = {}
        self._mag_data = {}

        # Ground truth state (optional, guarded before use)
        self._gt_state = None

        # ------------------------------------------------------------------ #
        # Sensor send timing                                                  #
        # ------------------------------------------------------------------ #
        self._t_sensor_sent: float | None = None

        # ------------------------------------------------------------------ #
        # PX4 computation latency samples                                     #
        # ------------------------------------------------------------------ #
        self._px4_latency_samples: collections.deque = collections.deque(maxlen=100)

        # ------------------------------------------------------------------ #
        # Rate monitoring                                                     #
        # ------------------------------------------------------------------ #
        self._last_arrival_mono: float | None = None   # monotonic time of previous msg
        self._last_arrival_utime: int | None = None    # sim time (µs) of previous msg

        # ------------------------------------------------------------------ #
        # Timing debug state                                                  #
        # ------------------------------------------------------------------ #
        # Monotonic time at which the most recent actuator message was received,
        # captured so the main thread can measure sim-side delay (actuator → sensor).
        self._t_actuator_recv_for_debug: float | None = None
        # Wall time of last [TIMING DEBUG] print, used as 2-second gate.
        self._last_debug_print_time: float = 0.0

        # ------------------------------------------------------------------ #
        # Time-drift monitoring (sim time vs PX4 time growth rate)           #
        # ------------------------------------------------------------------ #
        self._diag_ref_sim_utime: int | None = None
        self._diag_ref_px4_utime: int | None = None

    # ---------------------------------------------------------------------- #
    # Public properties                                                       #
    # ---------------------------------------------------------------------- #

    @property
    def port(self) -> int:
        return self._port

    @property
    def motor_commands(self) -> np.ndarray:
        """Return current motor commands as angular velocities [rad/s]."""
        return self._motor_commands

    @property
    def last_actuator_interval_ms(self):
        """Return wall-clock interval between last two HIL_ACTUATOR_CONTROLS."""
        return self._last_actuator_interval_ms

    @property
    def last_actuator_simdt_us(self):
        """Return PX4 sim-time delta between last two HIL_ACTUATOR_CONTROLS."""
        return self._last_actuator_simdt_us

    @property
    def current_utime(self):
        """Return current sim time as sent to PX4 in HIL_SENSOR, in µs."""
        return self._current_utime

    @property
    def raw_controls(self) -> np.ndarray:
        """Return raw control values [0..1] as received from PX4, before scaling."""
        return self._raw_controls

    @property
    def raw_mode(self) -> int:
        """Return MAV_MODE bitmask from last HIL_ACTUATOR_CONTROLS message."""
        return self._raw_mode

    @property
    def time_sync_offset_us(self):
        """
        Return difference between sim time and PX4's last echoed time.

        In a healthy lockstep this should be exactly one physics_dt (e.g.
        2000 µs). Any value other than that indicates a sync problem.
        """
        if self._last_actuator_time_usec is None:
            return None
        return self._current_utime - self._last_actuator_time_usec

    @property
    def px4_latency_stats(self):
        """Return (avg_ms, min_ms, max_ms) of PX4 computation latency, or None if empty."""
        samples = list(self._px4_latency_samples)
        if not samples:
            return None
        return (sum(samples) / len(samples), min(samples), max(samples))

    @property
    def t_last_sensor_sent_mono(self) -> 'float | None':
        """Return monotonic timestamp of last sensor send completion."""
        return self._t_sensor_sent

    @property
    def t_last_actuator_recv_mono(self) -> 'float | None':
        """Return monotonic timestamp of last actuator message received."""
        return self._last_arrival_mono

    # ---------------------------------------------------------------------- #
    # Lifecycle                                                               #
    # ---------------------------------------------------------------------- #

    def connect(self):
        """Open TCP connection."""
        if self._is_running:
            return
        print(f'[MAVLink] Vehicle {self._vehicle_id}: connecting to '
              f'{self._connection_str}')
        self._connection = mavutil.mavlink_connection(self._connection_str)
        self._is_running = True
        self._received_first_heartbeat = False
        self._received_first_actuator = False

    def wait_for_heartbeat(self, timeout: float = 30.0) -> bool:
        """Block until PX4 heartbeat received or timeout expires. Returns True on success."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self._connection is None:
                return False
            msg = self._connection.recv_match(blocking=True, timeout=0.5)
            if msg is not None and msg.get_type() == 'HEARTBEAT':
                self._received_first_heartbeat = True
                print(f'[MAVLink] Vehicle {self._vehicle_id}: received first heartbeat')
                return True
        return False

    def close(self):
        """Close the MAVLink connection."""
        if not self._is_running:
            return
        self._is_running = False
        if self._connection is not None:
            try:
                self._connection.close()  # type: ignore
            except Exception:
                pass
            self._connection = None

    # ---------------------------------------------------------------------- #
    # Sensor data ingestion (called from main thread)                        #
    # ---------------------------------------------------------------------- #

    def update_sensors(self, imu_data: dict, baro_data: dict, mag_data: dict):
        """Buffer sensor data for the next send cycle."""
        self._imu_data = imu_data
        self._new_imu = True
        self._received_first_imu = True

        self._baro_data = baro_data
        self._new_baro = True

        self._mag_data = mag_data
        self._new_mag = True

    def update_gps(self, gps_data: dict):
        """Buffer GPS data for the next send cycle."""
        self._gps_data = gps_data
        self._new_gps = True

    def update_visual_odometry(self, data: dict) -> None:
        self._visual_odometry_data = data

    def update_state(self, state: VehicleState):
        """Store ground truth for HIL_STATE_QUATERNION (not sent, kept for future)."""
        self._gt_state = state

    # ---------------------------------------------------------------------- #
    # Main-thread update                                                      #
    # ---------------------------------------------------------------------- #

    def update(self) -> bool:
        """
        Send HIL_SENSOR and apply any actuator reply available from PX4.

        Non-blocking: if no HIL_ACTUATOR_CONTROLS has arrived yet, the last
        known motor commands are kept. Returns True once the first actuator
        message has ever been received.
        """
        if not self._is_running or self._connection is None:
            return False

        if not self._received_first_heartbeat:
            return False

        if self._received_first_imu and not self._new_imu:
            return False

        # Send heartbeat at ~1 Hz
        now = time.time()
        if (now - self._last_heartbeat_time) > 1.0:
            self._send_heartbeat()
            self._last_heartbeat_time = now

        # Use Isaac Sim's authoritative simulation time
        self._current_utime = int(SimulationContext.instance().current_time * 1_000_000)

        # Snapshot actuator arrival time before sending sensor, so sim-side
        # delay is measured as: t_sensor_sent - t_actuator_recv_for_debug.
        t_actuator_snap = self._t_actuator_recv_for_debug

        # Send sensor messages
        self._send_sensor_msgs(self._current_utime)
        self._t_sensor_sent = time.monotonic()

        # --- NON-BLOCKING RECV: pick up latest actuator command if available ---
        msg = self._connection.recv_match(
            type='HIL_ACTUATOR_CONTROLS', blocking=False
        )

        if msg is not None:
            now_mono = time.monotonic()

            # PX4 computation latency
            px4_latency_ms = (now_mono - self._t_sensor_sent) * 1000.0
            self._px4_latency_samples.append(px4_latency_ms)

            # Rate intervals (in sim time)
            self._last_arrival_mono = now_mono
            self._last_arrival_utime = msg.time_usec

            # Wall-clock timing
            if self._last_actuator_wall_time is not None:
                self._last_actuator_interval_ms = (
                    time.time() - self._last_actuator_wall_time
                ) * 1000.0
            self._last_actuator_wall_time = time.time()
            if self._last_actuator_time_usec is not None:
                self._last_actuator_simdt_us = msg.time_usec - self._last_actuator_time_usec
            self._last_actuator_time_usec = msg.time_usec
            self._t_actuator_recv_for_debug = now_mono

            # Apply control
            self._apply_control(list(msg.controls), msg.mode)
            self._received_first_actuator = True

        # --- Periodic timing debug output (every 2 s) ---
        _now_debug = time.monotonic()
        if _now_debug - self._last_debug_print_time >= 2.0:
            self._last_debug_print_time = _now_debug

            # 1. PX4 round-trip latency: HIL_SENSOR sent → HIL_ACTUATOR_CONTROLS received
            _px4_samples = list(self._px4_latency_samples)
            if _px4_samples:
                _px4_rtt_ms = sum(_px4_samples) / len(_px4_samples)
            else:
                _px4_rtt_ms = float('nan')

            # 3. Sim-side delay: actuator received → sensor sent (this cycle)
            if t_actuator_snap is not None:
                _sim_delay_ms = (self._t_sensor_sent - t_actuator_snap) * 1000.0
            else:
                _sim_delay_ms = float('nan')

            # 4. Time sync offset: difference between current sim time and PX4's
            #    last echoed time_usec (healthy lockstep ≈ one physics_dt)
            sync_offset = self.time_sync_offset_us
            if sync_offset is not None:
                _sync_str = f'{int(sync_offset)} µs'
                if abs(sync_offset) > 10000:
                    _sync_str += ' WARNING'
            else:
                _sync_str = 'N/A'

            # 5. Time-drift ratio: PX4 time growth vs sim time growth over window
            #    Healthy: ≈ 1.0. Drift means PX4 is running faster/slower than sim.
            if (self._diag_ref_sim_utime is not None
                    and self._diag_ref_px4_utime is not None
                    and self._last_arrival_utime is not None):
                sim_delta = self._current_utime - self._diag_ref_sim_utime
                px4_delta = self._last_arrival_utime - self._diag_ref_px4_utime
                if sim_delta > 0 and px4_delta > 0:
                    _drift_ratio = px4_delta / sim_delta
                    _drift_str = f'{_drift_ratio:.3f}'
                    if abs(_drift_ratio - 1.0) > 0.05:
                        _drift_str += ' WARNING'
                else:
                    _drift_str = 'N/A'
            else:
                _drift_str = 'N/A'
            # Reset window references
            self._diag_ref_sim_utime = self._current_utime
            self._diag_ref_px4_utime = self._last_arrival_utime

            print(
                f'[TIMING DEBUG] Vehicle {self._vehicle_id}: '
                f'px4_rtt={_px4_rtt_ms:.2f} ms | '
                f'sim_delay={_sim_delay_ms:.2f} ms | '
                f'time_sync={_sync_str} | '
                f'time_drift={_drift_str}'
            )

        if self._use_gps:
            self._send_gps_msgs(self._current_utime)
        else:
            self._send_odometry_msg(self._current_utime)

        return self._received_first_actuator

    # ---------------------------------------------------------------------- #
    # Motor command application (main thread)                                 #
    # ---------------------------------------------------------------------- #

    def _apply_control(self, controls, mode):
        """Apply a decoded HIL_ACTUATOR_CONTROLS entry on the main thread."""
        self._raw_controls[:] = controls[:self._num_rotors]
        self._raw_mode = mode

        if mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED:
            self._armed = True
            fac = self._thr_mdl_fac
            for i in range(self._num_rotors):
                u = max(0.0, min(1.0, controls[i]))
                omega_norm = fac * u + (1.0 - fac) * u ** 0.5
                self._motor_commands[i] = (
                    self._omega_min + omega_norm * (self._omega_max - self._omega_min)
                )
        else:
            self._armed = False
            self._motor_commands[:] = 0.0

    # ---------------------------------------------------------------------- #
    # MAVLink send helpers (main thread)                                      #
    # ---------------------------------------------------------------------- #

    def _send_heartbeat(self):
        """Send MAVLink heartbeat."""
        self._connection.mav.heartbeat_send(  # type: ignore
            mavutil.mavlink.MAV_TYPE_GENERIC,
            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
            0, 0, 0,
        )

    def _send_sensor_msgs(self, time_usec: int):
        """Send HIL_SENSOR message with available sensor data."""
        fields_updated = 0

        if self._new_imu:
            fields_updated |= SensorSource.ACCEL | SensorSource.GYRO
            self._new_imu = False

        if self._new_mag:
            fields_updated |= SensorSource.MAG
            self._new_mag = False

        if self._new_baro:
            fields_updated |= SensorSource.BARO
            self._new_baro = False

        imu = self._imu_data
        baro = self._baro_data
        mag = self._mag_data

        try:
            self._connection.mav.hil_sensor_send(  # type: ignore
                time_usec,
                imu.get('linear_acceleration', [0, 0, 0])[0],
                imu.get('linear_acceleration', [0, 0, 0])[1],
                imu.get('linear_acceleration', [0, 0, 0])[2],
                imu.get('angular_velocity', [0, 0, 0])[0],
                imu.get('angular_velocity', [0, 0, 0])[1],
                imu.get('angular_velocity', [0, 0, 0])[2],
                mag.get('magnetic_field', [0, 0, 0])[0],
                mag.get('magnetic_field', [0, 0, 0])[1],
                mag.get('magnetic_field', [0, 0, 0])[2],
                baro.get('absolute_pressure', 0.0),
                0.0,  # diff_pressure
                baro.get('pressure_altitude', 0.0),
                baro.get('temperature', 0.0),
                fields_updated,
            )
        except Exception as e:
            print(f'[MAVLink] Vehicle {self._vehicle_id}: sensor send error: '
                  f'{e}')

    def _send_gps_msgs(self, time_usec: int):
        """Send HIL_GPS message if new data available."""
        if not self._new_gps:
            return
        self._new_gps = False
        gps = self._gps_data

        try:
            self._connection.mav.hil_gps_send(  # type: ignore
                time_usec,
                int(gps.get('fix_type', 3)),
                int(gps.get('latitude', 0) * 10_000_000),
                int(gps.get('longitude', 0) * 10_000_000),
                int(gps.get('altitude', 0) * 1000),
                int(gps.get('eph', 1.0)),
                int(gps.get('epv', 1.0)),
                int(gps.get('speed', 0) * 100),
                int(gps.get('velocity_north', 0) * 100),
                int(gps.get('velocity_east', 0) * 100),
                int(gps.get('velocity_down', 0) * 100),
                int(gps.get('cog', 0) * 100),
                int(gps.get('sattelites_visible', 10)),
            )
        except Exception as e:
            print(f'[MAVLink] Vehicle {self._vehicle_id}: GPS send error: {e}')

    def _send_odometry_msg(self, time_usec: int):
        """Send ODOMETRY message (ID 331) with visual odometry data."""
        if not self._visual_odometry_data:
            return
        data = self._visual_odometry_data

        try:
            self._connection.mav.odometry_send(  # type: ignore
                time_usec,
                1,  # frame_id: MAV_FRAME_LOCAL_NED = 1
                8,  # child_frame_id: MAV_FRAME_BODY_FRD = 8
                float(data.get('x', 0.0)),
                float(data.get('y', 0.0)),
                float(data.get('z', 0.0)),
                [
                    float(v)
                    for v in data.get('q', [1.0, 0.0, 0.0, 0.0])
                ],  # quaternion [w,x,y,z]
                float(data.get('vx', 0.0)),
                float(data.get('vy', 0.0)),
                float(data.get('vz', 0.0)),
                float(data.get('rollspeed', 0.0)),
                float(data.get('pitchspeed', 0.0)),
                float(data.get('yawspeed', 0.0)),
                [float('nan')] * 21,  # pose_covariance (unknown)
                [float('nan')] * 21,  # velocity_covariance (unknown)
                0,  # reset_counter
                0,  # estimator_type: MAV_ESTIMATOR_TYPE_UNKNOWN = 0
                -1,  # quality (-1 = unknown)
            )
        except Exception as e:
            print(f'[MAVLink] Vehicle {self._vehicle_id}: odometry send error: {e}')

    def send_ground_truth(self, state: VehicleState, time_usec: int):
        """Send HIL_STATE_QUATERNION for ground truth."""
        if self._connection is None:
            return

        att = state.get_attitude_ned_frd()  # [qx,qy,qz,qw] FRD→NED (body-to-world)
        ang_vel = state.get_angular_velocity_frd()
        lin_vel = state.get_linear_velocity_ned()

        # GPS lat/lon from ground truth (stored from update_gps)
        sim_lat = int(self._gps_data.get('latitude_gt', 0) * 10_000_000)
        sim_lon = int(self._gps_data.get('longitude_gt', 0) * 10_000_000)
        sim_alt = int(self._gps_data.get('altitude_gt', 0) * 1000)

        try:
            self._connection.mav.hil_state_quaternion_send(  # type: ignore
                time_usec,
                # MAVLink HIL_STATE_QUATERNION expects NED→FRD (world-to-body):
                # conjugate of body-to-world = negate vector part, keep scalar
                [att[3], -att[0], -att[1], -att[2]],
                ang_vel[0], ang_vel[1], ang_vel[2],
                sim_lat, sim_lon, sim_alt,
                int(lin_vel[0] * 100),
                int(lin_vel[1] * 100),
                int(lin_vel[2] * 100),
                0, 0,  # airspeed
                0, 0, 0,  # acceleration
            )
        except Exception as e:
            print(f'[MAVLink] Vehicle {self._vehicle_id}: ground truth send '
                  f'error: {e}')
