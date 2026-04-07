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

# Sensor models ported from PegasusSimulator (BSD-3-Clause, Marcelo Jacinto).
"""Simulated sensors: IMU, GPS, Barometer, Magnetometer with geo-magnetic utils."""

import numpy as np
from scipy.spatial.transform import Rotation

from .config import GPSOrigin, SensorParams, VisualOdometryParams
from .state import flu_to_frd, VehicleState

# ═══════════════════════════════════════════════════════════════════════════════
# Geomagnetic utilities (ported from Pegasus geo_mag_utils.py)
# ═══════════════════════════════════════════════════════════════════════════════

GRAVITY_VECTOR = np.array([0.0, 0.0, -9.80665])  # ENU, m/s^2
EARTH_RADIUS = 6353000.0  # m

SAMPLING_RES = 10.0
SAMPLING_MIN_LAT = -60
SAMPLING_MAX_LAT = 60
SAMPLING_MIN_LON = -180
SAMPLING_MAX_LON = 180

# WMM2018 lookup tables (10^5 x nanoTesla)
# noqa: E231
# fmt: off
DECLINATION_TABLE = [
    [47, 46, 45, 43, 42, 41, 39, 37, 33, 29, 23, 16, 10, 4, -1, -6, -10, -15,
     -20, -27, -34, -42, -49, -56, -62, -67, -72, -74, -75, -73, -61, -22, 26,
     42, 47, 48, 47],
    [31, 31, 31, 30, 30, 30, 30, 29, 27, 24, 18, 11, 3, -4, -9, -13, -15, -18,
     -21, -27, -33, -40, -47, -52, -56, -57, -56, -52, -44, -30, -14, 2, 14, 22,
     27, 30, 31],
    [22, 23, 23, 23, 22, 22, 22, 23, 22, 19, 13, 5, -4, -12, -17, -20, -22,
     -22, -23, -25, -30, -36, -41, -45, -46, -44, -39, -31, -21, -11, -3, 4, 10,
     15, 19, 21, 22],
    [17, 17, 17, 18, 17, 17, 17, 17, 16, 13, 8, -1, -10, -18, -22, -25, -26,
     -25, -22, -20, -21, -25, -29, -32, -31, -28, -23, -16, -9, -3, 0, 4, 7, 11,
     14, 16, 17],
    [13, 13, 14, 14, 14, 13, 13, 12, 11, 9, 3, -5, -14, -20, -24, -25, -24, -21,
     -17, -12, -9, -11, -14, -17, -18, -16, -12, -8, -3, -0, 1, 3, 6, 8, 11, 12,
     13],
    [11, 11, 11, 11, 11, 10, 10, 10, 9, 6, -0, -8, -15, -21, -23, -22, -19, -15,
     -10, -5, -2, -2, -4, -7, -9, -8, -7, -4, -1, 1, 1, 2, 4, 7, 9, 10, 11],
    [10, 9, 9, 9, 9, 9, 9, 8, 7, 3, -3, -10, -16, -20, -20, -18, -14, -9, -5,
     -2, 1, 2, 0, -2, -4, -4, -3, -2, -0, 0, 0, 1, 3, 5, 7, 9, 10],
    [9, 9, 9, 9, 9, 9, 9, 8, 6, 1, -4, -11, -16, -18, -17, -14, -10, -5, -2,
     -0, 2, 3, 2, 0, -1, -2, -2, -1, -0, -1, -1, -1, 1, 3, 6, 8, 9],
    [8, 9, 9, 10, 10, 10, 10, 8, 5, 0, -6, -12, -15, -16, -15, -11, -7, -4, -1,
     1, 3, 4, 3, 2, 1, 0, -0, -0, -1, -2, -3, -4, -2, 0, 3, 6, 8],
    [7, 9, 10, 11, 12, 12, 12, 9, 5, -1, -7, -13, -15, -15, -13, -10, -6, -3, 0,
     2, 3, 4, 4, 4, 3, 2, 1, 0, -1, -3, -5, -6, -6, -3, 0, 4, 7],
    [5, 8, 11, 13, 14, 15, 14, 11, 5, -2, -9, -15, -17, -16, -13, -10, -6, -3,
     0, 3, 4, 5, 6, 6, 6, 5, 4, 2, -1, -5, -8, -9, -9, -6, -3, 1, 5],
    [3, 8, 11, 15, 17, 17, 16, 12, 5, -4, -12, -18, -19, -18, -16, -12, -8, -4,
     -0, 3, 5, 7, 9, 10, 10, 9, 7, 4, -1, -6, -10, -12, -12, -9, -5, -1, 3],
    [3, 8, 12, 16, 19, 20, 18, 13, 4, -8, -18, -24, -25, -23, -20, -16, -11, -6,
     -1, 3, 7, 11, 14, 16, 17, 17, 14, 8, -0, -8, -13, -15, -14, -11, -7, -2, 3],
]

INCLINATION_TABLE = [
    [-78, -76, -74, -72, -70, -68, -65, -63, -60, -57, -55, -54, -54, -55, -56,
     -57, -58, -59, -59, -59, -59, -60, -61, -63, -66, -69, -73, -76, -79, -83,
     -86, -87, -86, -84, -82, -80, -78],
    [-72, -70, -68, -66, -64, -62, -60, -57, -54, -51, -49, -48, -49, -51, -55,
     -58, -60, -61, -61, -61, -60, -60, -61, -63, -66, -69, -72, -76, -78, -80,
     -81, -80, -79, -77, -76, -74, -72],
    [-64, -62, -60, -59, -57, -55, -53, -50, -47, -44, -41, -41, -43, -47, -53,
     -58, -62, -65, -66, -65, -63, -62, -61, -63, -65, -68, -71, -73, -74, -74,
     -73, -72, -71, -70, -68, -66, -64],
    [-55, -53, -51, -49, -46, -44, -42, -40, -37, -33, -30, -30, -34, -41, -48,
     -55, -60, -65, -67, -68, -66, -63, -61, -61, -62, -64, -65, -66, -66, -65,
     -64, -63, -62, -61, -59, -57, -55],
    [-42, -40, -37, -35, -33, -30, -28, -25, -22, -18, -15, -16, -22, -31, -40,
     -48, -55, -59, -62, -63, -61, -58, -55, -53, -53, -54, -55, -55, -54, -53,
     -51, -51, -50, -49, -47, -45, -42],
    [-25, -22, -20, -17, -15, -12, -10, -7, -3, 1, 3, 2, -5, -16, -27, -37, -44,
     -48, -50, -50, -48, -44, -41, -38, -38, -38, -39, -39, -38, -37, -36, -35,
     -35, -34, -31, -28, -25],
    [-5, -2, 1, 3, 5, 8, 10, 13, 16, 20, 21, 19, 12, 2, -10, -20, -27, -30, -30,
     -29, -27, -23, -19, -17, -17, -17, -18, -18, -17, -16, -16, -16, -16, -15,
     -12, -9, -5],
    [15, 18, 21, 22, 24, 26, 29, 31, 34, 36, 37, 34, 28, 20, 10, 2, -3, -5, -5,
     -4, -2, 2, 5, 7, 8, 7, 7, 6, 7, 7, 7, 6, 5, 6, 8, 11, 15],
    [31, 34, 36, 38, 39, 41, 43, 46, 48, 49, 49, 46, 42, 36, 29, 24, 20, 19, 20,
     21, 23, 25, 28, 30, 30, 30, 29, 29, 29, 29, 28, 27, 25, 25, 26, 28, 31],
    [43, 45, 47, 49, 51, 53, 55, 57, 58, 59, 59, 56, 53, 49, 45, 42, 40, 40, 40,
     41, 43, 44, 46, 47, 47, 47, 47, 47, 47, 47, 46, 44, 42, 41, 40, 42, 43],
    [53, 54, 56, 57, 59, 61, 64, 66, 67, 68, 67, 65, 62, 60, 57, 55, 55, 54, 55,
     56, 57, 58, 59, 59, 60, 60, 60, 60, 60, 60, 59, 57, 55, 53, 52, 52, 53],
    [62, 63, 64, 65, 67, 69, 71, 73, 75, 75, 74, 73, 70, 68, 67, 66, 65, 65, 65,
     66, 66, 67, 68, 68, 69, 70, 70, 71, 71, 70, 69, 67, 65, 63, 62, 62, 62],
    [71, 71, 72, 73, 75, 77, 78, 80, 81, 81, 80, 79, 77, 76, 74, 73, 73, 73, 73,
     73, 73, 74, 74, 75, 76, 77, 78, 78, 78, 78, 77, 75, 73, 72, 71, 71, 71],
]

STRENGTH_TABLE = [
    [62, 60, 58, 56, 54, 52, 49, 46, 43, 41, 38, 36, 34, 32, 31, 31, 30, 30, 30,
     31, 33, 35, 38, 42, 46, 51, 55, 59, 62, 64, 66, 67, 67, 66, 65, 64, 62],
    [59, 56, 54, 52, 50, 47, 44, 41, 38, 35, 32, 29, 28, 27, 26, 26, 26, 25, 25,
     26, 28, 30, 34, 39, 44, 49, 54, 58, 61, 64, 65, 66, 65, 64, 63, 61, 59],
    [54, 52, 49, 47, 45, 42, 40, 37, 34, 30, 27, 25, 24, 24, 24, 24, 24, 24, 24,
     24, 25, 28, 32, 37, 42, 48, 52, 56, 59, 61, 62, 62, 62, 60, 59, 56, 54],
    [49, 47, 44, 42, 40, 37, 35, 33, 30, 28, 25, 23, 22, 23, 23, 24, 25, 25, 26,
     26, 26, 28, 31, 36, 41, 46, 51, 54, 56, 57, 57, 57, 56, 55, 53, 51, 49],
    [43, 41, 39, 37, 35, 33, 32, 30, 28, 26, 25, 23, 23, 23, 24, 25, 26, 28, 29,
     29, 29, 30, 32, 36, 40, 44, 48, 51, 52, 52, 51, 51, 50, 49, 47, 45, 43],
    [38, 36, 35, 33, 32, 31, 30, 29, 28, 27, 26, 25, 24, 24, 25, 26, 28, 30, 31,
     32, 32, 32, 33, 35, 38, 42, 44, 46, 47, 46, 45, 45, 44, 43, 41, 40, 38],
    [34, 33, 32, 32, 31, 31, 31, 30, 30, 30, 29, 28, 27, 27, 27, 28, 29, 31, 32,
     33, 33, 33, 34, 35, 37, 39, 41, 42, 43, 42, 41, 40, 39, 38, 36, 35, 34],
    [33, 33, 32, 32, 33, 33, 34, 34, 35, 35, 34, 33, 32, 31, 30, 30, 31, 32, 33,
     34, 35, 35, 36, 37, 38, 40, 41, 42, 42, 41, 40, 39, 37, 36, 34, 33, 33],
    [34, 34, 34, 35, 36, 37, 39, 40, 41, 41, 40, 39, 37, 35, 35, 34, 35, 35, 36,
     37, 38, 39, 40, 41, 42, 43, 44, 45, 45, 45, 43, 41, 39, 37, 35, 34, 34],
    [37, 37, 38, 39, 41, 42, 44, 46, 47, 47, 46, 45, 43, 41, 40, 39, 39, 40, 41,
     41, 42, 43, 45, 46, 47, 48, 49, 50, 50, 50, 48, 46, 43, 41, 39, 38, 37],
    [42, 42, 43, 44, 46, 48, 50, 52, 53, 53, 52, 51, 49, 47, 45, 45, 44, 44, 45,
     46, 46, 47, 48, 50, 51, 53, 54, 55, 56, 55, 54, 52, 49, 46, 44, 43, 42],
    [48, 48, 49, 50, 52, 53, 55, 56, 57, 57, 56, 55, 53, 51, 50, 49, 48, 48, 48,
     49, 49, 50, 51, 53, 55, 56, 58, 59, 60, 60, 58, 56, 54, 52, 50, 49, 48],
    [54, 54, 54, 55, 56, 57, 58, 58, 59, 58, 58, 57, 56, 54, 53, 52, 51, 51, 51,
     51, 52, 53, 54, 55, 57, 58, 60, 61, 62, 61, 61, 59, 58, 56, 55, 54, 54],
]
# fmt: on


def _get_lookup_table_index(val, min_val, max_val):
    val = np.clip(val, min_val, max_val - SAMPLING_RES)
    return int((-min_val + val) / SAMPLING_RES)


def _get_table_data(lat, lon, table):
    if lat < -90.0 or lat > 90.0 or lon < -180.0 or lon > 180.0:
        return 0.0
    min_lat = int(lat / SAMPLING_RES) * SAMPLING_RES
    min_lon = int(lon / SAMPLING_RES) * SAMPLING_RES
    min_lat_idx = _get_lookup_table_index(min_lat, SAMPLING_MIN_LAT, SAMPLING_MAX_LAT)
    min_lon_idx = _get_lookup_table_index(min_lon, SAMPLING_MIN_LON, SAMPLING_MAX_LON)
    data_sw = table[min_lat_idx][min_lon_idx]
    data_se = table[min_lat_idx][min_lon_idx + 1]
    data_ne = table[min_lat_idx + 1][min_lon_idx + 1]
    data_nw = table[min_lat_idx + 1][min_lon_idx]
    lat_scale = np.clip((lat - min_lat) / SAMPLING_RES, 0.0, 1.0)
    lon_scale = np.clip((lon - min_lon) / SAMPLING_RES, 0.0, 1.0)
    data_min = lon_scale * (data_se - data_sw) + data_sw
    data_max = lon_scale * (data_ne - data_nw) + data_nw
    return lat_scale * (data_max - data_min) + data_min


def get_mag_declination(lat, lon):
    return _get_table_data(lat, lon, DECLINATION_TABLE)


def get_mag_inclination(lat, lon):
    return _get_table_data(lat, lon, INCLINATION_TABLE)


def get_mag_strength(lat, lon):
    return _get_table_data(lat, lon, STRENGTH_TABLE)


def reprojection(position, origin_lat_rad, origin_lon_rad):
    """Reproject local ENU position to (lat_rad, lon_rad)."""
    x_rad = position[1] / EARTH_RADIUS  # north
    y_rad = position[0] / EARTH_RADIUS  # east
    c = np.sqrt(x_rad**2 + y_rad**2)
    sin_c = np.sin(c)
    cos_c = np.cos(c)
    if c != 0.0:
        lat_rad = np.arcsin(
            cos_c * np.sin(origin_lat_rad)
            + (x_rad * sin_c * np.cos(origin_lat_rad)) / c
        )
        lon_rad = origin_lon_rad + np.arctan2(
            y_rad * sin_c,
            c * np.cos(origin_lat_rad) * cos_c
            - x_rad * np.sin(origin_lat_rad) * sin_c,
        )
    else:
        lat_rad = origin_lat_rad
        lon_rad = origin_lon_rad
    return lat_rad, lon_rad


# ═══════════════════════════════════════════════════════════════════════════════
# IMU Sensor
# ═══════════════════════════════════════════════════════════════════════════════


class IMUSensor:
    """Simulated IMU with gyro/accel bias random walk. Outputs in FRD body frame."""

    def __init__(self, params: SensorParams = SensorParams()):
        self._p = params
        self._gyro_bias = np.zeros(3)
        self._accel_bias = np.zeros(3)
        self._prev_linear_velocity = np.zeros(3)

    def update(self, state: VehicleState, dt: float) -> dict:
        """
        Compute noisy IMU readings.

        Args
        ----
        state
            Current vehicle state.
        dt
            Timestep in seconds.

        Returns
        -------
        dict
            dict with 'angular_velocity' (FRD) and 'linear_acceleration' (FRD).

        """
        p = self._p

        # Gyroscope noise
        tau_g = p.gyro_bias_correlation_time
        sigma_g_d = (1.0 / np.sqrt(dt)) * p.gyro_noise_density
        sigma_b_g = p.gyro_random_walk
        sigma_b_g_d = np.sqrt(
            -sigma_b_g**2 * tau_g / 2.0 * (np.exp(-2.0 * dt / tau_g) - 1.0)
        )
        phi_g_d = np.exp(-dt / tau_g)

        angular_velocity = np.zeros(3)
        for i in range(3):
            self._gyro_bias[i] = (
                phi_g_d * self._gyro_bias[i] + sigma_b_g_d * np.random.randn()
            )
            angular_velocity[i] = (
                state.angular_velocity[i]
                + sigma_g_d * np.random.randn()
                + self._gyro_bias[i]
            )

        # Accelerometer noise
        tau_a = p.accel_bias_correlation_time
        sigma_a_d = (1.0 / np.sqrt(dt)) * p.accel_noise_density
        sigma_b_a = p.accel_random_walk
        sigma_b_a_d = np.sqrt(
            -sigma_b_a**2 * tau_a / 2.0 * (np.exp(-2.0 * dt / tau_a) - 1.0)
        )
        phi_a_d = np.exp(-dt / tau_a)

        # Linear acceleration: differentiate velocity, subtract gravity, rotate to body
        linear_accel_inertial = (
            state.linear_velocity - self._prev_linear_velocity
        ) / dt - GRAVITY_VECTOR
        self._prev_linear_velocity = state.linear_velocity.copy()

        linear_accel_body = Rotation.from_quat(state.attitude).inv().apply(
            linear_accel_inertial
        )

        for i in range(3):
            self._accel_bias[i] = (
                phi_a_d * self._accel_bias[i] + sigma_b_a_d * np.random.randn()
            )
            linear_accel_body[i] += sigma_a_d * np.random.randn() + self._accel_bias[i]

        # FLU -> FRD
        angular_velocity_frd = flu_to_frd(angular_velocity)
        linear_accel_frd = flu_to_frd(linear_accel_body)

        return {
            'angular_velocity': angular_velocity_frd,
            'linear_acceleration': linear_accel_frd,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GPS Sensor
# ═══════════════════════════════════════════════════════════════════════════════


class GPSSensor:
    """Simulated GPS with position/velocity noise and lat/lon reprojection."""

    def __init__(
        self,
        params: SensorParams = SensorParams(),
        origin: GPSOrigin = GPSOrigin(),
    ):
        self._p = params
        self._origin = origin
        self._random_walk = np.zeros(3)
        self._gps_bias = np.zeros(3)

    def update(self, state: VehicleState, dt: float) -> dict:
        p = self._p
        origin = self._origin

        # Random walk and position noise
        self._random_walk[0] = p.gps_xy_random_walk * np.sqrt(dt) * np.random.randn()
        self._random_walk[1] = p.gps_xy_random_walk * np.sqrt(dt) * np.random.randn()
        self._random_walk[2] = p.gps_z_random_walk * np.sqrt(dt) * np.random.randn()

        noise_pos = np.array([
            p.gps_xy_noise_density * np.sqrt(dt) * np.random.randn(),
            p.gps_xy_noise_density * np.sqrt(dt) * np.random.randn(),
            p.gps_z_noise_density * np.sqrt(dt) * np.random.randn(),
        ])

        # Bias integration
        self._gps_bias += self._random_walk * dt - self._gps_bias / p.gps_correlation_time

        # Noisy position reprojection
        pos_noisy = state.position + noise_pos + self._gps_bias
        lat, lon = reprojection(
            pos_noisy, np.radians(origin.lat), np.radians(origin.lon)
        )

        # Ground truth
        lat_gt, lon_gt = reprojection(
            state.position, np.radians(origin.lat), np.radians(origin.lon)
        )

        vel = state.linear_velocity
        speed = np.linalg.norm(vel[:2])

        return {
            'fix_type': 3,
            'latitude': np.degrees(lat),
            'longitude': np.degrees(lon),
            'altitude': (state.position[2] + origin.alt - noise_pos[2]
                         + self._gps_bias[2]),
            'eph': 1.0,
            'epv': 1.0,
            'speed': speed,
            'velocity_north': vel[1],  # ENU y = North
            'velocity_east': vel[0],  # ENU x = East
            'velocity_down': -vel[2],
            'cog': 0.0,
            'sattelites_visible': 10,
            'latitude_gt': np.degrees(lat_gt),
            'longitude_gt': np.degrees(lon_gt),
            'altitude_gt': state.position[2] + origin.alt,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Visual Odometry Sensor
# ═══════════════════════════════════════════════════════════════════════════════


class VisualOdometrySensor:
    """Simulated visual odometry with position/velocity/attitude noise and bias random walk."""

    def __init__(self, params: VisualOdometryParams = VisualOdometryParams()):
        self._p = params
        self._pos_bias = np.zeros(3)
        self._vel_bias = np.zeros(3)

    def update(self, state: VehicleState, dt: float) -> dict:
        """
        Compute noisy visual odometry readings.

        Args
        ----
        state
            Current vehicle state.
        dt
            Timestep in seconds.

        Returns
        -------
        dict
            dict with position (x,y,z), velocity (vx,vy,vz), attitude (q),
            and angular rates (rollspeed, pitchspeed, yawspeed) in NED/FRD.

        """
        p = self._p

        # ENU position/velocity -> NED
        pos_ned = state.get_position_ned()
        vel_ned = state.get_linear_velocity_ned()

        # Position bias random walk
        tau_p = p.pos_correlation_time
        sigma_p_d = (1.0 / np.sqrt(dt)) * p.pos_noise_density
        sigma_b_p = p.pos_random_walk
        sigma_b_p_d = np.sqrt(
            -sigma_b_p**2 * tau_p / 2.0 * (np.exp(-2.0 * dt / tau_p) - 1.0)
        )
        phi_p_d = np.exp(-dt / tau_p)

        pos_noisy = np.zeros(3)
        for i in range(3):
            self._pos_bias[i] = (
                phi_p_d * self._pos_bias[i] + sigma_b_p_d * np.random.randn()
            )
            pos_noisy[i] = pos_ned[i] + sigma_p_d * np.random.randn() + self._pos_bias[i]

        # Velocity bias random walk
        tau_v = p.vel_correlation_time
        sigma_v_d = (1.0 / np.sqrt(dt)) * p.vel_noise_density
        sigma_b_v = p.vel_random_walk
        sigma_b_v_d = np.sqrt(
            -sigma_b_v**2 * tau_v / 2.0 * (np.exp(-2.0 * dt / tau_v) - 1.0)
        )
        phi_v_d = np.exp(-dt / tau_v)

        vel_noisy = np.zeros(3)
        for i in range(3):
            self._vel_bias[i] = (
                phi_v_d * self._vel_bias[i] + sigma_b_v_d * np.random.randn()
            )
            vel_noisy[i] = vel_ned[i] + sigma_v_d * np.random.randn() + self._vel_bias[i]

        # Attitude quaternion in NED/FRD with small orientation noise
        q_ned_frd = state.get_attitude_ned_frd()  # [qx, qy, qz, qw]
        sigma_att_d = (1.0 / np.sqrt(dt)) * p.att_noise_density
        att_rot = Rotation.from_quat(q_ned_frd)
        noise_rotvec = sigma_att_d * np.random.randn(3)
        att_noisy = (Rotation.from_rotvec(noise_rotvec) * att_rot).as_quat()
        # Return as [w, x, y, z]
        q_out = np.array([att_noisy[3], att_noisy[0], att_noisy[1], att_noisy[2]])

        # Angular rates in FRD body frame with noise
        ang_vel_frd = state.get_angular_velocity_frd()
        sigma_ang_d = (1.0 / np.sqrt(dt)) * p.att_noise_density
        ang_vel_noisy = ang_vel_frd + sigma_ang_d * np.random.randn(3)

        return {
            'x': pos_noisy[0],
            'y': pos_noisy[1],
            'z': pos_noisy[2],
            'vx': vel_noisy[0],
            'vy': vel_noisy[1],
            'vz': vel_noisy[2],
            'q': q_out,
            'rollspeed': ang_vel_noisy[0],
            'pitchspeed': ang_vel_noisy[1],
            'yawspeed': ang_vel_noisy[2],
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Barometer Sensor
# ═══════════════════════════════════════════════════════════════════════════════


class BarometerSensor:
    """Simulated barometer using ISA troposphere model."""

    TEMPERATURE_MSL = 288.15  # K
    PRESSURE_MSL = 101325.0  # Pa
    LAPSE_RATE = 0.0065  # K/m
    AIR_DENSITY_MSL = 1.225  # kg/m^3
    ABSOLUTE_ZERO_C = -273.15

    def __init__(
        self,
        params: SensorParams = SensorParams(),
        origin: GPSOrigin = GPSOrigin(),
    ):
        self._drift_rate = params.baro_drift_pa_per_sec
        self._origin_alt = origin.alt
        self._z_start = None
        self._drift_pa = 0.0
        self._rnd_use_last = False
        self._rnd_y2 = 0.0

    def update(self, state: VehicleState, dt: float) -> dict:
        if self._z_start is None:
            self._z_start = state.position[2]

        alt_rel = state.position[2] - self._z_start
        alt_amsl = self._origin_alt + alt_rel
        temp_local = self.TEMPERATURE_MSL - self.LAPSE_RATE * alt_amsl

        pressure_ratio = np.power(self.TEMPERATURE_MSL / temp_local, 5.2561)
        abs_pressure = self.PRESSURE_MSL / pressure_ratio

        # Box-Muller noise (matching Pegasus)
        if not self._rnd_use_last:
            w = 1.0
            while w >= 1.0:
                x1 = 2.0 * np.random.randn() - 1.0
                x2 = 2.0 * np.random.randn() - 1.0
                w = x1 * x1 + x2 * x2
            w = np.sqrt((-2.0 * np.log(w)) / w)
            y1 = x1 * w
            self._rnd_y2 = x2 * w
            self._rnd_use_last = True
        else:
            y1 = self._rnd_y2
            self._rnd_use_last = False

        noise_pa = y1
        self._drift_pa += self._drift_rate * dt
        abs_pressure_noisy = abs_pressure + noise_pa + self._drift_pa

        # hPa
        abs_pressure_hpa = abs_pressure_noisy * 0.01

        density_ratio = np.power(self.TEMPERATURE_MSL / temp_local, 4.256)
        air_density = self.AIR_DENSITY_MSL / density_ratio

        pressure_alt = alt_amsl - (noise_pa + self._drift_pa) / (
            np.linalg.norm(GRAVITY_VECTOR) * air_density
        )

        temp_celsius = temp_local + self.ABSOLUTE_ZERO_C

        return {
            'absolute_pressure': abs_pressure_hpa,
            'pressure_altitude': pressure_alt,
            'temperature': temp_celsius,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Magnetometer Sensor
# ═══════════════════════════════════════════════════════════════════════════════


class MagnetometerSensor:
    """Simulated magnetometer with geomagnetic lookup and noise."""

    def __init__(
        self,
        params: SensorParams = SensorParams(),
        origin: GPSOrigin = GPSOrigin(),
    ):
        self._p = params
        self._origin = origin
        self._bias = np.zeros(3)

    def update(self, state: VehicleState, dt: float) -> dict:
        origin = self._origin
        p = self._p

        lat, lon = reprojection(
            state.position, np.radians(origin.lat), np.radians(origin.lon)
        )

        declination_rad = np.radians(
            get_mag_declination(np.degrees(lat), np.degrees(lon))
        )
        inclination_rad = np.radians(
            get_mag_inclination(np.degrees(lat), np.degrees(lon))
        )
        strength_ga = 0.01 * get_mag_strength(np.degrees(lat), np.degrees(lon))

        H = strength_ga * np.cos(inclination_rad)
        Z = np.tan(inclination_rad) * H
        X = H * np.cos(declination_rad)
        Y = H * np.sin(declination_rad)

        # Magnetic field in NED inertial frame
        mag_inertial = np.array([X, Y, Z])

        # Rotate NED inertial field to FRD body frame via tested state helper
        rot_frd_to_ned = Rotation.from_quat(state.get_attitude_ned_frd())
        mag_body = rot_frd_to_ned.inv().apply(mag_inertial)

        # Add noise
        tau = p.mag_bias_correlation_time
        sigma_d = (1.0 / np.sqrt(dt)) * p.mag_noise_density
        sigma_b = p.mag_random_walk
        sigma_b_d = np.sqrt(-sigma_b**2 * tau / 2.0 * (np.exp(-2.0 * dt / tau) - 1.0))
        phi_d = np.exp(-dt / tau)

        mag_noisy = np.zeros(3)
        for i in range(3):
            self._bias[i] = phi_d * self._bias[i] + sigma_b_d * np.random.randn()
            mag_noisy[i] = mag_body[i] + sigma_d * np.random.randn() + self._bias[i]

        return {'magnetic_field': mag_noisy}
