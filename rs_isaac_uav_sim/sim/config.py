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
"""Configuration dataclasses for the GPU-compatible drone simulation plugin."""

from dataclasses import dataclass, field
from math import pi
from typing import Any, Dict

import numpy as np


@dataclass
class QuadrotorParams:
    """Physical parameters for a quadrotor vehicle."""

    mass: float = 1.0  # kg (total system mass from physics.usda)
    # Approximate combined inertia for the basic_quadrotor USD asset
    inertia: np.ndarray = field(
        default_factory=lambda: np.array([0.0052, 0.0052, 0.0082])
    )
    # Rotor positions in FLU body frame.
    # The USD asset uses Isaac Sim Z-up, X-forward, Y-left frame which matches FLU.
    # Motor1=(+X arm), Motor2=(+Y arm), Motor3=(-X arm), Motor4=(-Y arm)
    rotor_positions: np.ndarray = field(
        default_factory=lambda: np.array(
            [
                [0.17, 0.0, 0.002],   # Motor1: +X arm (front)
                [0.0, 0.17, 0.002],   # Motor2: +Y arm (left)
                [-0.17, 0.0, 0.002],  # Motor3: -X arm (rear)
                [0.0, -0.17, 0.002],  # Motor4: -Y arm (right)
            ]
        )
    )
    # Thrust coefficient: T = ct * omega^2
    ct: float = 1.1e-05
    # Torque coefficient: Q = cq * omega^2
    cq: float = 1.56e-07
    # Rotation directions (-1=CCW, +1=CW) viewed from above in FLU.
    # Motor1=CCW(-1), Motor2=CW(+1), Motor3=CCW(-1), Motor4=CW(+1)
    rot_dirs: list = field(default_factory=lambda: [-1, 1, -1, 1])
    # Control range [rad/s]: maps actuator input [0,1] -> [omega_min, omega_max]
    omega_min: float = 0.0
    omega_max: float = 1000.0
    # Linear drag coefficients [dx, dy, dz] in body frame [kg/s].
    # Applied as F_drag = -diag(drag) * v_body (force proportional to velocity).
    drag: list = field(default_factory=lambda: [0.0, 0.0, 0.0])
    # PX4 THR_MDL_FAC: thrust model factor interpolating between linear (0.0)
    # and quadratic (1.0) thrust curves.  Matches PX4 default of 0.0 (linear).
    thr_mdl_fac: float = 0.0
    # USD asset path relative to the assets directory.  Empty string means use
    # the default basic_quadrotor asset.
    usd_asset: str = ''

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'QuadrotorParams':
        """Construct from a dict, falling back to dataclass defaults for missing keys."""
        defaults = cls()
        inertia = d.get('inertia', None)
        rotor_positions = d.get('rotor_positions', None)
        return cls(
            mass=d.get('mass', defaults.mass),
            inertia=np.array(inertia) if inertia is not None else defaults.inertia,
            rotor_positions=(
                np.array(rotor_positions)
                if rotor_positions is not None
                else defaults.rotor_positions
            ),
            ct=d.get('ct', defaults.ct),
            cq=d.get('cq', defaults.cq),
            rot_dirs=d.get('rot_dirs', defaults.rot_dirs),
            omega_min=d.get('omega_min', defaults.omega_min),
            omega_max=d.get('omega_max', defaults.omega_max),
            drag=d.get('drag', defaults.drag),
            thr_mdl_fac=d.get('thr_mdl_fac', defaults.thr_mdl_fac),
            usd_asset=d.get('usd_asset', defaults.usd_asset),
        )


LOCALIZATION_MODES = ('gps', 'mocap')


@dataclass
class SensorParams:
    """Noise parameters for simulated sensors, ported from Pegasus defaults."""

    # Gyroscope
    gyro_noise_density: float = 2.0 * 35.0 / 3600.0 / 180.0 * pi  # ~0.000339 rad/s/sqrt(Hz)
    gyro_random_walk: float = 2.0 * 4.0 / 3600.0 / 180.0 * pi  # ~3.88e-05 rad/s*sqrt(Hz)
    gyro_bias_correlation_time: float = 1.0e3  # s
    gyro_turn_on_bias_sigma: float = 0.5 / 180.0 * pi  # rad/s

    # Accelerometer
    accel_noise_density: float = 2.0 * 2.0e-3  # m/s^2/sqrt(Hz)
    accel_random_walk: float = 2.0 * 3.0e-3  # m/s^2*sqrt(Hz)
    accel_bias_correlation_time: float = 300.0  # s
    accel_turn_on_bias_sigma: float = 20.0e-3 * 9.8  # m/s^2

    # GPS
    gps_xy_random_walk: float = 2.0  # (m/s)/sqrt(Hz)
    gps_z_random_walk: float = 4.0
    gps_xy_noise_density: float = 2.0e-4  # m/sqrt(Hz)
    gps_z_noise_density: float = 4.0e-4
    gps_vxy_noise_density: float = 0.2  # (m/s)/sqrt(Hz)
    gps_vz_noise_density: float = 0.4
    gps_correlation_time: float = 60.0  # s

    # Barometer
    baro_drift_pa_per_sec: float = 0.0

    # Magnetometer
    mag_noise_density: float = 0.4e-3  # gauss/sqrt(Hz)
    mag_random_walk: float = 6.4e-6  # gauss*sqrt(Hz)
    mag_bias_correlation_time: float = 6.0e2  # s

    # Localization mode (mutually exclusive). Selects how the simulated
    # vehicle's position estimate reaches PX4:
    #   - "gps":   simulated GPSSensor + magnetometer are streamed to PX4
    #              via HIL_GPS / HIL_SENSOR. PX4's EKF uses GPS+baro+mag.
    #              Matches the iris reference setup.
    #   - "mocap": no GPS or VISION_POSITION_ESTIMATE is sent over MAVLink.
    #              Isaac's ROS2PublishOdometry node exposes ground-truth
    #              odometry on /<drone>/isaac_odom; an external ROS bridge
    #              is expected to forward it (after any frame conversion
    #              and noise injection) to /fmu/in/vehicle_visual_odometry.
    #              Mirrors a real-hardware setup where motion-capture is
    #              the sole position source.
    localization_mode: str = 'gps'

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SensorParams':
        """Construct from a dict, falling back to dataclass defaults for missing keys."""
        defaults = cls()
        mode = d.get('localization_mode', defaults.localization_mode)
        if mode not in LOCALIZATION_MODES:
            raise ValueError(
                f'sensor.localization_mode must be one of {LOCALIZATION_MODES}, '
                f'got {mode!r}'
            )
        return cls(
            gyro_noise_density=d.get('gyro_noise_density', defaults.gyro_noise_density),
            gyro_random_walk=d.get('gyro_random_walk', defaults.gyro_random_walk),
            gyro_bias_correlation_time=d.get(
                'gyro_bias_correlation_time', defaults.gyro_bias_correlation_time
            ),
            gyro_turn_on_bias_sigma=d.get(
                'gyro_turn_on_bias_sigma', defaults.gyro_turn_on_bias_sigma
            ),
            accel_noise_density=d.get('accel_noise_density', defaults.accel_noise_density),
            accel_random_walk=d.get('accel_random_walk', defaults.accel_random_walk),
            accel_bias_correlation_time=d.get(
                'accel_bias_correlation_time', defaults.accel_bias_correlation_time
            ),
            accel_turn_on_bias_sigma=d.get(
                'accel_turn_on_bias_sigma', defaults.accel_turn_on_bias_sigma
            ),
            gps_xy_random_walk=d.get('gps_xy_random_walk', defaults.gps_xy_random_walk),
            gps_z_random_walk=d.get('gps_z_random_walk', defaults.gps_z_random_walk),
            gps_xy_noise_density=d.get('gps_xy_noise_density', defaults.gps_xy_noise_density),
            gps_z_noise_density=d.get('gps_z_noise_density', defaults.gps_z_noise_density),
            gps_vxy_noise_density=d.get('gps_vxy_noise_density', defaults.gps_vxy_noise_density),
            gps_vz_noise_density=d.get('gps_vz_noise_density', defaults.gps_vz_noise_density),
            gps_correlation_time=d.get('gps_correlation_time', defaults.gps_correlation_time),
            baro_drift_pa_per_sec=d.get('baro_drift_pa_per_sec', defaults.baro_drift_pa_per_sec),
            mag_noise_density=d.get('mag_noise_density', defaults.mag_noise_density),
            mag_random_walk=d.get('mag_random_walk', defaults.mag_random_walk),
            mag_bias_correlation_time=d.get(
                'mag_bias_correlation_time', defaults.mag_bias_correlation_time
            ),
            localization_mode=mode,
        )


MAVLINK_UPDATE_RATE: float = 500.0  # Hz — hardcoded, not configurable
MAVLINK_CONNECTION_TYPE: str = 'tcpin'  # Isaac Sim connects as TCP client to PX4


@dataclass
class MavlinkParams:
    """MAVLink communication parameters."""

    base_port: int = 4560
    connection_ip: str = 'localhost'
    num_rotors: int = 4

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MavlinkParams':
        """
        Construct from a dict, falling back to dataclass defaults for missing keys.

        Note: base_port, connection_ip and num_rotors are not YAML-configurable;
        they are set programmatically (from CLI args and rotor_positions length).
        """
        return cls()


@dataclass
class GPSOrigin:
    """GPS origin for the simulation. Override via the ``gps_origin`` block in the YAML config."""

    lat: float = 47.24926  # deg
    lon: float = -1.54844  # deg
    alt: float = 488.0  # m AMSL

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'GPSOrigin':
        """Construct from a dict, falling back to dataclass defaults for missing keys."""
        defaults = cls()
        return cls(
            lat=d.get('lat', defaults.lat),
            lon=d.get('lon', defaults.lon),
            alt=d.get('alt', defaults.alt),
        )


def load_config_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Read a YAML config file and return populated config dataclass instances.

    Returns a dict with keys 'quadrotor', 'sensor', 'mavlink', 'gps_origin',
    each being an instance of the corresponding dataclass.  Any key absent from
    the YAML falls back to the dataclass default.
    """
    import yaml  # stdlib

    with open(yaml_path, 'r') as f:
        loaded_data = yaml.safe_load(f)
        data: Dict[str, Any] = loaded_data if isinstance(loaded_data, dict) else {}

    return {
        'quadrotor': QuadrotorParams.from_dict(data.get('quadrotor', {})),
        'sensor': SensorParams.from_dict(data.get('sensor', {})),
        'mavlink': MavlinkParams.from_dict(data.get('mavlink', {})),
        'gps_origin': GPSOrigin.from_dict(data.get('gps_origin', {})),
    }
