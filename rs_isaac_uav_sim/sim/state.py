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

"""Vehicle state container with ENU/FLU <-> NED/FRD conversion helpers."""

import numpy as np
from scipy.spatial.transform import Rotation

# Rotation from ENU inertial to NED inertial frame
# +PI/2 about Z then +PI about X (symmetric: ENU->NED == NED->ENU)
_q_ENU_to_NED = np.array([0.70711, 0.70711, 0.0, 0.0])  # [qx,qy,qz,qw]
rot_ENU_to_NED = Rotation.from_quat(_q_ENU_to_NED)

# FLU <-> FRD is a 180° rotation about X: vectors transform as [x, -y, -z].
# For quaternions, right-multiplying by [1,0,0,0] (the 180°-about-X quaternion)
# maps [qx, qy, qz, qw] -> [qw, qz, -qy, -qx] (derived analytically).
# No Rotation object is needed; conversions are inlined for clarity and speed.


def flu_to_frd(v: np.ndarray) -> np.ndarray:
    """Convert a vector from FLU body frame to FRD body frame."""
    return np.array([v[0], -v[1], -v[2]])


class VehicleState:
    """
    Container for a single drone's kinematic state in ENU/FLU convention.

    Isaac Sim uses ENU (position) and FLU (body frame). PX4/MAVLink expects
    NED/FRD. This class stores the canonical ENU/FLU state and provides
    conversion accessors.
    """

    def __init__(self):
        # ENU world frame
        self.position = np.zeros(3)  # [x_east, y_north, z_up] m
        self.attitude = np.array([0.0, 0.0, 0.0, 1.0])  # [qx,qy,qz,qw] FLU-ENU
        self.linear_velocity = np.zeros(3)  # [vx,vy,vz] ENU m/s
        self.angular_velocity = np.zeros(3)  # [wx,wy,wz] FLU body rad/s

    @property
    def linear_body_velocity(self) -> np.ndarray:
        """Linear velocity in FLU body frame."""
        rot = Rotation.from_quat(self.attitude)
        return rot.inv().apply(self.linear_velocity)

    def get_attitude_ned_frd(self) -> np.ndarray:
        """Attitude quaternion [qx,qy,qz,qw] in NED-FRD convention."""
        qx, qy, qz, qw = self.attitude
        # ENU→FRD: right-multiply by 180°-about-X → [qw, qz, -qy, -qx]
        att_frd_enu = Rotation.from_quat([qw, qz, -qy, -qx])
        return (rot_ENU_to_NED * att_frd_enu).as_quat()

    def get_angular_velocity_frd(self) -> np.ndarray:
        """Angular velocity in FRD body frame."""
        w = self.angular_velocity
        return np.array([w[0], -w[1], -w[2]])

    def get_linear_velocity_ned(self) -> np.ndarray:
        """Linear velocity in NED inertial frame."""
        return rot_ENU_to_NED.apply(self.linear_velocity)

    def get_linear_body_velocity_ned_frd(self) -> np.ndarray:
        """Linear velocity in FRD body frame (NED-referenced rotation)."""
        att_ned_frd = Rotation.from_quat(self.get_attitude_ned_frd())
        vel_ned = self.get_linear_velocity_ned()
        return att_ned_frd.inv().apply(vel_ned)

    def get_position_ned(self) -> np.ndarray:
        """Position in NED inertial frame."""
        p = self.position
        return np.array([p[1], p[0], -p[2]])
