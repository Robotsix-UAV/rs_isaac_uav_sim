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

"""Unit tests for VehicleState frame-conversion helpers in state.py."""

import numpy as np
from numpy.testing import assert_allclose

from rs_isaac_uav_sim.sim.state import (
    flu_to_frd,
    rot_ENU_to_NED,
    VehicleState,
)
from scipy.spatial.transform import Rotation


def test_rot_enu_to_ned_east():
    # East vector [1,0,0] in ENU should be East [0,1,0] in NED
    result = rot_ENU_to_NED.apply(np.array([1.0, 0.0, 0.0]))
    assert_allclose(result, [0.0, 1.0, 0.0], atol=1e-6)


def test_rot_enu_to_ned_north():
    # North vector [0,1,0] in ENU should be North [1,0,0] in NED
    result = rot_ENU_to_NED.apply(np.array([0.0, 1.0, 0.0]))
    assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)


def test_rot_enu_to_ned_up():
    # Up vector [0,0,1] in ENU should map to [0,0,-1] in NED (negative Down)
    result = rot_ENU_to_NED.apply(np.array([0.0, 0.0, 1.0]))
    assert_allclose(result, [0.0, 0.0, -1.0], atol=1e-6)


def test_flu_to_frd():
    # FLU->FRD negates y and z
    assert_allclose(
        flu_to_frd(np.array([1.0, 2.0, 3.0])),
        [1.0, -2.0, -3.0],
        atol=1e-6
    )


def test_get_angular_velocity_frd():
    state = VehicleState()
    state.angular_velocity = np.array([1.0, 2.0, 3.0])  # FLU
    assert_allclose(
        state.get_angular_velocity_frd(),
        [1.0, -2.0, -3.0],
        atol=1e-6
    )


def test_get_linear_velocity_ned():
    state = VehicleState()
    state.linear_velocity = np.array([1.0, 0.0, 0.0])  # East in ENU
    # East in ENU should be East = [0,1,0] in NED
    assert_allclose(
        state.get_linear_velocity_ned(),
        [0.0, 1.0, 0.0],
        atol=1e-6
    )


def test_get_position_ned():
    state = VehicleState()
    state.position = np.array([3.0, 4.0, 5.0])  # [east, north, up] in ENU
    # NED: [north, east, down] = [4, 3, -5]
    assert_allclose(state.get_position_ned(), [4.0, 3.0, -5.0], atol=1e-6)


def test_attitude_ned_frd_facing_north():
    # Vehicle facing North in ENU: yaw=90 deg CCW about Z
    # quaternion [qx,qy,qz,qw] = [0, 0, sin(pi/4), cos(pi/4)]
    state = VehicleState()
    state.attitude = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    # When facing North, FRD body aligns with NED -> identity attitude
    q = state.get_attitude_ned_frd()
    rot = Rotation.from_quat(q)
    # Identity rotation: forward [1,0,0] in FRD stays [1,0,0] in NED (North)
    assert_allclose(rot.apply([1.0, 0.0, 0.0]), [1.0, 0.0, 0.0], atol=1e-6)
    assert_allclose(rot.apply([0.0, 1.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
    assert_allclose(rot.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-6)


def test_attitude_ned_frd_facing_east():
    # Vehicle at identity attitude [0,0,0,1]: facing East in ENU
    state = VehicleState()
    state.attitude = np.array([0.0, 0.0, 0.0, 1.0])
    q = state.get_attitude_ned_frd()
    rot = Rotation.from_quat(q)
    # Forward [1,0,0] in FRD should point East = [0,1,0] in NED
    assert_allclose(rot.apply([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-6)
    # Right [0,1,0] in FRD should point South = [-1,0,0] in NED
    assert_allclose(rot.apply([0.0, 1.0, 0.0]), [-1.0, 0.0, 0.0], atol=1e-6)
    # Down [0,0,1] in FRD should point Down = [0,0,1] in NED
    assert_allclose(rot.apply([0.0, 0.0, 1.0]), [0.0, 0.0, 1.0], atol=1e-6)


def test_get_linear_body_velocity_ned_frd_facing_north():
    # Vehicle facing North, moving North: body forward velocity is positive X
    state = VehicleState()
    state.attitude = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
    state.linear_velocity = np.array([0.0, 1.0, 0.0])  # North in ENU
    result = state.get_linear_body_velocity_ned_frd()
    # Forward (body X in FRD) should have positive component
    assert_allclose(result, [1.0, 0.0, 0.0], atol=1e-6)
