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
"""
Integration tests for MAVLink HIL scenarios.

Each test method starts a fresh headless Isaac Sim instance in setUp and
terminates it in tearDown, providing full isolation between tests.

Physical parameters (from rs_isaac_uav_sim/sim/config.py defaults):
    mass       = 1.0 kg
    ct         = 1.1e-05  (thrust coefficient: T = ct * omega^2)
    num_rotors = 4
    omega_min  = 0.0  rad/s
    omega_max  = 1000.0 rad/s

Hover omega = sqrt(mass * g / (num_rotors * ct))
           = sqrt(1.0 * 9.81 / (4 * 1.1e-5))
           ≈ 472.2 rad/s
Hover ctrl = hover_omega / omega_max ≈ 0.4722

Liftoff ctrl (115% thrust, omega scaled by sqrt(1.15)):
    = hover_omega * sqrt(1.15) / omega_max ≈ 0.5065
"""

import math
import os
import signal
import subprocess
import sys
import time
import unittest

# Physics constants
_MASS = 1.0
_G = 9.81
_CT = 1.1e-05
_NUM_ROTORS = 4
_OMEGA_MAX = 1000.0

_HOVER_OMEGA = math.sqrt(_MASS * _G / (_NUM_ROTORS * _CT))
_CTRL_HOVER = _HOVER_OMEGA / _OMEGA_MAX
_CTRL_LIFTOFF = _HOVER_OMEGA * math.sqrt(1.15) / _OMEGA_MAX


def _headless_args():
    """Return ['--headless'] unless RS_ISAAC_TEST_HEADLESS is set to a falsy value.

    Set RS_ISAAC_TEST_HEADLESS=0 (or 'false'/'no') to launch Isaac Sim with the
    GUI for local debugging. CI keeps the default (headless).
    """
    val = os.environ.get('RS_ISAAC_TEST_HEADLESS', '1').strip().lower()
    if val in ('0', 'false', 'no', ''):
        return []
    return ['--headless']


def _resolve_isaac_python():
    candidate = os.environ.get('ISAAC_SIM_PYTHON', '')
    if candidate and os.path.isfile(candidate):
        return candidate
    fallback = '/home/robotsix-docker/IsaacSim/_build/linux-x86_64/release/python.sh'
    if os.path.isfile(fallback):
        return fallback
    return None


def _resolve_sim_script():
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'isaac_run', 'scene_mavlink_sim.py',
    )


def _wait_for_state(autopilot, timeout=10.0):
    """Poll get_hil_state() until non-None or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        state = autopilot.get_hil_state()
        if state is not None:
            return state
        time.sleep(0.05)
    return None


def _wait_for_zero_velocity(autopilot, vel_tol=10, ang_tol=0.1, timeout=10.0):
    """Poll until all velocities and angular rates are within tolerance."""
    deadline = time.monotonic() + timeout
    state = None
    while time.monotonic() < deadline:
        state = autopilot.get_hil_state()
        if state is not None:
            if (
                abs(state['vx']) <= vel_tol
                and abs(state['vy']) <= vel_tol
                and abs(state['vz']) <= vel_tol
                and abs(state['rollspeed']) <= ang_tol
                and abs(state['pitchspeed']) <= ang_tol
                and abs(state['yawspeed']) <= ang_tol
            ):
                return state
        time.sleep(0.05)
    return state


class TestMavlinkScenarios(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._isaac_python = _resolve_isaac_python()
        if cls._isaac_python is None:
            raise unittest.SkipTest(
                'Isaac Sim python binary not found. Set ISAAC_SIM_PYTHON or ensure '
                '/home/robotsix-docker/IsaacSim/_build/linux-x86_64/release/python.sh exists.'
            )
        cls._sim_script = _resolve_sim_script()
        if not os.path.isfile(cls._sim_script):
            raise unittest.SkipTest(f'Simulation script not found at {cls._sim_script}')

    def setUp(self):
        """Start a fresh Isaac Sim subprocess for this test."""
        # start_new_session=True puts the child in its own process group so
        # tearDown can kill the entire group (python.sh + Isaac Sim grandchild).
        self._sim_process = subprocess.Popen(
            [self._isaac_python, self._sim_script, *_headless_args(), '--num_drones', '1'],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self._autopilot = None

    def tearDown(self):
        """Disconnect autopilot and terminate the simulation process group."""
        if self._autopilot is not None:
            try:
                self._autopilot.disconnect()
            except Exception:
                pass
            self._autopilot = None
        try:
            os.killpg(os.getpgid(self._sim_process.pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            self._sim_process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(self._sim_process.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass
            self._sim_process.wait()

    def _connect_autopilot(self, port=4561, timeout=60):
        """Create and connect a MockMavlinkAutopilot, storing it for tearDown."""
        sys.path.insert(
            0,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
        )
        from mocks.mock_mavlink_autopilot import MockMavlinkAutopilot
        ap = MockMavlinkAutopilot(port=port)
        ap.connect(timeout=timeout)
        self._autopilot = ap
        return ap

    def test_00_initial_state(self):
        _VEL_TOL = 10    # cm/s
        _ANG_TOL = 0.1   # rad/s

        autopilot = self._connect_autopilot()

        first = _wait_for_state(autopilot, timeout=5.0)
        self.assertIsNotNone(first, 'No HIL_STATE_QUATERNION received within 5s')

        state = _wait_for_zero_velocity(
            autopilot, vel_tol=_VEL_TOL, ang_tol=_ANG_TOL, timeout=10.0,
        )
        self.assertIsNotNone(state, 'No HIL_STATE_QUATERNION during settling wait')

        self.assertAlmostEqual(state['vx'], 0, delta=_VEL_TOL,
                               msg=f"Settled vx should be ~0 cm/s, got {state['vx']}")
        self.assertAlmostEqual(state['vy'], 0, delta=_VEL_TOL,
                               msg=f"Settled vy should be ~0 cm/s, got {state['vy']}")
        self.assertAlmostEqual(state['vz'], 0, delta=_VEL_TOL,
                               msg=f"Settled vz should be ~0 cm/s, got {state['vz']}")
        self.assertAlmostEqual(state['rollspeed'], 0, delta=_ANG_TOL,
                               msg=f'Settled rollspeed ~0 rad/s, got {state["rollspeed"]}')
        self.assertAlmostEqual(state['pitchspeed'], 0, delta=_ANG_TOL,
                               msg=f'Settled pitchspeed ~0 rad/s, got {state["pitchspeed"]}')
        self.assertAlmostEqual(state['yawspeed'], 0, delta=_ANG_TOL,
                               msg=f"Settled yawspeed should be ~0 rad/s, got {state['yawspeed']}")

        q = state['attitude_quaternion']
        qnorm = math.sqrt(sum(x * x for x in q))
        self.assertAlmostEqual(qnorm, 1.0, delta=0.01,
                               msg=f'Attitude quaternion norm should be 1.0, got {qnorm:.6f}')

        self.assertGreater(state['alt'], 0,
                           f"Altitude should be positive (WGS84 mm), got {state['alt']}mm")

    def test_01_liftoff(self):
        autopilot = self._connect_autopilot()

        first = _wait_for_state(autopilot, timeout=5.0)
        self.assertIsNotNone(first, 'No HIL_STATE_QUATERNION received within 5s')

        state = _wait_for_zero_velocity(autopilot, vel_tol=10, ang_tol=0.1, timeout=15.0)
        self.assertIsNotNone(state, 'No HIL_STATE_QUATERNION during settling wait')

        autopilot.arm()
        autopilot.set_controls([_CTRL_LIFTOFF] * 4)

        # Open-loop hover thrust over 2 s would let attitude perturbations
        # accumulate enough horizontal drift to clip the |vx|/|vy| < 50 cm/s
        # sanity bounds below. 1.0 s is plenty for vz to clear the -30 cm/s
        # threshold (≈-150 cm/s expected at 0.15 g) while keeping drift small.
        time.sleep(1.0)

        state = autopilot.get_hil_state()
        self.assertIsNotNone(state, 'No HIL_STATE_QUATERNION after liftoff command')

        autopilot.disarm()
        time.sleep(0.1)

        # vz in NED cm/s: negative means upward
        self.assertLess(
            state['vz'], -30,
            f"Expected upward velocity (vz < -30 cm/s NED), got {state['vz']} cm/s",
        )
        self.assertLess(
            abs(state['vx']), 50,
            f"Expected little horizontal velocity (|vx| < 50 cm/s), got {state['vx']} cm/s",
        )
        self.assertLess(
            abs(state['vy']), 50,
            f"Expected little horizontal velocity (|vy| < 50 cm/s), got {state['vy']} cm/s",
        )
        self.assertLess(
            abs(state['rollspeed']), 0.3,
            f"Expected no roll rotation (|rollspeed| < 0.3 rad/s), got {state['rollspeed']} rad/s",
        )
        self.assertLess(
            abs(state['pitchspeed']), 0.3,
            f'Expected |pitchspeed| < 0.3 rad/s, got {state["pitchspeed"]} rad/s',
        )
        self.assertLess(
            abs(state['yawspeed']), 0.3,
            f"Expected no yaw rotation (|yawspeed| < 0.3 rad/s), got {state['yawspeed']} rad/s",
        )
