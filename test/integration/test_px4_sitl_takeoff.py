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
Integration test: PX4 SITL arm + takeoff with GCS MAVLink client.

Architecture::

    Isaac Sim (HIL TCP :4561) <--HIL_SENSOR/HIL_ACTUATOR_CONTROLS--> PX4 SITL
                                                                            |
                                                                       UDP :14540
                                                                            |
                                                                      GCSMavlinkClient
                                                                  (arm + takeoff + verify)

Test steps:
    1. Arm PX4 SITL and switch to AUTO.TAKEOFF.
    2. Wait until drone reaches z < -2.0 (NED, negative = up).
    3. Check horizontal drift < 5m.
    4. Wait for PX4 to transition to AUTO.LOITER (takeoff complete).
    5. Collect position/velocity samples during hover and assert stabilization.

Prerequisites (skip if absent):
    ISAAC_SIM_PYTHON  – path to Isaac Sim python.sh binary
    PX4_SITL_BUILD_DIR – path to PX4 SITL build directory (contains bin/px4 and etc/)
"""

import os
import signal
import subprocess
import sys
import time
import unittest

# PX4 custom mode: AUTO.LOITER (main mode 4, sub mode 3)
PX4_CUSTOM_MODE_AUTO_LOITER = (3 << 24) | (4 << 16)


def _headless_args() -> list[str]:
    """Return ['--headless'] unless RS_ISAAC_TEST_HEADLESS is set to a falsy value.

    Set RS_ISAAC_TEST_HEADLESS=0 (or 'false'/'no') to launch Isaac Sim with the
    GUI for local debugging. CI keeps the default (headless).
    """
    val = os.environ.get('RS_ISAAC_TEST_HEADLESS', '1').strip().lower()
    if val in ('0', 'false', 'no', ''):
        return []
    return ['--headless']


def _resolve_isaac_python() -> str | None:
    candidate = os.environ.get('ISAAC_SIM_PYTHON', '')
    if candidate and os.path.isfile(candidate):
        return candidate
    fallback = '/home/robotsix-docker/IsaacSim/_build/linux-x86_64/release/python.sh'
    if os.path.isfile(fallback):
        return fallback
    return None


def _resolve_sim_script() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'isaac_run', 'scene_mavlink_sim.py',
    )


def _resolve_config() -> str:
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '..', '..', 'config', 'px4_iris.yaml',
    )


def _resolve_px4_binary(build_dir: str) -> str:
    return os.path.join(build_dir, 'bin', 'px4')


def _resolve_px4_rootfs(build_dir: str) -> str:
    return os.path.join(build_dir, 'etc')


def _kill_process_group(proc: subprocess.Popen, signum: int = signal.SIGTERM) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signum)
    except ProcessLookupError:
        pass


def _wait_for_position(gcs_client, timeout: float = 90.0) -> dict | None:
    """Poll until LOCAL_POSITION_NED is available (indicates EKF convergence)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pos = gcs_client.get_local_position()
        if pos is not None:
            return pos
        time.sleep(0.2)
    return None


def _wait_for_ready_to_arm(gcs_client, timeout: float = 60.0) -> bool:
    """Poll until PX4 reports MAV_STATE_STANDBY (ready to arm)."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gcs_client.is_ready_to_arm():
            return True
        time.sleep(0.2)
    return False


def _wait_for_armed(gcs_client, timeout: float = 30.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gcs_client.is_armed():
            return True
        time.sleep(0.1)
    return False


def _wait_for_loiter(gcs_client, timeout=30.0):
    """Poll heartbeat custom_mode until PX4 reports AUTO.LOITER."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        hb = gcs_client.get_px4_heartbeat()
        if hb is not None and hb['custom_mode'] == PX4_CUSTOM_MODE_AUTO_LOITER:
            return True
        time.sleep(0.1)
    return False


def _collect_hover_samples(gcs_client, duration=15.0, interval=0.5):
    """Sample LOCAL_POSITION_NED every interval seconds for duration seconds."""
    samples = []
    deadline = time.monotonic() + duration
    while time.monotonic() < deadline:
        pos = gcs_client.get_local_position()
        if pos is not None:
            samples.append(pos)
        time.sleep(interval)
    return samples


def _wait_for_altitude(gcs_client, z_threshold: float, timeout: float = 30.0) -> dict | None:
    """Poll LOCAL_POSITION_NED until z < z_threshold (NED: negative = up) or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        pos = gcs_client.get_local_position()
        if pos is not None and pos['z'] < z_threshold:
            return pos
        time.sleep(0.1)
    return gcs_client.get_local_position()


class TestPX4SitlTakeoff(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # --- Check ISAAC_SIM_PYTHON ---
        cls._isaac_python = _resolve_isaac_python()
        if cls._isaac_python is None:
            raise unittest.SkipTest(
                'Isaac Sim python binary not found. '
                'Set ISAAC_SIM_PYTHON or ensure the default path exists.'
            )

        # --- Check ISAAC_SIM_PYTHON ---
        px4_build_dir = os.environ.get('PX4_SITL_BUILD_DIR', '')
        if not px4_build_dir:
            raise unittest.SkipTest(
                'PX4_SITL_BUILD_DIR environment variable is not set. '
                'Set it to the PX4 SITL build directory (must contain bin/px4 and etc/).'
            )

        cls._px4_binary = _resolve_px4_binary(px4_build_dir)
        if not os.path.isfile(cls._px4_binary):
            raise unittest.SkipTest(
                f'PX4 binary not found at {cls._px4_binary}. '
                'Check PX4_SITL_BUILD_DIR.'
            )

        cls._px4_rootfs = _resolve_px4_rootfs(px4_build_dir)
        if not os.path.isdir(cls._px4_rootfs):
            raise unittest.SkipTest(
                f'PX4 rootfs (etc/) not found at {cls._px4_rootfs}. '
                'Check PX4_SITL_BUILD_DIR.'
            )

        cls._sim_script = _resolve_sim_script()
        if not os.path.isfile(cls._sim_script):
            raise unittest.SkipTest(
                f'Isaac Sim scene script not found at {cls._sim_script}.'
            )

        cls._config = _resolve_config()
        if not os.path.isfile(cls._config):
            raise unittest.SkipTest(
                f'PX4 iris config not found at {cls._config}.'
            )

    def setUp(self):
        """Launch Isaac Sim and PX4 SITL subprocesses for this test."""
        # Isaac Sim: TCP server on port 4561 (PX4 instance 1 → 4560 + 1)
        self._sim_process = subprocess.Popen(
            [
                self._isaac_python,
                self._sim_script,
                *_headless_args(),
                '--num_drones', '1',
                '--config', self._config,
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        # PX4 SITL: instance 1, GCS MAVLink on UDP 14550, HIL on TCP 4561
        px4_work_dir = '/tmp/px4_sitl_1'
        os.makedirs(px4_work_dir, exist_ok=True)

        px4_env = os.environ.copy()
        px4_env['PX4_SYS_AUTOSTART'] = '10016'
        px4_env['HEADLESS'] = '1'
        px4_env['PX4_INSTANCE'] = '1'
        # Disable health/mode arming checks at startup (no RC expected in HIL SITL)
        px4_env['PX4_PARAM_COM_ARM_HFLT_CHK'] = '0'
        px4_env['PX4_PARAM_COM_MODE_ARM_CHK'] = '0'

        self._px4_process = subprocess.Popen(
            [self._px4_binary, '-i', '1', self._px4_rootfs, '-w', px4_work_dir],
            cwd=px4_work_dir,
            env=px4_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

        self._gcs_client = None

    def tearDown(self):
        """Disconnect GCS client and kill both subprocesses cleanly."""
        if self._gcs_client is not None:
            try:
                self._gcs_client.disconnect()
            except Exception:
                pass
            self._gcs_client = None

        for proc in (self._px4_process, self._sim_process):
            _kill_process_group(proc, signal.SIGTERM)
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _kill_process_group(proc, signal.SIGKILL)
                proc.wait()

    def _connect_gcs(self, timeout: float = 120.0):
        """Import and connect GCSMavlinkClient, storing it for tearDown."""
        sys.path.insert(
            0,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'),
        )
        from mocks.gcs_mavlink_client import GCSMavlinkClient
        client = GCSMavlinkClient(host='0.0.0.0', port=14550)
        client.connect(timeout=timeout)
        self._gcs_client = client
        return client

    def test_arm_and_takeoff(self):
        # Connect GCS to PX4 (120s budget covers Isaac Sim + PX4 boot + HIL link)
        gcs = self._connect_gcs(timeout=120)

        # Confirm initial heartbeat was received
        hb = gcs.get_px4_heartbeat()
        self.assertIsNotNone(hb, 'No HEARTBEAT received from PX4')

        # Wait for LOCAL_POSITION_NED to appear (EKF converged, GPS lock)
        ready = _wait_for_position(gcs, timeout=90.0)
        self.assertIsNotNone(
            ready,
            'No LOCAL_POSITION_NED from PX4 within 90s (EKF not converged)'
        )

        # Wait for PX4 to report MAV_STATE_STANDBY (all arming checks passed)
        standby = _wait_for_ready_to_arm(gcs, timeout=60.0)
        self.assertTrue(standby, 'PX4 did not reach STANDBY state within 60s')

        # Arm, then immediately switch to AUTO.TAKEOFF
        gcs.arm()
        time.sleep(0.2)
        gcs.set_mode_auto_takeoff()

        armed = _wait_for_armed(gcs, timeout=30.0)
        self.assertTrue(
            armed,
            'PX4 did not report armed state within 30s after ARM command',
        )

        # --- Wait for climb: LOCAL_POSITION_NED z < -2.0 (NED, negative = up) ---
        pos = _wait_for_altitude(gcs, z_threshold=-2.0, timeout=30.0)

        self.assertIsNotNone(pos, 'No LOCAL_POSITION_NED received from PX4')
        self.assertLess(
            pos['z'],
            -2.0,
            f"Drone did not climb to 2m (z < -2.0 NED). Got z={pos['z']:.3f}m",
        )

        # --- Horizontal drift check ---
        self.assertLess(
            abs(pos['x']),
            5.0,
            f"Excessive x drift: {pos['x']:.3f}m (expected < 5.0m)",
        )
        self.assertLess(
            abs(pos['y']),
            5.0,
            f"Excessive y drift: {pos['y']:.3f}m (expected < 5.0m)",
        )

        # --- Wait for AUTO.LOITER (takeoff complete, drone holding position) ---
        in_loiter = _wait_for_loiter(gcs, timeout=30.0)
        self.assertTrue(
            in_loiter,
            'PX4 did not transition to AUTO.LOITER within 30s after takeoff'
        )

        # --- Collect stabilization samples during hover ---
        samples = _collect_hover_samples(gcs, duration=15.0, interval=0.5)
        self.assertGreater(len(samples), 5, 'Insufficient hover samples collected')

        # Extract altitude (z) and velocities
        z_values = [s['z'] for s in samples]
        vx_values = [s['vx'] for s in samples]
        vy_values = [s['vy'] for s in samples]
        vz_values = [s['vz'] for s in samples]

        # Altitude stability: z should stay < -1.5 (at least 1.5m up) throughout hover
        self.assertTrue(
            all(z < -1.5 for z in z_values),
            f'Drone lost altitude during hover. Min z={min(z_values):.3f}m '
            f'(expected all < -1.5)',
        )

        # Altitude stability: z should not drift above the takeoff target
        # by more than 2m (no runaway climb)
        self.assertTrue(
            all(z > -8.0 for z in z_values),
            f'Drone climbed too high during hover. Min z={min(z_values):.3f}m '
            f'(expected all > -8.0)',
        )

        # Velocity stabilization: average absolute velocity should be small
        avg_vx = sum(abs(v) for v in vx_values) / len(vx_values)
        avg_vy = sum(abs(v) for v in vy_values) / len(vy_values)
        avg_vz = sum(abs(v) for v in vz_values) / len(vz_values)

        self.assertLess(avg_vx, 1.0, f'Excessive average vx during hover: {avg_vx:.3f} m/s')
        self.assertLess(avg_vy, 1.0, f'Excessive average vy during hover: {avg_vy:.3f} m/s')
        self.assertLess(avg_vz, 0.5, f'Excessive average vz during hover: {avg_vz:.3f} m/s')
