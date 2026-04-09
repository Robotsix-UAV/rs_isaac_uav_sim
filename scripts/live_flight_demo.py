#!/usr/bin/env python3
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
Live flight demo: spawn Isaac Sim + PX4 SITL and auto-take-off.

Spawns one Isaac Sim window (or headless) running scene_mavlink_sim
alongside one PX4 SITL instance, connects a GCS MAVLink client over
UDP, waits for the EKF + arming checks to settle, arms PX4, switches
to AUTO.TAKEOFF, then sits in hover for the requested duration so the
end-to-end pipeline can be observed without any manual interaction.

Requires:
    DISPLAY            host X server (the docker-compose forwards it)
    ISAAC_SIM_PYTHON   path to Isaac Sim's python binary
    PX4_SITL_BUILD_DIR path to the PX4 SITL build dir (contains bin/px4)

Run:
    python3 scripts/live_flight_demo.py [--altitude 5.0] [--hover-seconds 60]

Press Ctrl-C at any time to land + tear everything down.
"""

import argparse
import os
import signal
import subprocess
import sys
import time

# Resolve the package root from this script's location: scripts/ is a
# direct child of the package directory.
PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCENE_SCRIPT = os.path.join(PKG_DIR, 'isaac_run', 'scene_mavlink_sim.py')
CONFIG_FILE = os.path.join(PKG_DIR, 'config', 'px4_iris.yaml')

# The GCS client lives in the package's test mocks; reuse it directly.
sys.path.insert(0, os.path.join(PKG_DIR, 'test'))
from mocks.gcs_mavlink_client import GCSMavlinkClient  # noqa: E402


def _kill_group(proc: subprocess.Popen, signum: int = signal.SIGTERM) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signum)
    except ProcessLookupError:
        pass


def _check_env() -> tuple[str, str]:
    isaac_python = os.environ.get('ISAAC_SIM_PYTHON', '')
    if not isaac_python or not os.path.isfile(isaac_python):
        sys.exit(
            'ISAAC_SIM_PYTHON is not set or does not point at a real file. '
            'Source the workspace env or run inside the workspace container.'
        )
    px4_dir = os.environ.get('PX4_SITL_BUILD_DIR', '')
    px4_bin = os.path.join(px4_dir, 'bin', 'px4')
    if not px4_dir or not os.path.isfile(px4_bin):
        sys.exit(
            'PX4_SITL_BUILD_DIR is not set or does not contain bin/px4. '
            'The workspace Dockerfile builds PX4 at /opt/PX4-Autopilot/'
            'build/px4_sitl_default.'
        )
    return isaac_python, px4_dir


def _start_isaac_sim(isaac_python: str, headless: bool) -> subprocess.Popen:
    cmd = [
        isaac_python,
        SCENE_SCRIPT,
        '--num_drones', '1',
        '--config', CONFIG_FILE,
    ]
    if headless:
        cmd.append('--headless')
    print(f'[demo] launching Isaac Sim ({SCENE_SCRIPT}) '
          f'headless={headless}')
    return subprocess.Popen(cmd, start_new_session=True)


def _start_px4_sitl(px4_dir: str) -> subprocess.Popen:
    px4_bin = os.path.join(px4_dir, 'bin', 'px4')
    rootfs = os.path.join(px4_dir, 'etc')
    work_dir = '/tmp/px4_sitl_1'
    os.makedirs(work_dir, exist_ok=True)
    env = os.environ.copy()
    env['PX4_SYS_AUTOSTART'] = '10016'
    env['HEADLESS'] = '1'
    env['PX4_INSTANCE'] = '1'
    env['PX4_PARAM_COM_ARM_HFLT_CHK'] = '0'
    env['PX4_PARAM_COM_MODE_ARM_CHK'] = '0'
    print(f'[demo] launching PX4 SITL ({px4_bin})')
    return subprocess.Popen(
        [px4_bin, '-i', '1', rootfs, '-w', work_dir],
        cwd=work_dir,
        env=env,
        start_new_session=True,
    )


def _wait_for_ready_to_arm(gcs: GCSMavlinkClient, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gcs.is_ready_to_arm():
            return True
        time.sleep(0.2)
    return False


def _wait_for_armed(gcs: GCSMavlinkClient, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if gcs.is_armed():
            return True
        time.sleep(0.1)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description='Live PX4 + Isaac Sim hover demo')
    parser.add_argument('--altitude', type=float, default=5.0,
                        help='Takeoff target altitude in meters (default: 5)')
    parser.add_argument('--hover-seconds', type=float, default=60.0,
                        help='Time to stay airborne after reaching target (default: 60)')
    parser.add_argument('--headless', action='store_true',
                        help='Run Isaac Sim without a GUI window')
    args = parser.parse_args()

    isaac_python, px4_dir = _check_env()

    sim = _start_isaac_sim(isaac_python, headless=args.headless)
    px4 = _start_px4_sitl(px4_dir)
    gcs: GCSMavlinkClient | None = None

    def shutdown(*_signals_unused: object) -> None:
        print('\n[demo] shutting down...')
        if gcs is not None:
            try:
                gcs.disarm()
            except Exception:
                pass
            try:
                gcs.disconnect()
            except Exception:
                pass
        for proc, name in ((px4, 'px4'), (sim, 'isaac_sim')):
            print(f'[demo]   killing {name}')
            _kill_group(proc, signal.SIGTERM)
        for proc in (px4, sim):
            try:
                proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                _kill_group(proc, signal.SIGKILL)
                proc.wait()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    try:
        print('[demo] connecting GCS to PX4 (UDP :14550) — '
              'allow up to ~2 min for Isaac Sim + PX4 boot...')
        gcs = GCSMavlinkClient(host='0.0.0.0', port=14550)
        gcs.connect(timeout=180.0)
        print('[demo] heartbeat received')

        print('[demo] waiting for PX4 to reach STANDBY (EKF + arming checks)...')
        if not _wait_for_ready_to_arm(gcs, timeout=120.0):
            raise RuntimeError('PX4 never reached STANDBY')
        print('[demo] PX4 ready to arm')

        gcs.arm()
        time.sleep(0.2)
        gcs.set_mode_auto_takeoff()
        if not _wait_for_armed(gcs, timeout=15.0):
            raise RuntimeError('PX4 did not report armed within 15 s')
        print(f'[demo] armed; AUTO.TAKEOFF to {args.altitude:.1f} m')

        # Sit in hover until either the timeout elapses or the user kills us.
        print(f'[demo] hovering for {args.hover_seconds:.0f} s — Ctrl-C to land')
        deadline = time.monotonic() + args.hover_seconds
        last_print = 0.0
        while time.monotonic() < deadline:
            time.sleep(0.5)
            now = time.monotonic()
            if now - last_print >= 5.0:
                pos = gcs.get_local_position()
                if pos is not None:
                    print(f'[demo]   z={pos["z"]:+.2f} m  '
                          f'vx={pos["vx"]:+.2f} vy={pos["vy"]:+.2f} '
                          f'vz={pos["vz"]:+.2f} m/s')
                last_print = now
    finally:
        shutdown()


if __name__ == '__main__':
    main()
