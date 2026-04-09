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

import threading
import time

from pymavlink import mavutil

# MAVLink mode flags
MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 0x01
MAV_MODE_FLAG_SAFETY_ARMED = 0x80

# MAVLink commands
MAV_CMD_COMPONENT_ARM_DISARM = 400
MAV_CMD_NAV_TAKEOFF = 22
MAV_CMD_DO_SET_MODE = 176

# PX4 custom mode encoding: (sub_mode << 24) | (main_mode << 16)
# PX4 main modes: 1=MANUAL, 2=ALTCTL, 3=POSCTL, 4=AUTO, 5=ACRO, 6=OFFBOARD, 7=STABILIZED
# PX4 AUTO sub_modes: 1=READY, 2=TAKEOFF, 3=LOITER, 4=MISSION, 5=RTL, 6=LAND
_PX4_MAIN_MODE_AUTO = 4
_PX4_SUB_MODE_AUTO_TAKEOFF = 2
PX4_CUSTOM_MODE_AUTO_TAKEOFF = (
    (_PX4_SUB_MODE_AUTO_TAKEOFF << 24) | (_PX4_MAIN_MODE_AUTO << 16)
)


class GCSMavlinkClient:
    """
    GCS-side MAVLink client that connects to PX4 SITL via UDP.

    Connects to PX4's standard GCS MAVLink endpoint (UDP port 14540).
    Provides methods to arm, take off, and read position/odometry feedback.
    All message reception runs in a background daemon thread.
    """

    def __init__(self, host='0.0.0.0', port=14550):
        self._host = host
        self._port = port

        self._mav = None
        self._recv_thread = None
        self._stop_event = threading.Event()

        self._recv_lock = threading.Lock()
        self._heartbeat = None
        self._local_position = None
        self._odometry = None

        self._heartbeat_event = threading.Event()
        self._ready_for_takeoff_event = threading.Event()

        self._send_thread = None
        self._last_heartbeat_sent = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def connect(self, timeout=120.0) -> None:
        """Open the UDP connection to PX4 and wait for the first HEARTBEAT."""
        self._mav = mavutil.mavlink_connection(
            f'udpin:{self._host}:{self._port}',
            source_system=255,
            source_component=190,
        )
        self._stop_event.clear()
        self._heartbeat_event.clear()
        self._ready_for_takeoff_event.clear()

        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            name='gcs-mav-recv',
            daemon=True,
        )
        self._send_thread = threading.Thread(
            target=self._send_loop,
            name='gcs-mav-send',
            daemon=True,
        )
        self._recv_thread.start()
        self._send_thread.start()

        if not self.wait_for_heartbeat(timeout=timeout):
            self.disconnect()
            raise RuntimeError(
                f'GCSMavlinkClient: no HEARTBEAT from PX4 within {timeout}s '
                f'at udp:{self._host}:{self._port}'
            )

    def disconnect(self) -> None:
        """Stop background threads and close the MAVLink connection."""
        self._stop_event.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=3.0)
            self._send_thread = None
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=3.0)
            self._recv_thread = None
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None

    def wait_for_heartbeat(self, timeout=30.0) -> bool:
        """Block until a HEARTBEAT from PX4 has been received or timeout expires."""
        return self._heartbeat_event.wait(timeout=timeout)

    def get_px4_heartbeat(self) -> dict | None:
        """Return a copy of the latest HEARTBEAT fields, or None."""
        with self._recv_lock:
            return dict(self._heartbeat) if self._heartbeat is not None else None

    def get_local_position(self) -> dict | None:
        """Return a copy of the latest LOCAL_POSITION_NED fields, or None."""
        with self._recv_lock:
            return (
                dict(self._local_position)
                if self._local_position is not None
                else None
            )

    def get_odometry(self) -> dict | None:
        """Return a copy of the latest ODOMETRY fields, or None."""
        with self._recv_lock:
            return dict(self._odometry) if self._odometry is not None else None

    def arm(self, force: bool = False) -> None:
        """
        Send MAV_CMD_COMPONENT_ARM_DISARM (arm=1) to PX4.

        Args:
        ----
        force:
            If True, bypass preflight checks using the PX4 force-arm magic
            value (param2=21196). Use only in simulation/testing.

        """
        self._send_command_long(
            command=MAV_CMD_COMPONENT_ARM_DISARM,
            param1=1.0,   # 1 = arm
            param2=21196.0 if force else 0.0,
        )

    def disarm(self) -> None:
        """Send MAV_CMD_COMPONENT_ARM_DISARM (arm=0) to PX4."""
        self._send_command_long(
            command=MAV_CMD_COMPONENT_ARM_DISARM,
            param1=0.0,  # 0 = disarm
        )

    def set_param_int(self, param_id: str, value: int) -> None:
        """Send PARAM_SET to PX4 to set an integer parameter."""
        if self._mav is None:
            return
        hb = self.get_px4_heartbeat()
        target_system = hb['system_id'] if hb is not None else 1
        self._mav.mav.param_set_send(
            target_system,
            1,
            param_id.encode('utf-8'),
            float(value),
            mavutil.mavlink.MAV_PARAM_TYPE_INT32,
        )

    def set_mode_auto_takeoff(self) -> None:
        """Switch PX4 to AUTO.TAKEOFF using PX4 custom mode encoding."""
        self._send_command_long(
            command=MAV_CMD_DO_SET_MODE,
            param1=float(MAV_MODE_FLAG_CUSTOM_MODE_ENABLED),
            param2=float(_PX4_MAIN_MODE_AUTO),
            param3=float(_PX4_SUB_MODE_AUTO_TAKEOFF),
        )

    def send_takeoff(self, altitude: float = 5.0) -> None:
        """Send MAV_CMD_NAV_TAKEOFF with the given altitude (meters)."""
        self._send_command_long(
            command=MAV_CMD_NAV_TAKEOFF,
            param7=altitude,
        )

    def is_armed(self) -> bool:
        """Return True if the latest HEARTBEAT shows the armed flag set."""
        hb = self.get_px4_heartbeat()
        if hb is None:
            return False
        return bool(hb['base_mode'] & MAV_MODE_FLAG_SAFETY_ARMED)

    def is_ready_to_arm(self) -> bool:
        """Return True if PX4 reports MAV_STATE_STANDBY (system ready, not armed)."""
        hb = self.get_px4_heartbeat()
        if hb is None:
            return False
        # MAV_STATE_STANDBY = 3
        return hb['system_status'] == 3

    def wait_for_ready_for_takeoff(self, timeout: float = 90.0) -> bool:
        """Block until PX4 sends STATUSTEXT 'Ready for takeoff!' or timeout."""
        return self._ready_for_takeoff_event.wait(timeout=timeout)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _send_command_long(
        self,
        command: int,
        param1: float = 0.0,
        param2: float = 0.0,
        param3: float = 0.0,
        param4: float = 0.0,
        param5: float = 0.0,
        param6: float = 0.0,
        param7: float = 0.0,
        confirmation: int = 0,
    ) -> None:
        if self._mav is None:
            return
        hb = self.get_px4_heartbeat()
        target_system = hb['system_id'] if hb is not None else 1
        target_component = 1
        self._mav.mav.command_long_send(
            target_system,
            target_component,
            command,
            confirmation,
            param1,
            param2,
            param3,
            param4,
            param5,
            param6,
            param7,
        )

    def _send_loop(self) -> None:
        """Periodically send a GCS heartbeat so PX4 considers the GCS connected."""
        while not self._stop_event.is_set():
            now = time.monotonic()
            if self._mav is not None and (now - self._last_heartbeat_sent) >= 1.0:
                try:
                    self._mav.mav.heartbeat_send(
                        mavutil.mavlink.MAV_TYPE_GCS,
                        mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                        0, 0, 0,
                    )
                except Exception:
                    break
                self._last_heartbeat_sent = now
            time.sleep(0.1)

    def _recv_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._mav is None:
                break
            try:
                msg = self._mav.recv_match(blocking=True, timeout=1.0)
            except Exception:
                break
            if msg is None:
                continue
            msg_type = msg.get_type()

            if msg_type == 'HEARTBEAT':
                data = {
                    'system_id': msg.get_srcSystem(),
                    'component_id': msg.get_srcComponent(),
                    'type': msg.type,
                    'autopilot': msg.autopilot,
                    'base_mode': msg.base_mode,
                    'custom_mode': msg.custom_mode,
                    'system_status': msg.system_status,
                }
                with self._recv_lock:
                    self._heartbeat = data
                self._heartbeat_event.set()

            elif msg_type == 'LOCAL_POSITION_NED':
                data = {
                    'time_boot_ms': msg.time_boot_ms,
                    'x': msg.x,
                    'y': msg.y,
                    'z': msg.z,
                    'vx': msg.vx,
                    'vy': msg.vy,
                    'vz': msg.vz,
                }
                with self._recv_lock:
                    self._local_position = data

            elif msg_type == 'STATUSTEXT':
                text = msg.text.rstrip('\x00').strip()
                if 'ready for takeoff' in text.lower():
                    self._ready_for_takeoff_event.set()

            elif msg_type == 'ODOMETRY':
                data = {
                    'time_usec': msg.time_usec,
                    'x': msg.x,
                    'y': msg.y,
                    'z': msg.z,
                    'vx': msg.vx,
                    'vy': msg.vy,
                    'vz': msg.vz,
                }
                with self._recv_lock:
                    self._odometry = data
