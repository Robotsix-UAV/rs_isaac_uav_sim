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

import socket
import threading
import time

from pymavlink import mavutil

MAV_MODE_FLAG_SAFETY_ARMED = 0x80


class MockMavlinkAutopilot:
    """
    Plain-Python mock autopilot that speaks MAVLink HIL over TCP.

    Connects as a TCP client to Isaac Sim (which is the TCP server).
    Sends HIL_ACTUATOR_CONTROLS and receives HIL_SENSOR, HIL_GPS,
    HIL_STATE_QUATERNION messages.
    """

    def __init__(self, host='127.0.0.1', port=4561, send_rate_hz=500):
        self._host = host
        self._port = port
        self._send_rate_hz = send_rate_hz

        self._mav = None
        self._armed = False
        self._controls = [0.0] * 8
        self._controls_lock = threading.Lock()

        self._hil_state = None
        self._hil_sensor = None
        self._hil_gps = None
        self._recv_lock = threading.Lock()

        self._stop_event = threading.Event()
        self._send_thread = None
        self._recv_thread = None
        self._first_message_event = threading.Event()
        self._last_heartbeat_time = 0.0

    def connect(self, timeout=30.0) -> None:
        deadline = time.monotonic() + timeout
        while True:
            try:
                self._mav = mavutil.mavlink_connection(
                    f'tcp:{self._host}:{self._port}',
                    source_system=1,
                    source_component=1,
                )
                break
            except (ConnectionRefusedError, socket.error):
                if time.monotonic() + 1.0 > deadline:
                    raise RuntimeError(
                        f'MockMavlinkAutopilot: TCP server not available after '
                        f'{timeout}s at tcp:{self._host}:{self._port}'
                    )
                time.sleep(1.0)

        self._stop_event.clear()
        self._first_message_event.clear()

        self._send_thread = threading.Thread(
            target=self._send_loop,
            name='mock-mav-send',
            daemon=True,
        )
        self._recv_thread = threading.Thread(
            target=self._recv_loop,
            name='mock-mav-recv',
            daemon=True,
        )
        self._send_thread.start()
        self._recv_thread.start()

        if not self.wait_for_first_message(timeout=timeout):
            self.disconnect()
            raise RuntimeError(
                f'MockMavlinkAutopilot: no HIL message received within '
                f'{timeout}s from tcp:{self._host}:{self._port}'
            )

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._send_thread is not None:
            self._send_thread.join(timeout=2.0)
            self._send_thread = None
        if self._recv_thread is not None:
            self._recv_thread.join(timeout=2.0)
            self._recv_thread = None
        if self._mav is not None:
            try:
                self._mav.close()
            except Exception:
                pass
            self._mav = None

    def arm(self) -> None:
        self._armed = True

    def disarm(self) -> None:
        self._armed = False

    def set_controls(self, controls: list) -> None:
        padded = list(controls) + [0.0] * 8
        with self._controls_lock:
            self._controls = padded[:8]

    def get_hil_state(self) -> dict | None:
        with self._recv_lock:
            return dict(self._hil_state) if self._hil_state is not None else None

    def get_hil_sensor(self) -> dict | None:
        with self._recv_lock:
            return dict(self._hil_sensor) if self._hil_sensor is not None else None

    def get_hil_gps(self) -> dict | None:
        with self._recv_lock:
            return dict(self._hil_gps) if self._hil_gps is not None else None

    def wait_for_first_message(self, timeout=30.0) -> bool:
        return self._first_message_event.wait(timeout=timeout)

    def _send_loop(self):
        interval = 1.0 / self._send_rate_hz
        while not self._stop_event.is_set():
            start = time.monotonic()
            if self._mav is not None:
                with self._controls_lock:
                    controls = list(self._controls)
                padded = controls + [0.0] * 8
                mode = MAV_MODE_FLAG_SAFETY_ARMED if self._armed else 0
                now = time.time()
                if now - self._last_heartbeat_time > 1.0:
                    try:
                        self._mav.mav.heartbeat_send(
                            mavutil.mavlink.MAV_TYPE_GCS,
                            mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                            0,
                            0,
                            0,
                        )
                    except Exception:
                        break
                    self._last_heartbeat_time = now
                try:
                    self._mav.mav.hil_actuator_controls_send(
                        time.time_ns() // 1000,
                        padded,
                        mode,
                        0,
                    )
                except Exception:
                    break
            elapsed = time.monotonic() - start
            remaining = interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    def _recv_loop(self):
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
            if msg_type == 'HIL_STATE_QUATERNION':
                data = {
                    'attitude_quaternion': list(msg.attitude_quaternion),
                    'rollspeed': msg.rollspeed,
                    'pitchspeed': msg.pitchspeed,
                    'yawspeed': msg.yawspeed,
                    'lat': msg.lat,
                    'lon': msg.lon,
                    'alt': msg.alt,
                    'vx': msg.vx,
                    'vy': msg.vy,
                    'vz': msg.vz,
                    'ind_airspeed': msg.ind_airspeed,
                    'true_airspeed': msg.true_airspeed,
                    'xacc': msg.xacc,
                    'yacc': msg.yacc,
                    'zacc': msg.zacc,
                }
                with self._recv_lock:
                    self._hil_state = data
                self._first_message_event.set()
            elif msg_type == 'HIL_SENSOR':
                data = {
                    'time_usec': msg.time_usec,
                    'xacc': msg.xacc,
                    'yacc': msg.yacc,
                    'zacc': msg.zacc,
                    'xgyro': msg.xgyro,
                    'ygyro': msg.ygyro,
                    'zgyro': msg.zgyro,
                    'xmag': msg.xmag,
                    'ymag': msg.ymag,
                    'zmag': msg.zmag,
                    'abs_pressure': msg.abs_pressure,
                    'diff_pressure': msg.diff_pressure,
                    'pressure_alt': msg.pressure_alt,
                    'temperature': msg.temperature,
                    'fields_updated': msg.fields_updated,
                }
                with self._recv_lock:
                    self._hil_sensor = data
                self._first_message_event.set()
            elif msg_type == 'HIL_GPS':
                data = {
                    'time_usec': msg.time_usec,
                    'fix_type': msg.fix_type,
                    'lat': msg.lat,
                    'lon': msg.lon,
                    'alt': msg.alt,
                    'eph': msg.eph,
                    'epv': msg.epv,
                    'vel': msg.vel,
                    'vn': msg.vn,
                    've': msg.ve,
                    'vd': msg.vd,
                    'cog': msg.cog,
                    'satellites_visible': msg.satellites_visible,
                }
                with self._recv_lock:
                    self._hil_gps = data
                self._first_message_event.set()
