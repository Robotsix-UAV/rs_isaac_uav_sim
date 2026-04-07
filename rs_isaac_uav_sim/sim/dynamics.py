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

"""Quadrotor dynamics: thrust/drag computation in body frame (FLU)."""

import numpy as np

from .config import QuadrotorParams
from .state import VehicleState


class QuadrotorDynamics:
    """Computes body-frame forces and torques from rotor angular velocities."""

    def __init__(self, params: QuadrotorParams = QuadrotorParams()):
        self.params = params
        self._drag_coeffs = np.diag(params.drag)

    def scale_motor_commands(self, controls: np.ndarray) -> np.ndarray:
        """
        Scale MAVLink actuator controls [0..1] to rotor angular velocities [rad/s].

        Applies the PX4 THR_MDL_FAC thrust model:
          - thr_mdl_fac=0 (linear):    omega_norm = sqrt(control)
            → T = ct * (omega_norm * omega_max)^2 = ct * omega_max^2 * control  (linear in control)
          - thr_mdl_fac=1 (quadratic): omega_norm = control
            → T = ct * (control * omega_max)^2  (quadratic in control)
          - intermediate:
            omega_norm = fac * control + (1 - fac) * sqrt(control)

        Args
        ----
        controls
            Raw HIL_ACTUATOR_CONTROLS values (4,) in [0, 1].

        Returns
        -------
        np.ndarray
            Rotor angular velocities in rad/s (4,), clipped to [omega_min,
            omega_max].

        """
        fac = self.params.thr_mdl_fac
        controls_clipped = np.clip(controls, 0.0, 1.0)
        omega_norm = fac * controls_clipped + (1.0 - fac) * np.sqrt(controls_clipped)
        omega_range = self.params.omega_max - self.params.omega_min
        omega = self.params.omega_min + omega_norm * omega_range
        return np.clip(omega, self.params.omega_min, self.params.omega_max)

    def compute_forces_and_torques(
        self, omega: np.ndarray, state: VehicleState
    ) -> tuple:
        """
        Compute total force and torque in FLU body frame.

        Args
        ----
        omega
            Rotor angular velocities [rad/s] shape (4,).
        state
            Current vehicle state (used for drag computation).

        Returns
        -------
        tuple
            (force_body_FLU, torque_body_FLU) each shape (3,).

        """
        p = self.params
        omega_sq = omega**2

        # Per-rotor thrust: T_i = ct * omega_i^2
        thrusts = p.ct * omega_sq  # (4,)

        # Total force along body Z (up in FLU)
        total_thrust = np.sum(thrusts)
        force_body = np.array([0.0, 0.0, total_thrust])

        # Torques from rotor arm cross products
        # tau = sum( r_i x [0,0,T_i] ) = sum( [r_iy*T_i, -r_ix*T_i, 0] )
        torque = np.zeros(3)
        for i in range(4):
            rx, ry, _ = p.rotor_positions[i]
            torque[0] += ry * thrusts[i]  # roll
            torque[1] += -rx * thrusts[i]  # pitch

        # Yaw torque from rolling moments: sum( cq * omega_i^2 * dir_i )
        for i in range(4):
            torque[2] += p.cq * omega_sq[i] * p.rot_dirs[i]

        # Linear drag in body frame: F_drag = -diag(drag) * v_body
        body_vel = state.linear_body_velocity
        drag_force = -self._drag_coeffs @ body_vel
        force_body += drag_force

        return force_body, torque
