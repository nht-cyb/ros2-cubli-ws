"""Sensor models for the simulated 3D Cubli: one ICM-42688-class IMU and three wheel encoders.

IMU frame = body frame (axes along the cube edges at the pivot corner). The IMU sits at
`imu_position` (default: the cube centre), so its accelerometer also feels the cube's own
rotation: f = omega_dot x r + omega x (omega x r) + g * up.

Noise levels for an ICM-42688-P at ~100 Hz bandwidth (datasheet densities 2.8 mdps/sqrt(Hz)
and 70 ug/sqrt(Hz)). Biases are what is left after a one-time accelerometer calibration;
the gyro bias is removed by the estimator's start-up calibration.
"""

from dataclasses import dataclass
import math

import numpy as np

from cubli_control import contact3d
from cubli_control.model3d import cross3, up_in_body


@dataclass(frozen=True)
class Sensor3DParams:
    imu_position: tuple = (0.075, 0.075, 0.075)    # pivot -> IMU, body frame [m]
    gyro_noise_std: float = 5e-4                    # [rad/s] per sample
    gyro_bias: tuple = (0.006, -0.004, 0.008)       # [rad/s] (~0.3-0.5 deg/s)
    accel_noise_std: float = 0.007                  # [m/s^2] per sample
    accel_bias: tuple = (0.02, -0.015, 0.025)       # [m/s^2]
    encoder_cpr: int = 400                          # counts per wheel turn (Nidec 100 PPR x4)
    g: float = 9.81


def ideal_imu(x, torque, p, imu_position, g=9.81, contacts=frozenset()):
    """Noise-free (gyro, accel) in the body frame for state x under wheel torques."""
    w = x[4:7]
    w_dot, _, _ = contact3d.accelerations(x, torque, p, frozenset(contacts))
    r = np.asarray(imu_position, float)
    accel = cross3(w_dot, r) + cross3(w, cross3(w, r)) + g * up_in_body(x[0:4])
    return w.copy(), accel


class Sensor3DModel:

    def __init__(self, params: Sensor3DParams, seed=None):
        self.p = params
        self.rng = np.random.default_rng(seed)

    def imu(self, x, torque, cube_params, fallen=False, contacts=frozenset()):
        """(gyro[3], accel[3]) as the IMU reports them. A fallen cube is still."""
        p = self.p
        if fallen:
            gyro, accel = np.zeros(3), p.g * up_in_body(x[0:4])
        else:
            gyro, accel = ideal_imu(x, torque, cube_params, p.imu_position, p.g, contacts)
        gyro = gyro + np.asarray(p.gyro_bias) + self.rng.normal(0.0, p.gyro_noise_std, 3)
        accel = accel + np.asarray(p.accel_bias) + self.rng.normal(0.0, p.accel_noise_std, 3)
        return gyro, accel

    def encoder_counts(self, wheel_angles) -> np.ndarray:
        cpr = self.p.encoder_cpr
        return np.floor(np.asarray(wheel_angles) * cpr / (2.0 * math.pi)).astype(int)

    def encoder_angles(self, wheel_angles) -> np.ndarray:
        return self.encoder_counts(wheel_angles) * 2.0 * math.pi / self.p.encoder_cpr
