"""Attitude estimation for the 3D Cubli: the "up" direction in the body frame (no ROS here).

The balance controller only needs gravity's direction relative to the cube, not yaw, so
the estimator tracks the body-frame up vector u (a unit vector) instead of a full attitude:

- Calibration: while the cube is held still, average the gyro (bias, 3 axes) and the
  accelerometer (starting up vector).
- Prediction: u_dot = u x omega with the bias-corrected gyro (exact rotation per sample).
- Correction: pull u towards the accelerometer's direction with time constant `tau`,
  after removing the IMU's own centripetal/tangential acceleration (lever-arm
  compensation). Skipped while the accelerometer clearly sees more than gravity.

The 3D version of estimator.TiltEstimator, which it reuses for wheel speeds.
"""

import math

import numpy as np

from cubli_control.estimator import WheelSpeedEstimator
from cubli_control.model3d import cross3


def rotate(v, axis_angle):
    """Rotate v by the rotation vector axis_angle (Rodrigues)."""
    angle = float(np.linalg.norm(axis_angle))
    if angle < 1e-12:
        return v.copy()
    k = axis_angle / angle
    c, s = math.cos(angle), math.sin(angle)
    return v * c + cross3(k, v) * s + k * (k @ v) * (1.0 - c)


class UpEstimator:

    def __init__(self, time_constant=0.5, imu_position=(0.075, 0.075, 0.075),
                 calib_samples=200, calib_max_rate_std=0.02, gate_accel=1.0, gate_rate=3.0,
                 g=9.81):
        self.tau = time_constant
        self.r = np.asarray(imu_position, float)
        self.calib_samples = calib_samples
        self.calib_max_rate_std = calib_max_rate_std
        self.gate_accel, self.gate_rate, self.g = gate_accel, gate_rate, g
        self.bias = np.zeros(3)
        self.up = np.array([0.0, 0.0, 1.0])
        self.rate = np.zeros(3)
        self.rate_dot = np.zeros(3)   # filtered angular acceleration (lever-arm compensation)
        self.ready = False
        self._calib = []

    def calibrate(self, gyro, accel) -> bool:
        """Feed one still sample. Returns True once calibration has finished."""
        self._calib.append((np.asarray(gyro, float), np.asarray(accel, float)))
        if len(self._calib) < self.calib_samples:
            return False
        gyros = np.array([s[0] for s in self._calib])
        if gyros.std(axis=0).max() > self.calib_max_rate_std:
            self._calib.clear()   # it moved; start again
            return False
        self.bias = gyros.mean(axis=0)
        accel = np.mean([s[1] for s in self._calib], axis=0)
        self.up = accel / np.linalg.norm(accel)
        self.ready = True
        return True

    def update(self, gyro, accel, dt):
        """Return (up vector, body rate) after one IMU sample."""
        rate = np.asarray(gyro, float) - self.bias
        if dt > 0.0:
            k = dt / (0.01 + dt)            # ~16 Hz low-pass on the differentiated gyro
            self.rate_dot += k * ((rate - self.rate) / dt - self.rate_dot)
        # A world-fixed vector seen from the body turns by -omega dt.
        self.up = rotate(self.up, -0.5 * (rate + self.rate) * dt)
        self.rate = rate

        f = (np.asarray(accel, float) - cross3(self.rate_dot, self.r)
             - cross3(rate, cross3(rate, self.r)))
        norm = np.linalg.norm(f)
        if abs(norm - self.g) < self.gate_accel and np.linalg.norm(rate) < self.gate_rate:
            alpha = self.tau / (self.tau + dt)
            self.up = alpha * self.up + (1.0 - alpha) * f / norm
        self.up /= np.linalg.norm(self.up)
        return self.up.copy(), self.rate.copy()


class WheelSpeeds:
    """Three WheelSpeedEstimators, one per wheel."""

    def __init__(self, window=8):
        self.wheels = [WheelSpeedEstimator(window) for _ in range(3)]

    def update(self, t, angles) -> np.ndarray:
        return np.array([w.update(t, a) for w, a in zip(self.wheels, angles)])

    @property
    def speeds(self) -> np.ndarray:
        return np.array([w.speed for w in self.wheels])
