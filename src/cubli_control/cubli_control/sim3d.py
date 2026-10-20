"""3D Cubli simulator core (no ROS): contact physics + Nidec motors + brakes + sensors.

Shared by sim3d_node.py, the tests and the GIF script, so all of them run the same plant.

Hardware non-idealities on top of the IMU/encoder models (sensors3d.py):
  - motor torque constant off by `torque_scale` (the controller assumes 1.0)
  - torque ripple: Gaussian, `torque_noise_rel` x |command| + `torque_noise_abs`
  - Nidec back-EMF limits (model3d.motor_limits)
  - brakes hand over only a fraction of the wheel momentum: efficiency ~ N(mean, std)
"""

from dataclasses import dataclass

import numpy as np

from cubli_control import contact3d
from cubli_control.model3d import Cubli3DParams, motor_limits, up_in_body
from cubli_control.sensors3d import Sensor3DModel, Sensor3DParams


@dataclass(frozen=True)
class HardwareNoise:
    torque_scale: float = 1.0
    torque_noise_rel: float = 0.0
    torque_noise_abs: float = 0.0
    brake_efficiency: float = 1.0
    brake_efficiency_std: float = 0.0


IDEAL = HardwareNoise()
REALISTIC = HardwareNoise(torque_scale=0.95, torque_noise_rel=0.03, torque_noise_abs=5e-4,
                          brake_efficiency=0.92, brake_efficiency_std=0.02)


class Simulator:

    def __init__(self, x0, p: Cubli3DParams = None, sensors: Sensor3DParams = None,
                 noise: HardwareNoise = REALISTIC, rate_hz=200.0, substeps=5, seed=None):
        self.p = p or Cubli3DParams()
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.sensors = Sensor3DModel(sensors or Sensor3DParams(), seed=self.rng.integers(1 << 31))
        self.dt = 1.0 / rate_hz
        self.substeps = substeps
        self.x = np.array(x0, float)
        self.contacts = contact3d.contacts_of(up_in_body(self.x[0:4]))
        self.t = 0.0
        self.applied = np.zeros(3)
        self.held = False
        self.brakes = []      # (t, wheels, efficiency, wheel speeds before)

    def brake(self, wheels):
        n = self.noise
        eff = float(np.clip(self.rng.normal(n.brake_efficiency, n.brake_efficiency_std), 0.0, 1.0))
        self.brakes.append((self.t, list(wheels), eff, self.x[7:10].copy()))
        self.x, self.contacts = contact3d.brake(self.x, self.p, wheels, eff, self.contacts)

    def motor_torque(self, command):
        n = self.noise
        torque = motor_limits(self.p, self.x[7:10], command)
        noise = self.rng.normal(0.0, 1.0, 3) * (n.torque_noise_rel * np.abs(torque)
                                                + n.torque_noise_abs)
        return n.torque_scale * torque + np.where(torque != 0.0, noise, 0.0)

    def step(self, command, brake=()):
        """Advance one control period. Returns the true torques applied."""
        if brake:
            self.brake(brake)
        torque = np.zeros(3) if self.held else self.motor_torque(command)
        if self.held:
            self.x[4:10] = 0.0
        else:
            h = self.dt / self.substeps
            for _ in range(self.substeps):
                self.x, self.contacts = contact3d.step(self.x, torque, h, self.p, self.contacts)
        self.applied = torque
        self.t += self.dt
        return torque

    def read_sensors(self):
        """(gyro, accel, encoder angles) for the current state."""
        still = self.held or len(self.contacts) >= 2
        if still:
            gyro, accel = self.sensors.imu(self.x, np.zeros(3), self.p, fallen=True)
        else:
            gyro, accel = self.sensors.imu(self.x, self.applied, self.p, contacts=self.contacts)
        return gyro, accel, self.sensors.encoder_angles(self.x[10:13])


def closed_loop(x0, seconds, controller=None, noise: HardwareNoise = REALISTIC, seed=0,
                hold=0.0, pushes=(), sensors: Sensor3DParams = None, on_step=None):
    """Run sim + estimator + JumpBalanceController with one control period of delay.

    hold: seconds held still at the start (for the gyro calibration when not on a face).
    pushes: [(t, world-frame angular velocity kick)]. on_step(sim, controller, est_up) is
    called every period (for logging). Returns (sim, controller).
    """
    from cubli_control.control3d import JumpBalanceController
    from cubli_control.estimator3d import UpEstimator, WheelSpeeds
    from cubli_control.model3d import quat_to_matrix

    sim = Simulator(x0, noise=noise, seed=seed, sensors=sensors)
    ctrl = controller or JumpBalanceController(sim.p)
    est, wheels = UpEstimator(imu_position=sim.sensors.p.imu_position), WheelSpeeds()
    command, brake = np.zeros(3), []
    for i in range(int(round(seconds / sim.dt))):
        sim.held = sim.t < hold
        for at, kick in pushes:
            if abs(sim.t - at) < sim.dt / 2:
                sim.x[4:7] += quat_to_matrix(sim.x[0:4]).T @ np.asarray(kick, float)
                sim.contacts = contact3d.lift_off(sim.x, np.zeros(3), sim.p, sim.contacts) \
                    if len(sim.contacts) < 2 else sim.contacts
        sim.step(command, brake)        # last period's command: one step of delay
        gyro, accel, enc = sim.read_sensors()
        speeds = wheels.update(sim.t, enc)
        up = None
        command, brake = np.zeros(3), []
        if not est.ready:
            est.calibrate(gyro, accel)
        else:
            up, rate = est.update(gyro, accel, sim.dt)
            command, brake = ctrl.step(sim.t, up, rate, speeds)
        if on_step:
            on_step(sim, ctrl, up)
    return sim, ctrl
