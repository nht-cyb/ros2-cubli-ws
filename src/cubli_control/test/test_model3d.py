"""Phase 4: 3D Cubli model, corner-balancing LQR and up-vector estimator (no ROS needed)."""

import math

import numpy as np
import pytest

from cubli_control.estimator3d import UpEstimator, WheelSpeeds, rotate
from cubli_control.lqr import discretize
from cubli_control.model3d import (
    FALL_TILT, Cubli3DParams, accelerations, energy, fallen_step, has_fallen, initial_state,
    linearized, lqr_gain_3d, motor_limits, quat_from_vectors, quat_to_matrix, rk4_step,
    state_vector, tilt_angle, tilt_basis, up_in_body, world_momentum)
from cubli_control.sensors3d import Sensor3DModel, Sensor3DParams

DT = 1.0 / 200.0
P = Cubli3DParams()
WEIGHTS = (100.0, 0.3, 1e-3, 100.0)
DIRECTIONS = np.linspace(0.0, 2.0 * math.pi, 6, endpoint=False)


def test_centre_of_mass_on_the_diagonal_and_upright_is_an_equilibrium():
    assert P.r_com == pytest.approx(np.full(3, P.r_com[0]))
    w_dot, ww_dot = accelerations(initial_state(P), np.zeros(3), P)
    assert np.abs(w_dot).max() < 1e-12 and np.abs(ww_dot).max() < 1e-12
    assert tilt_angle(P, up_in_body(initial_state(P, 0.05, 1.0)[0:4])) == pytest.approx(0.05)


def test_wheels_fit_in_the_cube_without_touching():
    assert P.wheel_clearance() >= 0.004
    assert Cubli3DParams(wheel_offset=0.045).wheel_clearance() < 0.0   # wheels would collide


def test_linearisation_matches_finite_differences():
    r_hat, a, b = tilt_basis(P)

    def f(z, torque):
        u = r_hat + z[0] * a + z[1] * b
        u /= np.linalg.norm(u)
        x = np.concatenate([quat_from_vectors(u, [0, 0, 1]), z[2:5], z[5:8], np.zeros(3)])
        w_dot, ww_dot = accelerations(x, torque, P)
        u_dot = np.cross(u, z[2:5])
        return np.concatenate([[a @ u_dot, b @ u_dot], w_dot, ww_dot])

    big_a, big_b = linearized(P)
    eps = 1e-6
    for i in range(8):
        dz = np.zeros(8)
        dz[i] = eps
        col = (f(dz, np.zeros(3)) - f(-dz, np.zeros(3))) / (2 * eps)
        assert col == pytest.approx(big_a[:, i], abs=1e-6)
    for i in range(3):
        dt = np.zeros(3)
        dt[i] = eps
        assert (f(np.zeros(8), dt) - f(np.zeros(8), -dt)) / (2 * eps) == pytest.approx(
            big_b[:, i], rel=1e-6)


def test_energy_and_momentum_conserved_without_friction():
    p = Cubli3DParams(c_b=0.0, c_w=0.0)
    x = initial_state(p, 0.3, 0.7)
    x[4:7] = [0.4, -0.2, 0.3]
    x[7:10] = [30.0, -50.0, 10.0]
    e0 = energy(x, p)
    for _ in range(400):
        x = rk4_step(x, np.zeros(3), DT, p)
    assert energy(x, p) == pytest.approx(e0, rel=1e-6)

    p0 = Cubli3DParams(c_b=0.0, c_w=0.0, g=0.0)   # no gravity: momentum about the pivot
    x = initial_state(p0, 0.3, 0.7)
    x[4:7] = [0.4, -0.2, 0.3]
    l0 = world_momentum(x, p0)
    for _ in range(400):
        x = rk4_step(x, np.array([0.01, -0.02, 0.005]), DT, p0)   # internal torques only
    assert world_momentum(x, p0) == pytest.approx(l0, abs=1e-9)


def test_upright_unstable_open_loop_and_stable_with_lqr():
    big_a, big_b = linearized(P)
    assert max(np.linalg.eigvals(big_a).real) > 5.0
    ad, bd = discretize(big_a, big_b, DT)
    k = lqr_gain_3d(P, DT, *WEIGHTS)
    assert k.shape == (3, 8)
    assert max(abs(np.linalg.eigvals(ad - bd @ k))) < 1.0


def test_motor_limits_include_back_emf():
    t = motor_limits(P, [0.0, 300.0, -600.0], [1.0, 1.0, 1.0])
    assert t == pytest.approx([P.max_torque, 0.05, P.max_torque])   # 0.1 * (1 - 0.5) = 0.05
    assert motor_limits(P, [600.0, 0.0, 0.0], [0.05, -1.0, 0.0])[0] == pytest.approx(0.0)


def simulate(tilt, direction, seconds=5.0, push=None, sensors=False, hold=1.2, seed=0):
    """Nonlinear closed loop with the Nidec limits and one step of control delay."""
    k = lqr_gain_3d(P, DT, *WEIGHTS)
    model = Sensor3DModel(Sensor3DParams(), seed=seed)
    est, wheels = UpEstimator(), WheelSpeeds()
    x = initial_state(P, tilt, direction)
    command, peak, errors = np.zeros(3), 0.0, []
    for i in range(int(seconds / DT)):
        now = i * DT
        held = sensors and now < hold        # held by hand while the gyro calibrates
        torque = np.zeros(3) if held else motor_limits(P, x[7:10], command)
        if push is not None and i == int((seconds / 2) / DT):
            x[4:7] += quat_to_matrix(x[0:4]).T @ np.asarray(push)
        if not held:
            for _ in range(5):
                x = rk4_step(x, torque, DT / 5, P)
        true_up = up_in_body(x[0:4])
        if sensors:
            gyro, accel = model.imu(x, torque, P, fallen=held)
            speeds = wheels.update(now, model.encoder_angles(x[10:13]))
            if not est.ready:
                est.calibrate(gyro, accel)
                continue
            up, rate = est.update(gyro, accel, DT)
            errors.append(math.acos(min(1.0, up @ true_up)))
        else:
            up, rate, speeds = true_up, x[4:7], x[7:10]
        command = -(k @ state_vector(P, up, rate, speeds))
        tilt_now = tilt_angle(P, true_up)
        peak = max(peak, tilt_now)
        if tilt_now > FALL_TILT:
            return None, peak, errors
    return x, peak, errors


@pytest.mark.parametrize('direction', DIRECTIONS)
def test_recovers_from_3_deg_in_any_direction(direction):
    x, _, _ = simulate(math.radians(3.0), direction)
    assert x is not None
    assert tilt_angle(P, up_in_body(x[0:4])) < math.radians(0.05)
    assert np.abs(x[7:10]).max() < 5.0          # wheels spun back down


def test_too_much_tilt_for_the_motors_falls():
    x, _, _ = simulate(0.1, 0.0, seconds=3.0)   # gravity needs ~0.16 N m, wheels give 0.08
    assert x is None


def test_balances_with_sensors_and_estimator_and_survives_a_push():
    x, peak, errors = simulate(0.03, 0.5, seconds=10.0, push=(0.15, 0.0, 0.0), sensors=True)
    assert x is not None
    assert tilt_angle(P, up_in_body(x[0:4])) < math.radians(0.1)
    assert peak < math.radians(2.0)
    assert np.degrees(np.sqrt(np.mean(np.square(errors[200:])))) < 0.5   # mostly accel bias


def test_estimator_calibration_rejects_motion_and_finds_bias():
    est = UpEstimator(calib_samples=50)
    rng = np.random.default_rng(1)
    bias = np.array([0.01, -0.02, 0.005])
    up = np.array([1.0, 1.0, 1.0]) / math.sqrt(3.0)
    assert not any(est.calibrate(bias + [0.5 * (i % 2), 0, 0], 9.81 * up) for i in range(50))
    done = [est.calibrate(bias + rng.normal(0, 5e-4, 3), 9.81 * up) for _ in range(50)]
    assert done[-1] and est.bias == pytest.approx(bias, abs=5e-4)
    assert est.up == pytest.approx(up)


def test_rotate_matches_quaternion_rotation():
    v = np.array([0.3, -0.4, 0.86])
    axis_angle = np.array([0.2, -0.1, 0.4])
    from cubli_control.model3d import quat_from_axis_angle
    q = quat_from_axis_angle(axis_angle, np.linalg.norm(axis_angle))
    assert rotate(v, axis_angle) == pytest.approx(quat_to_matrix(q) @ v)


def test_fallen_cube_only_moves_its_wheels():
    x = initial_state(P, FALL_TILT + 0.01, 0.0)
    assert has_fallen(x, P)
    x[4:7] = [0.3, 0.1, 0.0]
    out = fallen_step(x, np.array([0.01, 0.0, 0.0]), DT, P)
    assert np.all(out[4:7] == 0.0) and out[7] > 0.0
    assert np.all(out[0:4] == x[0:4])
