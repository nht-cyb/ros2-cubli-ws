"""3D Cubli contacts (face / edge / corner), brakes, edge balance and the jump-up sequence."""

import math

import numpy as np
import pytest

from cubli_control import contact3d as c
from cubli_control.control3d import (
    CORNER, EDGE, ControlParams, JumpBalanceController, corner_jump_speeds, edge_gain,
    edge_linearized, face_jump_speed)
from cubli_control.lqr import discretize
from cubli_control.model3d import Cubli3DParams, tilt_angle, up_in_body
from cubli_control.sim3d import IDEAL, REALISTIC, closed_loop

P = Cubli3DParams()
FRICTIONLESS = Cubli3DParams(c_b=0.0, c_w=0.0)
DT = 1.0 / 200.0


def energy(x, p):
    w, ww = x[4:7], x[7:10]
    return (0.5 * w @ p.theta_0 @ w + 0.5 * p.i_wheel * np.sum((w + ww) ** 2)
            + p.mass * p.g * c.com_height(x, p))


def run_free(x, contacts, p, seconds, dt=1e-3, stop=None):
    for _ in range(int(seconds / dt)):
        x, contacts = c.step(x, np.zeros(3), dt, p, contacts)
        if stop and stop(x, contacts):
            break
    return x, contacts


def test_lying_on_a_face_is_stable_and_pushes_on_both_vertices():
    x = c.face_state((0, 1))
    contacts = c.contacts_of(up_in_body(x[0:4]))
    assert contacts == {0, 1}
    forces = c.normal_forces(x, np.zeros(3), P, contacts)
    assert min(forces.values()) > 1.0
    x, contacts = run_free(x, contacts, P, 0.5)
    assert contacts == {0, 1} and np.all(x[4:7] == 0.0)


def test_edge_mode_conserves_energy_and_falls_onto_a_face_at_45_deg():
    x = c.edge_state(FRICTIONLESS, 0, 0.05)
    x[4] = 0.3
    e0 = energy(x, FRICTIONLESS)
    x1, contacts = run_free(x, frozenset({0}), FRICTIONLESS, 0.3)
    assert contacts == {0}
    assert energy(x1, FRICTIONLESS) == pytest.approx(e0, rel=1e-8)
    x2, contacts = run_free(x, frozenset({0}), FRICTIONLESS, 2.0, stop=lambda x, k: len(k) == 2)
    assert contacts == {0, 2}
    assert abs(c.edge_angle(P, 0, up_in_body(x2[0:4]))) == pytest.approx(math.pi / 4, abs=1e-6)


def test_falling_off_the_corner_lands_on_an_edge_keeping_momentum_about_it():
    x = c.edge_state(FRICTIONLESS, 0, 0.0)
    x[0:4] = c.state_from_up(up_in_body(x[0:4]) + np.array([0.02, 0.0, 0.0]))[0:4]
    x[4:7] = [-0.5, 0.2, -0.2]
    assert c.contacts_of(up_in_body(x[0:4])) == set()
    x, contacts = run_free(x, frozenset(), FRICTIONLESS, 1.0, stop=lambda x, k: bool(k))
    assert contacts == {0}
    assert x[5] == 0.0 and x[6] == 0.0      # only the rotation about the edge survives


@pytest.mark.parametrize('sign, lifts', [(1.0, True), (-1.0, False)])
def test_face_brake_lifts_onto_the_edge_only_in_the_right_direction(sign, lifts):
    x = c.face_state((0, 1))
    x[7] = sign * face_jump_speed(FRICTIONLESS, 0, face_up=2)
    x, contacts = c.brake(x, FRICTIONLESS, [0], 1.0, frozenset({0, 1}))
    assert contacts == ({0} if lifts else {0, 1})
    if lifts:   # a lossless brake at exactly that speed arrives at balance with no speed left
        best = [math.inf]

        def track(x, k):
            best[0] = min(best[0], abs(c.edge_angle(P, 0, up_in_body(x[0:4]))))
            return len(k) == 2
        run_free(x, contacts, FRICTIONLESS, 2.0, stop=track)
        assert best[0] < math.radians(1.0)


def test_corner_jump_speeds_lie_along_the_jump_axis_and_reach_the_corner():
    speeds = corner_jump_speeds(FRICTIONLESS, 0)
    n = c.edge_frame(FRICTIONLESS, 0)[2]
    assert speeds[0] == 0.0 and np.cross(speeds, n) == pytest.approx(np.zeros(3), abs=1e-9)
    x = c.edge_state(FRICTIONLESS, 0, 0.0)
    x[7:10] = speeds * 1.001
    x, contacts = c.brake(x, FRICTIONLESS, [1, 2], 1.0, frozenset({0}))
    assert contacts == set()
    best = [math.inf]

    def track(x, k):
        best[0] = min(best[0], tilt_angle(P, up_in_body(x[0:4])))
        return bool(k) and best[0] < 0.1
    run_free(x, contacts, FRICTIONLESS, 2.0, stop=track)
    assert best[0] < math.radians(1.0)


def test_brake_efficiency_scales_the_momentum_handed_over():
    x = c.face_state((0, 1))
    x[7] = 200.0
    full, _ = c.brake(x, P, [0], 1.0, frozenset({0, 1}))
    lossy, _ = c.brake(x, P, [0], 0.9, frozenset({0, 1}))
    assert lossy[4] == pytest.approx(0.9 * full[4]) and lossy[7] == 0.0


def test_edge_lqr_is_stable():
    big_a, big_b = edge_linearized(P, 0)
    assert max(np.linalg.eigvals(big_a).real) > 5.0
    ad, bd = discretize(big_a, big_b, DT)
    k = edge_gain(P, 0, DT, (100.0, 0.3, 1e-4), 100.0)
    assert max(abs(np.linalg.eigvals(ad - bd @ k[None, :]))) < 1.0


def test_balances_on_an_edge_with_noisy_sensors_and_motors():
    ctrl = JumpBalanceController(P, ControlParams(jump_to_corner=False))
    sim, ctrl = closed_loop(c.edge_state(P, 0, math.radians(2.0)), 6.0, controller=ctrl,
                            noise=REALISTIC, seed=3, hold=1.2)
    assert ctrl.mode == EDGE and sim.contacts == {0}
    assert abs(c.edge_angle(P, 0, up_in_body(sim.x[0:4]))) < math.radians(0.3)


@pytest.mark.parametrize('noise, prior, seconds', [(IDEAL, 1.0, 16.0), (REALISTIC, 0.9, 30.0)])
def test_jumps_from_a_face_to_the_edge_and_then_to_the_corner(noise, prior, seconds):
    # With a lossy, unknown brake the first jump may miss; there is time for retries.
    ctrl = JumpBalanceController(P, ControlParams(brake_efficiency=prior))
    sim, ctrl = closed_loop(c.face_state((0, 1)), seconds, controller=ctrl, noise=noise, seed=2)
    modes = [mode for _, mode, _ in ctrl.log]
    assert EDGE in modes and ctrl.mode == CORNER and not sim.contacts
    assert tilt_angle(P, up_in_body(sim.x[0:4])) < math.radians(0.2)
    if noise is REALISTIC:   # it learned the brake from its own measurements
        true = np.mean([b[2] for b in sim.brakes])
        assert ctrl.brake_efficiency == pytest.approx(true, abs=0.03)
