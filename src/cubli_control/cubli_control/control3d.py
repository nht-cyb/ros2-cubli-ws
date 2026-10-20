"""3D Cubli control: balance on an edge, jump face -> edge -> corner, balance on the corner.

Pure Python (no ROS), shared by controller3d_node.py, the tests and the GIF script. It only
sees what the estimator gives it: the body-frame up vector u, the body rate omega and the
three wheel speeds. Which vertices touch the floor follows from u (vertex i is at height
side * u_i), so the pose is known without any contact sensor.

Modes:
  FACE           lying on a face. After rest_time still (longer after failed jumps: it
                 backs off instead of looping), pick a floor edge i and spin wheel i
                 to the face -> edge jump speed (its reaction presses the cube into the floor)
  SPIN_EDGE      spinning up; brake once the speed has held for spin_settle_time
  JUMP_EDGE      wheel i braked: the cube flips onto edge i; the edge LQR takes over within
                 edge_catch of balance
  EDGE           balancing on edge i with wheel i (1D LQR about the edge). After edge_rest_time
                 it starts SPIN_CORNER (if jump_to_corner)
  SPIN_CORNER    still balancing with wheel i, the two other wheels spin up so that their
                 combined momentum lies along the corner-jump axis
  JUMP_CORNER    both braked at once: the cube lifts its far vertex and swings onto the
                 corner; the corner LQR takes over within corner_catch of upright
  CORNER         balancing on the corner with all three wheels (3D LQR)
Falls are caught where possible (a cube falling off the corner lands near edge balance),
but only while the cube is still moving towards balance, so a cube that is falling away
is let go instead of grabbed again on the next step.

Brakes are lossy, so the controller learns their efficiency: right after each brake it
compares the measured body rate with what a lossless brake would have given, and scales
the next spin-up targets by 1 / efficiency. On top of that each jump adapts its speed
margin: up after falling short, down after overshooting.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.linalg import solve_discrete_are

from cubli_control.contact3d import contacts_of, edge_angle, edge_frame
from cubli_control.lqr import discretize
from cubli_control.model3d import (
    Cubli3DParams, cross3, lqr_gain_3d, state_vector, tilt_angle, tilt_basis)

FACE, SPIN_EDGE, JUMP_EDGE, EDGE = 'FACE', 'SPIN_EDGE', 'JUMP_EDGE', 'EDGE'
SPIN_CORNER, JUMP_CORNER, CORNER = 'SPIN_CORNER', 'JUMP_CORNER', 'CORNER'


@dataclass
class ControlParams:
    rate_hz: float = 200.0
    max_torque: float = 0.08
    # Edge balance (1D LQR about the edge, wheel along the edge)
    edge_q: tuple = (100.0, 0.3, 1e-4)      # angle, rate, wheel speed
    edge_r: float = 100.0
    edge_catch: float = 0.4                 # rad: catch a cube moving towards balance
    edge_capture: float = 0.15              # rad: LQR takes over from the jump's energy control
    edge_disengage: float = 0.16            # rad: give up beyond this while tipping further
    # Corner balance (3D LQR)
    corner_q: tuple = (100.0, 0.3, 1e-3)    # tilt, rate, wheel speed
    corner_r: float = 100.0
    corner_catch: float = 0.25              # rad: catch a cube moving towards the corner
    corner_capture: float = 0.15            # rad: LQR takes over from the jump's energy control
    corner_disengage: float = 0.1           # rad
    corner_give_up: float = 0.3             # rad
    # Jumps
    jump_to_edge: bool = True
    jump_to_corner: bool = True
    rest_time: float = 1.0                  # s lying still before a face -> edge jump
    edge_rest_time: float = 2.0             # s balanced on the edge before the corner jump
    retry_backoff: float = 1.0              # extra rest [s] per failed jump in a row ...
    max_rest_time: float = 5.0              # ... up to this
    spin_torque: float = 0.03               # N m, gentle spin-up
    spin_gain: float = 0.01                 # N m per rad/s of speed error
    spin_tolerance: float = 2.0             # rad/s
    spin_settle_time: float = 0.15          # s
    spin_timeout: float = 10.0              # s
    edge_margin: float = 1.04               # x the ideal face -> edge wheel speed
    corner_margin: float = 1.03             # x the ideal edge -> corner wheel speeds
    margin_step: float = 0.01
    brake_efficiency: float = 0.9           # initial guess; learned from every brake
    energy_gain: float = 5.0                # N m per (J x rad/s): flight energy control
    jump_timeout: float = 1.5               # s from brake to catch
    floor_eps: float = 0.05                 # u_i below this: vertex i counts as on the floor


def edge_linearized(p: Cubli3DParams, i: int):
    """(A, B) for [phi, phi_dot, omega_w_i] balancing on edge i (same form as the 1D cube)."""
    _, r_perp, _ = edge_frame(p, i)
    inertia = p.theta_0[i, i]
    mg = p.mass * p.g * r_perp
    a = np.array([[0.0, 1.0, 0.0],
                  [mg / inertia, -p.c_b / inertia, p.c_w / inertia],
                  [-mg / inertia, p.c_b / inertia, -p.c_w / inertia - p.c_w / p.i_wheel]])
    b = np.array([[0.0], [-1.0 / inertia], [1.0 / inertia + 1.0 / p.i_wheel]])
    return a, b


def edge_gain(p: Cubli3DParams, i: int, dt: float, q, r: float) -> np.ndarray:
    ad, bd = discretize(*edge_linearized(p, i), dt)
    qm, rm = np.diag(q), np.array([[r]])
    s = solve_discrete_are(ad, bd, qm, rm)
    return np.linalg.solve(rm + bd.T @ s @ bd, bd.T @ s @ ad).ravel()


def face_jump_speed(p: Cubli3DParams, i: int, face_up: int) -> float:
    """Wheel-i speed [rad/s] whose momentum just lifts the cube from the face with up axis
    `face_up` onto edge i (no losses).

    The brake hands I_w omega_w to body + locked wheel: phi_dot = I_w omega_w / Theta_ii.
    Right after it the wheel is free again and keeps its own spin, so only
    1/2 Theta0_ii phi_dot^2 is there to lift the centre of mass."""
    _, r_perp, _ = edge_frame(p, i)
    rise = r_perp - p.r_com[face_up]
    rate = math.sqrt(2.0 * p.mass * p.g * rise / p.theta_0[i, i])
    return p.theta[i, i] * rate / p.i_wheel


def corner_jump_speeds(p: Cubli3DParams, i: int) -> np.ndarray:
    """Wheel speeds (wheel i = 0) whose momentum swings the cube from balance on edge i up to
    balance on the corner. The jump axis n = e_i x r_perp_hat is horizontal and an
    eigenvector of Theta, so the body turns about n alone after the brake."""
    _, r_perp, n = edge_frame(p, i)
    rise = np.linalg.norm(p.r_com) - r_perp
    lam = n @ p.theta_0 @ n            # the wheels spin freely again after the brake
    rate = math.sqrt(2.0 * p.mass * p.g * rise / lam)
    speeds = p.theta @ n * rate / p.i_wheel
    speeds[i] = 0.0
    return speeds


class JumpBalanceController:

    def __init__(self, p: Cubli3DParams = None, c: ControlParams = None):
        self.p = p or Cubli3DParams()
        self.c = c or ControlParams()
        dt = 1.0 / self.c.rate_hz
        self.k_corner = lqr_gain_3d(self.p, dt, *self.c.corner_q, self.c.corner_r)
        self.k_edge = [edge_gain(self.p, i, dt, self.c.edge_q, self.c.edge_r) for i in range(3)]
        self.r_hat = tilt_basis(self.p)[0]
        self.edge_margin, self.corner_margin = self.c.edge_margin, self.c.corner_margin
        self.mode, self.since, self.note = None, 0.0, ''
        self.edge = None              # edge index while on / jumping to an edge
        self.jump_face = None         # face we jumped from (to tell short from overshoot)
        self.target = np.zeros(3)
        self.settled_since = None
        self.still_since = None
        self.airborne = False         # the jump has left its starting pose
        self.corner_jumped = False    # CORNER was entered from a corner jump
        self.failures = 0             # failed jumps in a row (for the retry back-off)
        self.jump_side = None         # sign of the edge tilt the last face -> edge jump started at
        self.brake_efficiency = self.c.brake_efficiency
        self.brake_seen = []          # measured efficiencies
        self.expected_rate = None     # body rate a lossless brake would give
        self.peak_rate = 0.0
        self.brake_t = 0.0
        self.last_step_t = 0.0
        self.log = []                 # (t, mode, note)
        self.t = 0.0

    # ---- helpers -------------------------------------------------------------------
    def rest_needed(self) -> float:
        """Rest before the next jump: longer after each failed jump in a row."""
        c = self.c
        return min(c.rest_time + c.retry_backoff * self.failures, c.max_rest_time)

    def set_mode(self, mode, t, note=''):
        if mode != self.mode:
            self.mode, self.since, self.note = mode, t, note
            self.settled_since = None
            self.airborne = False
            self.log.append((round(t, 3), mode, note))

    def pose(self, u):
        """('face', contacts) / ('edge', i) / ('corner', None) from the up vector."""
        down = contacts_of(u, self.c.floor_eps)
        if len(down) >= 2:
            return 'face', down
        if len(down) == 1:
            return 'edge', next(iter(down))
        return 'corner', None

    def speed_loop(self, wheels, target):
        err = target - wheels
        return np.clip(self.c.spin_gain * err, -self.c.spin_torque, self.c.spin_torque)

    def settled(self, wheels, target, t, which):
        err = np.abs(target - wheels)[which]
        if np.all(err < self.c.spin_tolerance):
            if self.settled_since is None:
                self.settled_since = t
            return t - self.settled_since >= self.c.spin_settle_time
        self.settled_since = None
        return (t - self.since > self.c.spin_timeout
                and np.all(err < 1.5 * self.c.spin_tolerance))

    def start_brake(self, u, w, wheels, which, t):
        """Remember the momentum a lossless brake of `which` would hand over, to measure
        the real brake on the next sample."""
        delta = np.zeros(3)
        delta[which] = wheels[which]
        if len(which) == 1:
            axis = np.eye(3)[which[0]]          # face -> edge: turns about the edge
        else:
            axis = edge_frame(self.p, self.edge)[2]   # edge -> corner: the jump axis n
        self.brake_axis = axis
        self.brake_offered = float(axis @ (self.p.i_wheel * delta))
        self.brake_before = float(axis @ (self.p.theta @ w))
        self.brake_t = t
        self.expected_rate = 1.0   # marks a brake to measure
        return list(which)

    def measure_brake(self, u, w, t):
        """At the first sample where the brake shows (the brake command may reach the
        motors a sample or two late): momentum the body got about the jump axis, with
        one sample's gravity impulse added back, over what the wheels offered."""
        if self.expected_rate is None or t <= self.brake_t + 1e-9:
            return
        raw = self.brake_axis @ (self.p.theta @ w) - self.brake_before
        if abs(self.brake_offered) < 1e-6 or t - self.brake_t > 0.1:
            self.expected_rate = None          # nothing to learn from
            return
        if raw / self.brake_offered < 0.3:
            return                             # not braked yet
        g_torque = self.p.mass * cross3(self.p.r_com, -self.p.g * u)
        got = raw - (self.brake_axis @ g_torque) * (t - self.last_step_t)
        seen = float(np.clip(got / self.brake_offered, 0.5, 1.0))
        self.brake_seen.append(seen)
        self.brake_efficiency = float(np.mean(self.brake_seen[-5:]))
        self.expected_rate = None

    def edge_energy_torque(self, i, u, w) -> float:
        """Flight after a face -> edge brake: steer the energy about edge i to exactly what
        reaches balance. E = 1/2 Theta0_ii phi_dot^2 + M g r cos(phi), dE/dt = -T phi_dot."""
        _, r_perp, _ = edge_frame(self.p, i)
        phi, rate = edge_angle(self.p, i, u), w[i]
        mgr = self.p.mass * self.p.g * r_perp
        error = 0.5 * self.p.theta_0[i, i] * rate ** 2 + mgr * math.cos(phi) - mgr
        return float(np.clip(self.c.energy_gain * error * rate,
                             -self.c.max_torque, self.c.max_torque))

    def corner_energy_torque(self, u, w) -> np.ndarray:
        """Flight after an edge -> corner brake: the same about the jump axis n, with
        E = 1/2 lambda omega_n^2 + M g (r . u) and the wheel torques along n."""
        n = edge_frame(self.p, self.edge)[2]
        rate = w @ n
        lam = n @ self.p.theta_0 @ n
        mg = self.p.mass * self.p.g
        error = 0.5 * lam * rate ** 2 + mg * (self.p.r_com @ u) - mg * np.linalg.norm(self.p.r_com)
        tau = self.c.energy_gain * error * rate
        limit = self.c.max_torque / np.abs(n).max()
        return float(np.clip(tau, -limit, limit)) * n

    def edge_torque(self, i, u, w, wheels):
        phi = edge_angle(self.p, i, u)
        return -float(self.k_edge[i] @ [phi, w[i], wheels[i]]), phi

    def try_catch(self, u, w, t, note):
        """Engage edge or corner balance if the pose allows it and the cube is not falling
        away (or is already close). Returns True if engaged."""
        c = self.c
        kind, which = self.pose(u)
        if kind == 'corner':
            tilt = tilt_angle(self.p, u)
            tipping = np.cross(u, w) @ self.r_hat < 0.0
            if tilt < c.corner_catch and (not tipping or tilt < c.corner_disengage):
                self.set_mode(CORNER, t, note)
                return True
        if kind == 'edge':
            phi = edge_angle(self.p, which, u)
            tipping = phi * w[which] > 0.0
            if abs(phi) < c.edge_catch and (not tipping or abs(phi) < c.edge_disengage):
                self.edge = which
                self.set_mode(EDGE, t, note)
                return True
        return False

    # ---- main step -----------------------------------------------------------------
    def step(self, t, u, w, wheels):
        """One control step. Returns (torques[3], wheels to brake now (list))."""
        self.t = t
        u = np.asarray(u, float) / np.linalg.norm(u)
        w, wheels = np.asarray(w, float), np.asarray(wheels, float)
        c = self.c
        if self.mode is None:
            kind, _ = self.pose(u)
            self.set_mode(FACE, t, 'start')
            if kind != 'face':
                self.try_catch(u, w, t, 'start')
        torque, brake = np.zeros(3), []
        self.measure_brake(u, w, t)
        self.last_step_t = t

        if self.mode == FACE:
            torque = self.speed_loop(wheels, np.zeros(3))   # let the wheels run down
            kind, down = self.pose(u)
            still = kind == 'face' and np.linalg.norm(w) < 0.05
            if not still:
                self.still_since = None
                self.try_catch(u, w, t, 'caught')
            elif self.still_since is None:
                self.still_since = t
            elif c.jump_to_edge and t - self.still_since > self.rest_needed():
                face_up = 3 - sum(down)
                self.edge = min(down)
                phi = edge_angle(self.p, self.edge, u)
                self.target = np.zeros(3)
                self.target[self.edge] = (-math.copysign(1.0, phi) * self.edge_margin
                                          * face_jump_speed(self.p, self.edge, face_up)
                                          / self.brake_efficiency)
                self.jump_face = down
                self.jump_side = math.copysign(1.0, phi)
                self.still_since = None
                self.set_mode(SPIN_EDGE, t, f'edge {self.edge}, wheel target '
                              f'{self.target[self.edge]:.1f} rad/s')

        elif self.mode == SPIN_EDGE:
            torque = self.speed_loop(wheels, self.target)
            if self.settled(wheels, self.target, t, [0, 1, 2]):
                brake = self.start_brake(u, w, wheels, [self.edge], t)
                torque = np.zeros(3)
                self.set_mode(JUMP_EDGE, t, f'brake at {wheels[self.edge]:.1f} rad/s')

        elif self.mode == JUMP_EDGE:
            kind, which = self.pose(u)
            self.airborne |= kind != 'face'
            if self.airborne and kind == 'edge':
                torque[self.edge] = self.edge_energy_torque(self.edge, u, w)
            if kind == 'edge' and which == self.edge and \
                    abs(edge_angle(self.p, self.edge, u)) < c.edge_capture:
                self.set_mode(EDGE, t, f'caught on edge {self.edge} '
                              f'(margin {self.edge_margin:.2f}, '
                              f'brake efficiency {self.brake_efficiency:.2f})')
            elif t - self.since > c.jump_timeout or (kind == 'face' and self.airborne):
                if kind != 'face':
                    self.set_mode(FACE, t, 'edge jump timed out')
                else:
                    if set(which) == set(self.jump_face):
                        self.edge_margin += c.margin_step
                        self.failures += 1
                        note = 'fell short'
                    else:
                        self.edge_margin -= c.margin_step
                        self.failures += 1
                        note = 'overshot'
                    self.set_mode(FACE, t, f'{note}; edge margin now {self.edge_margin:.2f}')
                    self.try_catch(u, w, t, 'caught after the edge jump')

        elif self.mode in (EDGE, SPIN_CORNER):
            i = self.edge
            torque_i, phi = self.edge_torque(i, u, w, wheels)
            tipping = phi * w[i] > 0.0
            if abs(phi) > c.edge_catch or (abs(phi) > c.edge_disengage and tipping):
                note = ''
                if self.jump_side is not None and t - self.since < 1.0:
                    # Lost right after the jump's catch: the jump was off. Falling back to
                    # the side it came from means it was short.
                    short = math.copysign(1.0, phi) == self.jump_side
                    self.edge_margin += c.margin_step if short else -c.margin_step
                    self.failures += 1
                    note = (f' ({"short" if short else "overshot"}; edge margin now '
                            f'{self.edge_margin:.2f})')
                self.jump_side = None
                self.set_mode(FACE, t, f'lost edge balance at {math.degrees(phi):.1f} deg{note}')
            else:
                others = [j for j in range(3) if j != i]
                if self.mode == EDGE:
                    torque = self.speed_loop(wheels, np.zeros(3))
                    if c.jump_to_corner and t - self.since > max(c.edge_rest_time,
                                                                 self.rest_needed()):
                        self.jump_side = None
                        self.failures = 0     # the edge balance held
                        self.target = (self.corner_margin * corner_jump_speeds(self.p, i)
                                       / self.brake_efficiency)
                        self.set_mode(SPIN_CORNER, t, 'wheel targets '
                                      + ', '.join(f'{v:.1f}' for v in self.target[others]))
                else:
                    torque = self.speed_loop(wheels, self.target)
                    calm = abs(phi) < math.radians(0.5) and abs(w[i]) < 0.05
                    if self.settled(wheels, self.target, t, others) and calm:
                        brake = self.start_brake(u, w, wheels, others, t)
                        torque = np.zeros(3)
                        self.set_mode(JUMP_CORNER, t, 'brake at '
                                      + ', '.join(f'{wheels[j]:.1f}' for j in others))
                        return torque, brake
                torque[i] = torque_i

        elif self.mode == JUMP_CORNER:
            kind, which = self.pose(u)
            self.airborne |= kind == 'corner'
            if self.airborne and kind == 'corner':
                torque = self.corner_energy_torque(u, w)
            if kind == 'corner' and tilt_angle(self.p, u) < c.corner_capture:
                self.corner_jumped = True
                self.set_mode(CORNER, t, f'caught on the corner '
                              f'(margin {self.corner_margin:.2f}, '
                              f'brake efficiency {self.brake_efficiency:.2f})')
            elif self.airborne and kind != 'corner':
                if kind == 'edge' and which == self.edge:
                    self.corner_margin += c.margin_step
                    self.failures += 1
                    note = 'fell short'
                else:
                    self.corner_margin -= c.margin_step
                    self.failures += 1
                    note = 'overshot'
                self.set_mode(FACE, t, f'{note}; corner margin now {self.corner_margin:.2f}')
                self.try_catch(u, w, t, 'caught after the corner jump')
            elif t - self.since > c.jump_timeout:
                self.set_mode(FACE, t, 'corner jump timed out')

        elif self.mode == CORNER:
            tilt = tilt_angle(self.p, u)
            tipping = np.cross(u, w) @ self.r_hat < 0.0
            if tilt > c.corner_give_up or (tilt > c.corner_disengage and tipping):
                note = ''
                if self.corner_jumped and t - self.since < 1.0:
                    # Lost right after the catch: falling back towards the edge it came
                    # from means the jump was short.
                    r_perp = edge_frame(self.p, self.edge)[0]
                    short = u @ r_perp > self.r_hat @ r_perp
                    self.corner_margin += c.margin_step if short else -c.margin_step
                    self.failures += 1
                    note = (f' ({"short" if short else "overshot"}; corner margin now '
                            f'{self.corner_margin:.2f})')
                self.corner_jumped = False
                self.set_mode(FACE, t, f'lost corner balance at {math.degrees(tilt):.1f} deg{note}')
            else:
                torque = -(self.k_corner @ state_vector(self.p, u, w, wheels))
                if t - self.since > 2.0:
                    self.failures = 0

        return np.clip(torque, -c.max_torque, c.max_torque), brake
