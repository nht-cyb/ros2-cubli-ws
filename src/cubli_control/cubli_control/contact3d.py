"""Floor contact for the 3D Cubli: lying on a face, balancing on an edge, or on the corner.

The cube always pivots on the same corner (the body-frame origin). Its three neighbour
vertices v_i = side * e_i are the ones that can also touch the floor, and which of them
do sets the contact mode:

  {}      corner: free rotation about the pivot (model3d.dynamics)
  {i}     edge i: the edge along body axis i lies on the floor; the body can only turn
          about e_i:  Theta0_ii * phi_ddot = e_i . rhs
  {i, j}  face: the face spanned by e_i and e_j lies on the floor; the body is still

where rhs = -omega x L + M r x g_b - c_b omega - (T - c_w omega_w) is the corner model's
right-hand side. A vertex lifts off when the floor would have to pull on it (negative
normal force), and lands inelastically: on landing, the angular momentum about the new
contact edge is kept and the rest is lost.

Brakes act as impulses: locking a wheel hands its momentum to the body (a fraction
`efficiency` of it; the rest is lost, as in a real brake), and the contacts then decide
whether the cube lifts off.
"""

import math

import numpy as np

from cubli_control.model3d import (
    Cubli3DParams, cross3, quat_from_axis_angle, quat_from_vectors, quat_mul, quat_to_matrix,
    up_in_body)

CORNER = frozenset()
_E = np.eye(3)
EPS_FLOOR = 1e-9


def edge_frame(p: Cubli3DParams, i: int):
    """(r_perp, |r_perp|, n) for edge i: centre-of-mass offset from the edge, and the unit
    vector n = e_i x r_perp_hat. The edge tilt phi has u = cos(phi) r_perp_hat - sin(phi) n,
    so phi_dot = omega . e_i, and gravity gives +M g |r_perp| sin(phi) about e_i."""
    e = np.eye(3)[i]
    r_perp = p.r_com - (p.r_com @ e) * e
    norm = float(np.linalg.norm(r_perp))
    r_hat = r_perp / norm
    return r_hat, norm, cross3(e, r_hat)


def edge_angle(p: Cubli3DParams, i: int, u) -> float:
    """Tilt about edge i away from the centre of mass straight above the edge [rad]."""
    r_hat, _, n = edge_frame(p, i)
    return math.atan2(-(n @ u), r_hat @ u)


def edge_up(p: Cubli3DParams, i: int, phi: float) -> np.ndarray:
    r_hat, _, n = edge_frame(p, i)
    return math.cos(phi) * r_hat - math.sin(phi) * n


def state_from_up(u, yaw: float = 0.0) -> np.ndarray:
    """A still state whose body-frame up vector is u."""
    q = quat_from_vectors(u, [0.0, 0.0, 1.0])
    q = quat_mul(quat_from_axis_angle([0.0, 0.0, 1.0], yaw), q)
    return np.concatenate([q, np.zeros(9)])


def face_state(face=(0, 1), yaw: float = 0.0) -> np.ndarray:
    """Lying on the face spanned by body axes face[0], face[1]."""
    k = 3 - face[0] - face[1]
    return state_from_up(np.eye(3)[k], yaw)


def edge_state(p: Cubli3DParams, i: int, phi: float = 0.0, yaw: float = 0.0) -> np.ndarray:
    return state_from_up(edge_up(p, i, phi), yaw)


def contacts_of(u, eps: float = 1e-6) -> frozenset:
    """Neighbour vertices on (or below) the floor for body-frame up vector u."""
    return frozenset(i for i in range(3) if u[i] <= eps)


def _rhs(x, torque, p: Cubli3DParams):
    q, w, ww = x[0:4], x[4:7], x[7:10]
    momentum = p.theta @ w + p.i_wheel * ww
    g_body = -p.g * up_in_body(q)
    return (-cross3(w, momentum) + p.mass * cross3(p.r_com, g_body)
            - p.c_b * w - (np.asarray(torque, float) - p.c_w * ww))


def accelerations(x, torque, p: Cubli3DParams, contacts: frozenset):
    """(omega_dot, omega_w_dot, contact torque about the pivot) in this contact mode."""
    rhs = _rhs(x, torque, p)
    if not contacts:
        w_dot = p.theta_0_inv @ rhs
    elif len(contacts) == 1:
        (i,) = contacts
        w_dot = np.zeros(3)
        w_dot[i] = rhs[i] / p.theta_0[i, i]
    else:
        w_dot = np.zeros(3)
    tau_c = p.theta_0 @ w_dot - rhs
    ww_dot = (np.asarray(torque, float) - p.c_w * x[7:10]) / p.i_wheel - w_dot
    return w_dot, ww_dot, tau_c


def normal_forces(x, torque, p: Cubli3DParams, contacts: frozenset) -> dict:
    """Vertical floor force [N] on each contact vertex (negative: it would lift off)."""
    _, _, tau_c = accelerations(x, torque, p, contacts)
    u = up_in_body(x[0:4])
    return {i: float(u @ cross3(tau_c / p.side, _E[i])) for i in contacts}


def dynamics(x, torque, p: Cubli3DParams, contacts: frozenset, with_forces=False):
    """dx/dt in this contact mode; with_forces also returns the contact normal forces."""
    q, w = x[0:4], x[4:7]
    w_dot, ww_dot, tau_c = accelerations(x, torque, p, contacts)
    q_dot = 0.5 * quat_mul(q, np.concatenate([[0.0], w]))
    xdot = np.concatenate([q_dot, w_dot, ww_dot, x[7:10]])
    if not with_forces:
        return xdot
    u = up_in_body(q)
    return xdot, {i: float(u @ cross3(tau_c / p.side, _E[i])) for i in contacts}


def _snap(x, contacts: frozenset) -> np.ndarray:
    """Rotate the body (minimally) so the contact vertices sit exactly on the floor."""
    u = up_in_body(x[0:4])
    if max(abs(u[i]) for i in contacts) < 1e-9:
        return x
    u_new = u.copy()
    for i in contacts:
        u_new[i] = 0.0
    u_new /= np.linalg.norm(u_new)
    out = x.copy()
    out[0:4] = quat_mul(x[0:4], quat_from_vectors(u_new, u))
    out[0:4] /= np.linalg.norm(out[0:4])
    return out


def _project(x, p: Cubli3DParams, contacts: frozenset) -> np.ndarray:
    """Body rate after an inelastic landing: keep the momentum about the contact edge."""
    out = x.copy()
    w = x[4:7]
    if len(contacts) == 1:
        (i,) = contacts
        out[4:7] = 0.0
        out[4 + i] = (p.theta[i] @ w) / p.theta[i, i]
    elif len(contacts) >= 2:
        out[4:7] = 0.0
    return out


def lift_off(x, torque, p: Cubli3DParams, contacts: frozenset) -> frozenset:
    """Drop contacts the floor would have to pull on, until the rest push."""
    while contacts:
        forces = normal_forces(x, torque, p, contacts)
        lifting = [i for i, f in forces.items() if f < 0.0]
        if not lifting:
            break
        contacts = contacts - {min(lifting, key=lambda i: forces[i])}
    return contacts


def step(x, torque, dt: float, p: Cubli3DParams, contacts: frozenset):
    """One RK4 step with contacts. Returns (x, contacts)."""
    k1, forces = dynamics(x, torque, p, contacts, with_forces=True)
    if forces and min(forces.values()) < 0.0:   # a vertex lifts off: drop it, then redo k1
        contacts = lift_off(x, torque, p, contacts)
        k1 = dynamics(x, torque, p, contacts)
    k2 = dynamics(x + 0.5 * dt * k1, torque, p, contacts)
    k3 = dynamics(x + 0.5 * dt * k2, torque, p, contacts)
    k4 = dynamics(x + dt * k3, torque, p, contacts)
    out = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    out[0:4] /= np.linalg.norm(out[0:4])
    if len(contacts) >= 2:
        out[4:7] = 0.0
    u = up_in_body(out[0:4])
    landed = frozenset(i for i in range(3) if i not in contacts and u[i] <= EPS_FLOOR
                       and not _vertex_rising(out, i, out[4:7]))
    if landed:
        contacts = contacts | landed
        out = _project(out, p, contacts)
    if contacts:
        out = _snap(out, contacts)
    return out, contacts


def _vertex_rising(x, i: int, w) -> bool:
    """Whether vertex i moves up for body rate w (its height is side * u_i)."""
    return up_in_body(x[0:4]) @ cross3(w, _E[i]) > 1e-9


def brake(x, p: Cubli3DParams, wheels, efficiency: float, contacts: frozenset):
    """Lock the wheels in `wheels` (indices) at once. Returns (x, contacts)."""
    out = x.copy()
    delta = np.zeros(3)
    for i in wheels:
        delta[i] = x[7 + i]
        out[7 + i] = 0.0
    momentum = p.theta @ x[4:7] + efficiency * p.i_wheel * delta   # body + locked wheels
    w_free = np.linalg.solve(p.theta, momentum)
    remaining = frozenset(i for i in contacts if not _vertex_rising(x, i, w_free))
    if len(remaining) == 1:
        (i,) = remaining
        w = np.zeros(3)
        w[i] = momentum[i] / p.theta[i, i]
        # Turning about edge i must actually lift the other vertex off the face.
        if contacts - remaining and not all(_vertex_rising(x, j, w) for j in contacts - remaining):
            remaining = contacts
    if not remaining:
        out[4:7] = w_free
    elif len(remaining) == 1:
        out[4:7] = w
    else:
        out[4:7] = 0.0
    return out, remaining


def kinetic_energy_needed(p: Cubli3DParams, start_height: float, end_height: float) -> float:
    return p.mass * p.g * (end_height - start_height)


def com_height(x, p: Cubli3DParams) -> float:
    return float((quat_to_matrix(x[0:4]) @ p.r_com)[2])
