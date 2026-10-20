"""3D Cubli dynamics: a cube balancing on one corner with three reaction wheels.

Model after Muehlebach and D'Andrea, "Nonlinear Analysis and Control of a Reaction-Wheel-
Based 3-D Inverted Pendulum" (IEEE TCST 2017), with this build's hardware: three Nidec
24H motors driving the wheels and one IMU (ICM-42688-class) at the cube centre.

Body frame: origin at the pivot corner, axes along the three cube edges that meet there,
so the cube fills [0, side]^3. Wheel i spins about body axis i.

State x (13): [q (4, body -> world, w x y z), omega (3, body rate in body frame),
               omega_w (3, wheel speeds relative to the body), theta_w (3, wheel angles)]
Input T (3): motor torque on each wheel [N m]; the body feels -T.

With the angular momentum about the pivot L = Theta omega + I_w omega_w (body frame):
  Theta_0 omega_dot = -omega x L + M r_com x g_b - c_b omega - (T - c_w omega_w)
  I_w (omega_dot + omega_w_dot) = T - c_w omega_w
where Theta is the inertia of everything about the pivot, I_w the wheels' axial inertia
and Theta_0 = Theta - I_w * 1.
"""

from dataclasses import dataclass
from functools import cached_property
import math

import numpy as np

FALL_TILT = math.acos(math.sqrt(2.0 / 3.0))   # 35.26 deg: a second corner touches the floor


@dataclass(frozen=True)
class Cubli3DParams:
    side: float = 0.15            # cube edge [m]
    m_frame: float = 0.75         # MEASURE: frame, electronics, battery, motor stators [kg]
    m_wheel: float = 0.204        # MEASURE: one wheel + rotor [kg]
    i_wheel: float = 0.57e-3      # MEASURE: wheel axial inertia [kg m^2] (transverse = half)
    wheel_radius: float = 0.058   # [m]
    wheel_thickness: float = 0.010  # [m]
    # Wheel i sits at cube centre - wheel_offset along axis i, near the face at the pivot.
    # Two orthogonal wheels miss each other only if wheel_offset > wheel_radius.
    wheel_offset: float = 0.063   # [m]
    c_b: float = 1.0e-3           # pivot friction [N m s/rad]
    c_w: float = 0.05e-3          # wheel bearing friction [N m s/rad]
    g: float = 9.81
    max_torque: float = 0.08      # per wheel, below the Nidec's stall torque [N m]
    stall_torque: float = 0.10    # MEASURE: Nidec 24H at full effort, wheel stopped [N m]
    no_load_speed: float = 600.0  # MEASURE: Nidec 24H at full effort, no load [rad/s]

    @property
    def centre(self) -> np.ndarray:
        return np.full(3, self.side / 2.0)

    def wheel_centre(self, i: int) -> np.ndarray:
        c = self.centre.copy()
        c[i] -= self.wheel_offset
        return c

    def wheel_clearance(self) -> float:
        """Smallest gap [m] between two wheels, or between a wheel and the cube faces."""
        between = self.wheel_offset - self.wheel_radius
        to_face = self.side / 2 - self.wheel_offset - self.wheel_thickness / 2
        to_side = self.side / 2 - self.wheel_radius
        return min(between, to_face, to_side)

    @property
    def mass(self) -> float:
        return self.m_frame + 3 * self.m_wheel

    @cached_property
    def r_com(self) -> np.ndarray:
        """Pivot -> centre of mass, body frame. It lies on the body diagonal."""
        total = self.m_frame * self.centre
        for i in range(3):
            total = total + self.m_wheel * self.wheel_centre(i)
        return total / self.mass

    @cached_property
    def theta(self) -> np.ndarray:
        """Inertia of the whole cube (wheels locked) about the pivot, body frame."""
        def point(m, p):
            return m * (p @ p * np.eye(3) - np.outer(p, p))
        # Frame as a uniform solid cube (MEASURE: the real one is hollower, i.e. larger).
        inertia = self.m_frame * self.side ** 2 / 6.0 * np.eye(3) + point(self.m_frame, self.centre)
        for i in range(3):
            own = np.full(3, self.i_wheel / 2.0)
            own[i] = self.i_wheel
            inertia = inertia + np.diag(own) + point(self.m_wheel, self.wheel_centre(i))
        return inertia

    @cached_property
    def theta_0(self) -> np.ndarray:
        return self.theta - self.i_wheel * np.eye(3)

    @cached_property
    def theta_0_inv(self) -> np.ndarray:
        return np.linalg.inv(self.theta_0)


# ---- Quaternions (w, x, y, z) --------------------------------------------------------
def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz,
                     aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx,
                     aw * bz + ax * by - ay * bx + az * bw])


def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, float)
    n = np.linalg.norm(axis)
    if n < 1e-12 or angle == 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return np.concatenate([[math.cos(angle / 2)], math.sin(angle / 2) * axis / n])


def quat_from_vectors(u, v):
    """Shortest rotation taking unit vector u to unit vector v."""
    u = np.asarray(u, float) / np.linalg.norm(u)
    v = np.asarray(v, float) / np.linalg.norm(v)
    axis = np.cross(u, v)
    return quat_from_axis_angle(axis, math.atan2(np.linalg.norm(axis), u @ v))


def quat_to_matrix(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def cross3(a, b) -> np.ndarray:
    """np.cross for 3-vectors, without its overhead (it dominates the simulation time)."""
    return np.array([a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                     a[0] * b[1] - a[1] * b[0]])


def up_in_body(q) -> np.ndarray:
    """World +z expressed in the body frame (what a still accelerometer points at)."""
    w, x, y, z = q
    n = w * w + x * x + y * y + z * z   # bottom row of the rotation matrix
    return np.array([2 * (x * z - w * y), 2 * (y * z + w * x), w * w - x * x - y * y + z * z]) / n


# ---- Upright equilibrium and tilt coordinates -----------------------------------------
def tilt_basis(p: Cubli3DParams):
    """(r_hat, a, b): unit vector to the centre of mass and two unit vectors normal to it.

    Near upright the body-frame up vector is u = r_hat + p_a a + p_b b; (p_a, p_b) are
    the two tilt coordinates the controller works with (yaw is free and unobservable).
    """
    r_hat = p.r_com / np.linalg.norm(p.r_com)
    a = np.cross(r_hat, [0.0, 0.0, 1.0])
    a /= np.linalg.norm(a)
    b = np.cross(r_hat, a)
    return r_hat, a, b


def upright_quat(p: Cubli3DParams, yaw: float = 0.0):
    """Orientation with the centre of mass straight above the pivot."""
    q = quat_from_vectors(tilt_basis(p)[0], [0.0, 0.0, 1.0])
    return quat_mul(quat_from_axis_angle([0.0, 0.0, 1.0], yaw), q)


def tilted_quat(p: Cubli3DParams, tilt: float, direction: float = 0.0, yaw: float = 0.0):
    """Upright, then tipped by `tilt` rad towards the horizontal `direction` (rad)."""
    axis = [-math.sin(direction), math.cos(direction), 0.0]
    return quat_mul(quat_from_axis_angle(axis, tilt), upright_quat(p, yaw))


def tilt_angle(p: Cubli3DParams, u) -> float:
    """Angle between the body-frame up vector u and the pivot -> centre-of-mass line."""
    r_hat = tilt_basis(p)[0]
    u = np.asarray(u, float) / np.linalg.norm(u)
    return math.atan2(np.linalg.norm(np.cross(u, r_hat)), u @ r_hat)


def tilt_coords(p: Cubli3DParams, u) -> np.ndarray:
    _, a, b = tilt_basis(p)
    return np.array([a @ u, b @ u])


def initial_state(p: Cubli3DParams, tilt=0.0, direction=0.0, yaw=0.0) -> np.ndarray:
    return np.concatenate([tilted_quat(p, tilt, direction, yaw), np.zeros(9)])


# ---- Dynamics ------------------------------------------------------------------------
def accelerations(x, torque, p: Cubli3DParams):
    """(omega_dot, omega_w_dot) for state x and wheel torques."""
    q, w, ww = x[0:4], x[4:7], x[7:10]
    torque = np.asarray(torque, float)
    g_body = -p.g * up_in_body(q)
    momentum = p.theta @ w + p.i_wheel * ww
    wheel_drive = torque - p.c_w * ww
    w_dot = p.theta_0_inv @ (-cross3(w, momentum) + p.mass * cross3(p.r_com, g_body)
                             - p.c_b * w - wheel_drive)
    return w_dot, wheel_drive / p.i_wheel - w_dot


def dynamics(x, torque, p: Cubli3DParams) -> np.ndarray:
    q, w, ww = x[0:4], x[4:7], x[7:10]
    w_dot, ww_dot = accelerations(x, torque, p)
    q_dot = 0.5 * quat_mul(q, np.concatenate([[0.0], w]))
    return np.concatenate([q_dot, w_dot, ww_dot, ww])


def rk4_step(x, torque, dt: float, p: Cubli3DParams) -> np.ndarray:
    k1 = dynamics(x, torque, p)
    k2 = dynamics(x + 0.5 * dt * k1, torque, p)
    k3 = dynamics(x + 0.5 * dt * k2, torque, p)
    k4 = dynamics(x + dt * k3, torque, p)
    out = x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    out[0:4] /= np.linalg.norm(out[0:4])
    return out


def has_fallen(x, p: Cubli3DParams) -> bool:
    return tilt_angle(p, up_in_body(x[0:4])) >= FALL_TILT


def fallen_step(x, torque, dt: float, p: Cubli3DParams) -> np.ndarray:
    """Lying on the floor: the body stays put, only the wheels move."""
    out = x.copy()
    out[4:7] = 0.0
    ww_dot = (np.asarray(torque, float) - p.c_w * x[7:10]) / p.i_wheel
    out[7:10] = x[7:10] + dt * ww_dot
    out[10:13] = x[10:13] + dt * out[7:10]
    return out


def motor_limits(p: Cubli3DParams, wheel_speed, torque):
    """Clip commanded torques to what a Nidec at full effort can give at this speed."""
    wheel_speed = np.asarray(wheel_speed, float)
    hi = p.stall_torque * (1.0 - wheel_speed / p.no_load_speed)
    lo = p.stall_torque * (-1.0 - wheel_speed / p.no_load_speed)
    torque = np.clip(np.asarray(torque, float), -p.max_torque, p.max_torque)
    return np.clip(torque, lo, hi)


def energy(x, p: Cubli3DParams) -> float:
    """Kinetic + potential energy (constant without friction or torque)."""
    q, w, ww = x[0:4], x[4:7], x[7:10]
    kinetic = 0.5 * w @ p.theta_0 @ w + 0.5 * p.i_wheel * np.sum((w + ww) ** 2)
    height = (quat_to_matrix(q) @ p.r_com)[2]
    return kinetic + p.mass * p.g * height


def world_momentum(x, p: Cubli3DParams) -> np.ndarray:
    q, w, ww = x[0:4], x[4:7], x[7:10]
    return quat_to_matrix(q) @ (p.theta @ w + p.i_wheel * ww)


# ---- Linearisation about upright -------------------------------------------------------
def linearized(p: Cubli3DParams):
    """Continuous (A, B) for z = [p_a, p_b, omega (3), omega_w (3)], u = T (3)."""
    r_hat, a, b = tilt_basis(p)
    inv = p.theta_0_inv
    mgr = p.mass * p.g * np.linalg.norm(p.r_com)
    big_a = np.zeros((8, 8))
    big_b = np.zeros((8, 3))
    # Tilt: u_dot = u x omega, so p_a_dot = a . (r_hat x omega) = (a x r_hat) . omega.
    big_a[0, 2:5] = np.cross(a, r_hat)
    big_a[1, 2:5] = np.cross(b, r_hat)
    # Gravity torque M r x g_b = -M g |r| r_hat x u, with u = r_hat + p_a a + p_b b.
    gravity = -mgr * np.column_stack([np.cross(r_hat, a), np.cross(r_hat, b)])
    big_a[2:5, 0:2] = inv @ gravity
    big_a[2:5, 2:5] = -p.c_b * inv
    big_a[2:5, 5:8] = p.c_w * inv
    big_b[2:5, :] = -inv
    # Wheels: omega_w_dot = (T - c_w omega_w) / I_w - omega_dot.
    big_a[5:8, :] = -big_a[2:5, :]
    big_a[5:8, 5:8] -= p.c_w / p.i_wheel * np.eye(3)
    big_b[5:8, :] = np.eye(3) / p.i_wheel - big_b[2:5, :]
    return big_a, big_b


def lqr_gain_3d(p: Cubli3DParams, dt: float, q_tilt: float, q_rate: float,
                q_wheel_speed: float, r_torque: float) -> np.ndarray:
    """Discrete LQR gain (3 x 8) for T = -K z."""
    from scipy.linalg import solve_discrete_are

    from cubli_control.lqr import discretize

    ad, bd = discretize(*linearized(p), dt)
    q = np.diag([q_tilt] * 2 + [q_rate] * 3 + [q_wheel_speed] * 3)
    r = r_torque * np.eye(3)
    s = solve_discrete_are(ad, bd, q, r)
    return np.linalg.solve(r + bd.T @ s @ bd, bd.T @ s @ ad)


def state_vector(p: Cubli3DParams, u, w, ww) -> np.ndarray:
    """Controller state z from an up vector, body rate and wheel speeds."""
    return np.concatenate([tilt_coords(p, np.asarray(u) / np.linalg.norm(u)), w, ww])
