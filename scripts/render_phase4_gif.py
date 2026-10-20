#!/usr/bin/python3
"""Simulate the 3D Cubli jumping up face -> edge -> corner and render it as an animated GIF.

    python3 scripts/render_phase4_gif.py media/cubli3d_jump_up.gif

Runs the same plant, estimator and controller code as the ROS nodes (sim3d.closed_loop:
contact physics, Nidec motors with torque error and ripple, lossy brakes, IMU and encoder
noise, one step of control delay), offline so the run is repeatable. Lying on a face it
calibrates, jumps onto an edge, balances there, jumps onto the corner and is pushed.
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src',
                                'cubli_control'))
from cubli_control import contact3d  # noqa: E402
from cubli_control.model3d import quat_to_matrix  # noqa: E402
from cubli_control.sim3d import REALISTIC, closed_loop  # noqa: E402

WHEEL_COLOURS = ['#d64545', '#2e9e5b', '#3b6fd6']
STYLES = {
    'calib': ('Lying on a face: calibrating', '#7f7f7f'),
    'FACE': ('Lying on a face', '#7f7f7f'),
    'SPIN_EDGE': ('Spinning up the x wheel', '#e8a33d'),
    'JUMP_EDGE': ('Brake! Jumping onto the edge', '#d64545'),
    'EDGE': ('Balancing on the edge (x wheel)', '#2e9e5b'),
    'SPIN_CORNER': ('On the edge, spinning up y + z wheels', '#e8a33d'),
    'JUMP_CORNER': ('Brake y + z! Jumping onto the corner', '#d64545'),
    'CORNER': ('Balancing on the corner (3 wheels)', '#2e9e5b'),
    'push': ('Pushed!', '#8e44ad'),
}


def simulate(seconds, pushes, seed):
    log = []

    def on_step(sim, ctrl, up):
        log.append((sim.t, sim.x.copy(), ctrl.mode or 'calib',
                    contact3d.com_height(sim.x, sim.p)))
    sim, ctrl = closed_loop(contact3d.face_state((0, 1), yaw=0.6), seconds, noise=REALISTIC,
                            seed=seed, pushes=pushes, on_step=on_step)
    return sim.p, log, ctrl


def cube_faces(side):
    v = np.array([[i, j, k] for i in (0, side) for j in (0, side) for k in (0, side)])
    idx = [[0, 1, 3, 2], [4, 5, 7, 6], [0, 1, 5, 4], [2, 3, 7, 6], [0, 2, 6, 4], [1, 3, 7, 5]]
    return v, idx


def wheel_points(p, i, angle):
    """Rim circle and spoke of wheel i in the body frame."""
    centre = p.wheel_centre(i)
    e1, e2 = np.eye(3)[(i + 1) % 3], np.eye(3)[(i + 2) % 3]
    s = np.linspace(0, 2 * math.pi, 40)
    rim = centre + p.wheel_radius * (np.outer(np.cos(s), e1) + np.outer(np.sin(s), e2))
    tip = centre + 0.9 * p.wheel_radius * (math.cos(angle) * e1 + math.sin(angle) * e2)
    return rim, np.array([centre, tip])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('output')
    parser.add_argument('--seconds', type=float, default=22.0)
    parser.add_argument('--speed', type=float, default=2.0, help='playback speed factor')
    parser.add_argument('--fps', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--push-after', type=float, default=3.0,
                        help='push this long after reaching the corner [s]')
    args = parser.parse_args()

    # First run finds when it reaches the corner, so the push lands on a balanced cube.
    p, log, ctrl = simulate(args.seconds, [], args.seed)
    corner_t = next((t for t, mode, _ in ctrl.log if mode == 'CORNER'), None)
    pushes = [(corner_t + args.push_after, (0.25, -0.1, 0.0))] if corner_t else []
    if pushes:
        p, log, ctrl = simulate(args.seconds, pushes, args.seed)
    t = np.array([r[0] for r in log])
    states = [r[1] for r in log]
    modes = [r[2] for r in log]
    height = np.array([r[3] for r in log]) * 1000.0
    wheel_speeds = np.array([s[7:10] for s in states])

    fig = plt.figure(figsize=(8.8, 4.6), dpi=85)
    ax3d = fig.add_axes([0.0, 0.03, 0.47, 0.82], projection='3d')
    ax_h = fig.add_axes([0.58, 0.56, 0.39, 0.30])
    ax_w = fig.add_axes([0.58, 0.12, 0.39, 0.30], sharex=ax_h)
    fig.text(0.5, 0.95, '3D Cubli: face -> edge -> corner (sim: 3 x Nidec 24H + brakes, '
             'ICM-42688, noise on)', ha='center', fontsize=10.5, fontweight='bold')
    mode_text = fig.text(0.24, 0.87, '', ha='center', fontsize=11, fontweight='bold')
    info = fig.text(0.24, 0.035, '', ha='center', fontsize=9, color='#444444')

    ax_h.plot(t, height, color='#999999', alpha=0.2, lw=3)
    for level, name in ((p.r_com[2], 'face'), (contact3d.edge_frame(p, 0)[1], 'edge'),
                        (np.linalg.norm(p.r_com), 'corner')):
        ax_h.axhline(level * 1000.0, color='#2e9e5b', lw=0.7, ls='--', alpha=0.6)
        ax_h.text(t[-1], level * 1000.0 + 1.5, name, ha='right', fontsize=7, color='#2e9e5b')
    ax_h.set_ylabel('centre of mass\nheight [mm]', fontsize=9)
    ax_h.set_ylim(55, 125)
    for i in range(3):
        ax_w.plot(t, wheel_speeds[:, i], color=WHEEL_COLOURS[i], alpha=0.15, lw=1)
    ax_w.set_ylabel('wheel speed [rad/s]', fontsize=9)
    ax_w.set_xlabel('time [s]', fontsize=9)
    ax_h.set_xlim(0, t[-1])
    for ax in (ax_h, ax_w):
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=8)
        for at, _ in pushes:
            ax.axvline(at, color='#8e44ad', lw=0.8, alpha=0.7)
        for k in range(1, len(modes)):
            if modes[k] in ('JUMP_EDGE', 'JUMP_CORNER') and modes[k - 1] != modes[k]:
                ax.axvline(t[k], color='#d64545', lw=0.8, alpha=0.6)
    h_line, = ax_h.plot([], [], color='#34495e', lw=1.8)
    speed_lines = [ax_w.plot([], [], color=c, lw=1.4, label=n)[0]
                   for c, n in zip(WHEEL_COLOURS, ('x', 'y', 'z'))]
    ax_w.legend(loc='upper right', fontsize=8, ncol=3, framealpha=0.9)
    verts, faces = cube_faces(p.side)

    def draw(ft):
        i = min(np.searchsorted(t, ft), len(t) - 1)
        rot = quat_to_matrix(states[i][0:4])
        ax3d.cla()
        ax3d.set_xlim(-0.13, 0.13)
        ax3d.set_ylim(-0.13, 0.13)
        ax3d.set_zlim(0.0, 0.26)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.view_init(elev=14, azim=-60)
        ax3d.set_axis_off()
        g = np.linspace(-0.13, 0.13, 7)
        for v in g:
            ax3d.plot([v, v], [g[0], g[-1]], [0, 0], color='#cccccc', lw=0.6)
            ax3d.plot([g[0], g[-1]], [v, v], [0, 0], color='#cccccc', lw=0.6)
        world = verts @ rot.T
        ax3d.add_collection3d(Poly3DCollection(
            [world[f] for f in faces], facecolor='#c9d3e0', edgecolor='#34495e', lw=1.0,
            alpha=0.25))
        for w in range(3):
            rim, spoke = wheel_points(p, w, states[i][10 + w])
            ax3d.plot(*(rim @ rot.T).T, color=WHEEL_COLOURS[w], lw=2)
            ax3d.plot(*(spoke @ rot.T).T, color='#ff8c1a', lw=2.5)
        ax3d.scatter([0], [0], [0], color='#d64545', s=20)

        label = modes[i]
        if any(0.0 <= t[i] - at < 0.6 for at, _ in pushes):
            label = 'push'
        text, colour = STYLES.get(label, (label, '#333333'))
        mode_text.set_text(text)
        mode_text.set_color(colour)
        info.set_text(f't = {t[i]:.1f} s   centre of mass {height[i]:.0f} mm')
        h_line.set_data(t[:i + 1], height[:i + 1])
        for w, line in enumerate(speed_lines):
            line.set_data(t[:i + 1], wheel_speeds[:i + 1, w])

    frames = np.arange(0.0, t[-1], args.speed / args.fps)
    FuncAnimation(fig, draw, frames=frames, blit=False).save(
        args.output, writer=PillowWriter(fps=args.fps))
    print(f'Wrote {args.output} ({len(frames)} frames); modes: '
          + ', '.join(f'{m}@{tt:.1f}' for tt, m, _ in ctrl.log))


if __name__ == '__main__':
    main()
