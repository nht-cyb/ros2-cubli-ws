#!/usr/bin/python3
"""Monte Carlo of the 3D Cubli jump-up: face -> edge -> corner under sensor + hardware noise.

    python3 scripts/monte_carlo_3d.py 40 > tuning/phase4_monte_carlo.txt

Each run gets its own hardware: motor torque constant 0.90-1.05 x nominal, brake
efficiency mean 0.85-0.97 (+-0.02 per brake), IMU biases drawn around the datasheet
values, plus the per-sample noise, torque ripple and encoder quantisation of the sim.
The controller doesn't know any of it. Starts lying on a face, runs `--seconds` long.
"""

import argparse
import math
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'src',
                                'cubli_control'))
from cubli_control import contact3d  # noqa: E402
from cubli_control.model3d import tilt_angle, up_in_body  # noqa: E402
from cubli_control.sensors3d import Sensor3DParams  # noqa: E402
from cubli_control.sim3d import HardwareNoise, closed_loop  # noqa: E402


def run(args):
    seed, seconds = args
    rng = np.random.default_rng(1000 + seed)
    noise = HardwareNoise(torque_scale=rng.uniform(0.90, 1.05), torque_noise_rel=0.03,
                          torque_noise_abs=5e-4, brake_efficiency=rng.uniform(0.85, 0.97),
                          brake_efficiency_std=0.02)
    sensors = Sensor3DParams(gyro_bias=tuple(rng.normal(0.0, 0.006, 3)),
                             accel_bias=tuple(rng.normal(0.0, 0.025, 3)))
    tilts, wheel_max = [], [0.0]

    def on_step(sim, ctrl, up):
        wheel_max[0] = max(wheel_max[0], float(np.abs(sim.x[7:10]).max()))
        if ctrl.mode == 'CORNER':
            tilts.append((sim.t, tilt_angle(sim.p, up_in_body(sim.x[0:4]))))

    sim, ctrl = closed_loop(contact3d.face_state((0, 1)), seconds, noise=noise, seed=seed,
                            sensors=sensors, on_step=on_step)
    first = {m: next((t for t, mode, _ in ctrl.log if mode == m), None)
             for m in ('EDGE', 'CORNER')}
    edge_brakes = sum(1 for b in sim.brakes if len(b[1]) == 1)
    corner_brakes = sum(1 for b in sim.brakes if len(b[1]) == 2)
    end = [v for t, v in tilts if t > seconds - 3.0]
    on_corner = ctrl.mode == 'CORNER' and not sim.contacts
    return dict(seed=seed, scale=noise.torque_scale, eff=noise.brake_efficiency,
                learned=ctrl.brake_efficiency, edge=first['EDGE'], corner=first['CORNER'],
                edge_brakes=edge_brakes, corner_brakes=corner_brakes, on_corner=on_corner,
                rms=math.degrees(math.sqrt(np.mean(np.square(end)))) if end else float('nan'),
                wheel_max=wheel_max[0])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('runs', type=int, nargs='?', default=40)
    parser.add_argument('--seconds', type=float, default=30.0)
    args = parser.parse_args()
    with Pool() as pool:
        results = pool.map(run, [(s, args.seconds) for s in range(args.runs)])
    print(f'{args.runs} runs, {args.seconds:.0f} s each, start lying on a face')
    print('seed  torque  brake  learned  on edge  on corner  edge/corner brakes  '
          'corner tilt RMS (last 3 s)  max wheel')
    for r in results:
        fmt = lambda v: f'{v:6.2f} s' if v is not None else '     - '  # noqa: E731
        print(f'{r["seed"]:4d}  {r["scale"]:6.2f}  {r["eff"]:5.2f}  {r["learned"]:7.2f}  '
              f'{fmt(r["edge"])}  {fmt(r["corner"])}   '
              f'{r["edge_brakes"]:2d} / {r["corner_brakes"]:2d}'
              f'               {r["rms"]:7.3f} deg              {r["wheel_max"]:5.0f} rad/s'
              f'{"" if r["on_corner"] else "   NOT on the corner at the end"}')
    done = [r for r in results if r['on_corner']]
    times = [r['corner'] for r in done]
    first_try = sum(1 for r in done if r['edge_brakes'] == 1 and r['corner_brakes'] == 1)
    print(f'\non the corner at the end: {len(done)}/{len(results)}; first try (1 brake each): '
          f'{first_try}/{len(results)}')
    if times:
        print(f'time to the corner: median {np.median(times):.1f} s, max {max(times):.1f} s; '
              f'corner tilt RMS median {np.nanmedian([r["rms"] for r in done]):.3f} deg; '
              f'max wheel speed {max(r["wheel_max"] for r in results):.0f} rad/s')


if __name__ == '__main__':
    main()
