"""OmniReset-style reset families for Stack (paper Appendix A.2), no task script.

Each configuration samples cube poses, then *programmatically* places the robot:
- home:        robosuite home pose, cubes random.                 (eval only: the real task)
- reaching:    gripper near cube A (random offset above the grasp point, random aperture).
- grasp:       cube A held at a random point between its table spot and above cube B.
- near_goal:   cube A held on the descent path above B, depth fraction in [0, 1.1];
               with probability 0.3 the gripper is partly opened (so A may drop).

The arm is moved with the OSC controller in free space (a positioning move, not
a task demonstration); grasp families then teleport cube A between the fingers
and close the gripper. Configurations that fail validation are resampled.
States are saved in the same format as sgs.stack_task_ladder, so
sgs.train_stack_ladder trains on them unchanged. A separate eval split is drawn
from the same generator with a different seed (fresh, never trained on).

    ../robosuite/.venv/bin/python -m sgs.stack_omnireset --train-per-family 128 --eval-per-family 32
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import json
import time
from datetime import datetime
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import axisangle2quat, convert_quat

from uniform.check import check_stack
from uniform.stack_intermediate import eef_pose, cube_positions, metrics
from sgs.stack_task_ladder import new_env, set_scene, snapshot

TRAIN_FAMILIES = ('reaching', 'grasp', 'near_goal')
LIMITS = dict(home=200, reaching=200, grasp=150, near_goal=100)
ROOT = Path(__file__).resolve().parent.parent / 'stack_omnireset_pools'


class Reject(Exception):
    pass


def drive(env, target, ori, grip=-1., max_steps=150, tol=.005):
    """Move the end effector to `target` with OSC (2 cm/step cap), wrist at `ori`."""
    for _ in range(max_steps):
        pos, cur = eef_pose(env)
        if np.linalg.norm(target - pos) < tol:
            return
        rot = np.clip(orientation_error(ori, cur), -.1, .1) / .5
        env.step(np.r_[np.clip(target - pos, -.02, .02) / .05, rot, grip])
    raise Reject('positioning timeout')


def hold(env, ori, grip, steps):
    for _ in range(steps):
        rot = np.clip(orientation_error(ori, eef_pose(env)[1]), -.1, .1) / .5
        env.step(np.r_[0, 0, 0, rot, grip])


def place_cube(env, obj, pos, yaw):
    quat_wxyz = convert_quat(axisangle2quat(np.array([0, 0, yaw])), to='wxyz')
    env.sim.data.set_joint_qpos(obj.joints[0], np.r_[pos, quat_wxyz])
    env.sim.data.set_joint_qvel(obj.joints[0], np.zeros(6))
    env.sim.forward()


def sample_cubes(rng, half_range=.10, min_sep=.09):
    while True:
        a, b = rng.uniform(-half_range, half_range, 2), rng.uniform(-half_range, half_range, 2)
        if np.linalg.norm(a - b) >= min_sep:
            return a, b


def make_config(env, family, rng):
    a_xy, b_xy = sample_cubes(rng)
    set_scene(env, a_xy, b_xy)
    ori = eef_pose(env)[1]
    a, b = cube_positions(env)
    yaw = rng.uniform(-.2, .2)
    place_cube(env, env.cubeA, a, yaw)
    info = dict(family=family, a_xy=a_xy.tolist(), b_xy=b_xy.tolist(), yaw=float(yaw))
    if family == 'home':
        return info
    a_half, b_half = env.cubeA.top_offset[2], env.cubeB.top_offset[2]
    if family == 'reaching':
        target = a + np.r_[rng.uniform(-.015, .015, 2), -.005 + rng.uniform(0, .08)]
        drive(env, np.r_[target[:2], max(target[2], a[2]) + .1], ori)
        drive(env, target, ori)
        k = int(rng.integers(0, 6))                     # random aperture: 0-5 close steps
        hold(env, ori, 1., k)
        hold(env, ori, 0., 3)
        info.update(close_steps=k, eef_above_grasp=float(target[2] - a[2] + .005))
    else:
        carry = b[2] + b_half + a_half                  # A centre height when resting on B
        if family == 'grasp':
            lam = rng.uniform(0, 1)
            xy = (1 - lam) * a[:2] + lam * b[:2] + rng.uniform(-.01, .01, 2)
            z = carry + rng.uniform(.03, .12)
            info.update(path_fraction=float(lam))
        else:
            frac = rng.uniform(0, 1.1)
            xy = b[:2] + rng.uniform(-.01, .01, 2)
            z = carry + .003 + frac * .05
            info.update(depth_fraction=float(frac))
        grasp_eef = np.r_[xy, z - .005]                 # eef sits 5 mm below A's centre in a grasp
        drive(env, np.r_[xy, max(z, a[2]) + .1], ori)
        drive(env, grasp_eef, ori)
        place_cube(env, env.cubeA, np.r_[xy, z], yaw)
        # Put the fingers on the cube faces (a closed-gripper reset, as with IK resets);
        # waiting for the sign-incremental gripper to close would let the cube fall first.
        gripper = env.robots[0].gripper['right']
        for joint, q in zip(gripper.joints, (.0198, -.0198)):
            env.sim.data.set_joint_qpos(joint, q)
            env.sim.data.set_joint_qvel(joint, 0.)
        gripper.current_action = np.array([-1., 1.])
        env.sim.forward()
        hold(env, ori, 1., 8)
        if not env._check_grasp(env.robots[0].gripper, env.cubeA):
            raise Reject('grasp failed')
        if family == 'near_goal' and rng.uniform() < .3:
            j = int(rng.integers(1, 6))
            hold(env, ori, -1., j)
            info.update(open_steps=j)
    m = metrics(env)
    if not m['finite'] or m['min_contact_distance_m'] < -.005 or m['joint_limit_violations']:
        raise Reject('invalid state')
    if check_stack(env)['is_stacked']:
        raise Reject('already stacked')
    return info


def cell_key(m):
    """Coarse cell for cell-level SGS estimates (8 per family; paper Sec. 5 'cells').

    reaching:  gripper height above the grasp point (4 bins) x aperture (2)
    grasp:     fraction of the way from cube A's table spot to above B (8 bins)
    near_goal: depth fraction (4 bins) x partly released or not (2)
    """
    f = m['label']
    if f == 'reaching':
        return f'reaching_h{min(int(m["eef_above_grasp"] / .02), 3)}_c{int(m["close_steps"] >= 3)}'
    if f == 'grasp':
        return f'grasp_p{min(int(m["path_fraction"] * 8), 7)}'
    if f == 'near_goal':
        return f'near_goal_d{min(int(m["depth_fraction"] / .275), 3)}_o{int("open_steps" in m)}'
    return f


def _build(args):
    family, n, seed = args
    env = new_env()
    rng = np.random.default_rng(seed)
    out, rejects = [], 0
    while len(out) < n:
        try:
            info = make_config(env, family, rng)
        except Reject:
            rejects += 1
            continue
        env.timestep = 0
        out.append((info, snapshot(env)))
    env.close()
    return family, out, rejects


def build(train_per_family, eval_per_family, seed, hold_steps, out=None):
    jobs = [(f, train_per_family, seed * 100 + i) for i, f in enumerate(TRAIN_FAMILIES)]
    jobs += [(f, eval_per_family, seed * 100 + 50 + i)
             for i, f in enumerate(('home',) + TRAIN_FAMILIES)]
    splits = ['train'] * len(TRAIN_FAMILIES) + ['eval'] * (len(TRAIN_FAMILIES) + 1)
    t0 = time.time()
    with Pool(min(len(jobs), 7)) as pool:
        results = pool.map(_build, jobs)
    meta, states, grips = [], [], []
    for split, (family, items, rejects) in zip(splits, results):
        print(f'{split:5} {family:10} {len(items)} configs ({rejects} rejected)', flush=True)
        for info, snap in items:
            meta.append(dict(id=len(meta), split=split, label=family, limit=LIMITS[family],
                             steps_before_stack=LIMITS[family], **info))
            states.append(snap['state']); grips.append(snap['grip'])
    out = Path(out) if out else ROOT / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'pool.npz', states=np.stack(states), grips=np.stack(grips))
    (out / 'pool.json').write_text(json.dumps(dict(
        kind='stack-omnireset', seed=seed, hold_steps=hold_steps,
        labels=['home', *TRAIN_FAMILIES], train_families=list(TRAIN_FAMILIES),
        limits=LIMITS, configurations=meta), indent=1) + '\n')
    print(f'pool: {out} ({len(meta)} configurations, {time.time() - t0:.0f}s)', flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--train-per-family', type=int, default=128)
    p.add_argument('--eval-per-family', type=int, default=32)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--hold-steps', type=int, default=5)
    p.add_argument('--out', type=Path)
    a = p.parse_args()
    build(a.train_per_family, a.eval_per_family, a.seed, a.hold_steps, a.out)


if __name__ == '__main__':
    main()
