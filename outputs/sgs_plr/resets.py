"""Stack reset families (paper Appendix A.2), with random gripper aperture.

Reaching (two halves, chosen with probability 0.5):
  - table:    cube A rests on the table; gripper near it, random height above the grasp
              point and random aperture (0-5 close steps).
  - airborne: gripper 3-15 cm above the table over A's spot; cube A teleported between
              the open fingers and pinned in place while the gripper closes for 0-12 steps,
              then released. Small k: the cube falls and must be fetched. Large k: it is
              held, firmly or loosely. (Like the paper's IK resets: object anywhere in
              the workspace, gripper aligned, random aperture, may fall.)
Grasp / Near-Goal: cube A picked from the table with real OSC closes and carried toward /
  onto the descent path above cube B, then the gripper opens for 0-9 steps (random aperture:
  0-4 firm to light squeeze, 5-6 loose, 7-9 open). A cube that slips or drops is kept, as in the paper.

Every start records `finger_gap` (m) and `cube_held`; `cell_key` groups starts for
cell-level SGS (24 cells) and is stored in the pool metadata as `cell`.
"""
import numpy as np
from robosuite.utils.control_utils import orientation_error
from robosuite.utils.transform_utils import axisangle2quat, convert_quat

from uniform.check import check_stack
from uniform.stack_intermediate import eef_pose, cube_positions, metrics
from sgs.stack_omnireset import Reject, sample_cubes
from sgs.stack_task_ladder import set_scene


def drive(env, target, ori, grip=-1., max_steps=150, tol=.005):
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


def place_yaw(env, yaw):
    a, _ = cube_positions(env)
    quat = convert_quat(axisangle2quat(np.array([0, 0, yaw])), to='wxyz')
    env.sim.data.set_joint_qpos(env.cubeA.joints[0], np.r_[a, quat])
    env.sim.data.set_joint_qvel(env.cubeA.joints[0], np.zeros(6))
    env.sim.forward()


def close_on_table(env, ori):
    """Descend onto cube A and close with OSC. No joint/cube teleports."""
    a, _ = cube_positions(env)
    drive(env, a + np.array([0, 0, .08]), ori, grip=-1.)
    drive(env, a + np.array([0, 0, -.005]), ori, grip=-1.)
    for n in range(1, 16):
        hold(env, ori, 1., 1)
        if env._check_grasp(env.robots[0].gripper, env.cubeA):
            return n
    raise Reject('osc close failed')


def finger_gap(env):
    q = [env.sim.data.get_joint_qpos(j) for j in env.robots[0].gripper['right'].joints]
    return float(q[0] - q[1])


def held(env):
    return bool(env._check_grasp(env.robots[0].gripper, env.cubeA))


def pin_cube(env, pos, yaw):
    quat = convert_quat(axisangle2quat(np.array([0, 0, yaw])), to='wxyz')
    env.sim.data.set_joint_qpos(env.cubeA.joints[0], np.r_[pos, quat])
    env.sim.data.set_joint_qvel(env.cubeA.joints[0], np.zeros(6))
    env.sim.forward()


def airborne_reaching(env, ori, a, yaw, rng):
    """Cube A between the fingers 3-15 cm above the table; random close (pinned), then release."""
    height = rng.uniform(.03, .15)                      # cube bottom above the table
    centre = np.r_[a[:2] + rng.uniform(-.01, .01, 2), a[2] + height]
    eef = centre - np.r_[0, 0, .005]                    # eef sits 5 mm below A's centre in a grasp
    drive(env, np.r_[eef[:2], max(eef[2], a[2]) + .1], ori, grip=-1.)
    drive(env, eef, ori, grip=-1.)
    hold(env, ori, -1., 6)                              # fully open before the cube appears
    centre = eef_pose(env)[0] + np.r_[0, 0, .005]
    k = int(rng.integers(0, 13))
    pin_cube(env, centre, yaw)
    for _ in range(k):                                  # close while the cube is held in place
        hold(env, ori, 1., 1)
        pin_cube(env, centre, yaw)
    hold(env, ori, 0., 3)                               # release the pin; cube may fall
    return dict(airborne=True, close_steps=k, height=float(height))


def make_config(env, family, rng):
    a_xy, b_xy = sample_cubes(rng)
    set_scene(env, a_xy, b_xy)
    ori = eef_pose(env)[1]
    yaw = rng.uniform(-.2, .2)
    place_yaw(env, yaw)
    info = dict(family=family, a_xy=a_xy.tolist(), b_xy=b_xy.tolist(), yaw=float(yaw),
                grasp='osc-close')
    if family == 'home':
        return info
    a, b = cube_positions(env)
    if family == 'reaching' and rng.uniform() < .5:
        info.update(airborne_reaching(env, ori, a, yaw, rng))
    elif family == 'reaching':
        target = a + np.r_[rng.uniform(-.015, .015, 2), -.005 + rng.uniform(0, .08)]
        drive(env, np.r_[target[:2], max(target[2], a[2]) + .1], ori, grip=-1.)
        drive(env, target, ori, grip=-1.)
        k = int(rng.integers(0, 6))
        hold(env, ori, 1., k)
        hold(env, ori, 0., 3)
        info.update(airborne=False, close_steps=k, eef_above_grasp=float(target[2] - a[2] + .005))
    else:
        close_steps = close_on_table(env, ori)
        info['close_steps'] = close_steps
        a, b = cube_positions(env)
        a_half, b_half = env.cubeA.top_offset[2], env.cubeB.top_offset[2]
        carry = b[2] + b_half + a_half
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
        drive(env, np.r_[xy, max(z, a[2]) + .1], ori, grip=1.)
        drive(env, np.r_[xy, z - .005], ori, grip=1.)
        if not env._check_grasp(env.robots[0].gripper, env.cubeA):
            raise Reject('grasp lost during carry')
        # Random aperture. The carry leaves the command at fully closed; about 5 open steps
        # bring the fingers to just touching the cube, so 0-4 vary the squeeze, 5-6 are a
        # loose grip and 7-9 open the fingers (the cube usually drops).
        j = int(rng.integers(0, 10))
        hold(env, ori, -1., j)
        hold(env, ori, 0., 3)
        info.update(open_steps=j)
    info.update(finger_gap=finger_gap(env), cube_held=held(env))
    m = metrics(env)
    if not m['finite'] or m['min_contact_distance_m'] < -.005 or m['joint_limit_violations']:
        raise Reject('invalid state')
    if check_stack(env)['is_stacked']:
        raise Reject('already stacked')
    info['cell'] = cell_key(dict(info, label=family))
    info['cell_fine'] = cell_key(dict(info, label=family), fine=True)
    return info


def cell_key(m, fine=False):
    """Cells for cell-level SGS.

    coarse (24 cells, 21 occur):
      reaching:  airborne or on table (2) x low/high (2) x cube held or not (2)
      grasp:     path fraction (4 bins) x aperture firm (<5 open steps) or loose/open (2)
      near_goal: depth fraction (4 bins) x aperture (2)
    fine (~64 cells, for pools large enough to visit each often):
      reaching:  airborne (2) x height (4 bins) x held (2)
      grasp:     path fraction (8 bins) x aperture firm 0-2 / light 3-5 / open 6-9 (3)
      near_goal: depth fraction (8 bins) x aperture (3)
    """
    f = m['label']
    if f == 'reaching':
        h = m['height'] / .15 if m['airborne'] else (m['eef_above_grasp'] + .005) / .085
        hb = min(int(h * (4 if fine else 2)), 3 if fine else 1)
        return f'reaching_a{int(m["airborne"])}_h{hb}_g{int(m["cube_held"])}'
    if f in ('grasp', 'near_goal'):
        x = m['path_fraction'] if f == 'grasp' else m['depth_fraction'] / 1.1
        nb = 8 if fine else 4
        ap = min(m['open_steps'] // 3, 2) if fine else int(m['open_steps'] >= 5)
        return f'{f}_x{min(int(x * nb), nb - 1)}_o{ap}'
    return f
