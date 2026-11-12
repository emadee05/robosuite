"""Full-task Stack ladder: home-pose starts plus scripted pick-and-place rungs.

Adapts the Lift ladder (robosuite-scratchpad ladder-sgs-experiment, f628446) to
Stack. A privileged script runs reach -> descend -> close -> lift -> transport
-> lower -> open -> retreat from the home pose. Rung rN restores the simulator
state N control steps before the first frame of the script's final passing
physical-stack streak, so N counts the policy actions still needed.

Success is the conservative physical check in uniform.check (released, centred,
level, slow, supported), held for `hold_steps` consecutive control frames.
Reward is sparse: 1 on the success step, which also terminates the episode.

Build a pool (no training):
    ../robosuite/.venv/bin/python -m sgs.stack_task_ladder --scenes 8

Watch the scripted pick-and-place (macOS needs mjpython):
    ../robosuite/.venv/bin/mjpython -m sgs.stack_task_ladder --render
    ../robosuite/.venv/bin/mjpython -m sgs.stack_task_ladder --render --scene 0
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
import argparse
import hashlib
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces
from robosuite.utils.control_utils import orientation_error

from uniform.check import check_stack, StackStabilityTracker
from uniform.stack_intermediate import StackPreview, make_env, eef_pose, cube_positions, STATE_SPEC

logging.getLogger('robosuite').setLevel(logging.ERROR)
OBS_KEYS = ('robot0_proprio-state', 'object-state')
ROOT = Path(__file__).resolve().parent.parent / 'stack_task_ladders'


# ----------------------------------------------------------------- simulator
def new_env(seed=0):
    env = make_env(positions={'cubeA': [0., -.06], 'cubeB': [0., .06]})
    env.ignore_done = True
    env.horizon = 10 ** 9
    return env


def flatten(obs):
    return np.concatenate([obs[k].ravel() for k in OBS_KEYS]).astype(np.float32)


def set_scene(env, a_xy, b_xy):
    env.placement_initializer.positions = {'cubeA': np.asarray(a_xy, float),
                                           'cubeB': np.asarray(b_xy, float)}
    return env.reset()


def snapshot(env):
    model, data = env.sim.model._model, env.sim.data._data
    state = np.empty(mujoco.mj_stateSize(model, STATE_SPEC))
    mujoco.mj_getState(model, data, state, STATE_SPEC)
    return dict(state=state, grip=np.broadcast_to(env.robots[0].gripper['right'].current_action, (2,)).copy())


def restore(env, snap):
    """Soft reset, then overwrite physics; re-sync controller goals (about 1 ms)."""
    env.reset()
    mujoco.mj_setState(env.sim.model._model, env.sim.data._data, snap['state'], STATE_SPEC)
    env.sim.forward()
    robot = env.robots[0]
    robot.gripper['right'].current_action = snap['grip'].copy()
    robot.composite_controller.update_state()
    for controller in robot.part_controllers.values():
        controller.update(force=True)
    robot.composite_controller.reset()
    env.timestep, env.cur_time, env.done = 0, 0., False
    env._obs_cache = {}
    env._reset_observables()
    env.update_state()
    return env._get_observations(force_update=True)


def stacked_now(env):
    """Cheap geometric prefilter before the full contact-based physical check."""
    a, b = cube_positions(env)
    if np.linalg.norm(a[:2] - b[:2]) > .012 or a[2] < b[2] + .03:
        return {'is_stacked': False}
    return check_stack(env)


class Tracker(StackStabilityTracker):
    def update(self, env):
        return self.observe(stacked_now(env), int(env.timestep))


# -------------------------------------------------------------------- script
class PickPlaceScript:
    """Privileged feedback controller using the same OSC action scaling as PPO."""
    STAGES = ('hover', 'descend', 'close', 'lift', 'transport', 'lower', 'open', 'retreat')

    def __init__(self, env, step_m=.02):
        self.orientation = eef_pose(env)[1]
        self.step_m = step_m
        self.stage, self.n, self.ok = 'hover', 0, 0
        self.target = None
        self.timeouts = []

    def _advance(self, nxt):
        self.stage, self.n, self.ok = nxt, 0, 0

    def action(self, env):
        pos, ori = eef_pose(env)
        a, b = cube_positions(env)
        a_half, b_half = env.cubeA.top_offset[2], env.cubeB.top_offset[2]
        self.n += 1
        grip = -1.
        s = self.stage
        if s in ('hover', 'descend'):
            goal = a + np.array([0, 0, .08 if s == 'hover' else -.005])
            delta = goal - pos
            done = np.linalg.norm(delta) < (.008 if s == 'hover' else .005)
            self.ok = self.ok + 1 if done else 0
            if self.ok >= 2 or self.n >= 60:
                if self.ok < 2: self.timeouts.append(s)
                self._advance('descend' if s == 'hover' else 'close')
        elif s == 'close':
            delta, grip = np.zeros(3), 1.
            if self.n >= 12:
                self._advance('lift')
        elif s in ('lift', 'transport', 'lower'):
            grip = 1.
            carry = b[2] + b_half + a_half
            if s == 'lift':
                goal_a = np.array([a[0], a[1], carry + .05])
            elif s == 'transport':
                goal_a = np.array([b[0], b[1], carry + .05])
            else:
                goal_a = np.array([b[0], b[1], carry + .002])
            delta = goal_a - a
            if s == 'lower':
                delta[2] = np.clip(delta[2], -.01, .01)
            tol = .01 if s == 'lift' else .004
            self.ok = self.ok + 1 if np.linalg.norm(delta) < tol else 0
            if self.ok >= (1 if s == 'lift' else 3) or self.n >= 80:
                if self.ok < 1: self.timeouts.append(s)
                self._advance({'lift': 'transport', 'transport': 'lower', 'lower': 'open'}[s])
                if self.stage == 'open':
                    self.target = pos.copy()
        elif s == 'open':
            delta = self.target - pos
            if self.n >= 10:
                self._advance('retreat')
                self.target = pos + np.array([0, 0, .08])
        else:  # retreat, then hold
            delta = self.target - pos
        translation = np.clip(delta, -self.step_m, self.step_m)
        rotation = np.clip(orientation_error(self.orientation, ori), -.1, .1)
        return np.r_[translation / .05, rotation / .5, grip]


# ---------------------------------------------------------------------- pool
def sample_scenes(n, seed, half_range=.08, min_sep=.10):
    rng = np.random.default_rng(seed)
    scenes = []
    while len(scenes) < n:
        a, b = rng.uniform(-half_range, half_range, 2), rng.uniform(-half_range, half_range, 2)
        if np.linalg.norm(a - b) >= min_sep:
            scenes.append((a, b))
    return scenes


def run_script(env, scene, hold_steps, max_steps=500, preview=None):
    set_scene(env, *scene)
    script, tracker = PickPlaceScript(env), Tracker()
    tracker.required_steps = hold_steps
    traj = [snapshot(env)]
    stages = [script.stage]
    first = None
    if preview is not None:
        preview.update(env)
    for step in range(1, max_steps + 1):
        stages.append(script.stage)
        started = time.perf_counter()
        env.step(script.action(env))
        if preview is not None:
            preview.update(env)
            time.sleep(max(0., env.control_timestep - (time.perf_counter() - started)))
        report = tracker.update(env)
        if report['is_stacked'] and first is None:
            first = step                     # first frame of the current streak
        elif not report['is_stacked']:
            first = None
        if report['stable_now']:
            return traj, stages, first, script
        traj.append(snapshot(env))
    raise RuntimeError(f'script failed; stage={script.stage}, timeouts={script.timeouts}')


def preview_script(scenes, hold_steps, scene_id=None):
    """Replay PickPlaceScript in a MuJoCo window. Does not write a pool or train."""
    ids = list(range(len(scenes))) if scene_id is None else [scene_id]
    if any(i not in range(len(scenes)) for i in ids):
        raise ValueError(f'--scene must be in 0..{len(scenes) - 1}')
    preview = StackPreview()
    env = new_env()
    try:
        for sid in ids:
            print(f'Preview scene {sid}/{len(scenes) - 1}: hover → retreat until 5-frame stack',
                  flush=True)
            traj, stages, first, script = run_script(env, scenes[sid], hold_steps, preview=preview)
            print(f'  stacked at step {first}; final stage={script.stage}; '
                  f'timeouts={script.timeouts}', flush=True)
        print('Preview finished. Close the window or Ctrl-C.', flush=True)
        while preview.handle is not None and preview.handle.is_running():
            preview.update(env)
            time.sleep(0.05)
    finally:
        preview.close()
        env.close()


def build_pool(scenes, rungs, hold_steps, seed, out=None):
    env = new_env()
    configs, snaps, meta = [], [], []
    for sid, scene in enumerate(scenes):
        t0 = time.time()
        traj, stages, first, script = run_script(env, scene, hold_steps)
        # Index k in traj is the state BEFORE action k+1; rN starts N actions before `first`.
        for label in ['far'] + [f'r{n}' for n in rungs]:
            n = first if label == 'far' else int(label[1:])
            if n > first:
                raise RuntimeError(f'scene {sid}: rung {label} exceeds script length {first}')
            idx = first - n
            configs.append([*scene[0], *scene[1], sid, n])
            snaps.append(traj[idx])
            meta.append(dict(id=len(meta), scene=sid, label=label, steps_before_stack=n,
                             script_stage=stages[idx] if idx else 'home'))
        print(f'scene {sid}: stack begins at step {first}; timeouts={script.timeouts}; '
              f'{time.time() - t0:.1f}s', flush=True)
    env.close()
    out = Path(out) if out else ROOT / datetime.now().strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(out / 'pool.npz', configs=np.array(configs),
                        states=np.stack([s['state'] for s in snaps]),
                        grips=np.stack([s['grip'] for s in snaps]))
    manifest = dict(kind='stack-task-ladder', seed=seed, hold_steps=hold_steps,
                    rungs=list(rungs), n_scenes=len(scenes), configurations=meta,
                    labels=['far'] + [f'r{n}' for n in rungs])
    (out / 'pool.json').write_text(json.dumps(manifest, indent=1) + '\n')
    print('pool:', out, f'({len(meta)} configurations)', flush=True)
    return out


# ----------------------------------------------------------------------- env
class StackLadderEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, pool_dir, horizon=200, seed=0, slack=60, pos_only=True):
        super().__init__()
        self.slack, self.pos_only = slack, pos_only
        pool_dir = Path(pool_dir)
        self.manifest = json.loads((pool_dir / 'pool.json').read_text())
        with np.load(pool_dir / 'pool.npz') as z:
            self.snaps = [dict(state=s, grip=g) for s, g in zip(z['states'], z['grips'])]
        self.meta = self.manifest['configurations']
        self.env = new_env(seed)
        self.tracker = Tracker()
        self.tracker.required_steps = self.manifest['hold_steps']
        self.horizon = horizon
        obs = flatten(self.env.reset())
        self.orientation = eef_pose(self.env)[1]   # home wrist orientation (same as script)
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, np.float32)
        self.action_space = spaces.Box(-1, 1, (4 if pos_only else 7,), np.float32)
        self.config_id = None

    def reset(self, *, seed=None, options=None):
        self.config_id = int((options or {})['configuration_id'])
        obs = restore(self.env, self.snaps[self.config_id])
        self.tracker.reset()
        self.steps = 0
        # Per-start time limit: the script's remaining steps plus slack, capped.
        m = self.meta[self.config_id]
        self.limit = min(self.horizon, m['limit'] if 'limit' in m else m['steps_before_stack'] + self.slack)
        return flatten(obs), {}

    def step(self, action):
        action = np.clip(np.asarray(action, np.float64), -1, 1)
        if self.pos_only:   # policy controls xyz + gripper; wrist held at home orientation
            rot = np.clip(orientation_error(self.orientation, eef_pose(self.env)[1]), -.1, .1) / .5
            action = np.r_[action[:3], rot, action[3]]
        obs, _, _, _ = self.env.step(action)
        self.steps += 1
        report = self.tracker.update(self.env)
        terminated = bool(report['stable_now'])
        truncated = not terminated and self.steps >= self.limit
        return flatten(obs), float(terminated), terminated, truncated, {}

    def close(self):
        self.env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--scenes', type=int, default=8)
    p.add_argument('--rungs', default='1,3,6,10,15,20,25,30,40,50,60,80,100')
    p.add_argument('--hold-steps', type=int, default=5)
    p.add_argument('--seed', type=int, default=12345)
    p.add_argument('--out', type=Path)
    p.add_argument('--render', action='store_true',
                   help='Watch the scripted pick-and-place; on macOS launch with mjpython')
    p.add_argument('--scene', type=int, default=None,
                   help='With --render, only this scene index (default: all sampled scenes)')
    a = p.parse_args()
    scenes = sample_scenes(a.scenes, a.seed)
    if a.render:
        preview_script(scenes, a.hold_steps, a.scene)
        return
    rungs = [int(x) for x in a.rungs.split(',')]
    build_pool(scenes, rungs, a.hold_steps, a.seed, a.out)


if __name__ == '__main__':
    main()
