"""Step 1: build a frozen Stack placement ladder. No PPO training.

Rungs rN start N control steps before the first physically valid stacking frame
of the scripted placement. Already-valid stacks are excluded. The next step
checks restoration, sparse terminal reward and untrained-policy difficulty.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import copy
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from uniform.check import check_stack, StackStabilityTracker, DEFAULT_THRESHOLDS
from uniform.check_scripted_placement import PlacementScript
from uniform.stack_pool_env import StackPoolEnv
from uniform.stack_intermediate import make_env, integration_state, restore, metrics


RUNGS = (120, 90, 60, 40, 25, 10, 5, 1)
ANCHORS = ('tabletop', 'lifted', 'transport', 'placement-approach')
PPO_SETTINGS = dict(
    implementation='uniform.ladder_ppo: Damiano original PPO utilities',
    mentor_commit='f628446', actor=[256, 256], critic=[256, 256], activation='tanh',
    orthogonal_initialization=True, actor_output_gain=.01, log_std_init=-.5,
    num_envs=8, num_steps=64, rollout=512, num_minibatches=8, minibatch=64,
    update_epochs=10, lr=3e-4, adam_eps=1e-5, anneal_lr=True,
    gamma=.99, gae_lambda=.95, clip_coef=.2, ent_coef=0., vf_coef=.5,
    max_grad_norm=.5, target_kl=.05, observation_normalization=True,
    observation_clip=10., reward_scaling=True, device='cpu', torch_threads=4,
    horizon=200, control_freq=20,
    success='20 consecutive physically valid control frames within the episode',
    reward='0 until success, then 1 and terminated=True; no distance shaping',
    action_limits='Stock OSC_POSE: 5 cm and .5 rad per axis; full gripper',
    planned_first_training_sampler='uniform; no training in build/check commands',
    caveats=['Stack adaptation, not reproduction of the Lift result',
             'Assisted starts plus harder anchors; no full-task performance claim',
             'Success ends after 1 second; unlike earlier 1000-step stable-at-end metric'])


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(raw):
    return dict(integration_state=integration_state(raw), qpos=raw.sim.data.qpos.copy(),
                qvel=raw.sim.data.qvel.copy(),
                gripper_current_action=raw.robots[0].gripper['right'].current_action.copy())


def script_state(script):
    return dict(orientation=script.orientation.tolist(), stage=script.stage,
                stage_steps=script.stage_steps, stable=script.stable,
                timeouts=list(script.timeouts),
                target=None if script.target is None else script.target.tolist())


def resume_script(state):
    script = PlacementScript.__new__(PlacementScript)
    script.orientation = np.array(state['orientation'])
    script.stage = state['stage']
    script.stage_steps = state['stage_steps']
    script.stable = state['stable']
    script.timeouts = list(state['timeouts'])
    script.target = None if state['target'] is None else np.array(state['target'])
    return script


def physical_start(raw):
    physical = check_stack(raw)
    m = metrics(raw)
    if not m['finite'] or m['joint_limit_violations'] or m['min_contact_distance_m'] < -.005:
        raise ValueError('Nonfinite, joint-limit or penetration problem in ladder start')
    if physical['is_stacked']:
        raise ValueError('Ladder start already passes physical stacking checks')
    return physical


def build_pool(source, output=None):
    source = Path(source).resolve()
    root = Path(output).resolve() if output else Path(__file__).parent.parent/'stack_ladders'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    (root/'states').mkdir(); (root/'sources').mkdir()
    for name in ('stack_ladder.py', 'check_scripted_placement.py', 'stack_intermediate.py',
                 'stack_pool_env.py', 'check.py', 'ladder_ppo.py'):
        shutil.copyfile(Path(__file__).parent/name, root/'sources'/name)
    env = StackPoolEnv(source, reward_mode='sustained')
    entries = []
    manifest = dict(kind='stack-placement-ladder', version=1, status='building', task='Stack',
        source_pool=str(source), source_pool_sha256=digest(source),
        model_xml_sha256=env.pool['model_xml_sha256'], scenes=env.pool['scenes'],
        labels=list(ANCHORS)+[f'r{n}' for n in RUNGS], rungs=list(RUNGS),
        rung_definition='N steps before first valid physical frame; no valid stack at reset',
        thresholds=asdict(DEFAULT_THRESHOLDS), ppo_settings=PPO_SETTINGS, configurations=entries,
        note='Numerical-restoration and physical-continuation validation is in check_stack_ladder')
    (root/'pool.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('Building:', root, flush=True)
    try:
        shutil.copyfile(source.parent/'model.xml', root/'model.xml')
        for scene_id in range(8):
            scene_entries = [e for e in env.entries if e['scene_id'] == scene_id]
            for label in ANCHORS:
                base = next(e for e in scene_entries if e['stage'] == label)
                env.reset(options={'configuration_id': base['id']})
                physical_start(env.sim_env)
                append_state(root, entries, scene_id, label, snapshot(env.sim_env),
                             source_configuration_id=base['id'], script_continuation=None)
            script = PlacementScript(env.sim_env)  # last anchor is placement-approach
            trajectory = [(snapshot(env.sim_env), script_state(script))]
            first_valid = None
            for step in range(1, 401):
                _, reward, terminated, truncated, _ = env.step(script.action(env.sim_env))
                if terminated or truncated or not np.isfinite(reward):
                    raise RuntimeError('Script ended unexpectedly')
                if check_stack(env)['is_stacked']:
                    first_valid = step
                    break
                trajectory.append((snapshot(env.sim_env), script_state(script)))
            if first_valid is None or script.timeouts or first_valid < max(RUNGS):
                raise RuntimeError(f'Scene {scene_id}: failed script or insufficient trajectory for fixed rungs')
            for rung in RUNGS:
                index = first_valid-rung
                state, continuation = trajectory[index]
                restore(env.sim_env, state, manifest['model_xml_sha256'])
                physical_start(env.sim_env)
                append_state(root, entries, scene_id, f'r{rung}', state,
                             source_configuration_id=scene_id*4+3,
                             trajectory_step=index, first_valid_script_step=first_valid,
                             script_continuation=continuation)
            print(f'Scene {scene_id}: four anchors + eight rungs; first physical stack at step {first_valid}', flush=True)
        assert len(entries) == 96
        manifest['status'] = 'built_pending_validation'
        (root/'pool.json').write_text(json.dumps(manifest, indent=2)+'\n')
        (root/'pool.sha256').write_text(digest(root/'pool.json')+'\n')
        print('Built 96 starts. Next: python -m uniform.check_stack_ladder --pool', root/'pool.json', flush=True)
        return root
    finally:
        env.close()


def append_state(root, entries, scene_id, label, state, **extra):
    config_id = len(entries)
    path = root/'states'/f'{config_id:03d}_{label}.npz'
    np.savez_compressed(path, **state)
    entries.append(dict(id=config_id, scene_id=scene_id, label=label,
        state_file=str(path.relative_to(root)), state_sha256=digest(path), **extra))


class StackLadderEnv(gym.Env):
    """Uniform ladder starts; one sparse success reward and explicit timeouts.

    Physical streak always starts at zero; scripted prefix earns no reward.
    Does not feed stock Stack success or previous progress shaping to PPO.
    """
    metadata = {'render_modes': []}

    def __init__(self, pool_path, seed=0, horizon=200, stability_seconds=1.):
        super().__init__()
        if type(horizon) is not int or horizon < 1:
            raise ValueError('Positive integer horizon required')
        self.pool_path = Path(pool_path).resolve()
        self.pool_sha256 = digest(self.pool_path)
        if self.pool_sha256 != (self.pool_path.parent/'pool.sha256').read_text().strip():
            raise ValueError('Pool manifest hash mismatch')
        self.pool = json.loads(self.pool_path.read_text())
        if self.pool.get('kind') != 'stack-placement-ladder' or self.pool.get('status') != 'built_pending_validation':
            raise ValueError('Not a built Stack ladder')
        self.entries = self.pool['configurations']
        if len(self.entries) != 96 or [e['id'] for e in self.entries] != list(range(96)):
            raise ValueError('Expected 96 ordered configurations')
        self.states = []
        for entry in self.entries:
            path = self.pool_path.parent/entry['state_file']
            if digest(path) != entry['state_sha256']:
                raise ValueError('State hash mismatch')
            with np.load(path, allow_pickle=False) as archive:
                self.states.append({k: archive[k].copy() for k in archive.files})
        scene = self.pool['scenes'][0]
        self.sim_env = make_env(positions={'cubeA': scene['a'], 'cubeB': scene['b']})
        self.sim_env.ignore_done = True
        self.horizon = horizon
        self.tracker = StackStabilityTracker(stability_seconds, self.sim_env.control_freq)
        self.np_random = np.random.default_rng(seed)
        obs = self._flatten(self.sim_env.reset())
        self.observation_space = spaces.Box(-np.inf, np.inf, obs.shape, dtype=np.float32)
        low, high = self.sim_env.action_spec
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)
        self.needs_reset = True

    @staticmethod
    def _flatten(obs):
        return np.concatenate([obs[k].ravel() for k in StackPoolEnv.observation_keys]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options)-{'configuration_id'}:
            raise ValueError('Only configuration_id is supported')
        config_id = options.get('configuration_id')
        if config_id is None:
            config_id = int(self.np_random.integers(len(self.entries)))
        if type(config_id) is not int or not 0 <= config_id < len(self.entries):
            raise ValueError('Invalid configuration ID')
        self.configuration_id = config_id
        restore(self.sim_env, self.states[config_id], self.pool['model_xml_sha256'])
        physical_start(self.sim_env)
        self.tracker.reset()
        self.steps = 0; self.reward_sum = 0.; self.needs_reset = False
        return self._flatten(self.sim_env._get_observations(force_update=True)), dict(
            configuration_id=config_id, label=self.entries[config_id]['label'],
            pool_sha256=self.pool_sha256)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError('Reset before stepping')
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (7,) or not np.isfinite(action).all():
            raise ValueError('Expected seven finite actions')
        obs, _, _, _ = self.sim_env.step(np.clip(action, -1., 1.))
        self.steps += 1
        physical = self.tracker.update(self.sim_env)
        terminated = bool(physical['stable_now'])
        truncated = not terminated and self.steps >= self.horizon
        reward = float(terminated)
        self.reward_sum += reward
        entry = self.entries[self.configuration_id]
        info = dict(configuration_id=self.configuration_id, label=entry['label'],
            physical=physical, stock_success=bool(self.sim_env._check_success()))
        if terminated or truncated:
            self.needs_reset = True
            info.update(is_success=terminated, episode_outcome=dict(configuration_id=self.configuration_id,
                label=entry['label'], scene_id=entry['scene_id'], success=terminated,
                length=self.steps, reward_sum=self.reward_sum,
                longest_physical_stack_seconds=physical['longest_stable_seconds']))
        return self._flatten(obs), reward, terminated, bool(truncated), info

    def close(self):
        self.sim_env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-pool', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    build_pool(args.source_pool, args.output_dir)


if __name__ == '__main__':
    main()
