"""Gymnasium Stack adapter for the validated, frozen 32-state pool."""
import hashlib
import json
from pathlib import Path
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from uniform.stack_intermediate import make_env, restore
from uniform.uniform_env import EpisodeOutcome
from uniform.placement_reward import placement_geometry, progress_term
from uniform.check import StackStabilityTracker


class StackPoolEnv(gym.Env):
    metadata = {'render_modes': []}
    observation_keys = ('robot0_proprio-state', 'object-state')

    def __init__(self, pool_path, seed=42, shared=None, *, reward_mode='stock',
                 progress_gamma=.99, progress_weight=1., distance_scale=.1, configuration_ids=None,
                 stability_seconds=1.):
        super().__init__()
        if reward_mode not in ('stock', 'progress', 'sustained'):
            raise ValueError('reward_mode must be stock, progress or sustained')
        progress_term(.5, .5, progress_gamma, progress_weight)
        if not np.isfinite(distance_scale) or distance_scale <= 0:
            raise ValueError('distance_scale must be positive and finite')
        self.reward_mode = reward_mode
        self.progress_gamma = progress_gamma
        self.progress_weight = progress_weight
        self.distance_scale = distance_scale
        self.physical_tracker = StackStabilityTracker(stability_seconds, 20.) if reward_mode=='sustained' else None
        self.configuration_ids = list(range(32)) if configuration_ids is None else list(configuration_ids)
        if not self.configuration_ids or len(set(self.configuration_ids)) != len(self.configuration_ids) or any(
                type(i) is not int or not 0 <= i < 32 for i in self.configuration_ids):
            raise ValueError('Invalid configuration subset')
        if shared is not None and configuration_ids is not None:
            raise ValueError('Subset pilots use local uniform sampling, not shared SGS')
        self.pool_path = Path(pool_path)
        self.pool = json.loads(self.pool_path.read_text())
        self.pool_sha256 = hashlib.sha256(self.pool_path.read_bytes()).hexdigest()
        expected = (self.pool_path.parent/'pool.sha256').read_text().strip()
        if self.pool_sha256 != expected or self.pool.get('status') != 'validated' or self.pool.get('task') != 'Stack':
            raise ValueError('Invalid pool manifest/hash')
        self.entries = self.pool['configurations']
        if len(self.entries) != 32 or [e['id'] for e in self.entries] != list(range(32)):
            raise ValueError('Expected exactly 32 ordered configuration IDs')
        self.states = []
        for entry in self.entries:
            path = self.pool_path.parent/entry['state_file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['state_sha256']:
                raise ValueError('State checksum mismatch')
            with np.load(path, allow_pickle=False) as archive:
                self.states.append({k: archive[k].copy() for k in archive.files})
        self.shared = shared
        self.np_random = np.random.default_rng(seed)
        scene = self.pool['scenes'][0]
        self.sim_env = make_env(positions={'cubeA': scene['a'], 'cubeB': scene['b']})
        self.configuration_id = None
        self.needs_reset = True
        initial = self._flatten(self.sim_env.reset())
        self.observation_space = spaces.Box(-np.inf, np.inf, initial.shape, dtype=np.float32)
        low, high = self.sim_env.action_spec
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

    def _flatten(self, obs):
        return np.concatenate([obs[key].ravel() for key in self.observation_keys]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options)-{'configuration_id'}:
            raise ValueError('Only configuration_id is supported')
        selected = options.get('configuration_id')
        if selected is not None and self.shared is not None:
            raise ValueError('Training selection must use shared sampler')
        if self.shared is None:
            probabilities = [1/len(self.configuration_ids) if i in self.configuration_ids else 0. for i in range(32)]
            version = 0
        else:
            probabilities, version = self.shared.distribution()
        if selected is None:
            selected = int(self.np_random.choice(32, p=probabilities))
        if type(selected) is not int or not 0 <= selected < 32:
            raise ValueError('Configuration ID must be an integer 0–31')
        if selected not in self.configuration_ids:
            raise ValueError('Configuration ID is outside the selected subset')
        self.configuration_id = selected
        self.selection_probability = probabilities[selected]
        self.sampler_version = version
        restore(self.sim_env, self.states[selected], self.pool['model_xml_sha256'])
        if self.sim_env._check_success():
            raise ValueError('Reset state already meets Stack success')
        self.outcome = EpisodeOutcome()
        if self.physical_tracker is not None:
            self.physical_tracker.reset()
        self.initial_distance, self.previous_potential = placement_geometry(self.sim_env, self.distance_scale)
        self.reward_totals = dict(stock_reward_sum=0., task_reward_sum=0., shaping_reward_sum=0.,
                                 stock_task_reward_sum=0.)
        self.needs_reset = False
        obs = self.sim_env._get_observations(force_update=True)
        return self._flatten(obs), dict(configuration_id=selected,
            stage=self.entries[selected]['stage'], pool_sha256=self.pool_sha256)

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError('Reset before stepping a new episode')
        obs, stock_reward, done, info = self.sim_env.step(action)
        success = bool(self.sim_env._check_success())
        distance, potential = placement_geometry(self.sim_env, self.distance_scale)
        shaping = progress_term(self.previous_potential, potential, self.progress_gamma, self.progress_weight)
        physical = self.physical_tracker.update(self.sim_env) if self.physical_tracker is not None else None
        # Use a state-based per-frame task reward. Persistence earns repeated
        # rewards; waiting for a hidden streak before rewarding is unnecessary.
        task_reward = float(physical['is_stacked']) if physical is not None else float(success)
        reward = float(stock_reward) if self.reward_mode == 'stock' else task_reward+shaping
        self.previous_potential = potential
        self.reward_totals['stock_reward_sum'] += float(stock_reward)
        self.reward_totals['task_reward_sum'] += task_reward
        self.reward_totals['shaping_reward_sum'] += shaping
        self.reward_totals['stock_task_reward_sum'] += float(success)
        self.outcome.update(success, reward)
        stage = self.entries[self.configuration_id]['stage']
        info = dict(info, configuration_id=self.configuration_id, stage=stage, step_success=success,
                    reward_mode=self.reward_mode, stock_reward=float(stock_reward), task_reward=task_reward,
                    shaping_reward=shaping, placement_potential=potential, placement_distance_m=distance)
        if physical is not None:
            info.update(physical_step_success=physical['is_stacked'],
                        physical_stable_now=physical['stable_now'],
                        physical_success_ever=physical['stable_ever'])
        if done:
            self.needs_reset = True
            outcome = dict(configuration_id=self.configuration_id, stage=stage,
                success=self.outcome.success, length=self.outcome.steps, reward_sum=self.outcome.reward_sum,
                pool_sha256=self.pool_sha256, selection_probability=self.selection_probability,
                sampler_version_at_reset=self.sampler_version, reward_mode=self.reward_mode,
                initial_placement_distance_m=self.initial_distance,
                final_placement_distance_m=distance, **self.reward_totals)
            if physical is not None:
                outcome.update(stock_success_ever=self.outcome.success,
                    success=physical['stable_now'],
                    physical_success_ever=physical['stable_ever'],
                    physical_success_at_end=physical['stable_now'],
                    longest_physical_stack_seconds=physical['longest_stable_seconds'])
            info.update(is_success=outcome['success'], episode_outcome=outcome)
            if self.shared is not None:
                self.shared.record(self.configuration_id, outcome['success'])
        # These are artificial time-limit truncations, not terminal task states.
        # Keep Phi(next) here: SB3 bootstraps using the final observation/value.
        # A true terminal state would require its own zero-potential handling.
        return self._flatten(obs), float(reward), False, bool(done), info

    def close(self):
        self.sim_env.close()
