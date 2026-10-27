"""Uniform fixed-pool Lift, with episode success independent of dense reward."""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from uniform.fixed_pool import POOL_PATH, make_pool_environment, pool_hash


class EpisodeOutcome:
    def __init__(self):
        self.success = False
        self.steps = 0
        self.reward_sum = 0.0

    def update(self, success, reward):
        self.success = self.success or bool(success)
        self.steps += 1
        self.reward_sum += float(reward)


class UniformLiftEnv(gym.Env):
    metadata = {'render_modes': []}
    observation_keys = ('robot0_proprio-state', 'object-state')

    def __init__(self, seed=42, horizon=1000, pool_path=POOL_PATH):
        super().__init__()
        if horizon < 1:
            raise ValueError('horizon must be positive')
        self.sim_env, self.placement = make_pool_environment(pool_path)
        self.sim_env.horizon = horizon
        self.pool_sha256 = pool_hash(pool_path)
        # This RNG selects configuration IDs; the physical model seed stays fixed.
        self.np_random = np.random.default_rng(seed)
        self.configuration_id = None
        self.outcome = EpisodeOutcome()
        self.needs_reset = True
        initial = self._flatten(self.sim_env.reset())
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=initial.shape, dtype=np.float32)
        low, high = self.sim_env.action_spec
        self.action_space = spaces.Box(low.astype(np.float32), high.astype(np.float32), dtype=np.float32)

    def _flatten(self, obs):
        return np.concatenate([obs[k].ravel() for k in self.observation_keys]).astype(np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        if set(options) - {'configuration_id'}:
            raise ValueError('Only configuration_id is supported in reset options')
        # Explicit IDs allow evaluation to visit every pool item equally.
        selected = options.get('configuration_id')
        if selected is None:
            selected = int(self.np_random.integers(32))
        self.placement.select(selected)
        self.configuration_id = int(selected)
        obs = self.sim_env.reset()
        self.outcome = EpisodeOutcome()
        self.needs_reset = False
        return self._flatten(obs), {'configuration_id': self.configuration_id,
                                   'pool_sha256': self.pool_sha256}

    def step(self, action):
        if self.needs_reset:
            raise RuntimeError('Call reset before stepping a new episode')
        obs, reward, done, info = self.sim_env.step(action)
        step_success = bool(self.sim_env._check_success())
        self.outcome.update(step_success, reward)
        info = dict(info)
        info.update(configuration_id=self.configuration_id, step_success=step_success)
        if done:
            self.needs_reset = True
            # Exactly one terminal outcome; do not treat each successful step as a new trial.
            info['is_success'] = self.outcome.success
            info['episode_outcome'] = dict(configuration_id=self.configuration_id,
                success=self.outcome.success, length=self.outcome.steps,
                reward_sum=self.outcome.reward_sum, pool_sha256=self.pool_sha256)
        # Lift's done is its time limit. Success does not reset or terminate early.
        return self._flatten(obs), float(reward), False, bool(done), info

    def close(self):
        self.sim_env.close()
