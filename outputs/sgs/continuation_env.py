"""Report completed episodes before SubprocVecEnv automatically resets a worker."""
from uniform.uniform_env import UniformLiftEnv


class ContinuationEnv(UniformLiftEnv):
    def __init__(self, seed, pool_path, shared):
        super().__init__(seed=seed, horizon=1000, pool_path=pool_path)
        self.shared = shared

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            # Seed selection before sampling; super.reset also receives this seed.
            import numpy as np
            self.np_random = np.random.default_rng(seed)
        if options:
            raise ValueError('Training resets must use the shared sampler')
        probabilities, version = self.shared.distribution()
        selected = int(self.np_random.choice(32, p=probabilities))
        obs, info = super().reset(options={'configuration_id': selected})
        self.selection_probability = probabilities[selected]
        self.sampler_version = version
        return obs, info

    def step(self, action):
        result = super().step(action)
        info = result[-1]
        if 'episode_outcome' in info:
            outcome = info['episode_outcome']
            outcome.update(selection_probability=self.selection_probability,
                           sampler_version_at_reset=self.sampler_version)
            self.shared.record(outcome['configuration_id'], outcome['success'])
        return result
