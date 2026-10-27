"""One manager-owned sampler; serialized updates shared by all environment workers."""
from multiprocessing.managers import BaseManager
from threading import RLock
from sgs.sgs_sampler import SGSSampler


class SharedSampler:
    def __init__(self, mode, history_path, pool_hash, checkpoint_steps, kappa=1.0, temperature=2.0):
        if mode not in ('uniform', 'sgs'):
            raise ValueError(mode)
        self.mode = mode
        self.lock = RLock()
        self.sampler = SGSSampler(kappa=kappa, temperature=temperature)
        self.sampler.load_training_history(history_path, pool_sha256=pool_hash,
                                           through_transition=checkpoint_steps)
        self.initial_visits = self.sampler.total_visits.copy()
        self.updates = 0

    def record(self, configuration_id, success):
        with self.lock:
            self.sampler.update(configuration_id, success)
            self.updates += 1

    def distribution(self):
        with self.lock:
            p = self.sampler.probabilities() if self.mode == 'sgs' else [1/32]*32
            return p, self.updates

    def snapshot(self):
        with self.lock:
            state = self.sampler.snapshot()
            state.update(mode=self.mode, continuation_episodes=self.updates,
                         initial_visits=self.initial_visits,
                         sampling_probabilities=self.distribution()[0])
            return state


class SamplerManager(BaseManager):
    pass


SamplerManager.register('SharedSampler', SharedSampler)
