"""SGS equations (1–2), with Appendix B.2 floors; no simulator dependencies.

Manipulation defaults: H=100, t=.5, kappa=1, epsilon=1e-4, T=2.
Unobserved configurations use p=.5 (an explicit local initialization choice).
This object belongs in the coordinator, not separately in each worker.
"""
import json
import math
from collections import deque


class SGSSampler:
    def __init__(self, configurations=32, history_size=100, *, kappa=1.0, temperature=2.0):
        if configurations < 1 or history_size < 1:
            raise ValueError('Configuration count and history size must be positive')
        if not math.isfinite(kappa) or kappa < 0:
            raise ValueError('kappa must be finite and nonnegative')
        if not math.isfinite(temperature) or temperature < 1:
            raise ValueError('temperature must be finite and at least one')
        self.kappa = kappa
        self.temperature = temperature
        self.history = [deque(maxlen=history_size) for _ in range(configurations)]
        self.total_visits = [0] * configurations
        self.total_successes = [0] * configurations

    def update(self, configuration_id, success):
        if type(configuration_id) is not int or not 0 <= configuration_id < len(self.history):
            raise ValueError('Invalid configuration ID')
        if type(success) is not bool:
            raise ValueError('Success must be a Boolean episode outcome')
        self.history[configuration_id].append(success)
        self.total_visits[configuration_id] += 1
        self.total_successes[configuration_id] += int(success)

    def probabilities(self):
        rates = [sum(h)/len(h) if h else .5 for h in self.history]
        epsilon = 1e-4
        weights = [(p + epsilon)**(self.kappa*.5) * (1-p + epsilon)**(self.kappa*.5) for p in rates]
        logits = [math.log(max(w, epsilon) + epsilon)/self.temperature for w in weights]
        scaled = [math.exp(x - max(logits)) for x in logits]
        return [x/sum(scaled) for x in scaled]

    def snapshot(self):
        return dict(kappa=self.kappa, temperature=self.temperature, target=.5, epsilon=1e-4, history=[list(h) for h in self.history],
                    history_size=self.history[0].maxlen,
                    total_visits=self.total_visits.copy(),
                    total_successes=self.total_successes.copy(),
                    success_rates=[sum(h)/len(h) if h else None for h in self.history],
                    probabilities=self.probabilities())

    def load_training_history(self, path, *, pool_sha256, through_transition):
        """Seed from completed training episodes only, up to the source checkpoint."""
        with open(path) as stream:
            for line in stream:
                episode = json.loads(line)
                if episode['pool_sha256'] != pool_sha256:
                    raise ValueError('Training history belongs to a different pool')
                if episode['training_transitions'] <= through_transition:
                    self.update(episode['configuration_id'], episode['success'])
