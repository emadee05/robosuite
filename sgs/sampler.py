"""Samplers over the fixed configuration set: uniform baseline and SGS.

SGS = Success Guided Sampling, from "A Balanced Data Diet: Addressing the
Exploration Bottleneck in Mega-Scale RL for Robot Control" (Zhang, Guaman Castro
et al., CoRL 2026, arXiv:2610.12465).

The method is two equations.  For each configuration i the sampler keeps the
last H boolean episode outcomes and sets p_i to their mean.  Then (paper Eq. 1,
a Beta(1 + kappa*t, 1 + kappa*(1-t)) kernel in mode-concentration form):

    w_i = (p_i + eps)**(kappa*t) * (1 - p_i + eps)**(kappa*(1 - t))
    l_i = log(w_i + eps)

and (paper Eq. 2) the next configuration is drawn i.i.d. from a softmax over
those logits with temperature T:

    P(i) = exp(l_i / T) / sum_j exp(l_j / T)

t is the *target success rate*: the kernel peaks at p_i = t, so configurations
the policy solves ~t of the time are sampled most often, while fully-solved and
fully-unsolved ones keep a small but non-zero probability (that is what eps
guarantees).  kappa controls how sharply the kernel peaks; T how sharply the
softmax does.

Paper hyperparameters, for reference:
    manipulation: N = 32768, H = 100, T = 2, t = 0.5, eps = 1e-4, kappa = 1
    locomotion:   N = 104000, H = 100, T = 2, t = 0.66, eps = 1e-8, kappa = 5
Our defaults keep t/eps/kappa/T at the manipulation values but shrink N and H,
because at N = 32768 and H = 100 you need 3.3M episodes just to fill the
histories once.  See README for the arithmetic.
"""

from collections import deque

import numpy as np


class UniformSampler:
    """Baseline: every configuration equally likely, forever."""

    name = "uniform"

    def __init__(self, n_configs: int, seed: int = 0):
        self.n = n_configs
        self.rng = np.random.default_rng(seed)
        # Success bookkeeping is kept for logging even though it is never used
        # to sample, so the two runs produce identical diagnostics.
        self.history = [deque(maxlen=100) for _ in range(n_configs)]
        self.n_episodes = 0

    def probs(self) -> np.ndarray:
        return np.full(self.n, 1.0 / self.n)

    def sample(self, k: int = 1) -> np.ndarray:
        return self.rng.integers(0, self.n, size=k)

    def update(self, idx: int, success: bool) -> None:
        self.history[idx].append(bool(success))
        self.n_episodes += 1

    def success_rates(self) -> np.ndarray:
        return np.array([np.mean(h) if h else 0.0 for h in self.history])


class SGSSampler:
    """Success Guided Sampling (paper Eqs. 1 and 2)."""

    name = "sgs"

    def __init__(
        self,
        n_configs: int,
        target: float = 0.5,       # t
        kappa: float = 1.0,        # kappa
        eps: float = 1e-4,         # eps
        temperature: float = 2.0,  # T
        history: int = 50,         # H
        warmup_episodes: int = 0,
        seed: int = 0,
    ):
        self.n = n_configs
        self.t = float(target)
        self.kappa = float(kappa)
        self.eps = float(eps)
        self.T = float(temperature)
        self.H = int(history)
        # The paper does not say what to do with configurations that have no
        # outcomes yet (at N = 32768 and 1M parallel envs, every config is hit
        # almost immediately).  At N = 64 it matters, so we optionally draw the
        # first `warmup_episodes` uniformly to seed every history.  Set to 0 to
        # follow the paper literally (an empty history reads as p_i = 0).
        self.warmup = int(warmup_episodes)
        self.rng = np.random.default_rng(seed)
        self.history = [deque(maxlen=self.H) for _ in range(n_configs)]
        self.n_episodes = 0

    def success_rates(self) -> np.ndarray:
        """p_i: mean of the last H boolean outcomes (0.0 if never attempted)."""
        return np.array([np.mean(h) if h else 0.0 for h in self.history])

    def probs(self) -> np.ndarray:
        if self.n_episodes < self.warmup:
            return np.full(self.n, 1.0 / self.n)
        p = self.success_rates()
        # Eq. 1 evaluated in log space: log w_i, then l_i = log(w_i + eps).
        log_w = self.kappa * self.t * np.log(p + self.eps) + self.kappa * (
            1.0 - self.t
        ) * np.log(1.0 - p + self.eps)
        logits = np.logaddexp(log_w, np.log(self.eps))
        # Eq. 2: softmax(l / T).
        z = logits / self.T
        z -= z.max()
        e = np.exp(z)
        return e / e.sum()

    def sample(self, k: int = 1) -> np.ndarray:
        if self.n_episodes < self.warmup:
            return self.rng.integers(0, self.n, size=k)
        return self.rng.choice(self.n, size=k, p=self.probs())

    def update(self, idx: int, success: bool) -> None:
        self.history[idx].append(bool(success))
        self.n_episodes += 1


def make_sampler(kind: str, n_configs: int, seed: int, **kw):
    if kind == "uniform":
        return UniformSampler(n_configs, seed=seed)
    if kind == "sgs":
        return SGSSampler(n_configs, seed=seed, **kw)
    raise ValueError(f"unknown sampler {kind!r}")
