"""How many environment steps per second do we get, as a function of worker count?

Answers "how slow is this actually going to be" before you commit to a run.
Usage: python scripts/bench.py [--envs 1 4 8 16 32 64] [--steps 400]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sgs.configs import ConfigSpace, sample_config_set  # noqa: E402
from sgs.vec import ConfigVecEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[1, 4, 8, 16, 32, 64])
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--horizon", type=int, default=200)
    args = ap.parse_args()

    cfgs = sample_config_set(64, ConfigSpace(), seed=0)
    print(f"{'n_envs':>7} {'env-steps/s':>12} {'ms/iter':>9} {'1M steps in':>12}")
    for n in args.envs:
        venv = ConfigVecEnv(n, dict(horizon=args.horizon), seed=0)
        rng = np.random.default_rng(0)
        venv.reset_to(cfgs[rng.integers(0, len(cfgs), n)])
        act = np.zeros((n, venv.act_dim))
        venv.step(act)  # warm up numba/JIT in the workers
        t0 = time.time()
        for _ in range(args.steps):
            _, _, term, trunc, _ = venv.step(act)
            done = np.nonzero(term | trunc)[0]
            if len(done):
                venv.reset_to(cfgs[rng.integers(0, len(cfgs), len(done))], done)
        dt = time.time() - t0
        sps = n * args.steps / dt
        print(
            f"{n:>7} {sps:>12.0f} {dt / args.steps * 1000:>9.2f} "
            f"{1e6 / sps / 60:>9.1f} min"
        )
        venv.close()


if __name__ == "__main__":
    main()
