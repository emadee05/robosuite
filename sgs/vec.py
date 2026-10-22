"""Subprocess vector env where the *caller* chooses which configuration each
worker resets to.

Standard vector envs auto-reset with the env's own initial-state distribution,
which is precisely the thing we want to control.  So this one does not
auto-reset: `step` returns the true final observation of a finished episode
(which PPO needs to bootstrap a truncated return), and the trainer then calls
`reset_to` with the configurations its sampler picked.
"""

import multiprocessing as mp

import numpy as np

from .env import LiftConfigEnv


def _worker(remote, parent_remote, env_kwargs, seed):
    parent_remote.close()
    env = LiftConfigEnv(seed=seed, **env_kwargs)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "step":
                remote.send(env.step(data))
            elif cmd == "reset_to":
                remote.send(env.reset_to(data))
            elif cmd == "spec":
                remote.send((env.obs_dim, env.act_dim, env.horizon))
            elif cmd == "close":
                env.close()
                remote.close()
                break
            else:
                raise RuntimeError(f"unknown command {cmd!r}")
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()


class ConfigVecEnv:
    def __init__(self, num_envs: int, env_kwargs: dict, seed: int = 0):
        ctx = mp.get_context("spawn")
        self.num_envs = num_envs
        self.remotes, self.work_remotes = zip(
            *[ctx.Pipe() for _ in range(num_envs)]
        )
        self.procs = []
        for i, (wr, r) in enumerate(zip(self.work_remotes, self.remotes)):
            p = ctx.Process(
                target=_worker, args=(wr, r, env_kwargs, seed + i), daemon=True
            )
            p.start()
            self.procs.append(p)
            wr.close()
        self.remotes[0].send(("spec", None))
        self.obs_dim, self.act_dim, self.horizon = self.remotes[0].recv()

    def reset_to(self, configs: np.ndarray, env_ids=None) -> np.ndarray:
        """Reset the given envs (default: all) to the given configurations."""
        if env_ids is None:
            env_ids = range(self.num_envs)
        env_ids = list(env_ids)
        for cfg, i in zip(configs, env_ids):
            self.remotes[i].send(("reset_to", cfg))
        return np.stack([self.remotes[i].recv() for i in env_ids])

    def step(self, actions: np.ndarray):
        for remote, a in zip(self.remotes, actions):
            remote.send(("step", a))
        results = [remote.recv() for remote in self.remotes]
        obs, rew, term, trunc, infos = zip(*results)
        return (
            np.stack(obs),
            np.array(rew, dtype=np.float32),
            np.array(term, dtype=bool),
            np.array(trunc, dtype=bool),
            list(infos),
        )

    def close(self):
        for remote in self.remotes:
            try:
                remote.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()
