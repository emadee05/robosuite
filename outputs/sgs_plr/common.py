"""Shared evaluation helpers: fixed reference policies and per-family success."""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_key, '1')
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sgs.stack_task_ladder import StackLadderEnv
from uniform.ladder_ppo import Agent, RunningMeanStd

OUTPUTS = Path(__file__).resolve().parent.parent
RESULTS = Path(__file__).resolve().parent / 'results'
RUNS = OUTPUTS / 'stack_task_ladder_runs'


class Still:
    """Hold the arm still; gripper command -1 (open) or +1 (stay closed). 4-d pos-only actions."""

    def __init__(self, grip):
        self.grip = grip

    def __call__(self, obs):
        return np.array([0., 0., 0., self.grip])


class Net:
    """A PPO agent (trained checkpoint or fresh init), deterministic mean action."""

    def __init__(self, agent, obs_rms):
        self.agent, self.obs_rms = agent.eval(), obs_rms

    def __call__(self, obs):
        with torch.no_grad():
            x = torch.as_tensor(self.obs_rms.normalize(obs[None]), dtype=torch.float32)
            return self.agent.act(x, deterministic=True)[0][0].numpy()

    @classmethod
    def untrained(cls, obs_dim, act_dim, seed, hidden=256, log_std_init=-1.5):
        torch.manual_seed(seed)
        return cls(Agent(obs_dim, act_dim, hidden=hidden, log_std_init=log_std_init),
                   RunningMeanStd((obs_dim,)))

    @classmethod
    def from_run(cls, run, obs_dim, act_dim):
        import json
        args = json.loads((RUNS / run / 'args.json').read_text())
        ckpt = torch.load(RUNS / run / 'ckpt.pt', map_location='cpu', weights_only=False)
        agent = Agent(obs_dim, act_dim, hidden=args['hidden'], log_std_init=args['log_std_init'])
        agent.load_state_dict(ckpt['agent'])
        rms = RunningMeanStd((obs_dim,))
        rms.load_state_dict(ckpt['obs_rms'])
        return cls(agent, rms)


def episode(env, policy, config_id):
    obs, _ = env.reset(options={'configuration_id': int(config_id)})
    while True:
        obs, _, terminated, truncated, _ = env.step(policy(obs))
        if terminated or truncated:
            return bool(terminated)


def evaluate(env, policy, split='eval'):
    """Success per family over every start in `split` (one deterministic episode each)."""
    counts = defaultdict(lambda: [0, 0])
    for m in env.meta:
        if m.get('split', 'train') != split:
            continue
        counts[m['label']][0] += episode(env, policy, m['id'])
        counts[m['label']][1] += 1
    by = {lab: s / n for lab, (s, n) in counts.items()}
    total = sum(s for s, _ in counts.values()) / max(1, sum(n for _, n in counts.values()))
    return dict(all=total, by_label=by, n={lab: n for lab, (_, n) in counts.items()})


def make_env(pool):
    return StackLadderEnv(Path(pool), pos_only=True)
