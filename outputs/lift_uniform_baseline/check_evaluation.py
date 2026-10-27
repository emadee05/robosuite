"""Evaluate a fresh small PPO policy without learning; save the initial baseline."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import json
from datetime import datetime
from pathlib import Path
import torch
from stable_baselines3 import PPO
from uniform_env import UniformLiftEnv
from evaluate_policy import evaluate_policy


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    env = UniformLiftEnv(seed=42, horizon=1000)
    try:
        model = PPO('MlpPolicy', env, seed=42, device='cpu',
                    policy_kwargs={'net_arch': {'pi': [64, 64], 'vf': [64, 64]}},
                    n_steps=6144, batch_size=256, n_epochs=10, verbose=0)
        before = {name: value.detach().clone() for name, value in model.policy.named_parameters()}
        print('Evaluating an UNTRAINED policy: 32 configurations × 1,000 steps. No learn() call.', flush=True)
        report = evaluate_policy(model, env)
        assert report['episode_count'] == 32 and report['evaluation_transitions'] == 32000
        assert all(torch.equal(before[name], value.detach()) for name, value in model.policy.named_parameters())
        assert model.num_timesteps == 0
        report.update(policy='untrained PPO small 64-64', policy_seed=42, training_transitions=0,
                      parameters_unchanged=True,
                      caveat='One deterministic episode per fixed configuration; this is a pipeline baseline, not a statistical learning comparison.')
        output = Path(__file__).with_name('untrained_eval_' + datetime.now().strftime('%Y%m%d_%H%M%S') + '.json')
        output.write_text(json.dumps(report, indent=2) + '\n')
        print('PASS: all 32 configurations evaluated equally; policy weights unchanged')
        print(f'Starting success rate: {report["success_rate"]:.1%} ({report["successes"]}/32)')
        print('Saved:', output.resolve())
    finally:
        env.close()


if __name__ == '__main__':
    main()
