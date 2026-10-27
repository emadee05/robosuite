"""Evaluate a saved PPO with sampled actions; never calls learn()."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from uniform_env import UniformLiftEnv
from evaluate_policy import evaluate_policy
from fixed_pool import pool_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--checkpoint', default='ppo_001001472.zip')
    parser.add_argument('--episodes-per-configuration', type=int, default=5)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.episodes_per_configuration < 1 or args.seed < 0:
        parser.error('Episode count must be positive and seed nonnegative')
    root = args.run_dir.resolve()
    checkpoint = root / 'checkpoints' / args.checkpoint
    pool = root / 'train_pool.json'
    settings = json.loads((root / 'settings.json').read_text())
    assert pool_hash(pool) == settings['pool_sha256'], 'Training pool hash mismatch'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    model = PPO.load(checkpoint, device='cpu')
    # Seed after loading: loading can restore the model's original training seed.
    set_random_seed(args.seed)
    model.policy.set_training_mode(False)
    before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    training_steps = model.num_timesteps
    env = UniformLiftEnv(seed=args.seed, horizon=settings['episode_horizon'], pool_path=pool)
    try:
        count = 32 * args.episodes_per_configuration
        print(f'Sampled-action evaluation: {count} episodes, '
              f'{count * settings["episode_horizon"]:,} evaluation transitions. No training.', flush=True)
        report = evaluate_policy(model, env, args.episodes_per_configuration, deterministic=False)
        assert model.num_timesteps == training_steps
        assert all(torch.equal(before[k], v) for k, v in model.policy.state_dict().items())
        assert report['episode_count'] == count
        assert report['evaluation_transitions'] == count * settings['episode_horizon']
        report.update(checkpoint=str(checkpoint),
                      checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                      evaluation_seed=args.seed, training_transitions=training_steps,
                      parameters_unchanged=True,
                      caveat='Repeated fixed starts with stochastic actions; training-pool evaluation, not held-out generalization.')
        output = root / ('sampled_eval_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
        output.write_text(json.dumps(report, indent=2) + '\n')
        print('PASS: policy unchanged; all configurations evaluated equally')
        print(f'Sampled-action success: {report["successes"]}/{count} ({report["success_rate"]:.1%})')
        print('Saved:', output)
    finally:
        env.close()


if __name__ == '__main__':
    main()
