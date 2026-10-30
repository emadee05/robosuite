"""Check Stack reset/episode/evaluation behavior; no training."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from uniform.stack_pool_env import StackPoolEnv
from uniform.evaluate_stack import evaluate_stack


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    env = StackPoolEnv(args.pool)
    try:
        check_env(env, warn=True)
        print('PASS: SB3 compatibility', flush=True)
        for i in range(32):
            obs, info = env.reset(options={'configuration_id': i})
            np.testing.assert_allclose(env.sim_env.sim.data.qpos, env.states[i]['qpos'], atol=1e-12, rtol=0)
            assert info['stage'] == env.entries[i]['stage'] and np.isfinite(obs).all()
        print('PASS: all 32 IDs restore the expected state and stage', flush=True)
        initial, _ = env.reset(options={'configuration_id': 0})
        for step in range(1000):
            obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
            assert not terminated and truncated == (step==999)
            assert ('episode_outcome' in info) == (step==999)
            assert np.isfinite(obs).all() and np.isfinite(reward)
        assert info['episode_outcome']['length']==1000
        assert info['episode_outcome']['stage']=='tabletop'
        try:
            env.step(np.zeros(7, dtype=np.float32))
        except RuntimeError:
            pass
        else:
            raise AssertionError('Stepping a finished episode was allowed')
        repeated, _ = env.reset(options={'configuration_id': 0})
        np.testing.assert_allclose(repeated, initial, atol=1e-7, rtol=0)
        print('PASS: 1000-step truncation, one outcome, and fresh repeatable reset', flush=True)
        model = PPO('MlpPolicy', env, seed=42, device='cpu', n_steps=1536, batch_size=256,
                    policy_kwargs={'net_arch': {'pi':[64,64], 'vf':[64,64]}}, verbose=0)
        report = evaluate_stack(model, env, repeats=1, deterministic=False, seed=2027, ids=[0,1,2,3])
        assert report['episode_count']==4 and report['evaluation_transitions']==4000
        assert all(s['episodes']==1 for s in report['by_stage'].values())
        assert report['policy_unchanged']
        print('PASS: real four-stage sampled evaluation; policy unchanged', flush=True)
        print('Observation shape:',env.observation_space.shape, 'actions:', env.action_space.shape)
        print('No training performed.', flush=True)
    finally:
        env.close()


if __name__=='__main__':
    main()
