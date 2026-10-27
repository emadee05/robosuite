"""Validate all 32 starts and repeated resets. No PPO training or benchmark."""
import json
from pathlib import Path

import numpy as np
from fixed_pool import make_pool_environment, load_pool, pool_hash


def main():
    pool = load_pool()
    baselines = {}
    # Two independent instances verify repeatability across workers as well.
    for instance in range(2):
        env, sampler = make_pool_environment()
        try:
            for config in pool['configurations']:
                config_id = config['id']
                sampler.select(config_id)
                obs = env.reset()
                pos = env.sim.data.body_xpos[env.cube_body_id].copy()
                np.testing.assert_allclose(pos[:2], [config['cube_x_m'], config['cube_y_m']], atol=1e-8)
                assert not env._check_success(), f'ID {config_id} starts successful'
                assert all(np.isfinite(value).all() for value in obs.values())
                state = np.concatenate([env.sim.data.qpos.copy(), env.sim.data.qvel.copy()])
                if instance == 0:
                    baselines[config_id] = state
                else:
                    np.testing.assert_allclose(state, baselines[config_id], rtol=0, atol=1e-10)
                initial_obs = {k: v.copy() for k,v in obs.items()}
                # Allow the cube to settle after its stock 1 cm starting clearance.
                for _ in range(10):
                    stepped, reward, _, _ = env.step(np.zeros(env.action_dim))
                    assert np.isfinite(reward)
                    assert all(np.isfinite(value).all() for value in stepped.values())
                settled = env.sim.data.body_xpos[env.cube_body_id].copy()
                assert settled[2] > 0.8, f'ID {config_id}: cube fell below table'
                assert not env._check_success(), f'ID {config_id}: zero-action start succeeds'
                obs = env.reset()
                restored = np.concatenate([env.sim.data.qpos.copy(), env.sim.data.qvel.copy()])
                np.testing.assert_allclose(restored, baselines[config_id], rtol=0, atol=1e-10)
                for key in initial_obs:
                    np.testing.assert_allclose(obs[key], initial_obs[key], rtol=0, atol=1e-10)
            print(f'PASS: instance {instance + 1}: all 32 placements, finite steps, repeatable resets', flush=True)
        finally:
            env.close()
    report = {'status': 'passed', 'pool_sha256': pool_hash(), 'configurations': 32,
              'instances_checked': 2, 'training_performed': False}
    Path(__file__).with_name('pool_validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print('PASS: identical starts across two independent environments')
    print('Pool SHA256:', pool_hash())
    print('No policy was trained. Next: uniform episode sampling and success evaluation.')


if __name__ == '__main__':
    main()
