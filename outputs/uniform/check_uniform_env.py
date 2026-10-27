"""Check Gymnasium compatibility, reproducible sampling and episode accounting."""
import json
from pathlib import Path
import numpy as np
from stable_baselines3.common.env_checker import check_env
from uniform.uniform_env import UniformLiftEnv, EpisodeOutcome


def main():
    # Synthetic outcomes test bookkeeping, not robot ability.
    tracker = EpisodeOutcome()
    for success in [False, True, False]:
        tracker.update(success, 0.25)
    assert tracker.success and tracker.steps == 3 and tracker.reward_sum == 0.75
    assert not EpisodeOutcome().success
    print('PASS: success is retained through the episode and cleared for a new episode')
    env = UniformLiftEnv(horizon=20)
    try:
        check_env(env, warn=True)
        print('PASS: Stable-Baselines3 environment compatibility')
        sequences = []
        for _ in range(2):
            ids = []
            for index in range(12):
                _, info = env.reset(seed=123 if index == 0 else None)
                ids.append(info['configuration_id'])
            sequences.append(ids)
        assert sequences[0] == sequences[1] and len(set(sequences[0])) > 1
        print('PASS: seeded uniform ID selection is reproducible:', sequences[0])
        outcomes = []
        for configuration_id in range(32):
            obs, info = env.reset(options={'configuration_id': configuration_id})
            assert info['configuration_id'] == configuration_id
            seen_success = False
            reward_sum = 0.0
            for step in range(1, 21):
                obs, reward, terminated, truncated, info = env.step(np.zeros(7, dtype=np.float32))
                assert env.observation_space.contains(obs) and np.isfinite(obs).all()
                assert np.isfinite(reward) and not terminated and truncated == (step == 20)
                assert info['configuration_id'] == configuration_id
                seen_success |= info['step_success']
                reward_sum += reward
                assert ('episode_outcome' in info) == truncated
            outcome = info['episode_outcome']
            assert outcome['success'] == seen_success == info['is_success']
            assert outcome['length'] == 20 and np.isclose(outcome['reward_sum'], reward_sum)
            outcomes.append(outcome)
            try:
                env.step(np.zeros(7))
            except RuntimeError:
                pass
            else:
                raise AssertionError('Stepping past the horizon must require a reset')
        print('PASS: all 32 explicit evaluation IDs and one terminal outcome per episode')
        report = dict(status='passed', pool_sha256=env.pool_sha256,
                      check_horizon=20, training_performed=False, zero_action_outcomes=outcomes)
        Path(__file__).with_name('uniform_validation.json').write_text(json.dumps(report, indent=2)+'\n')
        print('No PPO training performed. The training episode horizon will remain 1,000 steps.')
    finally:
        env.close()


if __name__ == '__main__':
    main()
