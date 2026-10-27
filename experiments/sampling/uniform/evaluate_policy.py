"""Balanced evaluation over the training pool; no learning or sampler updates."""
from datetime import datetime, timezone
import time
import numpy as np


def evaluate_policy(model, env, episodes_per_configuration=1, *, deterministic=True):
    if episodes_per_configuration < 1:
        raise ValueError('episodes_per_configuration must be positive')
    episodes = []
    started = time.perf_counter()
    for configuration_id in range(32):
        for repeat in range(episodes_per_configuration):
            obs, _ = env.reset(options={'configuration_id': configuration_id})
            finished = False
            while not finished:
                action, _ = model.predict(obs, deterministic=deterministic)
                if not np.isfinite(action).all():
                    raise ValueError('Policy produced a nonfinite action')
                obs, reward, terminated, truncated, info = env.step(action)
                if not np.isfinite(obs).all() or not np.isfinite(reward):
                    raise ValueError('Nonfinite evaluation observation or reward')
                finished = terminated or truncated
            outcome = dict(info['episode_outcome'], repeat=repeat + 1)
            episodes.append(outcome)
        recent = episodes[-episodes_per_configuration:]
        print(f'Evaluation configuration {configuration_id:02d}: '
              f'{sum(e["success"] for e in recent)}/{episodes_per_configuration} successful', flush=True)
    per_configuration = []
    for configuration_id in range(32):
        matching = [e for e in episodes if e['configuration_id'] == configuration_id]
        per_configuration.append(dict(configuration_id=configuration_id,
            episodes=len(matching), successes=sum(e['success'] for e in matching),
            success_rate=float(np.mean([e['success'] for e in matching]))))
    return dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),
        pool_sha256=env.pool_sha256, distribution='equal visits to all 32 training-pool IDs',
        held_out=False, deterministic_policy=deterministic,
        success_definition='Lift success condition true on at least one step; no early termination',
        episode_count=len(episodes), evaluation_transitions=sum(e['length'] for e in episodes),
        successes=sum(e['success'] for e in episodes),
        success_rate=float(np.mean([e['success'] for e in episodes])),
        mean_episode_reward=float(np.mean([e['reward_sum'] for e in episodes])),
        wall_seconds=time.perf_counter()-started,
        per_configuration=per_configuration, episodes=episodes)
