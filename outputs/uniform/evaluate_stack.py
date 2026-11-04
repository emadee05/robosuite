"""Equal exposure evaluation of Stack pool, reporting each starting stage."""
import random
import time
import numpy as np
import torch
from dataclasses import asdict
from stable_baselines3.common.utils import set_random_seed
from uniform.build_stack_pool import STAGES
from uniform.check import StackStabilityTracker, DEFAULT_THRESHOLDS


def stage_summary(episodes):
    result = {}
    for stage in STAGES:
        selected = [e for e in episodes if e['stage'] == stage]
        successes = sum(e['success'] for e in selected)
        result[stage] = dict(episodes=len(selected), successes=successes,
            success_rate=successes/len(selected) if selected else None,
            mean_episode_reward=float(np.mean([e['reward_sum'] for e in selected])) if selected else None)
    return result


def evaluate_stack(model, env, *, repeats=1, deterministic=True, seed=2027, ids=None,
                   physical_checks=False, stability_seconds=1.):
    if repeats < 1:
        raise ValueError('repeats must be positive')
    ids = list(range(32)) if ids is None else list(ids)
    if not ids or len(set(ids)) != len(ids) or any(type(i) is not int or not 0 <= i < 32 for i in ids):
        raise ValueError('Invalid evaluation IDs')
    if env.shared is not None:
        raise ValueError('Evaluation must not use a training sampler')
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    mode = model.policy.training
    steps = model.num_timesteps
    before = {k: v.detach().clone() for k, v in model.policy.state_dict().items()}
    episodes = []
    start = time.perf_counter()
    try:
        set_random_seed(seed)
        for config_id in ids:
            # Per-configuration action seed prevents previous IDs' outcomes from
            # changing the random action stream used on this configuration.
            set_random_seed(seed+config_id)
            for repeat in range(repeats):
                obs, _ = env.reset(options={'configuration_id': config_id})
                tracker = StackStabilityTracker(stability_seconds, env.sim_env.control_freq) if physical_checks else None
                failed_counts = {}
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=deterministic)
                    if not np.isfinite(action).all():
                        raise ValueError('Nonfinite action')
                    obs, reward, terminated, truncated, info = env.step(action)
                    if not np.isfinite(obs).all() or not np.isfinite(reward):
                        raise ValueError('Nonfinite evaluation output')
                    done = terminated or truncated
                    if tracker is not None:
                        physical = tracker.update(env)
                        for condition in physical['failed_conditions']:
                            failed_counts[condition] = failed_counts.get(condition, 0)+1
                outcome = dict(info['episode_outcome'], repeat=repeat+1)
                if tracker is not None:
                    outcome.update(physical_success_ever=physical['stable_ever'],
                        physical_success_at_end=physical['stable_now'],
                        longest_physical_stack_seconds=physical['longest_stable_seconds'],
                        final_physical_conditions=physical['conditions'],
                        physical_failed_frame_counts=failed_counts)
                episodes.append(outcome)
            recent = episodes[-repeats:]
            print(f'  ID {config_id:02d} ({recent[0]["stage"]}): '
                  f'{sum(e["success"] for e in recent)}/{repeats} successful'
                  + (' (stable at end)' if env.reward_mode=='sustained' else ' (stock flag)'), flush=True)
            if physical_checks:
                print(f'    Physical: sustained ever {sum(e["physical_success_ever"] for e in recent)}/{repeats}; '
                      f'stable at end {sum(e["physical_success_at_end"] for e in recent)}/{repeats}', flush=True)
        assert model.num_timesteps == steps
        assert all(torch.equal(before[k], v) for k, v in model.policy.state_dict().items())
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        model.policy.set_training_mode(mode)
    report = dict(deterministic_policy=deterministic, evaluation_seed=seed,
        reward_mode=env.reward_mode,
        pool_sha256=env.pool_sha256, held_out=False, configuration_ids=ids,
        success_definition=('Physical stack sustained for the configured duration at episode end'
                            if env.reward_mode=='sustained' else 'Stock Stack success reached at least once in 1000 steps; no early termination'),
        task_success_metric='stable_at_episode_end' if env.reward_mode=='sustained' else 'stock_success_ever',
        episodes=episodes, episode_count=len(episodes), successes=sum(e['success'] for e in episodes),
        success_rate=sum(e['success'] for e in episodes)/len(episodes),
        evaluation_transitions=sum(e['length'] for e in episodes),
        by_stage=stage_summary(episodes), wall_seconds=time.perf_counter()-start,
        policy_unchanged=True)
    report['stock_successes'] = sum(e.get('stock_success_ever',e['success']) for e in episodes)
    report['stock_success_rate'] = report['stock_successes']/len(episodes)
    if physical_checks:
        def physical_summary(selected):
            n = len(selected)
            ever = sum(e['physical_success_ever'] for e in selected)
            end = sum(e['physical_success_at_end'] for e in selected)
            return dict(episodes=n, sustained_successes=ever, stable_at_end_successes=end,
                        sustained_success_rate=ever/n if n else None,
                        stable_at_end_success_rate=end/n if n else None,
                        mean_longest_stack_seconds=float(np.mean([e['longest_physical_stack_seconds'] for e in selected])) if n else None)
        report['physical'] = dict(physical_summary(episodes),
            required_seconds=stability_seconds, thresholds=asdict(DEFAULT_THRESHOLDS),
            by_stage={stage:physical_summary([e for e in episodes if e['stage']==stage]) for stage in STAGES},
            by_configuration={str(i):physical_summary([e for e in episodes if e['configuration_id']==i]) for i in ids})
    return report
