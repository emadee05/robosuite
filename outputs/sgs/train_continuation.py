"""Matched uniform/SGS continuations. Explicit command required; no resume support."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import shutil
import time
from datetime import datetime
from functools import partial
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan
from uniform.train_uniform import Outcomes, write_json
from uniform.uniform_env import UniformLiftEnv
from sgs.continuation_env import ContinuationEnv
from sgs.shared_sampler import SamplerManager
from uniform.fixed_pool import pool_hash
from uniform.evaluate_policy import evaluate_policy


class ProgressPPO(PPO):
    """Print after the optimizer phase, without changing rollout/update scheduling."""

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ['cycle_progress']

    def train(self):
        super().train()
        progress = getattr(self, 'cycle_progress', None)
        if progress is None:
            return
        now = time.perf_counter()
        elapsed = now - progress['last_time']
        progress['training_seconds'] += elapsed
        progress['last_time'] = now
        added = self.num_timesteps - progress['base']
        episodes = int(progress['outcomes'].visits.sum())
        print(f"{progress['mode']}: cycle {added // 6144}/{progress['target'] // 6144}; "
              f"+{added:,}/{progress['target']:,} transitions; "
              f"training {progress['training_seconds'] / 60:.1f} min; "
              f"cycle {elapsed:.1f}s ({6144 / elapsed:.1f} transitions/s); "
              f"{episodes} completed episodes", flush=True)


def evaluate(model, env, seed, repeats, deterministic):
    state = random.getstate(), np.random.get_state(), torch.get_rng_state()
    mode = model.policy.training
    try:
        set_random_seed(seed)
        return evaluate_policy(model, env, repeats, deterministic=deterministic)
    finally:
        random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
        model.policy.set_training_mode(mode)


def run_branch(args, root, mode, source, checkpoint, settings, target):
    folder = root/mode
    folder.mkdir()
    (folder/'checkpoints').mkdir()
    (folder/'evaluations').mkdir()
    pool = root/'train_pool.json'
    manager = SamplerManager(ctx=mp.get_context('spawn'))
    manager.start()
    vec = env = None
    status = {'status': 'running'}
    write_json(folder/'status.json', status)
    try:
        shared = manager.SharedSampler(mode, str(source/'training_episodes.jsonl'),
                                       pool_hash(pool), settings['actual_target_steps'],
                                       args.sgs_kappa, args.sgs_temperature)
        vec = SubprocVecEnv([partial(ContinuationEnv, args.seed+i, pool, shared)
                            for i in range(4)], start_method='spawn')
        model = ProgressPPO.load(checkpoint, env=VecCheckNan(vec, raise_exception=True, check_inf=True),
                         device='cpu', force_reset=True)
        base = model.num_timesteps
        assert base == settings['actual_target_steps']
        assert model.n_steps * model.n_envs == 6144
        assert model.batch_size == 256 and model.n_epochs == 10
        # PPO.load restores policy AND optimizer. New worker episodes begin here.
        model.set_random_seed(args.seed)
        env = UniformLiftEnv(seed=args.eval_seed, pool_path=pool, horizon=1000)
        write_json(folder/'sampler_initial.json', shared.snapshot())
        history = []
        training_seconds = 0.0
        schedule = [0] + list(range(args.eval_every_cycles*6144, target, args.eval_every_cycles*6144)) + [target]
        with (folder/'training_episodes.jsonl').open('w') as stream:
            callback = Outcomes(stream)
            model.cycle_progress = dict(mode=mode, base=base, target=target,
                                        outcomes=callback, training_seconds=0.0)
            for added in schedule:
                if added:
                    started = time.perf_counter()
                    model.cycle_progress['last_time'] = started
                    model.learn(total_timesteps=base+added-model.num_timesteps,
                                reset_num_timesteps=False, callback=callback)
                    training_seconds += time.perf_counter()-started
                assert model.num_timesteps == base+added
                assert all(torch.isfinite(p).all() for p in model.policy.parameters())
                model.save(folder/'checkpoints'/f'ppo_{model.num_timesteps:09d}')
                snapshot = shared.snapshot()
                assert snapshot['continuation_episodes'] == int(callback.visits.sum())
                write_json(folder/f'sampler_{added:09d}.json', snapshot)
                if not args.skip_evaluation:
                    for deterministic, repeats, label in ((True, 1, 'deterministic'),
                                                          (False, args.eval_repeats, 'sampled')):
                        print(f'{mode}: +{added:,} transitions; {label} evaluation', flush=True)
                        report = evaluate(model, env, args.eval_seed, repeats, deterministic)
                        report.update(additional_training_transitions=added,
                                      training_transitions=model.num_timesteps,
                                      evaluation_seed=args.eval_seed)
                        write_json(folder/'evaluations'/f'{label}_{added:09d}.json', report)
                        history.append({k:v for k,v in report.items() if k not in ('episodes','per_configuration')})
                write_json(folder/'results.json', dict(mode=mode, training_seconds=training_seconds,
                    additional_training_transitions=added, training_transitions=model.num_timesteps,
                    completed_training_episodes=int(callback.visits.sum()),
                    visits_by_configuration=callback.visits.tolist(),
                    training_successes_by_configuration=callback.successes.tolist(), evaluations=history))
                print(f'{mode}: +{added:,} transitions; {callback.visits.sum()} completed continuation episodes', flush=True)
        status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed', error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        if vec is not None:
            if status['status'] == 'completed':
                vec.close()
            else:
                for process in vec.processes:
                    if process.is_alive(): process.terminate()
                for process in vec.processes:
                    process.join(timeout=5)
                    if process.is_alive(): process.kill(); process.join()
        manager.shutdown()
        write_json(folder/'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=Path(__file__).parent.parent/'lift_uniform_baseline'/'uniform_run_20261026_205152')
    parser.add_argument('--additional-steps', type=int, default=1000000)
    parser.add_argument('--mode', choices=['both','uniform','sgs'], default='both')
    parser.add_argument('--sgs-kappa', type=float, default=1.0)
    parser.add_argument('--sgs-temperature', type=float, default=2.0)
    parser.add_argument('--seed', type=int, default=43)
    parser.add_argument('--eval-seed', type=int, default=2027)
    parser.add_argument('--eval-every-cycles', type=int, default=30)
    parser.add_argument('--eval-repeats', type=int, default=5)
    parser.add_argument('--skip-evaluation', action='store_true', help='Smoke checks only')
    args = parser.parse_args()
    if min(args.additional_steps, args.eval_every_cycles, args.eval_repeats) < 1 or min(args.seed,args.eval_seed) < 0:
        parser.error('Invalid budget, interval, repetitions, or seed')
    if not math.isfinite(args.sgs_kappa) or args.sgs_kappa < 0 or not math.isfinite(args.sgs_temperature) or args.sgs_temperature < 1:
        parser.error('SGS kappa must be finite and nonnegative; temperature must be finite and >= 1')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    source = args.source_run.resolve()
    settings = json.loads((source/'settings.json').read_text())
    assert settings['workers'] == 4 and settings['episode_horizon'] == 1000
    assert pool_hash(source/'train_pool.json') == settings['pool_sha256']
    checkpoint = source/'checkpoints'/f'ppo_{settings["actual_target_steps"]:09d}.zip'
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    target = math.ceil(args.additional_steps/6144)*6144
    root = (Path(__file__).parent.parent/'lift_uniform_baseline')/('continuation_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    root.mkdir()
    shutil.copyfile(source/'train_pool.json', root/'train_pool.json')
    (root/'sources').mkdir()
    for package in ('uniform', 'sgs'):
        target = root/'sources'/package
        target.mkdir()
        for path in (Path(__file__).parent.parent/package).glob('*.py'):
            shutil.copyfile(path, target/path.name)
    shutil.copyfile(source/'train_pool.json', root/'sources'/'uniform'/'train_pool.json')
    write_json(root/'settings.json', dict(source_run=str(source), checkpoint=str(checkpoint),
        checkpoint_sha256=digest, source_settings=settings, additional_steps=target,
        requested_additional_steps=args.additional_steps, workers=4, seed=args.seed,
        eval_seed=args.eval_seed, eval_repeats=args.eval_repeats, eval_every_cycles=args.eval_every_cycles,
        skip_evaluation=args.skip_evaluation, mode=args.mode,
        sgs=dict(history=100,target=.5,kappa=args.sgs_kappa,epsilon=1e-4,temperature=args.sgs_temperature),
        history_initialization='source completed training episodes; excludes evaluation',
        reset_behavior='Fresh episodes in both branches; optimizer retained',
        concurrency='Updates serialized by manager; simultaneous worker order is scheduler-dependent',
        branch_order=['uniform','sgs'] if args.mode=='both' else [args.mode]))
    print('Results:', root, flush=True)
    for mode in (['uniform','sgs'] if args.mode=='both' else [args.mode]):
        run_branch(args, root, mode, source, checkpoint, settings, target)
    print('Completed comparison:', root, flush=True)


if __name__ == '__main__':
    main()
