"""Uniform Stack warmup -> matched uniform and SGS continuations, CPU/headless.

Run as a module from hep/outputs. Does not use any Lift checkpoint. Snapshots the
validated pool and source before training. No hardware sudo or separate monitor.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
import hashlib
import json
import math
import multiprocessing as mp
import shutil
import time
import traceback
from datetime import datetime, timezone
from functools import partial
from pathlib import Path
import numpy as np
import torch
import stable_baselines3 as sb3
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan
from uniform.stack_pool_env import StackPoolEnv
from uniform.evaluate_stack import evaluate_stack, stage_summary
from uniform.build_stack_pool import STAGES
from uniform.train_uniform import write_json
from sgs.shared_sampler import SamplerManager

ROLLOUT = 6144


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    """Stable fingerprint of policy/optimizer tensors, without pickle identities."""
    digest = hashlib.sha256()
    def visit(x):
        if isinstance(x, torch.Tensor):
            a = x.detach().cpu().numpy()
            digest.update(str(a.dtype).encode()); digest.update(str(a.shape).encode()); digest.update(a.tobytes())
        elif isinstance(x, dict):
            for key in sorted(x, key=repr):
                digest.update(repr(key).encode()); visit(x[key])
        elif isinstance(x, (list, tuple)):
            for item in x: visit(item)
        else:
            digest.update(repr(x).encode())
    visit(value)
    return digest.hexdigest()


class EpisodeLog(BaseCallback):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.records = []
        self.visits = [0]*32
        self.successes = [0]*32

    def _on_rollout_start(self):
        self.model.rollout_clock = time.perf_counter()

    def _on_step(self):
        for worker, info in enumerate(self.locals['infos']):
            if 'episode_outcome' in info:
                record = dict(info['episode_outcome'], worker=worker, training_transitions=self.num_timesteps)
                self.records.append(record)
                index = record['configuration_id']
                self.visits[index] += 1
                self.successes[index] += int(record['success'])
                self.stream.write(json.dumps(record)+'\n')
                self.stream.flush()
        return True


class StackPPO(sb3.PPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['stack_tracking', 'rollout_clock']

    def train(self):
        super().train()
        if not all(torch.isfinite(p).all() for p in self.policy.parameters()):
            raise ValueError('Nonfinite PPO weights')
        tracker = getattr(self, 'stack_tracking', None)
        if tracker is None:
            return
        elapsed = time.perf_counter()-self.rollout_clock
        tracker['training_seconds'] += elapsed
        added = self.num_timesteps-tracker['base']
        snapshot = tracker['shared'].snapshot()
        callback = tracker['callback']
        assert snapshot['continuation_episodes'] == len(callback.records)
        record = dict(cycle=added//ROLLOUT, additional_transitions=added,
            cumulative_transitions=self.num_timesteps, cycle_seconds=elapsed,
            cycle_transitions_per_second=ROLLOUT/elapsed,
            training_seconds=tracker['training_seconds'], completed_episodes=len(callback.records),
            by_stage=stage_summary(callback.records),
            sampling_probabilities=snapshot['sampling_probabilities'],
            estimated_success_rates=snapshot['success_rates'])
        tracker['stream'].write(json.dumps(record)+'\n'); tracker['stream'].flush()
        print(f'{tracker["name"]}: cycle {added//ROLLOUT}/{tracker["target"]//ROLLOUT}; '
              f'+{added:,}/{tracker["target"]:,} transitions; '
              f'{ROLLOUT/elapsed:.0f}/s; {len(callback.records)} episodes', flush=True)


def evaluate_checkpoint(model, env, folder, added, args, final):
    reports = []
    for deterministic, repeats, label in ((True,1,'deterministic'),
            (False,args.final_eval_repeats if final else args.eval_repeats,'sampled')):
        print(f'Evaluating {folder.name} +{added:,}: {label}, {repeats} episode(s) per ID', flush=True)
        report = evaluate_stack(model, env, repeats=repeats, deterministic=deterministic,
                                seed=args.eval_seed)
        report.update(training_transitions=model.num_timesteps, additional_training_transitions=added)
        write_json(folder/'evaluations'/f'{label}_{added:09d}.json', report)
        print(f'  overall: {report["successes"]}/{report["episode_count"]} ({report["success_rate"]:.1%})', flush=True)
        for stage, values in report['by_stage'].items():
            print(f'  {stage}: {values["successes"]}/{values["episodes"]} ({values["success_rate"]:.1%})', flush=True)
        reports.append({k:v for k,v in report.items() if k!='episodes'})
    return reports


def run_phase(args, root, name, budget, source=None):
    folder = root/name
    folder.mkdir()
    for part in ('checkpoints','evaluations'):
        (folder/part).mkdir()
    pool = root/'pool'/'pool.json'
    manager = SamplerManager(ctx=mp.get_context('spawn'))
    manager.start()
    model = vec = evaluation_env = None
    status = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat())
    write_json(folder/'status.json', status)
    history = []
    phase_start = time.perf_counter()
    seed = args.seed if source is None else args.continuation_seed
    source_steps = 0 if source is None else json.loads((source/'results.json').read_text())['training_transitions']
    try:
        shared = manager.SharedSampler('sgs' if name=='sgs' else 'uniform',
            None if source is None else str(source/'training_episodes.jsonl'),
            sha256(pool), source_steps, args.sgs_kappa, args.sgs_temperature)
        vec = SubprocVecEnv([partial(StackPoolEnv, pool, seed+i, shared) for i in range(4)], start_method='spawn')
        checked = VecCheckNan(vec, raise_exception=True, check_inf=True)
        if source is None:
            model = StackPPO('MlpPolicy', checked, device='cpu', seed=seed,
                policy_kwargs={'net_arch': {'pi':[64,64], 'vf':[64,64]}},
                n_steps=1536, batch_size=256, n_epochs=10, learning_rate=3e-4,
                gamma=.99, gae_lambda=.95, clip_range=.2, ent_coef=0., vf_coef=.5,
                max_grad_norm=.5, target_kl=None, verbose=0)
        else:
            checkpoint = source/'final_model.zip'
            model = StackPPO.load(checkpoint, env=checked, device='cpu', force_reset=True)
            expected = json.loads((source/'final_checkpoint.json').read_text())
            assert sha256(checkpoint) == expected['file_sha256']
            assert fingerprint(model.policy.state_dict()) == expected['policy_sha256']
            assert fingerprint(model.policy.optimizer.state_dict()) == expected['optimizer_sha256']
            assert model.num_timesteps == source_steps
            model.set_random_seed(seed)
        base = model.num_timesteps
        assert model.n_envs == 4 and model.n_steps*4 == ROLLOUT
        write_json(folder/'initial_checkpoint.json', dict(training_transitions=base,
            policy_sha256=fingerprint(model.policy.state_dict()),
            optimizer_sha256=fingerprint(model.policy.optimizer.state_dict()),
            source=None if source is None else str(source/'final_model.zip'), seed=seed))
        write_json(folder/'sampler_initial.json', shared.snapshot())
        evaluation_env = StackPoolEnv(pool, seed=args.eval_seed)
        schedule = [0]+list(range(args.eval_every_cycles*ROLLOUT, budget, args.eval_every_cycles*ROLLOUT))+[budget]
        with (folder/'training_episodes.jsonl').open('w') as episodes, (folder/'progress.jsonl').open('w') as progress:
            callback = EpisodeLog(episodes)
            model.stack_tracking = dict(base=base, target=budget, name=name, shared=shared,
                                        callback=callback, stream=progress, training_seconds=0.)
            for added in schedule:
                if added:
                    model.learn(total_timesteps=base+added-model.num_timesteps,
                                reset_num_timesteps=False, callback=callback)
                assert model.num_timesteps == base+added
                path = folder/'checkpoints'/f'ppo_{model.num_timesteps:09d}.zip'
                model.save(path)
                if added == budget:
                    shutil.copyfile(path, folder/'final_model.zip')
                    write_json(folder/'final_checkpoint.json', dict(training_transitions=model.num_timesteps,
                        file_sha256=sha256(folder/'final_model.zip'),
                        policy_sha256=fingerprint(model.policy.state_dict()),
                        optimizer_sha256=fingerprint(model.policy.optimizer.state_dict())))
                snapshot = shared.snapshot()
                assert snapshot['continuation_episodes'] == len(callback.records)
                write_json(folder/f'sampler_{added:09d}.json', snapshot)
                # Persist results before potentially lengthy evaluation.
                result = dict(phase=name, source_training_transitions=base,
                    additional_training_transitions=added, training_transitions=model.num_timesteps,
                    training_seconds=model.stack_tracking['training_seconds'],
                    completed_training_episodes=len(callback.records),
                    visits_by_configuration=callback.visits,
                    training_successes_by_configuration=callback.successes,
                    training_by_stage=stage_summary(callback.records), evaluations=history)
                write_json(folder/'results.json', result)
                if not args.skip_evaluation:
                    history.extend(evaluate_checkpoint(model, evaluation_env, folder, added, args, added==budget))
                    write_json(folder/'results.json', result)
            status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed_or_interrupted', error=repr(error))
        (folder/'error.txt').write_text(traceback.format_exc())
        raise
    finally:
        if evaluation_env is not None: evaluation_env.close()
        if vec is not None:
            if status['status']=='completed':
                vec.close()
            else:
                for process in vec.processes:
                    if process.is_alive(): process.terminate()
                for process in vec.processes:
                    process.join(timeout=5)
                    if process.is_alive(): process.kill(); process.join(timeout=5)
        manager.shutdown()
        status.update(wall_seconds=time.perf_counter()-phase_start,
                      finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(folder/'status.json', status)
    return folder


def freeze_pool(source, destination):
    pool = json.loads(source.read_text())
    if pool.get('status')!='validated' or pool.get('configuration_count')!=32:
        raise ValueError('Expected a validated 32-state Stack pool')
    if sha256(source)!=(source.parent/'pool.sha256').read_text().strip():
        raise ValueError('Pool hash mismatch')
    destination.mkdir()
    (destination/'states').mkdir()
    for name in ('pool.json','pool.sha256','model.xml'):
        shutil.copyfile(source.parent/name, destination/name)
    for entry in pool['configurations']:
        path = source.parent/entry['state_file']
        if sha256(path)!=entry['state_sha256']: raise ValueError('State hash mismatch')
        target = destination/entry['state_file']
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--warmup-steps', type=int, default=3000000)
    parser.add_argument('--additional-steps', type=int, default=3000000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--continuation-seed', type=int, default=43)
    parser.add_argument('--eval-seed', type=int, default=2027)
    parser.add_argument('--eval-every-cycles', type=int, default=100)
    parser.add_argument('--eval-repeats', type=int, default=2)
    parser.add_argument('--final-eval-repeats', type=int, default=5)
    parser.add_argument('--sgs-kappa', type=float, default=5.)
    parser.add_argument('--sgs-temperature', type=float, default=2.)
    parser.add_argument('--skip-evaluation', action='store_true', help='Integration checks only')
    args = parser.parse_args()
    if min(args.warmup_steps,args.additional_steps,args.eval_every_cycles,args.eval_repeats,args.final_eval_repeats)<1:
        parser.error('Budgets and evaluation settings must be positive')
    if min(args.seed,args.continuation_seed,args.eval_seed)<0 or args.eval_seed+31 >= 2**32:
        parser.error('Invalid seed')
    if not math.isfinite(args.sgs_kappa) or args.sgs_kappa<0 or not math.isfinite(args.sgs_temperature) or args.sgs_temperature<1:
        parser.error('Invalid SGS weighting')
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    warmup_budget = math.ceil(args.warmup_steps/ROLLOUT)*ROLLOUT
    continuation_budget = math.ceil(args.additional_steps/ROLLOUT)*ROLLOUT
    root = args.output_dir or Path(__file__).parent.parent/'stack_comparisons'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    freeze_pool(args.pool.resolve(), root/'pool')
    (root/'sources').mkdir()
    for package in ('uniform','sgs'):
        destination = root/'sources'/package
        destination.mkdir()
        for path in (Path(__file__).parent.parent/package).glob('*.py'):
            shutil.copyfile(path, destination/path.name)
    settings = dict(task='Stack', pool_sha256=sha256(root/'pool/pool.json'),
        warmup_transitions=warmup_budget, additional_transitions_per_branch=continuation_budget,
        seed=args.seed, continuation_seed=args.continuation_seed, evaluation_seed=args.eval_seed,
        workers=4, device='cpu', actor=[64,64], critic=[64,64], rollout=ROLLOUT,
        minibatch=256, epochs=10, horizon=1000, control_freq=20, dense_rewards=True,
        eval_every_cycles=args.eval_every_cycles, eval_repeats=args.eval_repeats,
        final_eval_repeats=args.final_eval_repeats, skip_evaluation=args.skip_evaluation,
        sgs=dict(kappa=args.sgs_kappa, temperature=args.sgs_temperature, history=100, target=.5, epsilon=1e-4),
        sampler_history='Warmup completed training outcomes only; never evaluation outcomes',
        evaluation='Same training IDs, equal exposure, by starting stage; tabletop reported separately',
        caveats=['One warmup and continuation seed pair; no held-out generalization test',
                 'New episodes at each continuation; policy and optimizer retained',
                 'Manager update order can vary with worker scheduling',
                 'Checkpoints persist; automatic mid-phase crash resume is not implemented'],
        versions=dict(sb3=sb3.__version__, torch=torch.__version__))
    write_json(root/'settings.json', settings)
    status = dict(status='running', active_phase='warmup')
    write_json(root/'status.json', status)
    print('Results:', root, flush=True)
    print(f'Uniform warmup: {warmup_budget:,}; EACH continuation: {continuation_budget:,}', flush=True)
    try:
        source = run_phase(args, root, 'warmup', warmup_budget)
        for name in ('uniform','sgs'):
            status['active_phase'] = name
            write_json(root/'status.json', status)
            run_phase(args, root, name, continuation_budget, source=source)
        summary = {}
        for name in ('warmup','uniform','sgs'):
            result = json.loads((root/name/'results.json').read_text())
            final = [e for e in result['evaluations'] if e['additional_training_transitions']==result['additional_training_transitions']]
            summary[name] = dict(training_transitions=result['training_transitions'],
                training_seconds=result['training_seconds'], training_by_stage=result['training_by_stage'],
                final_evaluations=final)
        write_json(root/'comparison.json', summary)
        status.update(status='completed', active_phase=None)
    except BaseException as error:
        status.update(status='failed_or_interrupted', error=repr(error))
        raise
    finally:
        write_json(root/'status.json', status)
    print('Completed Stack comparison:', root, flush=True)


if __name__=='__main__':
    main()
