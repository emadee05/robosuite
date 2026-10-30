"""Bounded saved-progress-policy continuation with strict placement evaluation.

Uniform over eight placement starts. Reward, observations and PPO objective
unchanged. Physical stacking criteria are evaluation-only. No SGS.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import math
import shutil
import time
import traceback
from dataclasses import asdict
from datetime import datetime,timezone
from functools import partial
from pathlib import Path
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv,VecCheckNan
from uniform.stack_pool_env import StackPoolEnv
from uniform.evaluate_stack import evaluate_stack
from uniform.check import DEFAULT_THRESHOLDS
from uniform.train_uniform import write_json
from sgs.train_stack_comparison import StackPPO,EpisodeLog,freeze_pool,fingerprint,sha256,ROLLOUT


class LongerPlacementPPO(StackPPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['placement_progress']

    def train(self):
        super().train()
        track=getattr(self,'placement_progress',None)
        if track is None:return
        elapsed=time.perf_counter()-self.rollout_clock
        track['training_seconds']+=elapsed
        added=self.num_timesteps-track['base']
        record=dict(additional_training_transitions=added,training_transitions=self.num_timesteps,
            cycle=added//ROLLOUT,cycle_seconds=elapsed,transitions_per_second=ROLLOUT/elapsed,
            completed_training_episodes=len(track['callback'].records),
            raw_successful_training_episodes=sum(track['callback'].successes))
        track['stream'].write(json.dumps(record)+'\n');track['stream'].flush()
        print(f'Placement: cycle {added//ROLLOUT}/{track["budget"]//ROLLOUT}; '
              f'+{added:,}/{track["budget"]:,} transitions; {ROLLOUT/elapsed:.0f}/s; '
              f'{record["completed_training_episodes"]} episodes',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run',type=Path,required=True)
    parser.add_argument('--checkpoint-steps',type=int,default=202752)
    parser.add_argument('--additional-steps',type=int,default=1000000)
    parser.add_argument('--workers',type=int,choices=[1,2,4],default=4)
    parser.add_argument('--seed',type=int,default=43)
    parser.add_argument('--eval-seed',type=int,default=2027)
    parser.add_argument('--eval-every-cycles',type=int,default=20)
    parser.add_argument('--eval-repeats',type=int,default=2)
    parser.add_argument('--final-eval-repeats',type=int,default=5)
    parser.add_argument('--stability-seconds',type=float,default=1.)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--skip-evaluation',action='store_true',help='Short integration checks only')
    args=parser.parse_args()
    if min(args.additional_steps,args.eval_every_cycles,args.eval_repeats,args.final_eval_repeats)<1:
        parser.error('Budgets and evaluation settings must be positive')
    if args.checkpoint_steps<0 or min(args.seed,args.eval_seed)<0 or max(args.seed,args.eval_seed)+32>=2**32:
        parser.error('Invalid checkpoint or seed')
    if not math.isfinite(args.stability_seconds) or args.stability_seconds<=0:
        parser.error('Stability duration must be positive and finite')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    source=args.source_run.resolve()
    settings=json.loads((source/'settings.json').read_text())
    ids=settings['configuration_ids']
    source_pool=json.loads((source/'pool/pool.json').read_text())
    if len(ids)!=8 or any(source_pool['configurations'][i]['stage']!='placement-approach' for i in ids):
        raise ValueError('Source must be the eight-placement-state reward pilot')
    checkpoint=source/'progress/checkpoints'/f'ppo_{args.checkpoint_steps:09d}.zip'
    reference=StackPPO.load(checkpoint,device='cpu')
    if reference.num_timesteps!=args.checkpoint_steps or reference.gamma!=settings['gamma']:
        raise ValueError('Source checkpoint does not match its settings')
    initial=dict(training_transitions=reference.num_timesteps,
        policy_sha256=fingerprint(reference.policy.state_dict()),
        optimizer_sha256=fingerprint(reference.policy.optimizer.state_dict()))
    del reference
    budget=math.ceil(args.additional_steps/ROLLOUT)*ROLLOUT
    root=args.output_dir or Path(__file__).parent.parent/'longer_placement_runs'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'checkpoints').mkdir();(root/'evaluations').mkdir();(root/'sources').mkdir()
    freeze_pool(source/'pool/pool.json',root/'pool')
    shutil.copyfile(checkpoint,root/'source_checkpoint.zip')
    for package in ('uniform','sgs'):
        destination=root/'sources'/package;destination.mkdir()
        for path in (Path(__file__).parent.parent/package).glob('*.py'):
            shutil.copyfile(path,destination/path.name)
    write_json(root/'settings.json',dict(source_run=str(source),source_checkpoint=str(checkpoint),
        source_checkpoint_sha256=sha256(root/'source_checkpoint.zip'),
        source_training_transitions=args.checkpoint_steps,requested_additional_steps=args.additional_steps,
        actual_additional_steps=budget,final_training_transitions=args.checkpoint_steps+budget,
        configuration_ids=ids,workers=args.workers,rollout=ROLLOUT,seed=args.seed,eval_seed=args.eval_seed,
        reward_mode='progress',progress_weight=settings['progress_weight'],
        distance_scale_m=settings['distance_scale_m'],gamma=settings['gamma'],
        eval_every_cycles=args.eval_every_cycles,eval_repeats=args.eval_repeats,
        final_eval_repeats=args.final_eval_repeats,skip_evaluation=args.skip_evaluation,
        stability_seconds=args.stability_seconds,physical_thresholds=asdict(DEFAULT_THRESHOLDS),
        experiment='Additional PPO training only; reward unchanged; uniform sampling, no SGS',
        caveats=['Physical checker is evaluation-only; task reward still uses stock success',
                 'Known assisted starts; no whole-task/generalization conclusion',
                 'Best-checkpoint selection is not an independent final test',
                 'Fresh episodes on load; no automatic crash resume']))
    vec=env=model=None
    status=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'status.json',status)
    history=[];best={}
    print('Output:',root,flush=True)
    print(f'Source: {args.checkpoint_steps:,}; additional: {budget:,}; final: {args.checkpoint_steps+budget:,}',flush=True)
    print('Physical checks are evaluation-only; rewards are unchanged.',flush=True)
    started=time.perf_counter()
    try:
        kwargs=dict(reward_mode='progress',configuration_ids=ids,progress_gamma=settings['gamma'],
            progress_weight=settings['progress_weight'],distance_scale=settings['distance_scale_m'])
        vec=SubprocVecEnv([partial(StackPoolEnv,root/'pool/pool.json',args.seed+i,**kwargs)
                          for i in range(args.workers)],start_method='spawn')
        model=LongerPlacementPPO.load(root/'source_checkpoint.zip',device='cpu',
            env=VecCheckNan(vec,raise_exception=True,check_inf=True),force_reset=True,
            n_steps=ROLLOUT//args.workers)
        assert fingerprint(model.policy.state_dict())==initial['policy_sha256']
        assert fingerprint(model.policy.optimizer.state_dict())==initial['optimizer_sha256']
        assert model.n_steps*model.n_envs==ROLLOUT
        model.set_random_seed(args.seed)
        write_json(root/'initial_checkpoint.json',initial)
        env=StackPoolEnv(root/'pool/pool.json',seed=args.eval_seed,**kwargs)
        schedule=[0]+list(range(args.eval_every_cycles*ROLLOUT,budget,args.eval_every_cycles*ROLLOUT))+[budget]
        with (root/'training_episodes.jsonl').open('w') as episodes,(root/'progress.jsonl').open('w') as progress:
            callback=EpisodeLog(episodes)
            model.placement_progress=dict(base=args.checkpoint_steps,budget=budget,
                                          callback=callback,stream=progress,training_seconds=0.)
            for added in schedule:
                if added:
                    model.learn(total_timesteps=args.checkpoint_steps+added-model.num_timesteps,
                                reset_num_timesteps=False,callback=callback)
                assert model.num_timesteps==args.checkpoint_steps+added
                path=root/'checkpoints'/f'ppo_{model.num_timesteps:09d}.zip'
                model.save(path)
                if added==budget:shutil.copyfile(path,root/'final_model.zip')
                result=dict(additional_training_transitions=added,training_transitions=model.num_timesteps,
                    training_seconds=model.placement_progress['training_seconds'],
                    completed_training_episodes=len(callback.records),raw_training_successes=sum(callback.successes),
                    visits_by_configuration=callback.visits,evaluations=history,best_checkpoints=best)
                write_json(root/'results.json',result)
                if args.skip_evaluation:continue
                for deterministic,label,repeats in ((True,'deterministic',1),
                        (False,'sampled',args.final_eval_repeats if added==budget else args.eval_repeats)):
                    print(f'Evaluating +{added:,} ({model.num_timesteps:,} total): {label}',flush=True)
                    report=evaluate_stack(model,env,ids=ids,repeats=repeats,deterministic=deterministic,
                        seed=args.eval_seed,physical_checks=True,stability_seconds=args.stability_seconds)
                    report.update(training_transitions=model.num_timesteps,additional_training_transitions=added)
                    write_json(root/'evaluations'/f'{label}_{model.num_timesteps:09d}.json',report)
                    history.append({k:v for k,v in report.items() if k!='episodes'})
                    physical=report['physical']
                    print(f'  Raw success {report["successes"]}/{report["episode_count"]}; '
                          f'physical sustained {physical["sustained_successes"]}/{report["episode_count"]}; '
                          f'stable at end {physical["stable_at_end_successes"]}/{report["episode_count"]}',flush=True)
                    rank=(physical['stable_at_end_success_rate'],physical['sustained_success_rate'])
                    old=best.get(label)
                    if old is None or rank>tuple(old['rank']):
                        shutil.copyfile(path,root/f'best_{label}.zip')
                        best[label]=dict(rank=list(rank),training_transitions=model.num_timesteps,
                            additional_training_transitions=added,episode_count=report['episode_count'],
                            stable_at_end_successes=physical['stable_at_end_successes'],
                            sustained_successes=physical['sustained_successes'])
                    write_json(root/'results.json',result)
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed_or_interrupted',error=repr(error))
        (root/'error.txt').write_text(traceback.format_exc());raise
    finally:
        if env is not None:env.close()
        if vec is not None:
            if status['status']=='completed':vec.close()
            else:
                for process in vec.processes:
                    if process.is_alive():process.terminate()
                for process in vec.processes:
                    process.join(timeout=5)
                    if process.is_alive():process.kill();process.join(timeout=5)
        status.update(wall_seconds=time.perf_counter()-started,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(root/'status.json',status)
    print('Completed bounded longer-placement test:',root,flush=True)


if __name__=='__main__':main()
