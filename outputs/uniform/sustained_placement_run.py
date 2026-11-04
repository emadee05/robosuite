"""Fresh PPO + optimizer learning physical placement, with strict evaluations.

Uniform sampling over eight known placement-approach starts. No checkpoint is
loaded and no SGS is used. A valid physical frame earns task reward 1; sustained
and end-of-episode stability are the reported episode success criteria.
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
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import SubprocVecEnv,VecCheckNan
from uniform.stack_pool_env import StackPoolEnv
from uniform.evaluate_stack import evaluate_stack
from uniform.check import DEFAULT_THRESHOLDS
from uniform.train_progress_pilot import PilotPPO
from uniform.train_uniform import write_json
from sgs.train_stack_comparison import EpisodeLog,freeze_pool,fingerprint,sha256,ROLLOUT


def main(argv=None, *, environment_class=StackPoolEnv, environment_options=None,
         action_constraints=None, output_family='sustained_placement_runs', default_steps=1000000):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=default_steps)
    parser.add_argument('--workers',type=int,choices=[1,2,4],default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--eval-seed',type=int,default=2027)
    parser.add_argument('--eval-every-cycles',type=int,default=20)
    parser.add_argument('--eval-repeats',type=int,default=2)
    parser.add_argument('--final-eval-repeats',type=int,default=5)
    parser.add_argument('--stability-seconds',type=float,default=1.)
    parser.add_argument('--progress-weight',type=float,default=1.)
    parser.add_argument('--distance-scale',type=float,default=.1)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--skip-evaluation',action='store_true',help='Short integration tests only')
    args=parser.parse_args(argv)
    if min(args.steps,args.eval_every_cycles,args.eval_repeats,args.final_eval_repeats)<1:
        parser.error('Budgets and evaluation settings must be positive')
    if min(args.seed,args.eval_seed)<0 or max(args.seed,args.eval_seed)+32>=2**32:
        parser.error('Invalid seed')
    if not math.isfinite(args.progress_weight) or args.progress_weight<0 or not math.isfinite(args.distance_scale) or args.distance_scale<=0:
        parser.error('Invalid potential weight or distance scale')
    if not math.isfinite(args.stability_seconds) or args.stability_seconds<=0:
        parser.error('Stability duration must be finite and positive')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=args.output_dir or Path(__file__).parent.parent/output_family/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root=root.resolve();root.mkdir(parents=True,exist_ok=False)
    freeze_pool(args.pool.resolve(),root/'pool')
    pool=json.loads((root/'pool/pool.json').read_text())
    ids=[e['id'] for e in pool['configurations'] if e['stage']=='placement-approach']
    if len(ids)!=8:raise ValueError('Expected eight placement-approach states')
    target=math.ceil(args.steps/ROLLOUT)*ROLLOUT
    (root/'sources').mkdir()
    for package in ('uniform','sgs'):
        destination=root/'sources'/package;destination.mkdir()
        for path in (Path(__file__).parent.parent/package).glob('*.py'):
            shutil.copyfile(path,destination/path.name)
    settings=dict(task='Stack',mode='sustained',pool_sha256=sha256(root/'pool/pool.json'),
        configuration_ids=ids,requested_steps=args.steps,actual_target_steps=target,
        initialization='Fresh actor, critic AND optimizer; no saved policy loaded',
        workers=args.workers,seed=args.seed,eval_seed=args.eval_seed,
        actor=[64,64],critic=[64,64],device='cpu',rollout=ROLLOUT,minibatch=256,epochs=10,
        learning_rate=3e-4,gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,
        vf_coef=.5,max_grad_norm=.5,horizon=1000,control_freq=20,
        progress_weight=args.progress_weight,distance_scale_m=args.distance_scale,
        reward='float(current_frame_passes_physical_checks) + weight*(gamma*Phi(next)-Phi(current))',
        potential='exp(-cube_center_distance_to_stacked_goal / distance_scale)',
        no_stock_task_bonus=True,task_success_metric='stable_at_episode_end',
        stability_seconds=args.stability_seconds,physical_thresholds=asdict(DEFAULT_THRESHOLDS),
        eval_every_cycles=args.eval_every_cycles,eval_repeats=args.eval_repeats,
        final_eval_repeats=args.final_eval_repeats,skip_evaluation=args.skip_evaluation,
        sampling='Uniform over eight assisted placement starts; no SGS',
        time_limit='Truncated at 1000 steps; final potential retained; SB3 value bootstrap',
        versions=dict(sb3=sb3.__version__,torch=torch.__version__),
        caveats=['Known assisted starts, not full-task or held-out evaluation',
                 'Physical frame reward is state-based; duration is evaluated separately',
                 'Best-checkpoint selection is optimistic, not independent validation',
                 'No automatic mid-run resume'])
    if action_constraints is not None:
        settings['action_constraints']=action_constraints
    write_json(root/'settings.json',settings)
    folder=root/'sustained';folder.mkdir()
    (folder/'checkpoints').mkdir();(folder/'evaluations').mkdir()
    status=dict(status='running',started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'status.json',status)
    vec=env=model=None
    history=[];best={}
    start=time.perf_counter()
    print('Output:',root,flush=True)
    print(f'FRESH policy and optimizer; {target:,} training transitions; {args.workers} workers',flush=True)
    print('Task reward: physical frame valid. Episode success: physical stack stable at episode end.',flush=True)
    try:
        kwargs=dict(reward_mode='sustained',configuration_ids=ids,progress_gamma=.99,
            progress_weight=args.progress_weight,distance_scale=args.distance_scale,
            stability_seconds=args.stability_seconds)
        kwargs.update(environment_options or {})
        vec=SubprocVecEnv([partial(environment_class,root/'pool/pool.json',args.seed+i,**kwargs)
                          for i in range(args.workers)],start_method='spawn')
        # Constructor, deliberately not PPO.load(): no old policy/optimizer state.
        model=PilotPPO('MlpPolicy',VecCheckNan(vec,raise_exception=True,check_inf=True),
            device='cpu',seed=args.seed,n_steps=ROLLOUT//args.workers,batch_size=256,n_epochs=10,
            policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}},learning_rate=3e-4,
            gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,
            max_grad_norm=.5,target_kl=None,verbose=0)
        assert model.num_timesteps==0 and not model.policy.optimizer.state_dict()['state']
        assert model.gamma==kwargs['progress_gamma'] and model.n_steps*model.n_envs==ROLLOUT
        write_json(folder/'initial_checkpoint.json',dict(training_transitions=0,
            optimizer_state_entries=0,loaded_checkpoint=None,
            policy_sha256=fingerprint(model.policy.state_dict()),
            optimizer_sha256=fingerprint(model.policy.optimizer.state_dict())))
        env=environment_class(root/'pool/pool.json',seed=args.eval_seed,**kwargs)
        schedule=[0]+list(range(args.eval_every_cycles*ROLLOUT,target,args.eval_every_cycles*ROLLOUT))+[target]
        with (folder/'training_episodes.jsonl').open('w') as episodes,(folder/'progress.jsonl').open('w') as progress:
            callback=EpisodeLog(episodes)
            model.pilot_progress=dict(mode='sustained',target=target,callback=callback,stream=progress,seconds=0.)
            for step in schedule:
                if step:
                    model.learn(total_timesteps=step-model.num_timesteps,reset_num_timesteps=False,callback=callback)
                assert model.num_timesteps==step
                path=folder/'checkpoints'/f'ppo_{step:09d}.zip';model.save(path)
                if step==target:shutil.copyfile(path,folder/'final_model.zip')
                result=dict(reward_mode='sustained',training_transitions=step,
                    training_seconds=model.pilot_progress['seconds'],completed_training_episodes=len(callback.records),
                    stable_at_end_training_episodes=sum(callback.successes),
                    sustained_ever_training_episodes=sum(e['physical_success_ever'] for e in callback.records),
                    stock_successful_training_episodes=sum(e['stock_success_ever'] for e in callback.records),
                    visits_by_configuration=callback.visits,evaluations=history,best_checkpoints=best)
                write_json(folder/'results.json',result);write_json(root/'results.json',result)
                if args.skip_evaluation:continue
                for deterministic,label,repeats in ((True,'deterministic',1),
                        (False,'sampled',args.final_eval_repeats if step==target else args.eval_repeats)):
                    print(f'Evaluating sustained at {step:,}: {label}',flush=True)
                    report=evaluate_stack(model,env,ids=ids,repeats=repeats,deterministic=deterministic,
                        seed=args.eval_seed,physical_checks=True,stability_seconds=args.stability_seconds)
                    assert report['successes']==report['physical']['stable_at_end_successes']
                    report.update(training_transitions=step)
                    write_json(folder/'evaluations'/f'{label}_{step:09d}.json',report)
                    history.append({k:v for k,v in report.items() if k!='episodes'})
                    physical=report['physical'];n=report['episode_count']
                    print(f'  Stable at end {physical["stable_at_end_successes"]}/{n}; '
                          f'sustained ever {physical["sustained_successes"]}/{n}; '
                          f'original stock flag {report["stock_successes"]}/{n}',flush=True)
                    rank=(physical['stable_at_end_success_rate'],physical['sustained_success_rate'])
                    previous=best.get(label)
                    if previous is None or rank>tuple(previous['rank']):
                        shutil.copyfile(path,folder/f'best_{label}.zip')
                        best[label]=dict(rank=list(rank),training_transitions=step,
                            episode_count=n,stable_at_end_successes=physical['stable_at_end_successes'],
                            sustained_successes=physical['sustained_successes'])
                    write_json(folder/'results.json',result);write_json(root/'results.json',result)
        assert model.policy.optimizer.state_dict()['state']
        write_json(folder/'final_checkpoint.json',dict(training_transitions=model.num_timesteps,
            policy_sha256=fingerprint(model.policy.state_dict()),
            optimizer_sha256=fingerprint(model.policy.optimizer.state_dict()),
            optimizer_state_entries=len(model.policy.optimizer.state_dict()['state'])))
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
        status.update(wall_seconds=time.perf_counter()-start,finished_utc=datetime.now(timezone.utc).isoformat())
        write_json(root/'status.json',status)
    print('Completed fresh sustained-placement run:',root,flush=True)


if __name__=='__main__':main()
