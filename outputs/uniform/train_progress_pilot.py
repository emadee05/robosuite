"""Matched stock-vs-progress PPO pilot on eight placement-approach starts.

Uniform sampling only, fresh small policies, no SGS and no old checkpoints.
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
from datetime import datetime
from functools import partial
from pathlib import Path
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv,VecCheckNan
from uniform.stack_pool_env import StackPoolEnv
from uniform.evaluate_stack import evaluate_stack
from uniform.train_uniform import write_json
from sgs.train_stack_comparison import StackPPO,EpisodeLog,freeze_pool,fingerprint,ROLLOUT


class PilotPPO(StackPPO):
    def _excluded_save_params(self):
        return super()._excluded_save_params()+['pilot_progress']

    def train(self):
        super().train()
        tracking=getattr(self,'pilot_progress',None)
        if tracking is None:return
        elapsed=time.perf_counter()-self.rollout_clock
        tracking['seconds']+=elapsed
        item=dict(training_transitions=self.num_timesteps,cycle=self.num_timesteps//ROLLOUT,
                  cycle_seconds=elapsed,transitions_per_second=ROLLOUT/elapsed,
                  completed_episodes=len(tracking['callback'].records))
        tracking['stream'].write(json.dumps(item)+'\n');tracking['stream'].flush()
        print(f'{tracking["mode"]}: cycle {item["cycle"]}/{tracking["target"]//ROLLOUT}; '
              f'{self.num_timesteps:,} transitions; {item["transitions_per_second"]:.0f}/s; '
              f'{item["completed_episodes"]} completed episodes',flush=True)


def run_mode(args,root,mode,target,ids):
    folder=root/mode;folder.mkdir()
    (folder/'checkpoints').mkdir();(folder/'evaluations').mkdir()
    vec=env=model=None
    status=dict(status='running');write_json(folder/'status.json',status)
    history=[]
    try:
        kwargs=dict(reward_mode=mode,configuration_ids=ids,progress_gamma=.99,
                    progress_weight=args.progress_weight,distance_scale=args.distance_scale)
        vec=SubprocVecEnv([partial(StackPoolEnv,root/'pool/pool.json',args.seed+i,**kwargs)
                          for i in range(4)],start_method='spawn')
        model=PilotPPO('MlpPolicy',VecCheckNan(vec,raise_exception=True,check_inf=True),
            device='cpu',seed=args.seed,n_steps=1536,batch_size=256,n_epochs=10,
            policy_kwargs={'net_arch':{'pi':[64,64],'vf':[64,64]}},learning_rate=3e-4,
            gamma=.99,gae_lambda=.95,clip_range=.2,ent_coef=0.,vf_coef=.5,max_grad_norm=.5,
            target_kl=None,verbose=0)
        assert model.gamma==kwargs['progress_gamma']
        initial=dict(policy_sha256=fingerprint(model.policy.state_dict()),
                     optimizer_sha256=fingerprint(model.policy.optimizer.state_dict()))
        write_json(folder/'initial_checkpoint.json',initial)
        if mode=='progress' and (root/'stock/initial_checkpoint.json').exists():
            assert initial==json.loads((root/'stock/initial_checkpoint.json').read_text())
        env=StackPoolEnv(root/'pool/pool.json',seed=args.eval_seed,**kwargs)
        schedule=[0]+list(range(args.eval_every_cycles*ROLLOUT,target,args.eval_every_cycles*ROLLOUT))+[target]
        with (folder/'training_episodes.jsonl').open('w') as episodes,(folder/'progress.jsonl').open('w') as progress:
            callback=EpisodeLog(episodes)
            model.pilot_progress=dict(mode=mode,target=target,callback=callback,stream=progress,seconds=0.)
            for checkpoint in schedule:
                if checkpoint:
                    model.learn(total_timesteps=checkpoint-model.num_timesteps,
                                reset_num_timesteps=False,callback=callback)
                model.save(folder/'checkpoints'/f'ppo_{checkpoint:09d}.zip')
                if checkpoint==target:model.save(folder/'final_model.zip')
                result=dict(reward_mode=mode,training_transitions=model.num_timesteps,
                    training_seconds=model.pilot_progress['seconds'],completed_training_episodes=len(callback.records),
                    successes=sum(callback.successes),visits_by_configuration=callback.visits,
                    training_successes_by_configuration=callback.successes,evaluations=history)
                write_json(folder/'results.json',result)
                if not args.skip_evaluation:
                    for deterministic,label,repeats in ((True,'deterministic',1),
                        (False,'sampled',args.final_eval_repeats if checkpoint==target else args.eval_repeats)):
                        print(f'{mode}: evaluating {checkpoint:,} transitions; {label}',flush=True)
                        report=evaluate_stack(model,env,ids=ids,repeats=repeats,
                                              deterministic=deterministic,seed=args.eval_seed)
                        report.update(reward_mode=mode,training_transitions=checkpoint)
                        write_json(folder/'evaluations'/f'{label}_{checkpoint:09d}.json',report)
                        history.append({k:v for k,v in report.items() if k!='episodes'})
                        print(f'{mode}: {report["successes"]}/{report["episode_count"]} '
                              f'({report["success_rate"]:.1%}) successful',flush=True)
                    write_json(folder/'results.json',result)
        status['status']='completed'
    except BaseException as error:
        status.update(status='failed_or_interrupted',error=repr(error))
        (folder/'error.txt').write_text(traceback.format_exc());raise
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
        write_json(folder/'status.json',status)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=200000)
    parser.add_argument('--mode',choices=['both','stock','progress'],default='both')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--eval-seed',type=int,default=2027)
    parser.add_argument('--eval-every-cycles',type=int,default=10)
    parser.add_argument('--eval-repeats',type=int,default=2)
    parser.add_argument('--final-eval-repeats',type=int,default=5)
    parser.add_argument('--progress-weight',type=float,default=1.)
    parser.add_argument('--distance-scale',type=float,default=.1)
    parser.add_argument('--skip-evaluation',action='store_true',help='Integration checks only')
    args=parser.parse_args()
    if min(args.steps,args.eval_every_cycles,args.eval_repeats,args.final_eval_repeats)<1:
        parser.error('Budgets and intervals must be positive')
    if min(args.seed,args.eval_seed)<0 or max(args.seed,args.eval_seed)+32>=2**32:
        parser.error('Invalid seed')
    if not math.isfinite(args.progress_weight) or args.progress_weight<0 or not math.isfinite(args.distance_scale) or args.distance_scale<=0:
        parser.error('Invalid potential weight or distance scale')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    root=Path(__file__).parent.parent/'stack_reward_pilots'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    freeze_pool(args.pool.resolve(),root/'pool')
    pool=json.loads((root/'pool/pool.json').read_text())
    ids=[entry['id'] for entry in pool['configurations'] if entry['stage']=='placement-approach']
    assert len(ids)==8
    target=math.ceil(args.steps/ROLLOUT)*ROLLOUT
    (root/'sources').mkdir()
    for package in ('uniform','sgs'):
        destination=root/'sources'/package;destination.mkdir()
        for path in (Path(__file__).parent.parent/package).glob('*.py'):
            shutil.copyfile(path,destination/path.name)
    write_json(root/'settings.json',dict(task='Stack',configuration_ids=ids,requested_steps=args.steps,
        actual_steps_per_mode=target,mode=args.mode,seed=args.seed,eval_seed=args.eval_seed,
        eval_every_cycles=args.eval_every_cycles,eval_repeats=args.eval_repeats,
        final_eval_repeats=args.final_eval_repeats,skip_evaluation=args.skip_evaluation,
        progress_weight=args.progress_weight,distance_scale_m=args.distance_scale,gamma=.99,
        formula='float(stock_success) + weight*(gamma*Phi(next)-Phi(current)); Phi=exp(-distance/scale)',
        time_limit='1000-step truncation; retain final potential; SB3 bootstraps final value',
        selection='Uniform over eight placement starts; no SGS',
        comparison='Fresh matching actor/critic/optimizer seeds; compare binary success, not cross-mode returns',
        generalization='Known placement starts only; not full task or held-out evaluation'))
    print('Output:',root,flush=True)
    print('Placement IDs:',ids,'; transitions per mode:',f'{target:,}',flush=True)
    status=dict(status='running');write_json(root/'status.json',status)
    try:
        for mode in (('stock','progress') if args.mode=='both' else (args.mode,)):
            status['active_mode']=mode;write_json(root/'status.json',status)
            run_mode(args,root,mode,target,ids)
        status.update(status='completed',active_mode=None)
    except BaseException as error:
        status.update(status='failed_or_interrupted',error=repr(error));raise
    finally:write_json(root/'status.json',status)
    print('Completed reward pilot:',root,flush=True)


if __name__=='__main__':main()
