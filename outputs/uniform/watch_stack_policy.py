"""Display saved reward-pilot PPO episodes without video recording or training.

Defaults: both progress checkpoints, all eight trained placement starts, mean
(deterministic) actions. One persistent MuJoCo window; read-only saved artifacts.
"""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from uniform.stack_pool_env import StackPoolEnv
from uniform.stack_intermediate import StackPreview
from uniform.check import StackStabilityTracker


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True,help='Saved stack_reward_pilots/<timestamp> folder')
    parser.add_argument('--mode',choices=['stock','progress','sustained'],default='progress')
    parser.add_argument('--checkpoints',nargs='+',type=int,default=[122880,202752])
    parser.add_argument('--ids',nargs='+',type=int,help='Default: all eight placement IDs from run settings')
    parser.add_argument('--sampled',action='store_true',help='Sample actions instead of choosing the mean')
    parser.add_argument('--repeats',type=int,default=1)
    parser.add_argument('--pause',action='store_true',help='Press Enter in terminal to start each episode')
    parser.add_argument('--speed',type=float,default=1.,help='Playback speed; 1 = real time')
    parser.add_argument('--headless',action='store_true',help='Check episodes without opening a viewer')
    parser.add_argument('--stability-seconds',type=float,default=1.,help='Consecutive physical stack duration required')
    args=parser.parse_args()
    if args.repeats<1 or not np.isfinite(args.speed) or args.speed<=0:
        parser.error('Repeats and speed must be positive')
    if not np.isfinite(args.stability_seconds) or args.stability_seconds<=0:
        parser.error('Stability duration must be finite and positive')
    root=args.run.resolve()
    settings=json.loads((root/'settings.json').read_text())
    ids=settings['configuration_ids'] if args.ids is None else args.ids
    if not ids or any(i not in settings['configuration_ids'] for i in ids):
        parser.error('IDs must belong to the configurations used to train this pilot')
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    models={}
    originals={}
    for steps in args.checkpoints:
        path=root/args.mode/'checkpoints'/f'ppo_{steps:09d}.zip'
        model=PPO.load(path,device='cpu')
        if model.num_timesteps!=steps:raise ValueError('Checkpoint timestep mismatch')
        models[steps]=model
        originals[steps]={k:v.detach().clone() for k,v in model.policy.state_dict().items()}
    environment_class=StackPoolEnv
    action_options={}
    if 'action_constraints' in settings:
        from uniform.constrained_movement import ConstrainedStackEnv
        limits=settings['action_constraints']
        if limits['type']!='cartesian_delta_scaling':
            raise ValueError('Unknown saved action constraint type')
        environment_class=ConstrainedStackEnv
        action_options=dict(translation_limit_m=limits['translation_limit_m'],
                            rotation_limit_rad=limits['rotation_limit_rad'])
    env=environment_class(root/'pool/pool.json',reward_mode=args.mode,
        progress_gamma=settings['gamma'],progress_weight=settings['progress_weight'],
        distance_scale=settings['distance_scale_m'],configuration_ids=settings['configuration_ids'],
        stability_seconds=settings.get('stability_seconds',1.),**action_options)
    preview=None if args.headless else StackPreview()
    count=len(ids)*len(models)*args.repeats
    print(f'Displaying {count} episodes; {"sampled" if args.sampled else "deterministic"} actions. '
          'No training, no video recording, no result files written.',flush=True)
    try:
        # Show the same configuration at both checkpoints back-to-back.
        for config_id in ids:
            for steps,model in models.items():
                set_random_seed(settings['eval_seed']+config_id)
                for repeat in range(args.repeats):
                    obs,info=env.reset(options={'configuration_id':config_id})
                    print(f'ID {config_id:02d}; {args.mode}; checkpoint {steps:,}; '
                          f'{info["stage"]}; attempt {repeat+1}/{args.repeats}',flush=True)
                    if preview is not None:preview.update(env.sim_env)
                    if args.pause:input('Press Enter in the terminal to start this episode... ')
                    done=False
                    checker=StackStabilityTracker(required_seconds=args.stability_seconds,
                                                   control_freq=env.sim_env.control_freq)
                    successes=streak=longest=0
                    previous=False
                    previous_stable=False
                    while not done:
                        start=time.perf_counter()
                        action,_=model.predict(obs,deterministic=not args.sampled)
                        if not np.isfinite(action).all():raise ValueError('Nonfinite action')
                        obs,reward,terminated,truncated,info=env.step(action)
                        if not np.isfinite(obs).all() or not np.isfinite(reward):
                            raise ValueError('Nonfinite observation/reward')
                        done=terminated or truncated
                        success=info['step_success']
                        strict=checker.update(env)
                        successes+=int(success)
                        streak=streak+1 if success else 0
                        longest=max(longest,streak)
                        if success!=previous:
                            print(f'  step {env.outcome.steps}: stock success flag -> {success}',flush=True)
                            if success and strict['failed_conditions']:
                                print('    Physical conditions failing: '+', '.join(strict['failed_conditions']),flush=True)
                        previous=success
                        if strict['stable_now']!=previous_stable:
                            print(f'  step {env.outcome.steps}: sustained physical stack -> {strict["stable_now"]}',flush=True)
                        previous_stable=strict['stable_now']
                        if preview is not None:
                            preview.update(env.sim_env)
                            time.sleep(max(0.,.05/args.speed-(time.perf_counter()-start)))
                    print(f'  Episode ended: stock success ever={successes>0}; '
                          f'successful steps={successes}; longest stock-flag streak={longest} '
                          f'({longest/20:.2f}s); success at end={success}; '
                          f'final placement distance={info["placement_distance_m"]:.3f}m',flush=True)
                    print(f'  Task success ({args.mode} definition)={info["is_success"]}',flush=True)
                    print(f'  Physical stack: sustained ever={strict["stable_ever"]}; '
                          f'sustained at episode end={strict["stable_now"]}; '
                          f'longest valid streak={strict["longest_stable_seconds"]:.2f}s',flush=True)
        for steps,model in models.items():
            assert model.num_timesteps==steps
            assert all(torch.equal(originals[steps][k],v) for k,v in model.policy.state_dict().items())
        print('Finished. Policies/pool unchanged. Physical checks are separate from the stock success flag.',flush=True)
    except KeyboardInterrupt:
        print('Viewer stopped by user. No training performed.',flush=True)
    finally:
        env.close()
        if preview is not None:preview.close()


if __name__=='__main__':main()
