"""Behavior and discounted-return checks for placement progress reward; no PPO."""
import argparse
import json
import math
from datetime import datetime
from pathlib import Path
import numpy as np
from uniform.stack_pool_env import StackPoolEnv
from uniform.stack_intermediate import command, eef_pose
from uniform.placement_reward import progress_term


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    args=parser.parse_args()
    gamma=.99
    # A move-and-return cycle must have the same shaping return as staying at
    # the same potential for the same duration, rather than harvest a bonus.
    cycle=progress_term(.2,.4)+gamma*progress_term(.4,.2)
    stationary=progress_term(.2,.2)+gamma*progress_term(.2,.2)
    assert math.isclose(cycle,stationary,abs_tol=1e-12)
    assert progress_term(.6,.6)<0 and progress_term(.6,.7)>0 and progress_term(.6,.3)<0
    output=Path(__file__).parent.parent/'stack_reward_checks'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True)
    records={}
    endings={}
    for mode in ('stock','progress'):
        env=StackPoolEnv(args.pool,reward_mode=mode)
        try:
            for behavior in ('hold','release','raise','away-release'):
                env.reset(options={'configuration_id':3})
                initial_phi=env.previous_potential
                pos,ori=eef_pose(env.sim_env)
                target=pos.copy()
                if behavior=='raise': target[2]+=.15
                if behavior=='away-release': target[0]+=.12
                total=discounted=shape_discounted=task_discounted=0.
                with (output/f'{mode}_{behavior}.jsonl').open('w') as stream:
                    for step in range(1000):
                        if behavior=='hold': action=np.r_[np.zeros(6),1.]
                        elif behavior=='release': action=np.r_[np.zeros(6),-1.]
                        elif behavior=='raise': action=command(env.sim_env,target,ori,1.)
                        else: action=command(env.sim_env,target,ori,1. if step<80 else -1.)
                        _,reward,terminated,truncated,info=env.step(action)
                        assert not terminated and truncated==(step==999)
                        total+=reward;discounted+=gamma**step*reward
                        shape_discounted+=gamma**step*info['shaping_reward']
                        task_discounted+=gamma**step*info['task_reward']
                        stream.write(json.dumps(dict(step=step,reward=reward,**info))+'\n')
                # Includes the truncation transition: Phi is not incorrectly
                # zeroed on an artificial time limit, and SB3 can bootstrap.
                expected=gamma**1000*env.previous_potential-initial_phi
                assert math.isclose(shape_discounted,expected,abs_tol=1e-10)
                if mode=='progress':
                    assert math.isclose(discounted,task_discounted+shape_discounted,abs_tol=1e-10)
                if mode=='stock':
                    assert math.isclose(total,info['episode_outcome']['stock_reward_sum'],abs_tol=1e-8)
                    endings[behavior]=env.sim_env.sim.data.qpos.copy()
                else:
                    np.testing.assert_allclose(env.sim_env.sim.data.qpos,endings[behavior],atol=1e-10,rtol=0)
                records[f'{mode}/{behavior}']=dict(return_sum=total,discounted_return=discounted,
                    discounted_shaping=shape_discounted,success=info['is_success'],
                    task_reward_sum=info['episode_outcome']['task_reward_sum'])
                print(f'{mode}/{behavior}: return {total:.2f}; discounted {discounted:.2f}; success {info["is_success"]}',flush=True)
        finally: env.close()
    assert records['progress/release']['success']
    assert not any(records[f'progress/{behavior}']['success'] for behavior in ('hold','raise','away-release'))
    assert records['progress/hold']['return_sum']<0
    assert records['progress/release']['discounted_return']>max(
        records[f'progress/{behavior}']['discounted_return'] for behavior in ('hold','raise','away-release'))
    (output/'results.json').write_text(json.dumps(dict(status='passed',records=records),indent=2)+'\n')
    print('PASS: physics unchanged by reward mode; hold has no positive reward stream; successful release preferred.',flush=True)
    print('PASS: discounted shaping identity including time limit; no move-return reward farming.',flush=True)
    print('Saved:',output,flush=True)


if __name__=='__main__':main()
