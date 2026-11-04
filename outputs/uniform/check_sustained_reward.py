"""Physical reward/episode semantics checks. No PPO training."""
import argparse
import json
from pathlib import Path
from datetime import datetime
import numpy as np
from uniform.stack_pool_env import StackPoolEnv
from uniform.check import check_stack


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool',type=Path,required=True)
    args=parser.parse_args()
    folder=Path(__file__).parent.parent/'sustained_reward_checks'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder.mkdir(parents=True)
    reports={};endings={}
    # Old progress mode is exercised too: reward changes must not alter physics.
    for mode in ('progress','sustained'):
        env=StackPoolEnv(args.pool,reward_mode=mode)
        try:
            for behavior in ('hold','release','release-then-disrupt'):
                env.reset(options={'configuration_id':3})
                for step in range(1000):
                    if behavior=='release-then-disrupt' and step==950:
                        # Deliberate negative fixture: translate both cubes off
                        # the table, without preserving a successful end state.
                        raw=env.sim_env
                        for obj in (raw.cubeA,raw.cubeB):
                            joint=raw.sim.model.joint_name2id(obj.joints[0])
                            raw.sim.data.qpos[int(raw.sim.model.jnt_qposadr[joint])]+=1.
                        raw.sim.forward()
                    grip=1. if behavior=='hold' else -1.
                    _,reward,terminated,truncated,info=env.step(np.r_[np.zeros(6),grip])
                    assert not terminated and truncated==(step==999)
                    if mode=='sustained':
                        valid=check_stack(env)['is_stacked']
                        assert info['task_reward']==float(valid)
                        assert abs(reward-(float(valid)+info['shaping_reward']))<1e-12
                    else:
                        assert info['task_reward']==float(info['step_success'])
                outcome=info['episode_outcome']
                reports[f'{mode}/{behavior}']=outcome
                if mode=='progress':endings[behavior]=env.sim_env.sim.data.qpos.copy()
                else:
                    np.testing.assert_allclose(env.sim_env.sim.data.qpos,endings[behavior],atol=1e-10,rtol=0)
                    if behavior=='hold':
                        assert outcome['task_reward_sum']==0 and not outcome['success']
                    elif behavior=='release':
                        assert outcome['task_reward_sum']>900 and outcome['success'] and outcome['physical_success_ever']
                    else:
                        assert not outcome['success'] and outcome['physical_success_ever'] and outcome['stock_success_ever']
                        assert not outcome['physical_success_at_end']
                    assert info['is_success']==outcome['physical_success_at_end']
                    env.reset(options={'configuration_id':3})
                    assert not env.physical_tracker.ever and env.physical_tracker.streak==0
                print('PASS',mode,behavior,'task-reward sum',outcome['task_reward_sum'],'episode success',outcome['success'],flush=True)
        finally:env.close()
    (folder/'validation.json').write_text(json.dumps(dict(status='passed',episodes=reports),indent=2)+'\n')
    print('PASS: new reward is physical-frame-based; episode success requires sustained stack at end; physics unchanged.',flush=True)
    print('Saved:',folder,flush=True)


if __name__=='__main__':main()
