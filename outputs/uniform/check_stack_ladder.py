"""Validate the Stack ladder and measure untrained PPO success by rung.

No learn/update call. Uses Damiano's actual actor/critic and initial action
distribution, with initial (unfitted) observation statistics. Rates are a small
diagnostic on the known pool, not learning results or generalization evidence.
"""
import argparse
from collections import defaultdict
from datetime import datetime
import json
from pathlib import Path
import time
import numpy as np
import torch
from stable_baselines3.common.env_checker import check_env
from uniform.stack_ladder import StackLadderEnv, PPO_SETTINGS, resume_script
from uniform.ladder_ppo import Agent, RunningMeanStd
from uniform.stack_intermediate import StackPreview


def validate_restoration(env, ids):
    continuations = []
    for config_id in ids:
        endings = []
        for repeat in range(2):
            obs, _ = env.reset(options={'configuration_id':config_id})
            state = env.states[config_id]
            np.testing.assert_allclose(env.sim_env.sim.data.qpos, state['qpos'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(env.sim_env.sim.data.qvel, state['qvel'], atol=1e-12, rtol=0)
            assert env.tracker.streak == 0 and not env.tracker.ever
            for _ in range(5):
                obs, reward, terminated, truncated, _ = env.step(np.r_[np.zeros(6), -1.])
                assert not terminated and not truncated and reward == 0
                assert np.isfinite(obs).all()
            endings.append(env.sim_env.sim.data.qpos.copy())
        np.testing.assert_allclose(endings[0], endings[1], atol=1e-8, rtol=0)
        entry = env.entries[config_id]
        if entry['script_continuation'] is not None:
            env.reset(options={'configuration_id':config_id})
            script = resume_script(entry['script_continuation'])
            total = 0.
            while True:
                obs, reward, terminated, truncated, info = env.step(script.action(env.sim_env))
                assert np.isfinite(obs).all() and reward in (0.,1.)
                total += reward
                if terminated or truncated:
                    break
            success = terminated and not truncated and total == 1. and not script.timeouts
            continuations.append(dict(configuration_id=config_id, label=entry['label'],
                scene_id=entry['scene_id'], success=success, length=env.steps,
                stage_timeouts=script.timeouts))
            if not success:
                raise AssertionError(f'Script continuation failed: {continuations[-1]}')
            try:
                env.step(np.zeros(7))
            except RuntimeError:
                pass
            else:
                raise AssertionError('Terminal episode accepted another step')
        print(f'PASS restore ID {config_id:02d} ({entry["label"]})'
              + ('; scripted continuation succeeds' if entry['script_continuation'] else ''), flush=True)
    # Holding from the original above-target start must not earn sparse success.
    hold_id = next(e['id'] for e in env.entries if e['label'] == 'placement-approach')
    env.reset(options={'configuration_id':hold_id})
    total = 0.
    for step in range(env.horizon):
        _, reward, terminated, truncated, _ = env.step(np.r_[np.zeros(6),1.])
        total += reward
        assert not terminated and truncated == (step == env.horizon-1)
    assert total == 0.
    print('PASS holding earns zero reward; timeout is truncated, not terminated',flush=True)
    return continuations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--sampled-repeats', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--ids', type=int, nargs='+', help='Optional small diagnostic subset')
    parser.add_argument('--skip-baseline', action='store_true')
    parser.add_argument('--render', action='store_true', help='Display baseline episodes with mjpython')
    args = parser.parse_args()
    if args.sampled_repeats < 1 or args.seed < 0:
        parser.error('Positive repeats and nonnegative seed required')
    torch.set_num_threads(PPO_SETTINGS['torch_threads'])
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    env = StackLadderEnv(args.pool)
    ids = list(range(len(env.entries))) if args.ids is None else args.ids
    if not ids or len(set(ids)) != len(ids) or any(i not in range(len(env.entries)) for i in ids):
        parser.error('Choose distinct valid configuration IDs')
    root = args.pool.resolve().parent/'checks'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True, exist_ok=False)
    print('Output:', root, flush=True)
    preview = None
    try:
        check_env(env, warn=True)
        continuations = validate_restoration(env, ids)
        report = dict(status='physical_checks_passed', pool_sha256=env.pool_sha256,
            configuration_ids=ids, scripted_continuations=continuations, no_training=True,
            ppo_settings=PPO_SETTINGS, baseline=[], by_label={})
        report_path = root/'report.json'
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        if args.skip_baseline:
            print('PASS validation. Baseline skipped. Report:', report_path,flush=True)
            return
        agent = Agent(env.observation_space.shape[0],7,hidden=256,log_std_init=-.5).eval()
        obs_rms = RunningMeanStd((env.observation_space.shape[0],))
        initial = {k:v.detach().clone() for k,v in agent.state_dict().items()}
        report.update(observation_statistics='Unfitted: mean zero, variance one; frozen throughout baseline',
                      sampled_repeats=args.sampled_repeats, seed=args.seed)
        torch.save(dict(agent=agent.state_dict(),obs_rms=obs_rms.state_dict(),
                        settings=PPO_SETTINGS, training_transitions=0),root/'untrained_policy.pt')
        preview = StackPreview() if args.render else None
        with (root/'episodes.jsonl').open('w') as stream:
            for deterministic in (True,False):
                mode = 'deterministic' if deterministic else 'sampled'
                counts = defaultdict(lambda: [0,0])
                for config_id in ids:
                    for repeat in range(1 if deterministic else args.sampled_repeats):
                        torch.manual_seed(args.seed+config_id*1000+repeat)
                        obs, _ = env.reset(options={'configuration_id':config_id})
                        while True:
                            started = time.perf_counter()
                            with torch.no_grad():
                                action, _, _, _ = agent.act(torch.as_tensor(obs_rms.normalize(obs))[None],
                                                           deterministic=deterministic)
                            obs, _, terminated, truncated, info = env.step(action[0].numpy())
                            if not np.isfinite(obs).all():
                                raise RuntimeError('Nonfinite baseline observation')
                            if preview:
                                preview.update(env.sim_env)
                                time.sleep(max(0.,.05-(time.perf_counter()-started)))
                            if terminated or truncated:
                                break
                        outcome = dict(info['episode_outcome'], mode=mode, repeat=repeat+1)
                        report['baseline'].append(outcome)
                        stream.write(json.dumps(outcome)+'\n');stream.flush()
                        counts[outcome['label']][0] += int(outcome['success'])
                        counts[outcome['label']][1] += 1
                report['by_label'][mode] = {label:dict(successes=s, episodes=n, success_rate=s/n)
                                            for label,(s,n) in counts.items()}
                report_path.write_text(json.dumps(report,indent=2)+'\n')
                print(f'UNTRAINED {mode} by start:',flush=True)
                for label in env.pool['labels']:
                    if label in counts:
                        s,n=counts[label]
                        print(f'  {label:20s} {s}/{n} ({s/n:.1%})',flush=True)
        assert all(torch.equal(initial[k],v) for k,v in agent.state_dict().items())
        assert obs_rms.count == 1e-4
        report['status'] = 'completed'
        report['policy_and_normalizer_unchanged'] = True
        report_path.write_text(json.dumps(report,indent=2)+'\n')
        print('PASS policy/normalizer unchanged. No PPO training. Report:', report_path,flush=True)
    finally:
        env.close()
        if preview:
            preview.close()


if __name__ == '__main__':
    main()
