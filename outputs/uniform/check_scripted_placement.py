"""Audit complete placement from saved starts; no policy training or state edits.

Uses the current sustained reward and checks, optionally the saved action limits
from a PPO run. Logs every control step and three scripted/control behaviors.
Run with mjpython and --render to watch without recording video.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from robosuite.utils.control_utils import orientation_error
from uniform.stack_pool_env import StackPoolEnv
from uniform.constrained_movement import ConstrainedStackEnv
from uniform.stack_intermediate import StackPreview, eef_pose, cube_positions
from uniform.check import check_stack, DEFAULT_THRESHOLDS


class PlacementScript:
    """Feedback align/lower, then open, withdraw, and stay clear.

    Uses simulator poses, so this is an oracle feasibility check, not a learned
    baseline or a demonstration of PPO performance. All changes use env.step().
    """
    def __init__(self, raw):
        self.orientation = eef_pose(raw)[1]
        self.stage = 'align'
        self.stage_steps = self.stable = 0
        self.timeouts = []
        self.target = None

    def action(self, raw):
        pos, ori = eef_pose(raw)
        a, b = cube_positions(raw)
        self.stage_steps += 1
        if self.stage in ('align', 'lower'):
            goal = b.copy()
            goal[2] += raw.cubeB.top_offset[2] - raw.cubeA.bottom_offset[2]
            goal[2] += .05 if self.stage == 'align' else .002
            error = goal - a
            translation = np.clip(error, -.005, .005)
            if self.stage == 'lower':
                translation[2] = np.clip(error[2], -.001, .001)
            self.stable = self.stable + 1 if np.linalg.norm(error) < .003 else 0
            if self.stable >= 8 or self.stage_steps >= 160:
                if self.stable < 8:
                    self.timeouts.append(self.stage)
                self.stage = 'lower' if self.stage == 'align' else 'open'
                self.stage_steps = self.stable = 0
                if self.stage == 'open':
                    self.target = pos.copy()
            grip = 1.
        else:
            if self.stage == 'open' and self.stage_steps >= 40:
                self.stage = 'withdraw'
                self.stage_steps = 0
                self.target = pos + np.array([0., 0., .12])
            translation = np.clip(self.target - pos, -.005, .005)
            grip = -1.
        rotation = np.clip(orientation_error(self.orientation, ori), -.05, .05)
        # Output in original OSC normalized units; caller inversely maps to any
        # constrained policy space before stepping the environment.
        return np.r_[translation/.05, rotation/.5, grip]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--pool', type=Path)
    source.add_argument('--run', type=Path, help='Use frozen pool, reward and action limits from a saved run')
    parser.add_argument('--ids', nargs='+', type=int)
    parser.add_argument('--behaviors', nargs='+', choices=['scripted', 'hold', 'release'],
                        default=['scripted', 'hold', 'release'])
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    settings = json.loads((args.run/'settings.json').read_text()) if args.run else {}
    pool = args.run/'pool/pool.json' if args.run else args.pool
    manifest = json.loads(pool.read_text())
    ids = args.ids if args.ids is not None else [e['id'] for e in manifest['configurations']
                                               if e['stage'] == 'placement-approach']
    valid_ids = [e['id'] for e in manifest['configurations'] if e['stage'] == 'placement-approach']
    if not ids or len(set(ids)) != len(ids) or any(i not in valid_ids for i in ids):
        parser.error('Choose distinct placement-approach configuration IDs')
    limits = settings.get('action_constraints')
    env_options = {}
    cls = StackPoolEnv
    scales = np.ones(7)
    if limits:
        if limits['type'] != 'cartesian_delta_scaling':
            parser.error('Unsupported saved action mapping')
        cls = ConstrainedStackEnv
        env_options = {k: limits[k] for k in ('translation_limit_m', 'rotation_limit_rad')}
        scales[:3] = limits['translation_limit_m']/.05
        scales[3:6] = limits['rotation_limit_rad']/.5
    root = Path(__file__).parent.parent/'scripted_placement_checks'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    root.mkdir(parents=True)
    env = cls(pool, reward_mode='sustained', configuration_ids=valid_ids,
              progress_gamma=settings.get('gamma', .99),
              progress_weight=settings.get('progress_weight', 1.),
              distance_scale=settings.get('distance_scale_m', .1),
              stability_seconds=settings.get('stability_seconds', 1.), **env_options)
    preview = StackPreview() if args.render else None
    reports = []
    print('Output:', root, flush=True)
    try:
        with (root/'steps.jsonl').open('w') as stream:
            for config_id in ids:
                for behavior in args.behaviors:
                    env.reset(options={'configuration_id': config_id})
                    script = PlacementScript(env.sim_env)
                    failures = Counter()
                    reward_sum = discounted_return = 0.
                    minimum_distance = float('inf')
                    first_open = first_physical = None
                    for step in range(1000):
                        started = time.perf_counter()
                        raw_action = script.action(env.sim_env) if behavior == 'scripted' else np.r_[
                            np.zeros(6), 1. if behavior == 'hold' else -1.]
                        policy_action = np.clip(raw_action/scales, -1., 1.)
                        obs, reward, terminated, truncated, info = env.step(policy_action)
                        if not np.isfinite(obs).all() or not np.isfinite(reward):
                            raise RuntimeError('Nonfinite simulation output')
                        if terminated or truncated != (step == 999):
                            raise RuntimeError('Unexpected episode boundary')
                        physical = check_stack(env)
                        failures.update(physical['failed_conditions'])
                        distance = info['placement_distance_m']
                        minimum_distance = min(minimum_distance, distance)
                        reward_sum += reward
                        discounted_return += env.progress_gamma**step * reward
                        if first_open is None and policy_action[-1] < 0:
                            first_open = step+1
                        if first_physical is None and physical['is_stacked']:
                            first_physical = step+1
                        stream.write(json.dumps(dict(configuration_id=config_id, behavior=behavior,
                            step=step+1, stage=script.stage if behavior == 'scripted' else behavior,
                            policy_action=policy_action.tolist(), controller_action=(policy_action*scales).tolist(),
                            placement_distance_m=distance, reward=reward,
                            task_reward=info['task_reward'], shaping_reward=info['shaping_reward'],
                            physical=physical))+'\n')
                        if preview:
                            preview.update(env.sim_env)
                            time.sleep(max(0., .05-(time.perf_counter()-started)))
                    result = dict(info['episode_outcome'], behavior=behavior,
                        discounted_return=discounted_return, total_reward=reward_sum,
                        minimum_placement_distance_m=minimum_distance,
                        first_open_step=first_open, first_physical_stack_step=first_physical,
                        failed_condition_frames=dict(failures),
                        scripted_stage_timeouts=script.timeouts if behavior == 'scripted' else [])
                    reports.append(result)
                    print(f'ID {config_id:02d} {behavior}: stable at end={result["success"]}; '
                          f'longest={result["longest_physical_stack_seconds"]:.2f}s; '
                          f'return={reward_sum:.2f}; stage timeouts={result["scripted_stage_timeouts"]}',flush=True)
        summary = dict(status='completed', pool_path=str(pool.resolve()), pool_sha256=env.pool_sha256,
            action_constraints=limits, gamma=env.progress_gamma,
            reward_mode=env.reward_mode, progress_weight=env.progress_weight,
            distance_scale_m=env.distance_scale,
            stability_seconds=settings.get('stability_seconds', 1.),
            physical_thresholds=asdict(DEFAULT_THRESHOLDS),
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            no_training=True, oracle_controller=True, episodes=reports)
        (root/'report.json').write_text(json.dumps(summary,indent=2)+'\n')
        print('Completed physical audit. No policy trained. Report:',root/'report.json',flush=True)
    finally:
        env.close()
        if preview:
            preview.close()


if __name__ == '__main__':
    main()
