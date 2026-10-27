"""Fixed-pool uniform PPO baseline with checkpoints and balanced evaluation."""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import argparse
import json
import math
import random
import shutil
import time
from datetime import datetime, timezone
from functools import partial
from pathlib import Path

import numpy as np
import torch
import stable_baselines3 as sb3
import gymnasium
import robosuite
import mujoco
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecCheckNan
from fixed_pool import POOL_PATH, pool_hash
from uniform_env import UniformLiftEnv
from evaluate_policy import evaluate_policy

ROLLOUT = 6144


def write_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2) + '\n')
    temporary.replace(path)


def create_env(seed, pool_path):
    return UniformLiftEnv(seed=seed, horizon=1000, pool_path=pool_path)


class Outcomes(BaseCallback):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.visits = np.zeros(32, dtype=int)
        self.successes = np.zeros(32, dtype=int)

    def _on_step(self):
        for worker, info in enumerate(self.locals['infos']):
            if 'episode_outcome' not in info:
                continue
            outcome = dict(info['episode_outcome'], worker=worker,
                           training_transitions=self.num_timesteps)
            index = outcome['configuration_id']
            self.visits[index] += 1
            self.successes[index] += int(outcome['success'])
            self.stream.write(json.dumps(outcome)+'\n')
            self.stream.flush()
        return True


def evaluate_without_changing_training_rng(model, env):
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    training_mode = model.policy.training
    try:
        return evaluate_policy(model, env)
    finally:
        random.setstate(state[0])
        np.random.set_state(state[1])
        torch.set_rng_state(state[2])
        model.policy.set_training_mode(training_mode)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--steps', type=int, default=300000)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every-cycles', type=int, default=10)
    parser.add_argument('--output-dir', type=Path, default=Path(__file__).parent /
                        ('uniform_run_' + datetime.now().strftime('%Y%m%d_%H%M%S')))
    args = parser.parse_args()
    if args.steps < 1 or args.workers < 1 or ROLLOUT % args.workers or args.eval_every_cycles < 1 or args.seed < 0:
        parser.error('Positive steps/evaluation interval; workers must divide 6144; seed must be nonnegative')
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root/'checkpoints').mkdir()
    (root/'evaluations').mkdir()
    (root/'sources').mkdir()
    pool_path = root/'train_pool.json'
    shutil.copyfile(POOL_PATH, pool_path)
    for name in ('fixed_pool.py', 'uniform_env.py', 'evaluate_policy.py', 'train_uniform.py'):
        shutil.copyfile(Path(__file__).with_name(name), root/'sources'/name)
    target = math.ceil(args.steps/ROLLOUT)*ROLLOUT
    interval = args.eval_every_cycles*ROLLOUT
    schedule = list(range(interval, target, interval)) + [target]
    settings = dict(requested_steps=args.steps, actual_target_steps=target,
        seed=args.seed, workers=args.workers, pool_sha256=pool_hash(pool_path),
        sampler='independent uniform sampling with replacement, probability 1/32 per ID per reset',
        actor=[64,64], critic=[64,64], rollout=ROLLOUT, minibatch=256, epochs=10,
        episode_horizon=1000, control_freq=20, dense_rewards=True, device='cpu',
        success='success condition true at least once during episode; no early termination',
        evaluation_steps=[0]+schedule, evaluation_episodes=32,
        evaluation='deterministic actions, one episode per training-pool ID; not held-out',
        versions=dict(sb3=sb3.__version__, gymnasium=gymnasium.__version__,
                      torch=torch.__version__, robosuite=robosuite.__version__, mujoco=mujoco.__version__))
    write_json(root/'settings.json', settings)
    status = dict(status='starting', started_utc=datetime.now(timezone.utc).isoformat())
    write_json(root/'status.json', status)
    vec = evaluation_env = None
    started = time.perf_counter()
    training_seconds = evaluation_seconds = 0.0
    history = []
    completed = False
    print(f'Target: {target:,} training transitions (rounded to complete rollouts).', flush=True)
    print('Evaluation checkpoints:', [0]+schedule, flush=True)
    try:
        vec = SubprocVecEnv([partial(create_env, args.seed+i, pool_path) for i in range(args.workers)], start_method='spawn')
        checked = VecCheckNan(vec, raise_exception=True, check_inf=True)
        evaluation_env = create_env(args.seed, pool_path)
        model = sb3.PPO('MlpPolicy', checked, seed=args.seed, device='cpu',
            policy_kwargs={'net_arch': {'pi':[64,64], 'vf':[64,64]}},
            n_steps=ROLLOUT//args.workers, batch_size=256, n_epochs=10,
            learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, target_kl=None, verbose=0)
        with (root/'training_episodes.jsonl').open('w') as stream:
            callback = Outcomes(stream)
            for checkpoint in [0]+schedule:
                if checkpoint:
                    segment_start = time.perf_counter()
                    model.learn(total_timesteps=checkpoint-model.num_timesteps,
                                reset_num_timesteps=False, callback=callback)
                    training_seconds += time.perf_counter()-segment_start
                    assert model.num_timesteps == checkpoint
                    assert all(torch.isfinite(p).all().item() for p in model.policy.parameters())
                model.save(root/'checkpoints'/f'ppo_{checkpoint:09d}')
                print(f'\nEvaluating at {checkpoint:,} training transitions...', flush=True)
                evaluation = evaluate_without_changing_training_rng(model, evaluation_env)
                evaluation_seconds += evaluation['wall_seconds']
                evaluation.update(training_transitions=checkpoint, seed=args.seed,
                                  cumulative_training_seconds=training_seconds,
                                  wall_seconds_since_start=time.perf_counter()-started)
                write_json(root/'evaluations'/f'eval_{checkpoint:09d}.json', evaluation)
                history.append({k:v for k,v in evaluation.items() if k not in ('episodes','per_configuration')})
                write_json(root/'results.json', dict(evaluations=history,
                    completed_training_episodes=int(callback.visits.sum()),
                    visits_by_configuration=callback.visits.tolist(),
                    training_successes_by_configuration=callback.successes.tolist(),
                    training_seconds=training_seconds, evaluation_seconds=evaluation_seconds))
                print(f'Checkpoint {checkpoint:,}: evaluation success {evaluation["successes"]}/32 '
                      f'({evaluation["success_rate"]:.1%}); training episodes {callback.visits.sum()}', flush=True)
            completed = True
            status['status'] = 'completed'
    except BaseException as error:
        status.update(status='failed_or_interrupted', error=repr(error))
        raise
    finally:
        if evaluation_env is not None:
            evaluation_env.close()
        if vec is not None:
            if completed:
                vec.close()
            else:
                for process in vec.processes:
                    if process.is_alive(): process.terminate()
                for process in vec.processes:
                    process.join(timeout=5)
                    if process.is_alive(): process.kill(); process.join(timeout=5)
        status['finished_utc'] = datetime.now(timezone.utc).isoformat()
        write_json(root/'status.json', status)
    print('Completed. Results:', root, flush=True)


if __name__ == '__main__':
    main()
