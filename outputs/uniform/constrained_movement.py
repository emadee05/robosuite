"""Fresh placement PPO with smaller Cartesian action requests.

Run from outputs: ../robosuite/.venv/bin/python -m uniform.constrained_movement
--pool stack_pools/<pool>/pool.json. Defaults to 200,000 requested transitions.
Training, evaluation and playback must all use this action mapping.
"""
import os
for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
            'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import math
import numpy as np
from uniform.stack_pool_env import StackPoolEnv


TRANSLATION_LIMIT_M = .02
ROTATION_LIMIT_RAD = math.radians(10.)
ACTION_CONSTRAINTS = dict(
    type='cartesian_delta_scaling', translation_limit_m=TRANSLATION_LIMIT_M,
    rotation_limit_rad=ROTATION_LIMIT_RAD, gripper_unchanged=True,
    description='Per-axis controller requests, not guaranteed physical displacement')


def scale_action(action, translation_limit_m=TRANSLATION_LIMIT_M,
                 rotation_limit_rad=ROTATION_LIMIT_RAD):
    """Map normalized policy actions into the existing OSC controller input.

    The controller maps input 1 to 5 cm or .5 radians. PPO still samples and
    computes log probabilities in its original seven-dimensional action space.
    Gripper input keeps the full [-1, 1] range. No array is changed in place.
    """
    if not math.isfinite(translation_limit_m) or not 0 < translation_limit_m <= .05:
        raise ValueError('Translation limit must be in (0, .05] metres')
    if not math.isfinite(rotation_limit_rad) or not 0 < rotation_limit_rad <= .5:
        raise ValueError('Rotation limit must be in (0, .5] radians')
    requested = np.asarray(action, dtype=np.float64)
    if requested.shape != (7,) or not np.isfinite(requested).all():
        raise ValueError('Expected seven finite action values')
    applied = np.clip(requested, -1., 1.).copy()
    applied[:3] *= translation_limit_m / .05
    applied[3:6] *= rotation_limit_rad / .5
    return applied


class ConstrainedStackEnv(StackPoolEnv):
    """Same starts, observations and reward; only the action mapping changes."""

    def __init__(self, *args, translation_limit_m=TRANSLATION_LIMIT_M,
                 rotation_limit_rad=ROTATION_LIMIT_RAD, **kwargs):
        # Validate before constructing a simulator.
        scale_action(np.zeros(7), translation_limit_m, rotation_limit_rad)
        self.translation_limit_m = translation_limit_m
        self.rotation_limit_rad = rotation_limit_rad
        super().__init__(*args, **kwargs)
        if self.action_space.shape != (7,):
            self.close()
            raise ValueError('Constrained movement requires Panda OSC_POSE actions')

    def step(self, action):
        applied = scale_action(action, self.translation_limit_m, self.rotation_limit_rad)
        return super().step(applied)


def main():
    from uniform.sustained_placement_run import main as run_fresh
    print('Constrained controller requests: ±2 cm translation and ±10 degrees rotation '
          'per axis per action; full gripper control.', flush=True)
    run_fresh(environment_class=ConstrainedStackEnv,
              action_constraints=ACTION_CONSTRAINTS,
              output_family='constrained_movement_runs', default_steps=200000)


if __name__ == '__main__':
    main()
