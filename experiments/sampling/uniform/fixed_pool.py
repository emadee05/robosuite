"""Repeatable Lift configuration pool, shared by future uniform/SGS samplers."""
import hashlib
import json
from pathlib import Path

import numpy as np
import robosuite as suite
from robosuite.utils.placement_samplers import ObjectPositionSampler

POOL_PATH = Path(__file__).with_name('train_pool.json')


def load_pool(path=POOL_PATH):
    data = json.loads(Path(path).read_text())
    configs = data['configurations']
    if data['schema_version'] != 1 or data['task'] != 'Lift' or data['robot'] != 'Panda':
        raise ValueError('Unsupported pool schema or environment')
    if len(configs) != 32 or [c['id'] for c in configs] != list(range(32)):
        raise ValueError('Expected 32 configurations with IDs 0 through 31')
    for c in configs:
        values = [c['cube_x_m'], c['cube_y_m'], c['cube_yaw_rad']]
        if not np.isfinite(values).all() or max(abs(v) for v in values[:2]) > 0.03:
            raise ValueError('Invalid placement or position outside the documented pool region')
    if len({(c['cube_x_m'], c['cube_y_m'], c['cube_yaw_rad']) for c in configs}) != 32:
        raise ValueError('Duplicate configurations')
    return data


class FixedPoolPlacement(ObjectPositionSampler):
    """Select an ID before reset; episode selection probabilities come later."""
    def __init__(self, pool):
        self.pool = pool
        self.configuration_id = 0
        super().__init__(name='FixedPoolPlacement', reference_pos=pool['table_offset_m'],
                         z_offset=pool['cube_bottom_clearance_m'])

    def select(self, configuration_id):
        if not isinstance(configuration_id, (int, np.integer)) or not 0 <= configuration_id < 32:
            raise ValueError('Configuration ID must be an integer from 0 through 31')
        self.configuration_id = int(configuration_id)

    def sample(self, fixtures=None, reference=None, on_top=True):
        if fixtures or reference is not None or not on_top:
            raise ValueError('This sampler only supports the standalone Lift table scene')
        if len(self.mujoco_objects) != 1:
            raise ValueError('Expected exactly one Lift cube')
        obj = self.mujoco_objects[0]
        config = self.pool['configurations'][self.configuration_id]
        pos = np.array(self.reference_pos, dtype=float)
        pos[:2] += [config['cube_x_m'], config['cube_y_m']]
        pos[2] += self.z_offset - obj.bottom_offset[2]
        yaw = config['cube_yaw_rad']
        quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])  # MuJoCo wxyz
        return {obj.name: (pos, quat, obj)}


def make_pool_environment(path=POOL_PATH):
    pool = load_pool(path)
    placement = FixedPoolPlacement(pool)
    env = suite.make('Lift', robots='Panda', seed=pool['model_seed'],
                     initialization_noise=None, hard_reset=False,
                     placement_initializer=placement, has_renderer=False,
                     has_offscreen_renderer=False, use_camera_obs=False,
                     use_object_obs=True, reward_shaping=True, control_freq=20,
                     horizon=1000, ignore_done=False)
    return env, placement


def pool_hash(path=POOL_PATH):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
