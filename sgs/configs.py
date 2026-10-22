"""The fixed set of task configurations that both samplers draw from.

In the SGS paper a task configuration is tau = (s0, g, e): initial state, goal,
environment configuration. robosuite's Lift has no goal variation (the goal is
always "cube 4cm above the table"), so here tau varies only the initial state:

    cube_x, cube_y     where the cube starts on the table (metres, table frame)
    cube_zrot          cube yaw (radians)
    arm_dq[0..6]       offset added to the Panda's nominal initial joint angles (radians)

That is 10 numbers per configuration.  The spread of cube_x / cube_y is what
creates a *difficulty gradient*: a cube 5cm in front of the gripper is easy, a
cube 40cm off to the side is near-impossible for an untrained policy.  Without
such a gradient there is no "frontier" for SGS to find and it cannot beat
uniform sampling -- see README.
"""

from dataclasses import dataclass, asdict

import numpy as np

CONFIG_DIM = 10


@dataclass
class ConfigSpace:
    """Ranges the fixed configuration set is drawn from (uniformly, once)."""

    # robosuite's own Lift randomisation is x,y in [-0.03, 0.03]: a 6cm x 6cm box.
    # That is far too narrow to produce a difficulty gradient, so we widen it.
    cube_x_range: tuple = (-0.20, 0.20)
    cube_y_range: tuple = (-0.20, 0.20)
    cube_zrot_range: tuple = (0.0, np.pi / 2)
    arm_dq_scale: float = 0.15  # per-joint uniform offset in [-scale, +scale] rad

    def to_dict(self):
        return asdict(self)


def sample_config_set(n: int, space: ConfigSpace, seed: int) -> np.ndarray:
    """Draw the fixed pool of n configurations. Called once, before training."""
    rng = np.random.default_rng(seed)
    cfgs = np.empty((n, CONFIG_DIM), dtype=np.float64)
    cfgs[:, 0] = rng.uniform(*space.cube_x_range, size=n)
    cfgs[:, 1] = rng.uniform(*space.cube_y_range, size=n)
    cfgs[:, 2] = rng.uniform(*space.cube_zrot_range, size=n)
    cfgs[:, 3:] = rng.uniform(-space.arm_dq_scale, space.arm_dq_scale, size=(n, 7))
    return cfgs


def difficulty_proxy(cfgs: np.ndarray) -> np.ndarray:
    """Cheap scalar for plotting only: distance of the cube from the table centre.

    Not used by any sampler -- it exists so you can ask "did SGS concentrate on
    the configurations I think are hard?" without hand-labelling them.
    """
    return np.linalg.norm(cfgs[:, :2], axis=1)
