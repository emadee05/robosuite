"""Bounded placement potential, with discounted potential-based shaping.

No reward for grasping or remaining lifted. Stack success supplies the task
reward. This heuristic uses cube-center distance and does not prove stability.
"""
import math
import numpy as np


def placement_geometry(env, distance_scale=.1):
    if not math.isfinite(distance_scale) or distance_scale <= 0:
        raise ValueError('distance_scale must be finite and positive')
    a = env.sim.data.body_xpos[env.cubeA_body_id]
    b = env.sim.data.body_xpos[env.cubeB_body_id]
    goal = b.copy()
    # The present pool uses upright, fixed-size cubes. Goal is the center of A
    # when its bottom rests on B's top, not the center of B itself.
    goal[2] += env.cubeB.top_offset[2] - env.cubeA.bottom_offset[2]
    distance = float(np.linalg.norm(a-goal))
    return distance, math.exp(-distance/distance_scale)


def progress_term(previous, current, gamma=.99, weight=1.):
    if not 0 < gamma < 1 or not math.isfinite(weight) or weight < 0:
        raise ValueError('Require 0 < gamma < 1 and finite nonnegative weight')
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (previous,current)):
        raise ValueError('Placement potential must be finite and within [0, 1]')
    return weight*(gamma*current-previous)
