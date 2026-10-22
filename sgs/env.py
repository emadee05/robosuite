"""robosuite Panda Lift, headless, state observations, resettable to a chosen tau.

Two things differ from stock `suite.make("Lift")`:

1. `reset_to(config)` puts the cube and the arm exactly where a configuration
   says, instead of drawing from robosuite's own placement initializer.  That is
   what lets a sampler control the training distribution.
2. `hard_reset=False`, so a reset is ~1.5ms instead of ~250ms (the MuJoCo model
   is built once and only the state is rewritten).  The cost is that nothing
   which lives in the model XML -- cube size, mass, friction -- can vary across
   configurations.  If you want those axes, set hard_reset=True and pay for it.
"""

import os

# State observations only, so no GL context is ever needed.  This must be set
# before robosuite imports mujoco.  Note we *force* it rather than setdefault:
# importing mujoco rewrites MUJOCO_GL to whatever backend it picked ("egl"
# here), and spawned worker processes inherit that rewritten value and then
# die on import because this box has no EGL libraries.  Override with
# SGS_MUJOCO_GL=egl if you later install GL and want to render.
os.environ["MUJOCO_GL"] = os.environ.get("SGS_MUJOCO_GL", "disable")
os.environ["OMP_NUM_THREADS"] = os.environ.get("OMP_NUM_THREADS", "1")
os.environ["MKL_NUM_THREADS"] = os.environ.get("MKL_NUM_THREADS", "1")

import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config

OBS_KEYS = ("robot0_proprio-state", "object-state")


class LiftConfigEnv:
    """Single Lift environment addressed by task configuration."""

    def __init__(
        self,
        horizon: int = 200,
        control_freq: int = 20,
        reward_shaping: bool = True,
        terminate_on_success: bool = False,
        seed: int = 0,
    ):
        controller_config = load_composite_controller_config(
            controller="BASIC", robot="Panda"
        )
        self.env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=controller_config,
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=reward_shaping,
            control_freq=control_freq,
            horizon=horizon,
            hard_reset=False,
            ignore_done=False,
            # Deterministic given a configuration: the config supplies all the
            # arm variation we want, so robosuite should add none of its own.
            initialization_noise=None,
        )
        self.env.seed = seed
        self.horizon = horizon
        self.terminate_on_success = terminate_on_success
        self.robot = self.env.robots[0]
        self.nominal_qpos = np.array(self.robot.init_qpos, dtype=np.float64).copy()
        self.cube_joint = self.env.cube.joints[0]
        gid = self.env.sim.model.geom_name2id(self.env.cube.name + "_g0")
        self.cube_half_z = float(self.env.sim.model.geom_size[gid][2])
        self.table_z = float(self.env.table_offset[2])

        obs = self.env.reset()
        self.obs_dim = int(sum(obs[k].size for k in OBS_KEYS))
        self.act_dim = int(self.env.action_dim)
        self._ep_success = False
        self._ep_len = 0

    # ---------------------------------------------------------------- helpers
    def _flat_obs(self, obs) -> np.ndarray:
        return np.concatenate([np.asarray(obs[k]).ravel() for k in OBS_KEYS]).astype(
            np.float32
        )

    # ------------------------------------------------------------------- API
    def reset_to(self, config: np.ndarray) -> np.ndarray:
        """config = [cube_x, cube_y, cube_zrot, arm_dq(7)] (see sgs/configs.py)."""
        config = np.asarray(config, dtype=np.float64)
        # Set the arm BEFORE reset so robosuite's own reset path syncs the
        # controller to the new joint angles for us.
        self.robot.init_qpos = self.nominal_qpos + config[3:10]
        obs = self.env.reset()

        x, y, zrot = config[0], config[1], config[2]
        quat = np.array([np.cos(zrot / 2.0), 0.0, 0.0, np.sin(zrot / 2.0)])
        pose = np.concatenate([[x, y, self.table_z + self.cube_half_z], quat])
        self.env.sim.data.set_joint_qpos(self.cube_joint, pose)
        self.env.sim.data.set_joint_qvel(self.cube_joint, np.zeros(6))
        self.env.sim.forward()

        obs = self.env._get_observations(force_update=True)
        self._ep_success = False
        self._ep_len = 0
        return self._flat_obs(obs)

    def step(self, action: np.ndarray):
        """Returns (obs, reward, terminated, truncated, info).

        `info["is_success"]` is only meaningful on the final step of an episode:
        it is True if the Lift success condition held at any point during it.
        That boolean is the single signal SGS consumes.
        """
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        obs, reward, _, _ = self.env.step(action)
        self._ep_len += 1
        success_now = bool(self.env._check_success())
        self._ep_success |= success_now

        terminated = bool(self.terminate_on_success and success_now)
        truncated = (not terminated) and self._ep_len >= self.horizon
        info = {}
        if terminated or truncated:
            info = {
                "is_success": self._ep_success,
                "episode_length": self._ep_len,
            }
        return self._flat_obs(obs), float(reward), terminated, truncated, info

    def close(self):
        try:
            self.env.close()
        except Exception:
            pass
