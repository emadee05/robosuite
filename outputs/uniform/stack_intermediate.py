"""Create and validate Panda Stack lifted/above-target starts. No policy training.

The scripted policy uses privileged simulator positions and the initial wrist
orientation. It is deliberately limited to a fixed, easy scene. Generation does
not teleport objects after reset. Restores begin fresh episodes with achieved-pose OSC
controller goals; they are not an exact replay of historical controller goals.
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'

import argparse
import hashlib
import json
import time
from datetime import datetime
from pathlib import Path

import mujoco
import numpy as np
import robosuite as suite
from robosuite.controllers import load_composite_controller_config
from robosuite.utils.placement_samplers import ObjectPositionSampler
from robosuite.utils.control_utils import orientation_error


POSITIONS = {'cubeA': [0.0, -0.05], 'cubeB': [0.08, 0.06]}
STATE_SPEC = mujoco.mjtState.mjSTATE_INTEGRATION


class StackPreview:
    """One viewer with separate display data, unaffected by environment resets."""

    def __init__(self):
        self.handle = None
        self.model = self.data = None
        self.model_hash = None
        self.source_model = None

    def update(self, env):
        source = env.sim.model._model
        if source is not self.source_model:
            digest = hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()
            if self.model_hash is not None and digest != self.model_hash:
                raise ValueError('Preview cannot switch to a different Stack model')
            self.model_hash = digest
            self.source_model = source
        if self.handle is None:
            from mujoco import viewer
            # Keep the model alive after the generation environment is closed.
            # Display data is separate, so viewer interaction cannot alter physics.
            self.model = source
            self.data = mujoco.MjData(self.model)
            self.handle = viewer.launch_passive(self.model, self.data,
                                              show_left_ui=False, show_right_ui=False)
            self.handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.handle.cam.fixedcamid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, 'agentview')
            self.handle.opt.geomgroup[0] = 0
        if not self.handle.is_running():
            raise KeyboardInterrupt('Preview window closed')
        with self.handle.lock():
            mujoco.mj_setState(self.model, self.data, integration_state(env), STATE_SPEC)
            mujoco.mj_forward(self.model, self.data)
        self.handle.sync()

    def close(self):
        if self.handle is not None:
            self.handle.close()


class StackPlacement(ObjectPositionSampler):
    def __init__(self, positions=None):
        supplied = POSITIONS if positions is None else positions
        if set(supplied) != {'cubeA', 'cubeB'}:
            raise ValueError('Supply cubeA and cubeB XY positions')
        self.positions = {name: np.asarray(xy, dtype=float).copy() for name, xy in supplied.items()}
        if any(xy.shape != (2,) or not np.isfinite(xy).all() or np.max(np.abs(xy)) > .15
               for xy in self.positions.values()):
            raise ValueError('Cube positions must be finite XY pairs within +/-15cm')
        if np.linalg.norm(self.positions['cubeA']-self.positions['cubeB']) < .075:
            raise ValueError('Initial cubes need at least 7.5cm center separation')
        super().__init__(name='OneStackScene', reference_pos=[0, 0, .8], z_offset=.01)

    def sample(self, fixtures=None, reference=None, on_top=True):
        if fixtures or reference is not None or not on_top:
            raise ValueError('Only the fixed standalone table scene is supported')
        placements = {}
        for obj in self.mujoco_objects:
            xy = self.positions[obj.name]
            pos = np.array([xy[0], xy[1], .8 + .01 - obj.bottom_offset[2]])
            placements[obj.name] = (pos, np.array([1., 0, 0, 0]), obj)
        return placements


def make_env(preview=None, positions=None):
    config = load_composite_controller_config(robot='Panda')
    arm = config['body_parts']['right'] if 'right' in config['body_parts'] else config['body_parts']['arms']['right']
    # Explicit world-frame delta commands, with the stock +/-5cm and .5rad scales.
    arm.update(type='OSC_POSE', input_type='delta', input_ref_frame='world')
    env = suite.make('Stack', robots='Panda', controller_configs=config,
        initialization_noise=None, hard_reset=False, seed=42,
        placement_initializer=StackPlacement(positions), has_renderer=False,
        has_offscreen_renderer=False, use_camera_obs=False, use_object_obs=True,
        reward_shaping=True, control_freq=20, horizon=1000, ignore_done=False)
    env._script_preview = preview
    return env


def eef_pose(env):
    site = env.robots[0].eef_site_id['right']
    return (env.sim.data.site_xpos[site].copy(),
            env.sim.data.site_xmat[site].reshape(3, 3).copy())


def cube_positions(env):
    return (env.sim.data.body_xpos[env.cubeA_body_id].copy(),
            env.sim.data.body_xpos[env.cubeB_body_id].copy())


def metrics(env):
    a, b = cube_positions(env)
    eef, _ = eef_pose(env)
    robot = env.robots[0]
    raw_model, data = env.sim.model._model, env.sim.data._data
    bad_contacts = []
    depths = []
    for contact in data.contact[:data.ncon]:
        names = [mujoco.mj_id2name(raw_model, mujoco.mjtObj.mjOBJ_GEOM, int(i)) or ''
                 for i in (contact.geom1, contact.geom2)]
        depths.append(float(contact.dist))
        is_robot = any(n.startswith(robot.robot_model.naming_prefix) or
                       n.startswith(robot.gripper['right'].naming_prefix) for n in names)
        forbidden = any(n in env.cubeB.contact_geoms or n.startswith('table') for n in names)
        if is_robot and forbidden:
            bad_contacts.append(names)
    # Only scalar hinge/slide joints have a direct qpos-versus-range comparison.
    violations = []
    for joint in range(raw_model.njnt):
        if raw_model.jnt_limited[joint] and raw_model.jnt_type[joint] in (
                mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            q = data.qpos[raw_model.jnt_qposadr[joint]]
            low, high = raw_model.jnt_range[joint]
            if q < low-.003 or q > high+.003:
                violations.append(int(joint))
    return dict(cube_a=a.tolist(), cube_b=b.tolist(), eef=eef.tolist(),
        grasp=bool(env._check_grasp(robot.gripper, env.cubeA)),
        stack_success=bool(env._check_success()),
        cube_bottom_clearance_m=float(a[2] + env.cubeA.bottom_offset[2] - env.table_offset[2]),
        target_xy_error_m=float(np.linalg.norm(a[:2]-b[:2])),
        target_vertical_gap_m=float(a[2] + env.cubeA.bottom_offset[2]
                                    - b[2] - env.cubeB.top_offset[2]),
        min_contact_distance_m=min(depths, default=0.),
        forbidden_contacts=bad_contacts, joint_limit_violations=violations,
        finite=bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()
                    and np.isfinite(data.ctrl).all()))


def step_checked(env, action):
    started = time.perf_counter()
    obs, reward, done, _ = env.step(action)
    if done or not np.isfinite(reward) or not all(np.isfinite(v).all() for v in obs.values()):
        raise RuntimeError('Nonfinite output or unexpected episode end')
    m = metrics(env)
    if not m['finite'] or m['forbidden_contacts'] or m['joint_limit_violations']:
        raise RuntimeError(f'Invalid motion: {m}')
    if m['min_contact_distance_m'] < -.005:
        raise RuntimeError(f'Contact penetration exceeds 5mm: {m}')
    if env._script_preview is not None:
        env._script_preview.update(env)
        # Pace at the 20Hz control rate so the routine can be watched.
        time.sleep(max(0., env.control_timestep - (time.perf_counter()-started)))
    return m


def command(env, target, orientation, grip):
    pos, ori = eef_pose(env)
    # Limit each request to 1cm translation and .1rad rotation per control step.
    delta = np.clip(target-pos, -.01, .01) / .05
    rotation = np.clip(orientation_error(orientation, ori), -.1, .1) / .5
    return np.r_[delta, rotation, grip]


def move(env, target, orientation, grip, label, log, max_steps=120, require_grasp=False):
    stable = 0
    for index in range(max_steps):
        m = step_checked(env, command(env, target, orientation, grip))
        if require_grasp and not m['grasp']:
            raise RuntimeError(f'{label}: grasp lost during transport')
        pos, ori = eef_pose(env)
        error = float(np.linalg.norm(pos-target))
        angle_error = float(np.linalg.norm(orientation_error(orientation, ori)))
        log.write(json.dumps(dict(stage=label, step=env.timestep, **m))+'\n')
        stable = stable+1 if error < .004 and angle_error < .05 else 0
        if stable >= 8:
            print(f'PASS {label}: {index+1} steps; position error {error*1000:.1f}mm', flush=True)
            return
    raise RuntimeError(f'{label} timed out; position error={error:.4f}m, orientation error={angle_error:.3f}')


def hold(env, steps, log=None, label='hold', target=None, orientation=None):
    records = []
    for _ in range(steps):
        action = np.r_[np.zeros(6), 1.] if target is None else command(env, target, orientation, 1.)
        m = step_checked(env, action)
        if log:
            log.write(json.dumps(dict(stage=label, step=env.timestep, **m))+'\n')
        records.append(m)
    return records


def require_stable(records, reference):
    a0, b0 = np.array(reference['cube_a']), np.array(reference['cube_b'])
    max_drift = max(float(np.linalg.norm(np.array(m['cube_a'])-a0)) for m in records)
    target_drift = max(float(np.linalg.norm(np.array(m['cube_b'])-b0)) for m in records)
    if not all(m['grasp'] and not m['stack_success'] and m['cube_bottom_clearance_m'] > .08 for m in records):
        raise RuntimeError('Grasp/lift lost, or start was already successful')
    if max_drift > .015 or target_drift > .005:
        raise RuntimeError(f'Unstable hold: cube drift={max_drift:.4f}, target drift={target_drift:.4f}m')
    return dict(max_cube_drift_m=max_drift, max_target_drift_m=target_drift,
                minimum_bottom_clearance_m=min(m['cube_bottom_clearance_m'] for m in records),
                grasp_present_all_steps=True, stack_success_any_step=False)


def require_above_target(records):
    """Require proximity to the target throughout the restored hold, not only once."""
    max_xy = max(m['target_xy_error_m'] for m in records)
    gaps = [m['target_vertical_gap_m'] for m in records]
    if max_xy > .012 or min(gaps) < .02 or max(gaps) > .08:
        raise RuntimeError(f'Not stably above target: xy={max_xy:.4f}m; gap={min(gaps):.4f}–{max(gaps):.4f}m')
    return dict(max_target_xy_error_m=max_xy, minimum_target_gap_m=min(gaps),
                maximum_target_gap_m=max(gaps))


def save_candidate(env, path):
    np.savez_compressed(path, integration_state=integration_state(env),
        qpos=env.sim.data.qpos.copy(), qvel=env.sim.data.qvel.copy(),
        gripper_current_action=env.robots[0].gripper['right'].current_action.copy())


def validate_candidate(env, path, reference, xml_hash, repeats, hold_steps, label):
    trials, endings = [], []
    with np.load(path, allow_pickle=False) as archive:
        for repeat in range(repeats):
            restore(env, archive, xml_hash)
            np.testing.assert_allclose(env.sim.data.qpos, archive['qpos'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(env.sim.data.qvel, archive['qvel'], atol=1e-12, rtol=0)
            records = hold(env, hold_steps)
            summary = require_stable(records, reference)
            if label == 'above-target':
                summary.update(require_above_target(records))
            trials.append(dict(repeat=repeat+1, **summary))
            endings.append(env.sim.data.qpos.copy())
            print(f'PASS {label} restore {repeat+1}: {hold_steps} closed-gripper neutral steps; '
                  f'max cube drift {summary["max_cube_drift_m"]*1000:.2f}mm', flush=True)
    for ending in endings[1:]:
        np.testing.assert_allclose(ending, endings[0], atol=1e-7, rtol=0)
    return trials


def integration_state(env):
    state = np.empty(mujoco.mj_stateSize(env.sim.model._model, STATE_SPEC))
    mujoco.mj_getState(env.sim.model._model, env.sim.data._data, state, STATE_SPEC)
    return state


def restore(env, archive, expected_xml_hash):
    env.reset()
    actual = hashlib.sha256(env.sim.model.get_xml().encode()).hexdigest()
    if actual != expected_xml_hash:
        raise ValueError('Saved state requires the same compiled model XML')
    raw_model, data = env.sim.model._model, env.sim.data._data
    mujoco.mj_setState(raw_model, data, archive['integration_state'], STATE_SPEC)
    env.sim.forward()
    robot = env.robots[0]
    robot.gripper['right'].current_action = archive['gripper_current_action'].copy()
    # Refresh controller caches BEFORE resetting goals to the achieved robot pose.
    robot.composite_controller.update_state()
    for controller in robot.part_controllers.values():
        controller.update(force=True)
    robot.composite_controller.reset()
    env.timestep, env.cur_time, env.done = 0, 0., False
    env._obs_cache = {}
    env._reset_observables()
    env.update_state()
    obs = env._get_observations(force_update=True)
    if not all(np.isfinite(v).all() for v in obs.values()):
        raise RuntimeError('Nonfinite restored observation')
    if env.timestep != 0 or env.done:
        raise RuntimeError('Episode counters not reset')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--restore-repeats', type=int, default=3)
    parser.add_argument('--hold-steps', type=int, default=100)
    parser.add_argument('--stage', choices=['lifted', 'above-target'], default='lifted',
                        help='above-target additionally transports and lowers the held cube without release')
    parser.add_argument('--render', action='store_true',
                        help='Open the MuJoCo viewer; on macOS launch with mjpython')
    args = parser.parse_args()
    if args.restore_repeats < 2 or not 1 <= args.hold_steps <= 500:
        parser.error('At least two restores; hold steps must be between 1 and 500')
    folder = args.output_dir or Path(__file__).parent.parent/'stack_intermediate_results'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', task='Stack', robot='Panda', seed=42,
        positions_xy_m=POSITIONS, controller='OSC_POSE delta world; stock gains and scaling',
        robosuite_version=suite.__version__, mujoco_version=mujoco.__version__,
        state_spec='mjSTATE_INTEGRATION', hold_steps=args.hold_steps,
        thresholds=dict(minimum_cube_bottom_clearance_m=.08, max_cube_drift_m=.015,
                        max_target_drift_m=.005, max_contact_penetration_m=.005,
                        joint_limit_tolerance_m_or_rad=.003),
        render=args.render, stage=args.stage, no_training=True,
        purpose='Validate lifted/above-target resets, not stacking performance')
    (folder/'source.py').write_text(Path(__file__).read_text())
    env = None
    preview = StackPreview() if args.render else None
    started = time.perf_counter()
    print('Output:', folder, flush=True)
    try:
        env = make_env(preview=preview)
        env.reset()
        _, orientation = eef_pose(env)
        with (folder/'trajectory.jsonl').open('w') as log:
            # Let the cubes settle while opening the fingers.
            for _ in range(20):
                step_checked(env, np.r_[np.zeros(6), -1.])
            cube, _ = cube_positions(env)
            above = cube + [0, 0, .12]
            move(env, above, orientation, -1., 'approach', log)
            grasp_target = cube.copy()
            move(env, grasp_target, orientation, -1., 'lower', log)
            for _ in range(30):
                step_checked(env, command(env, grasp_target, orientation, 1.))
            if not metrics(env)['grasp']:
                raise RuntimeError('Both finger groups are not contacting cubeA after closing')
            print('PASS close: both finger groups contact cubeA', flush=True)
            lifted = grasp_target + [0, 0, .15]
            move(env, lifted, orientation, 1., 'lift', log)
            hold(env, 40, log, target=lifted, orientation=orientation)
            candidate = metrics(env)
            if not candidate['grasp'] or candidate['cube_bottom_clearance_m'] < .08 or candidate['stack_success']:
                raise RuntimeError(f'Invalid candidate: {candidate}')
            xml = env.sim.model.get_xml()
            (folder/'model.xml').write_text(xml)
            xml_hash = hashlib.sha256(xml.encode()).hexdigest()
            if args.stage == 'above-target':
                report['lifted_candidate'] = candidate
                save_candidate(env, folder/'lifted_candidate_state.npz')
                # Track the cube-to-gripper offset instead of assuming the cube
                # center coincides exactly with the end-effector site.
                cube_a, cube_b = cube_positions(env)
                eef, _ = eef_pose(env)
                offset = eef-cube_a
                transport_target = eef.copy()
                transport_target[:2] = cube_b[:2] + offset[:2]
                move(env, transport_target, orientation, 1., 'transport above green cube',
                     log, require_grasp=True)
                # Leave a 5cm gap between the red cube bottom and green cube top.
                cube_a, cube_b = cube_positions(env)
                eef, _ = eef_pose(env)
                offset = eef-cube_a
                desired_cube = cube_b.copy()
                desired_cube[2] += env.cubeB.top_offset[2] - env.cubeA.bottom_offset[2] + .05
                near_target = desired_cube + offset
                move(env, near_target, orientation, 1., 'lower to placement approach',
                     log, require_grasp=True)
                hold_records = hold(env, 40, log, label='above-target hold',
                                    target=near_target, orientation=orientation)
                require_stable(hold_records, hold_records[0])
                report['above_target_generation'] = require_above_target(hold_records)
                candidate = metrics(env)
                print(f'PASS above-target: xy error {candidate["target_xy_error_m"]*1000:.1f}mm; '
                      f'vertical gap {candidate["target_vertical_gap_m"]*1000:.1f}mm', flush=True)
            save_candidate(env, folder/'candidate_state.npz')
        report.update(candidate=candidate, model_xml_sha256=xml_hash,
                      generation_steps=env.timestep, controller_goals='reset to achieved pose on restore')
        # Fresh independent environment; same model and explicit controller configuration.
        env.close()
        env = make_env(preview=preview)
        if args.stage == 'above-target':
            report['lifted_restore_trials'] = validate_candidate(env,
                folder/'lifted_candidate_state.npz', report['lifted_candidate'], xml_hash,
                args.restore_repeats, args.hold_steps, 'lifted')
        trials = validate_candidate(env, folder/'candidate_state.npz', candidate, xml_hash,
                                    args.restore_repeats, args.hold_steps, args.stage)
        # Negative control: the same validator must reject a deliberately released grasp.
        print('Checking intentional release: the fingers will open to test rejection.', flush=True)
        with np.load(folder/'candidate_state.npz', allow_pickle=False) as archive:
            restore(env, archive, xml_hash)
            released = []
            for _ in range(30):
                released.append(step_checked(env, np.r_[np.zeros(6), -1.]))
                if not released[-1]['grasp']:
                    break
        try:
            require_stable(released, candidate)
        except RuntimeError:
            print('PASS negative control: opened grasp rejected', flush=True)
        else:
            raise RuntimeError('Validator incorrectly accepted a released grasp')
        (folder/'candidate_state.npz').rename(folder/'validated_state.npz')
        if args.stage == 'above-target':
            (folder/'lifted_candidate_state.npz').rename(folder/'validated_lifted_state.npz')
        report.update(status='passed', restore_trials=trials, repeatability_qpos_atol=1e-7,
                      negative_control_open_grasp_rejected=True)
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if env is not None:
            env.close()
        if preview is not None:
            preview.close()
        report['wall_seconds'] = time.perf_counter()-started
        (folder/'validation.json').write_text(json.dumps(report, indent=2)+'\n')
    print(f'PASS: {args.stage} intermediate state validated. No policy trained.', flush=True)
    print('Report:', folder/'validation.json', flush=True)


if __name__ == '__main__':
    main()
