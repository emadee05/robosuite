"""Physically generate and restore-check 32 Stack starts: 8 scenes x 4 stages.

Run from hep/outputs: ../robosuite/.venv/bin/python -m uniform.build_stack_pool
No PPO/SGS training. Difficulty labels describe starting stages, not measured
learning difficulty. All starts are paused/settled, with fixed cube yaw and size.
"""
import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from uniform.stack_intermediate import (
    StackPreview, command, cube_positions, eef_pose, hold, make_env, metrics,
    move, np, restore, save_candidate, step_checked, require_stable, suite, mujoco,
)

STAGES = ('tabletop', 'lifted', 'transport', 'placement-approach')
# Small reproducible scene changes around the validated grasp scene. All eight
# scenes are retained or the build fails: no success-based selection/replacement.
SCENES = (
    dict(a=[-.020, -.065], b=[.065, .055], fraction=.35, offset=[0., 0.], gap=.050),
    dict(a=[-.010, -.055], b=[.080, .045], fraction=.40, offset=[.020, 0.], gap=.050),
    dict(a=[.000, -.050], b=[.080, .060], fraction=.45, offset=[-.020, 0.], gap=.050),
    dict(a=[.010, -.065], b=[.065, .075], fraction=.50, offset=[0., .020], gap=.050),
    dict(a=[.020, -.045], b=[.095, .050], fraction=.55, offset=[0., -.020], gap=.050),
    dict(a=[-.015, -.040], b=[.060, .065], fraction=.60, offset=[.014, .014], gap=.040),
    dict(a=[.005, -.075], b=[.090, .070], fraction=.65, offset=[-.014, .014], gap=.060),
    dict(a=[.015, -.055], b=[.070, .080], fraction=.50, offset=[0., 0.], gap=.065),
)


def write_json(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    temp.replace(path)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_records(env, records, reference, stage, scene):
    if not records or any(not m['finite'] or m['stack_success'] or m['forbidden_contacts']
                          or m['joint_limit_violations'] or m['min_contact_distance_m'] < -.005
                          for m in records):
        raise RuntimeError('Invalid or already-successful captured state')
    if stage == 'tabletop':
        # A supported tabletop start intentionally has no grasp and no lift.
        drifts = [max(np.linalg.norm(np.array(m[key])-reference[key]) for m in records)
                  for key in ('cube_a', 'cube_b')]
        if max(drifts) > .003 or any(m['grasp'] or abs(m['cube_bottom_clearance_m']) > .003
                                     for m in records):
            raise RuntimeError('Tabletop cubes are not stably supported')
        # Also check actual contact, rather than accepting a floating cube.
        if not all(env.check_contact(obj, 'table_collision') for obj in (env.cubeA, env.cubeB)):
            raise RuntimeError('Both tabletop cubes must contact the table')
        return dict(max_cube_drift_m=float(drifts[0]), max_target_drift_m=float(drifts[1]),
                    both_cubes_supported=True, stack_success_any_step=False)
    summary = require_stable(records, reference)
    if stage == 'placement-approach':
        errors = [float(np.linalg.norm(np.array(m['cube_a'])[:2]-np.array(m['cube_b'])[:2]
                                       - scene['offset'])) for m in records]
        gaps = [m['target_vertical_gap_m'] for m in records]
        if max(errors) > .012 or max(abs(g-scene['gap']) for g in gaps) > .015 or min(gaps) < .02:
            raise RuntimeError('Placement approach drifted outside its intended offset/gap')
        summary.update(max_intended_offset_error_m=max(errors),
                       minimum_target_gap_m=min(gaps), maximum_target_gap_m=max(gaps))
    return summary


def capture(env, folder, scene_id, stage, scene, log):
    # Paused checkpoints are intentional. These are not arbitrary moving frames.
    if stage != 'tabletop':
        pos, ori = eef_pose(env)
        records = hold(env, 40, log, label=f'{stage}-settle', target=pos, orientation=ori)
    else:
        records = [metrics(env)]
    reference = metrics(env)
    validate_records(env, records, reference, stage, scene)
    config_id = scene_id*4 + STAGES.index(stage)
    path = folder/'states'/f'candidate_{config_id:02d}.npz'
    save_candidate(env, path)
    print(f'Captured ID {config_id:02d}: scene {scene_id}, {stage}', flush=True)
    return dict(id=config_id, scene_id=scene_id, stage=stage,
                candidate_file=str(path.relative_to(folder)),
                generation_step=env.timestep, reference=reference)


def generate_scene(env, folder, scene_id, scene):
    env.reset()
    _, orientation = eef_pose(env)
    generated = []
    with (folder/'trajectories'/f'scene_{scene_id:02d}.jsonl').open('w') as log:
        for _ in range(30):
            m = step_checked(env, np.r_[np.zeros(6), -1.])
            log.write(json.dumps(dict(stage='tabletop-settle', step=env.timestep, **m))+'\n')
        generated.append(capture(env, folder, scene_id, 'tabletop', scene, log))
        cube, _ = cube_positions(env)
        move(env, cube+[0, 0, .12], orientation, -1., 'approach', log)
        move(env, cube.copy(), orientation, -1., 'lower', log)
        for _ in range(30):
            m = step_checked(env, command(env, cube, orientation, 1.))
            log.write(json.dumps(dict(stage='close', step=env.timestep, **m))+'\n')
        if not metrics(env)['grasp']:
            raise RuntimeError(f'Scene {scene_id}: failed to grasp cubeA')
        move(env, cube+[0, 0, .15], orientation, 1., 'lift', log, require_grasp=True)
        generated.append(capture(env, folder, scene_id, 'lifted', scene, log))
        a, b = cube_positions(env)
        start, _ = eef_pose(env)
        destination = start.copy()
        destination[:2] += b[:2]-a[:2]
        halfway = start + scene['fraction']*(destination-start)
        move(env, halfway, orientation, 1., 'partial transport', log, require_grasp=True)
        generated.append(capture(env, folder, scene_id, 'transport', scene, log))
        # Move horizontally at the safe lifted height before descending.
        a, b = cube_positions(env)
        pos, _ = eef_pose(env)
        above = pos.copy()
        above[:2] += b[:2]+np.array(scene['offset'])-a[:2]
        move(env, above, orientation, 1., 'above approach location', log, require_grasp=True)
        a, b = cube_positions(env)
        pos, _ = eef_pose(env)
        desired_cube = np.r_[b[:2]+scene['offset'],
                             b[2]+env.cubeB.top_offset[2]-env.cubeA.bottom_offset[2]+scene['gap']]
        move(env, desired_cube+(pos-a), orientation, 1., 'placement approach', log, require_grasp=True)
        generated.append(capture(env, folder, scene_id, 'placement-approach', scene, log))
    return generated


def validate_state(env, folder, entry, scene, xml_hash, repeats, steps):
    summaries, endings = [], []
    path = folder/entry['candidate_file']
    with np.load(path, allow_pickle=False) as archive:
        for repeat in range(repeats):
            restore(env, archive, xml_hash)
            np.testing.assert_allclose(env.sim.data.qpos, archive['qpos'], atol=1e-12, rtol=0)
            np.testing.assert_allclose(env.sim.data.qvel, archive['qvel'], atol=1e-12, rtol=0)
            if entry['stage'] == 'tabletop':
                records = [step_checked(env, np.r_[np.zeros(6), -1.]) for _ in range(steps)]
            else:
                records = hold(env, steps)
            summaries.append(dict(repeat=repeat+1,
                **validate_records(env, records, entry['reference'], entry['stage'], scene)))
            endings.append(env.sim.data.qpos.copy())
    for ending in endings[1:]:
        np.testing.assert_allclose(ending, endings[0], atol=1e-7, rtol=0)
    final = folder/'states'/f'state_{entry["id"]:02d}.npz'
    path.rename(final)
    entry.pop('candidate_file')
    entry.update(state_file=str(final.relative_to(folder)), state_sha256=digest(final),
                 validation=dict(status='passed', repeats=repeats, hold_steps=steps,
                                 repeatability_qpos_atol=1e-7, trials=summaries))
    print(f'PASS ID {entry["id"]:02d} {entry["stage"]}: {repeats} restores x {steps} steps', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--render', action='store_true', help='On macOS launch with mjpython; substantially slower')
    args = parser.parse_args()
    folder = args.output_dir or Path(__file__).parent.parent/'stack_pools'/datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    folder = folder.resolve()
    folder.mkdir(parents=True, exist_ok=False)
    for name in ('states', 'trajectories', 'sources'):
        (folder/name).mkdir()
    for name in ('build_stack_pool.py', 'stack_intermediate.py'):
        source = Path(__file__).with_name(name)
        (folder/'sources'/name).write_text(source.read_text())
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  expected_configurations=32, completed_configurations=0, no_training=True)
    manifest = dict(schema_version=1, task='Stack', robot='Panda', pool_id='stack_stages32_v1',
        model_seed=42, robosuite_version=suite.__version__, mujoco_version=mujoco.__version__,
        state_spec='mjSTATE_INTEGRATION', controller='OSC_POSE delta world; stock Panda gains',
        stages=list(STAGES), configurations_per_stage=8, scenes=list(SCENES),
        capture_protocol='Settled scripted prefixes; no release; fixed yaw, size, friction and robot initial pose',
        assistance='Privileged object positions used only by scripted state generator',
        difficulty='Stage labels are not measured difficulty; all candidates retained or build fails',
        restoration='Same model XML; reset counters and controller goals to achieved pose; restore gripper command',
        validation_settings=dict(repeats=3, hold_steps=100, max_penetration_m=.005,
            joint_limit_tolerance=.003, held_max_cube_drift_m=.015, max_target_drift_m=.005,
            held_min_table_clearance_m=.08, tabletop_max_drift_m=.003,
            placement_offset_tolerance_m=.012, placement_gap_tolerance_m=.015),
        evaluation='Training pool only; no held-out states generated', configurations=[])
    preview = StackPreview() if args.render else None
    env = None
    started = time.perf_counter()
    print('Building 32 states (8 scenes x 4 stages). No PPO training.', flush=True)
    print('Output:', folder, flush=True)
    write_json(folder/'build_report.json', report)
    try:
        for scene_id, scene in enumerate(SCENES):
            print(f'Scene {scene_id+1}/8', flush=True)
            positions = {'cubeA': scene['a'], 'cubeB': scene['b']}
            env = make_env(preview=preview, positions=positions)
            entries = generate_scene(env, folder, scene_id, scene)
            xml = env.sim.model.get_xml()
            xml_hash = hashlib.sha256(xml.encode()).hexdigest()
            if scene_id == 0:
                (folder/'model.xml').write_text(xml)
                manifest['model_xml_sha256'] = xml_hash
            elif xml_hash != manifest['model_xml_sha256']:
                raise RuntimeError('Scene changes unexpectedly modified model XML')
            env.close()
            env = make_env(preview=preview, positions=positions)
            for entry in entries:
                validate_state(env, folder, entry, scene, xml_hash, 3, 100)
                manifest['configurations'].append(entry)
                report['completed_configurations'] += 1
                write_json(folder/'build_report.json', report)
            env.close()
            env = None
        assert len(manifest['configurations']) == 32
        assert [c['id'] for c in manifest['configurations']] == list(range(32))
        for stage in STAGES:
            assert sum(c['stage']==stage for c in manifest['configurations']) == 8
        # Exact duplicate physics states would defeat the purpose of the pool.
        states = [np.load(folder/c['state_file'], allow_pickle=False) for c in manifest['configurations']]
        try:
            fingerprints = [hashlib.sha256(s['qpos'].tobytes()+s['qvel'].tobytes()).hexdigest() for s in states]
            assert len(set(fingerprints)) == 32, 'Duplicate physical states'
        finally:
            for state in states: state.close()
        manifest.update(status='validated', configuration_count=32)
        write_json(folder/'pool.json', manifest)
        (folder/'pool.sha256').write_text(digest(folder/'pool.json')+'\n')
        report.update(status='passed', pool_sha256=digest(folder/'pool.json'))
    except BaseException as error:
        report.update(status='failed', error=repr(error))
        raise
    finally:
        if env is not None: env.close()
        if preview is not None: preview.close()
        report['wall_seconds'] = time.perf_counter()-started
        write_json(folder/'build_report.json', report)
    print('PASS: 32 distinct states; 8 per stage; 96 restoration trials.', flush=True)
    print('Pool:', folder/'pool.json', flush=True)


if __name__ == '__main__':
    main()
