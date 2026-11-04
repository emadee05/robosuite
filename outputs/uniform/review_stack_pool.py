"""Review selected saved Stack pool IDs without regenerating states or training."""
import argparse
import hashlib
import json
from pathlib import Path

from uniform.stack_intermediate import StackPreview, make_env, metrics, np, restore, step_checked
from uniform.build_stack_pool import validate_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True, help='Path to saved pool.json')
    parser.add_argument('--ids', nargs='+', type=int, default=[0, 1, 2, 3])
    parser.add_argument('--seconds', type=float, default=5., help='Simulated hold seconds per sample (0.05–30)')
    parser.add_argument('--pause', action='store_true', help='Wait for Enter before holding each sample')
    parser.add_argument('--headless', action='store_true', help='Check selected samples without a viewer')
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or not .05 <= args.seconds <= 30:
        parser.error('--seconds must be between .05 and 30')
    pool_path = args.pool.resolve()
    folder = pool_path.parent
    pool = json.loads(pool_path.read_text())
    if pool.get('status') != 'validated' or pool.get('task') != 'Stack' or pool.get('configuration_count') != 32:
        parser.error('Expected a validated 32-state Stack pool')
    expected = (folder/'pool.sha256').read_text().strip()
    if hashlib.sha256(pool_path.read_bytes()).hexdigest() != expected:
        raise ValueError('Pool manifest checksum mismatch')
    entries = {c['id']: c for c in pool['configurations']}
    if any(i not in entries for i in args.ids):
        parser.error('IDs must exist in this pool (0–31)')
    preview = None if args.headless else StackPreview()
    env = None
    try:
        for config_id in args.ids:
            entry = entries[config_id]
            scene = pool['scenes'][entry['scene_id']]
            path = folder/entry['state_file']
            if hashlib.sha256(path.read_bytes()).hexdigest() != entry['state_sha256']:
                raise ValueError(f'ID {config_id}: state archive checksum mismatch')
            env = make_env(preview=preview, positions={'cubeA': scene['a'], 'cubeB': scene['b']})
            with np.load(path, allow_pickle=False) as archive:
                restore(env, archive, pool['model_xml_sha256'])
                np.testing.assert_allclose(env.sim.data.qpos, archive['qpos'], atol=1e-12, rtol=0)
                np.testing.assert_allclose(env.sim.data.qvel, archive['qvel'], atol=1e-12, rtol=0)
            print(f'Viewing ID {config_id:02d}: scene {entry["scene_id"]}, {entry["stage"]}', flush=True)
            if preview is not None:
                preview.update(env)
            if args.pause:
                input('State restored. Press Enter to run the holding check and continue... ')
            grip = -1. if entry['stage'] == 'tabletop' else 1.
            records = [step_checked(env, np.r_[np.zeros(6), grip])
                       for _ in range(max(1, round(args.seconds*20)))]
            summary = validate_records(env, records, entry['reference'], entry['stage'], scene)
            print(f'PASS ID {config_id:02d}: max cube drift {summary["max_cube_drift_m"]*1000:.2f}mm', flush=True)
            env.close()
            env = None
        print('Review complete. Saved pool unchanged; no training performed.', flush=True)
    finally:
        if env is not None:
            env.close()
        if preview is not None:
            preview.close()


if __name__ == '__main__':
    main()
