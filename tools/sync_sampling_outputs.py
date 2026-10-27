"""Copy the maintained sibling outputs code into this fork; leave artifacts local."""
from pathlib import Path
import shutil


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo.parent / 'outputs' / 'lift_uniform_baseline'
    destination = repo / 'outputs' / 'lift_uniform_baseline'
    if not source.is_dir():
        raise SystemExit(f'Local source folder not found: {source}')
    paths = sorted(source.glob('*.py')) + [source / 'train_pool.json']
    if not paths[-1].is_file():
        raise SystemExit('Missing train_pool.json; nothing copied')
    destination.mkdir(parents=True, exist_ok=True)
    for path in paths:
        shutil.copy2(path, destination / path.name)
    print(f'Synced {len(paths)} source/config files. No results or checkpoints copied.')


if __name__ == '__main__':
    main()
