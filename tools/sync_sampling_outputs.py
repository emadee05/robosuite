"""Publish maintained sibling uniform/SGS source without training artifacts."""
from pathlib import Path
import shutil


def main():
    repo = Path(__file__).resolve().parents[1]
    source = repo.parent / 'outputs'
    destination = repo / 'outputs'
    paths = [source/'README.md', source/'requirements.txt', source/'uniform/train_pool.json']
    for package in ('uniform', 'sgs'):
        paths.extend(sorted((source/package).glob('*.py')))
    if not all(path.is_file() for path in paths) or not (source/'sgs/__init__.py').is_file():
        raise SystemExit('Missing maintained source files; nothing copied')
    for path in paths:
        target = destination/path.relative_to(source)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    print(f'Synced {len(paths)} source/config/doc files. Results and checkpoints excluded.')


if __name__ == '__main__':
    main()
