"""Re-evaluate finished runs' final checkpoints on another pool's evaluation split.

Also reports each run's training success (last log row, mean over training
families) so the train-vs-fresh gap is visible.

    PYTHONPATH=. ../robosuite/.venv/bin/python -m sgs_plr.rescore --pool stack_omnireset_pools/f1 \
        o1_uniform_s0 o1_plr_s0 o1_sgscells_s0
"""
import argparse
import csv
import json
from multiprocessing import get_context
from pathlib import Path

import numpy as np

from sgs_plr.common import RESULTS, RUNS, Net, evaluate, make_env


def train_success(run):
    rows = list(csv.DictReader(open(RUNS / run / 'log.csv')))
    vals = [float(v) for k, v in rows[-1].items() if k.startswith('train_') and v not in ('', 'nan')]
    return float(np.mean(vals)) if vals else float('nan')


def _run(args):
    pool, run, split = args
    env = make_env(pool)
    r = evaluate(env, Net.from_run(run, env.observation_space.shape[0], 4), split)
    env.close()
    return run, dict(r, train_success=train_success(run))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--split', default='eval')
    p.add_argument('--procs', type=int, default=4)
    p.add_argument('--tag', default='')
    a = p.parse_args()
    with get_context('spawn').Pool(a.procs) as pool:
        res = dict(pool.map(_run, [(a.pool, r, a.split) for r in a.runs]))
    labels = list(next(iter(res.values()))['by_label'])
    print(f'{"run":<20}{"train":>7}{"fresh":>7}{"gap":>7}' + ''.join(f'{l:>11}' for l in labels))
    for n, r in res.items():
        print(f'{n:<20}{r["train_success"]:>7.3f}{r["all"]:>7.3f}{r["train_success"] - r["all"]:>7.3f}'
              + ''.join(f'{r["by_label"][l]:>11.3f}' for l in labels))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'rescore_{a.pool.name}_{a.split}{a.tag}.json'
    out.write_text(json.dumps(dict(pool=str(a.pool), split=a.split, runs=res), indent=1) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
