"""Success floors on a pool's evaluation split: what no learning already achieves.

Policies: hold still + open, hold still + stay closed, and 3 untrained PPO inits
(deterministic). A trained run only counts as learning where it beats all of them.

    PYTHONPATH=. ../robosuite/.venv/bin/python -m sgs_plr.floor --pool stack_omnireset_pools/f1
"""
import argparse
import json
from multiprocessing import get_context
from pathlib import Path

from sgs_plr.common import RESULTS, Net, Still, evaluate, make_env

POLICIES = ['still_open', 'still_closed', 'untrained_s0', 'untrained_s1', 'untrained_s2']


def _run(args):
    pool, name, split = args
    env = make_env(pool)
    if name.startswith('still'):
        policy = Still(-1. if name == 'still_open' else 1.)
    else:
        policy = Net.untrained(env.observation_space.shape[0], 4, seed=int(name[-1]))
    r = evaluate(env, policy, split)
    env.close()
    return name, r


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--split', default='eval')
    a = p.parse_args()
    with get_context('spawn').Pool(len(POLICIES)) as pool:
        res = dict(pool.map(_run, [(a.pool, n, a.split) for n in POLICIES]))
    labels = list(next(iter(res.values()))['by_label'])
    print(f'{"policy":<14}{"all":>6}' + ''.join(f'{l:>11}' for l in labels))
    for n, r in res.items():
        print(f'{n:<14}{r["all"]:>6.3f}' + ''.join(f'{r["by_label"][l]:>11.3f}' for l in labels))
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f'floor_{a.pool.name}_{a.split}.json'
    out.write_text(json.dumps(dict(pool=str(a.pool), split=a.split, policies=res), indent=1) + '\n')
    print('wrote', out)


if __name__ == '__main__':
    main()
