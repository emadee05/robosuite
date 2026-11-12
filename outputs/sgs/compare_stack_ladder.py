"""Summarise uniform vs SGS Stack-ladder runs (reads evals.json and log.csv only).

    ../robosuite/.venv/bin/python -m sgs.compare_stack_ladder v1_uniform_s0 v1_sgs_s0 [--plot out.png]
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

RUNS = Path(__file__).resolve().parent.parent / 'stack_task_ladder_runs'


def load(name):
    run = RUNS / name
    evals = json.loads((run / 'evals.json').read_text())
    rows = list(csv.DictReader(open(run / 'log.csv')))
    args = json.loads((run / 'args.json').read_text())
    return args, evals, rows


def summary(name):
    args, evals, rows = load(name)
    steps = np.array([e['step'] for e in evals])
    allv = np.array([e['all'] for e in evals])
    # Area under the all-pool curve, normalised by the step range (mean success over training).
    auc = float(np.trapezoid(allv, steps) / (steps[-1] - steps[0])) if len(steps) > 1 else float(allv[0])
    return dict(name=name, sampler=args['sampler'], seed=args['seed'], labels=args['labels'],
                final=float(allv[-1]), last3=float(allv[-3:].mean()), auc=auc,
                by_label_last3={l: float(np.mean([e['by_label'][l] for e in evals[-3:]])) for l in evals[-1]['by_label']},
                first_50=next((int(s) for s, v in zip(steps, allv) if v >= .5), None),
                share_last={l: float(rows[-1][f'share_{l}']) if f'share_{l}' in rows[-1] else float('nan') for l in args['labels']},
                evals=evals)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('runs', nargs='+')
    p.add_argument('--plot', type=Path)
    a = p.parse_args()
    S = [summary(n) for n in a.runs]
    labels = [l for l in S[0]['labels'] if l in S[0]['by_label_last3']]
    print(f'{"run":<24}{"final":>7}{"last3":>7}{"AUC":>7}{"≥0.5 at":>9}')
    for s in S:
        print(f'{s["name"]:<24}{s["final"]:>7.3f}{s["last3"]:>7.3f}{s["auc"]:>7.3f}{str(s["first_50"]):>9}')
    print('\neval success by start label (mean of last 3 evals)')
    print(f'{"run":<24}' + ''.join(f'{l:>6}' for l in labels))
    for s in S:
        print(f'{s["name"]:<24}' + ''.join(f'{s["by_label_last3"][l]:>6.2f}' for l in labels))
    print('\nshare of last 200 training episodes by start label')
    for s in S:
        print(f'{s["name"]:<24}' + ''.join(f'{s["share_last"][l]:>6.2f}' for l in labels))
    print('\nper-variant means (run names grouped without their _s<seed> suffix)')
    groups = {}
    for s in S:
        groups.setdefault(s['name'].rsplit('_s', 1)[0], []).append(s)
    for variant, g in groups.items():
        if len(g) > 1:
            print(f'{variant:>18}: n={len(g)}  final {np.mean([s["final"] for s in g]):.3f}±{np.std([s["final"] for s in g]):.3f}'
                  f'  last3 {np.mean([s["last3"] for s in g]):.3f}  AUC {np.mean([s["auc"] for s in g]):.3f}')
    if a.plot:
        try:
            import matplotlib
        except ImportError:
            print('matplotlib is not installed; skipping --plot')
            return
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = {'uniform': '#7f7f7f', 'sgs': '#1f77b4', 'plr': '#d62728'}
        for s in S:
            ax.plot([e['step'] for e in s['evals']], [e['all'] for e in s['evals']],
                    color=colors[s['sampler']], alpha=.8, label=f'{s["sampler"]} seed {s["seed"]}')
        ax.set_xlabel('training transitions'); ax.set_ylabel('all-pool eval success')
        ax.set_ylim(0, 1); ax.grid(alpha=.3); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(a.plot, dpi=130)
        print('plot:', a.plot)


if __name__ == '__main__':
    main()
