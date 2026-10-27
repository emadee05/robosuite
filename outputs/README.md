# Lift simulation, uniform sampling, and SGS

The maintained local source is `hep/outputs/lift_uniform_baseline` alongside
this robosuite checkout. This folder publishes those source files unchanged.
Results, checkpoints, logs, caches, and run source snapshots are excluded.

## Setup and run

From the robosuite checkout root:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r outputs/requirements.txt
cd outputs/lift_uniform_baseline
python check_fixed_pool.py
python check_uniform_env.py
python train_uniform.py --steps 1000000 --workers 4
```

`sgs_sampler.py` and `shared_sampler.py` implement SGS and worker coordination.
`train_continuation.py` compares matched uniform/SGS continuations. Use
`--source-run /path/to/uniform_run` to select an existing baseline.
`evaluate_sampled.py --run-dir /path/to/uniform_run` evaluates a saved policy.

The original `check_sgs_sampler.py` and `check_shared_sampler.py` require the
saved `uniform_run_20261026_205152` beside the scripts; these are experiment
history checks. They intentionally retain the original behavior. Saved runs
are not included on GitHub. On the original machine, ignored directory shortcuts
provide access to those runs without copying results or checkpoints.

## Publish local edits

Continue editing `hep/outputs/lift_uniform_baseline`. From the robosuite root:

```sh
python tools/sync_sampling_outputs.py
git add outputs
git commit -m "Update sampling experiments"
git push
```

The sync copies only the top-level Python scripts and `train_pool.json`,
never run directories or training artifacts.
