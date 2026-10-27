# Uniform and SGS experiments

Maintained code lives in `hep/outputs/uniform` and `hep/outputs/sgs`.
The fork publishes the same folders under `outputs/`.

- `uniform/`: Panda Lift environment, fixed configuration pool, uniform PPO training,
  evaluation utilities, and simulation checks.
- `sgs/`: SGS sampler, shared worker coordination, matched continuation training,
  and checks against the existing baseline history.
- `lift_uniform_baseline/`: existing local results, checkpoints and saved-run source
  snapshots. These remain in place and are excluded from Git.

## Setup

From the robosuite checkout root:

```sh
source .venv/bin/activate
# For a new environment, first run: python3.11 -m venv .venv
python -m pip install -e .
python -m pip install -r outputs/requirements.txt
```

## Work with your local code and existing results

From the robosuite checkout, switch to the maintained sibling outputs folder:

```sh
cd ../outputs
../robosuite/.venv/bin/python -m uniform.check_fixed_pool
../robosuite/.venv/bin/python -m uniform.check_uniform_env
../robosuite/.venv/bin/python -m sgs.check_sgs_sampler
../robosuite/.venv/bin/python -m sgs.check_shared_sampler
```

SGS checks read `lift_uniform_baseline/uniform_run_20261026_205152`.
`check_sgs_sampler` also writes its sampler snapshot to that existing run.
Training commands (start only when ready):

```sh
../robosuite/.venv/bin/python -m uniform.train_uniform --steps 1000000 --workers 4
../robosuite/.venv/bin/python -m sgs.train_continuation --source-run lift_uniform_baseline/uniform_run_20261026_205152 --mode both
```

Default new training outputs remain under `lift_uniform_baseline/`.
To evaluate an existing checkpoint without training:

```sh
../robosuite/.venv/bin/python -m uniform.evaluate_sampled --run-dir lift_uniform_baseline/uniform_run_20261026_205152
```

Run scripts as modules from the outputs folder so both packages can be imported.
For another machine, run from the fork's `outputs/` folder using your environment's
`python`, and provide your own saved-run path. Checkpoints and histories are not
uploaded. The original SGS history checks require the original baseline data.

## Publish edits

From `hep/robosuite`:

```sh
python tools/sync_sampling_outputs.py
git add outputs
git commit -m "Update uniform and SGS code"
git push
```

The sync includes source, the fixed pool, requirements and this README only.
Older benchmark scripts elsewhere in hep/outputs remain separate experiments.
