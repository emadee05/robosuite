# RoboSuite Lift: uniform and SGS sampling

Panda Lift simulation with a fixed pool of 32 cube placements, uniform PPO training,
and matched uniform/SGS continuations. SGS tracks the latest 100 episode outcomes
per configuration and favors intermediate success rates. Unobserved configurations
start at an assumed success rate of 0.5.

## Setup

From the robosuite repository root, use Python 3.11 (the version used for the local experiments):

```sh
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python -m pip install -r experiments/sampling/requirements.txt
cd experiments/sampling
```

MuJoCo runs headless with state observations; no camera or display is required.
The additional dependency versions match the original local environment. The simulator
comes from this checkout, installed in editable mode.

## Run checks

Sampler checks use small synthetic histories and only the Python standard library:

```sh
python -m sgs.check_sgs_sampler
python -m sgs.check_shared_sampler
```

Simulation and evaluation checks require the installed dependencies:

```sh
python -m uniform.check_fixed_pool
python -m uniform.check_uniform_env
python -m uniform.check_evaluation
```

These checks validate pool placements, Gymnasium compatibility, seeded sampling,
episode accounting, evaluation behavior, SGS probabilities, checkpoint history
filtering, and shared updates from four worker processes. They do not train a policy.

## Train and compare

Generate your own baseline (four workers are required for subsequent comparison):

```sh
python -m uniform.train_uniform --steps 1000000 --workers 4 --output-dir runs/baseline
python -m sgs.train_continuation --source-run runs/baseline --mode both --additional-steps 1000000
```

Training budgets round up to multiples of 6,144 transitions. Continuations load the
baseline checkpoint and completed training episodes up to that checkpoint, retaining
the optimizer and starting fresh episodes in both branches. Evaluation episodes do
not initialize SGS. Concurrent worker report order can vary with process scheduling.
Run these commands from `experiments/sampling`. Use `--help` on training and sampled-evaluation scripts for other options.

## Folders

- `uniform/`: fixed placement pool, robot environment, uniform PPO training,
  evaluation utilities, and simulation checks.
- `sgs/`: SGS sampler, shared worker coordination, matched uniform/SGS
  continuation training, sampler checks, and synthetic test history.

SGS training reuses the environment and evaluation utilities in `uniform/`.
Run scripts as modules from `experiments/sampling` using the commands above.
Source snapshots preserve this package structure.

Training outputs, checkpoints, local environments, and benchmark artifacts are excluded
from Git. Original saved experiments are not required for the checks. Continuation
training and sampled policy evaluation require a baseline generated as above.
