# Uniform and SGS experiments

Maintained code lives in `hep/outputs/uniform` and `hep/outputs/sgs`.
The fork publishes the same folders under `outputs/`.

- `uniform/`: Panda Lift environment, fixed configuration pool, uniform PPO training,
  evaluation utilities, and simulation checks.
- `sgs/`: SGS sampler, shared worker coordination, matched continuation training,
  and checks against the existing baseline history.
- `lift_uniform_baseline/`: existing local results, checkpoints and saved-run source
  snapshots. These remain in place and are excluded from Git.

## Recorded runs and results (2026-10-30)

This branch publishes Emily's maintained `uniform/` and `sgs/` experiment
packages under `outputs/`. Run commands below from that directory after
installing robosuite and `requirements.txt`. Local paths beginning `hep/` describe
the original workspace; on another machine use this repository's `outputs/`.
The destination repository's existing `sgs/`, `scripts/` and `runs/` are separate
experiments and are preserved.

Results below were read from saved JSON reports, without retraining.
[run_results_summary.json](run_results_summary.json) contains the extracted
evaluation counts and budgets, including short integration checks. Run IDs
identify local artifact directories; checkpoints, raw trajectories, machine logs,
and saved-run source snapshots are excluded. The Stack pool can be rebuilt with
`uniform.build_stack_pool`; the original validated pool files remain local.
All reported evaluations reuse training starts, not held-out configurations.

| Experiment / local run ID | Training transitions | Final evaluation result |
| --- | ---: | --- |
| Lift uniform `uniform_run_20261026_205152` | 1,001,472 | Deterministic stock success 1/32 (3.125%); peak recorded deterministic checkpoint 4/32 |
| Stack warmup `20261029_014819_838968/warmup` | 3,004,416 | Stock success: deterministic 0/32; sampled 22/160 (13.75%) |
| Stack uniform continuation, same run `/uniform` | 6,008,832 cumulative | Stock success: deterministic 0/32; sampled 0/160 |
| Stack SGS continuation, same run `/sgs` | 6,008,832 cumulative | Stock success: deterministic 0/32; sampled 1/160 (0.625%) |
| Placement stock pilot `20261029_073556_017334/stock` | 202,752 | Stock success: deterministic 0/8; sampled 2/40 (5%) |
| Placement progress pilot, same run `/progress` | 202,752 | Stock success: deterministic 0/8; sampled 19/40 (47.5%); strict physical checks were not recorded |
| Longer-placement check `20261029_094709_763037` | 208,896 cumulative (+6,144) | Stock success 2/8 deterministic, 4/8 sampled; stable-at-end 0/8 in both |
| Fresh sustained reward `20261029_103446_851055` | 1,001,472 | Physical stable-at-end: deterministic 0/8; sampled 0/40 |
| Constrained movement `20261029_111835_622460` | 61,440 | Physical stable-at-end: deterministic 2/8; sampled 4/16 |
| Constrained movement `20261029_112325_665874` | 202,752 | Physical stable-at-end: deterministic 0/8; sampled 2/40 |
| Constrained movement `20261029_114520_061880` | 202,752 | Physical stable-at-end: deterministic 0/8; sampled 0/40 |

The constrained runs are separate trials; their saved action limits must be used
for playback (earlier defaults were 5 mm / 3 degrees, later 2 cm / 10 degrees).
Short 6,144- and 12,288-transition runs are integration checks, not learning
comparisons. Despite the longer-placement script's example million-step budget,
the saved continuation above actually added only 6,144 steps.

The Stack comparison uses 32 states (eight scenes, four stages). Both continuation
branches start from identical warmup policy and optimizer states, with the same
continuation seed. Neither achieves tabletop evaluation success. One seed pair
and 0 versus 1 sampled success do not establish an SGS advantage. High dense
reward and a stock success flag do not establish a released, stable stack.
The later physical metric requires a released, supported, aligned, low-motion
stack sustained for one second and still valid at the end of a 1,000-step episode.
Placement-only results use eight assisted starts and do not measure full-task
stacking. Differences between reward variants are exploratory, not controlled
multi-seed evidence.

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

## First Stack intermediate state

From `hep/outputs`, run a short headless scripted grasp/lift/hold and reset check:

```sh
../robosuite/.venv/bin/python -m uniform.stack_intermediate
```

To watch in a real-time viewer on macOS, use MuJoCo's Python launcher:

```sh
../robosuite/.venv/bin/mjpython -m uniform.stack_intermediate --render
```

The preview uses one persistent `agentview` window with separate display data,
so environment resets do not launch additional viewers on macOS. Closing that
window interrupts the check. The final deliberate gripper opening is a negative
validation check, not an attempt to stack the cube.

To additionally transport the held red cube above the green cube and save a
second start, run:

```sh
../robosuite/.venv/bin/mjpython -m uniform.stack_intermediate --stage above-target --render
```

The script moves horizontally at the lifted height, then lowers to an intended
5cm gap above the target. Both lifted and above-target states receive three
independent five-second restore checks. The above-target check additionally
requires horizontal cube-center error <=12mm and a vertical gap between 2cm and
8cm throughout the hold. It saves `validated_lifted_state.npz` and
`validated_state.npz` (the above-target state) only after all checks pass.
The same command without `--render`, launched with `python`, runs headless.
The routine does not intentionally release onto the target or demonstrate
successful stacking; the final brief opening remains a grasp-rejection test.

This uses one fixed Panda Stack scene, world-frame delta OSC pose control, and
privileged cube positions. It physically grasps and lifts the red cube; it does
not attempt stacking or train a policy. Three restorations each run 100 neutral
closed-gripper control steps (5 simulated seconds). Checks require a retained
two-finger grasp, lifted cube, no initial/task success, no robot–table or
robot–target contacts, finite states, bounded penetration and drift, and
repeatable final joint positions. A deliberately opened grasp must be rejected.

New artifacts live under `stack_intermediate_results/<timestamp>/`: `validation.json`,
`trajectory.jsonl`, `model.xml`, a source snapshot, and (only after passing)
`validated_state.npz`. Existing Lift runs remain untouched. The archive stores
MuJoCo integration state, not just observations, plus the gripper command.
Restore calls require the same model XML; controller goals reset to the achieved
pose and episode counters reset to zero. This validates one local held start,
not stacking ability, a general scripted expert, or transfer to other placements.

## Build a 32-state Stack pool

From `hep/outputs`:

```sh
../robosuite/.venv/bin/python -m uniform.build_stack_pool
```

`uniform/build_stack_pool.py` defines eight deterministic scenes and captures
four paused checkpoints from each physically executed trajectory: tabletop,
lifted, partway through transport, and placement approach. Cube positions,
transport fractions (35–65%), placement offsets (up to 2cm), and target gaps
(4–6.5cm) vary across scenes. Cube yaw, geometry, friction, and the robot's
initial pose stay fixed. These labels are candidate stages, not measured
difficulty or evidence of an SGS advantage.

Each of the 32 states must pass three independent 100-control-step restores
(5 simulated seconds each) in a fresh environment. Tabletop checks require
supported, stationary cubes and no grasp; held checkpoints require a retained
grasp and clearance. Placement checks use the intended offset rather than
requiring perfect alignment. Every stage also checks finite physics, joint
limits, unintended contacts, penetration, no pre-existing success, and repeatable
final joint positions. Failed scenes are not silently replaced with easier ones.

The builder writes a new `stack_pools/<timestamp>/` directory containing:

- `pool.json`: published only after all 32 states pass; IDs, stages, scene
  specifications, per-state hashes, and detailed validation trials.
- `states/state_00.npz` through `state_31.npz`: MuJoCo integration states and
  gripper commands, compatible with `stack_intermediate.restore()`.
- `model.xml`, `pool.sha256`, `build_report.json`, trajectory logs, and source
  snapshots. A failed build retains diagnostics but publishes no complete pool.

For a slower real-time preview, use:

```sh
../robosuite/.venv/bin/mjpython -m uniform.build_stack_pool --render
```

This may take about ten simulated minutes to watch because it includes all
96 restoration trials. It runs no PPO/SGS training, does not create held-out
evaluation states, and does not connect the pool to the existing Lift wrappers.
Do not change the code directory while a build is running.

To inspect four existing starts without rebuilding the pool, substitute your
saved `pool.json` path below:

```sh
../robosuite/.venv/bin/mjpython -m uniform.review_stack_pool \
  --pool stack_pools/<timestamp>/pool.json --ids 0 1 2 3 --pause
```

IDs 0–3 show all four stages of scene 0. The viewer restores each state directly,
waits for Enter in the terminal, then performs a five-second hold. Change `--ids`
to inspect other starts (for example, `0 9 18 31` spans four scenes and stages).
This verifies manifest/archive hashes and model compatibility and leaves the
saved pool untouched. Use normal `python` and `--headless` for a windowless check.

## Overnight Stack PPO comparison

The Stack pipeline uses the validated 32-state pool, not a Lift checkpoint or the
proposed larger table. Run from `hep/outputs` (replace the pool path if needed):

```sh
caffeinate -i ../robosuite/.venv/bin/python -m sgs.train_stack_comparison \
  --pool stack_pools/20261029_012238_498673/pool.json \
  --warmup-steps 3000000 \
  --additional-steps 3000000 \
  --sgs-kappa 5 --sgs-temperature 2 \
  --eval-every-cycles 100 --eval-repeats 2 --final-eval-repeats 5
```

Phases run sequentially without more input:

1. `warmup`: fresh CPU PPO, uniform episode resets over the 32 IDs.
2. `uniform`: reload the warmup final policy AND optimizer; continue uniformly.
3. `sgs`: independently reload that same warmup checkpoint; use a shared SGS
   sampler initialized only from warmup completed training episodes.

Budgets round up to complete 6,144-transition rollouts. The example collects
3,004,416 warmup transitions plus 3,004,416 additional transitions in EACH branch.
Both branches use continuation seed 43, new episodes, four processes, small
64x64 actor/critic networks, batch 256, and ten PPO epochs. Warmup seed is 42.
Stack observations have 73 components; the action space has 7 components.

Evaluation occurs initially, every 100 cycles, and at the phase end: one
mean-action episode per ID plus two sampled-action episodes per ID (five at the
final checkpoint). All episodes last 1,000 transitions, even after success.
Success means the stock Stack success condition occurred at least once.
Evaluation restores each ID equally, reports each starting stage separately,
preserves training RNG and policy weights, and never changes sampler history.
Tabletop results assess whole-task behavior on the eight known tabletop starts;
these are not held-out placements. This is one seed pair, not a multi-seed claim.

Each new `stack_comparisons/<timestamp>/` contains a frozen pool and source
snapshots, `settings.json`, root/phase `status.json`, and:

- `warmup/`, `uniform/`, `sgs/`: full PPO checkpoints, `final_model.zip`, evaluation
  JSON, `results.json`, and `training_episodes.jsonl` with ID/stage/outcome.
- `progress.jsonl` in each phase: one row after each complete PPO cycle, including
  throughput, stage counts/success, sampling probabilities, and success estimates.
- Sampler snapshots at checkpoints: initialization and recent per-ID histories.
- `initial_checkpoint.json` and `final_checkpoint.json`: policy/optimizer hashes
  verify both continuations begin with the warmup weights and optimizer state.
- `comparison.json`: final deterministic/sampled evaluation by stage for all phases.

Checkpoints/results are written before evaluation. A failure writes `error.txt`
and status; previously completed checkpoints remain available. Automatic
mid-phase resumption is not implemented. Existing Lift runs are untouched.
`caffeinate` prevents idle sleep while the process runs; keep the laptop plugged
in with its lid open. No sudo, GUI, or hardware-monitor terminal is required.
Several hours are expected, not a fixed wall-clock deadline.

Checks without an overnight training run:

```sh
../robosuite/.venv/bin/python -m uniform.check_stack_ppo \
  --pool stack_pools/20261029_012238_498673/pool.json

# Short actual PPO integration test: 12,288 transitions per phase, no evaluation.
../robosuite/.venv/bin/python -m sgs.train_stack_comparison \
  --pool stack_pools/20261029_012238_498673/pool.json \
  --warmup-steps 12288 --additional-steps 12288 --skip-evaluation
```

The short test is not a learning comparison. Do not move/edit the source folder
while a run is active, because spawned workers need to import the modules.

## Placement progress reward pilot

`StackPoolEnv` preserves `reward_mode='stock'` as its default. The explicit
`progress` option replaces the stock dense reward with:

```text
reward = float(stock_stack_success) + weight * (gamma * Phi(next) - Phi(current))
Phi = exp(-distance_between_cube_A_center_and_stacked_goal_center / distance_scale)
```

Defaults are weight 1, gamma .99, and distance scale .1m. The stacked goal center
accounts for the heights of both cubes. There is no per-step grasp/lift bonus.
Stock reward, task reward, shaping contribution, potential and placement distance
are recorded separately; the returned reward depends on the selected mode.
Discounted return, not raw sums alone, determines the incentive check. A
nonnegative potential gives a small negative term while stationary. Artificial
1000-step truncations retain their final potential and SB3 bootstraps the final
value; a true task termination would need explicit terminal-potential handling.

First check the incentives using physically simulated trajectories, with no PPO:

```sh
../robosuite/.venv/bin/python -m uniform.check_progress_reward \
  --pool stack_pools/20261029_012238_498673/pool.json
```

It compares holding, rising farther, releasing over the target, and releasing
away from the target under both reward modes. It checks identical physics across
modes, the discounted shaping identity including the time limit, and that
move-and-return cycles cannot harvest shaping reward. Artifacts go to a new
`stack_reward_checks/<timestamp>/`. Passing these checks does not prove learning.

Then run a small matched learning pilot (four CPU workers, uniform sampling):

```sh
caffeinate -i ../robosuite/.venv/bin/python -m uniform.train_progress_pilot \
  --pool stack_pools/20261029_012238_498673/pool.json \
  --mode both --steps 200000 --eval-every-cycles 10 \
  --eval-repeats 2 --final-eval-repeats 5
```

This trains two fresh small PPO policies, with matching initial weights,
optimizer state, and seeds: `stock` then `progress`. Each gets 202,752 training
transitions from only the eight placement-approach IDs (3,7,...,31). PPO gamma
matches the shaping gamma. Initial, periodic, and final deterministic/sampled
evaluations visit those eight IDs equally, without updating weights. Evaluate
success rates rather than comparing reward magnitudes across modes. Both
policies start from scratch; no overnight checkpoint is continued.

Results are in `stack_reward_pilots/<timestamp>/`, with separate stock/progress
checkpoints, evaluations, cycle logs, and episode outcomes including reward
components and initial/final placement distances. These are known assisted
placement starts, not a full-task or generalization test; no SGS is used.
The old overnight runner continues to use stock reward until explicitly revised.

For a short training integration check only, use `--steps 12288 --skip-evaluation`.

Watch a completed reward-pilot policy without recording video:

```sh
../robosuite/.venv/bin/mjpython -m uniform.watch_stack_policy \
  --run stack_reward_pilots/<timestamp> \
  --mode progress --checkpoints 122880 202752 --ids 3 7 --pause
```

This displays the same ID at each checkpoint back-to-back with deterministic
actions, one 1000-step (50 simulated second) episode per checkpoint/ID. Omit
`--ids` to display all eight configurations (16 episodes for two checkpoints).
Press Enter in the terminal to start each episode. `--sampled --repeats 5`
uses the evaluation seed and action randomness instead. `--speed 2` requests
double-speed playback; without a viewer, use normal Python and `--headless`.
Closing the window or Ctrl-C stops viewing. It opens one persistent window,
writes no video/results, and does not change models or the pool. The terminal
reports stock success flag changes, longest stock-flag streak, final placement
distance, and the independent physical checks described below.

## Physical stacking check

`uniform/check.py` provides two evaluation-only APIs:

```python
from uniform.check import check_stack, StackStabilityTracker

frame = check_stack(env)  # accepts StackPoolEnv or the raw robosuite Stack env
print(frame['is_stacked'], frame['failed_conditions'])

tracker = StackStabilityTracker(required_seconds=1.0, control_freq=20)
tracker.reset()  # at every episode reset
# After EVERY environment control step:
report = tracker.update(env)
# At episode end, distinguish temporary success from retained completion:
print(report['stable_ever'], report['stable_now'], report['longest_stable_seconds'])
```

The frame check requires red-on-green contact with an upward-facing support
normal, <=1cm horizontal error, <=4mm bottom-to-top surface gap, cube faces
within 10 degrees of level, and no robot/gripper contact with either cube.
Green must contact and rest on the table, both cube footprints must be within
the table boundary, and red must not contact the table directly. Both cubes
must have linear speed <=2cm/s and angular speed <=.25rad/s, with relative speed
<=2cm/s. Positions, velocities and orientations must be finite; cube-contact
penetration must be <=5mm. Orientations are included when calculating corners
and surface heights. `StackThresholds` exposes these conservative thresholds;
this is an explicit diagnostic definition, not proof of all possible stable
stacks or a replacement for visual review.

The tracker requires 20 consecutive passing control observations at 20Hz. A
failing frame clears `stable_now`; `stable_ever` remains latched. Repeated checks
at the same step do not increase duration, skipped observations break the streak,
and the tracker must be reset for each episode. It samples control frames,
not every underlying physics substep. The viewer uses this tracker and reports
its metrics separately from stock success; training rewards, SGS outcomes and
the existing evaluation JSON success definition remain unchanged.

Validate the checker against a physically executed release and deliberate
negative fixtures (no training):

```sh
../robosuite/.venv/bin/python -m uniform.check \
  --pool stack_pools/20261029_012238_498673/pool.json
```

It accepts a settled released stack, rejects a held cube, side contact,
unsupported/off-table cubes, excessive translation/rotation and tilt, and checks
streak bookkeeping. Perturbed negative fixtures are tests only and never enter
the training pool. Reports go to `stack_check_results/<timestamp>/validation.json`.

## Bounded longer-placement continuation

`uniform/test_longer_placement.py` continues a saved **progress** reward pilot
checkpoint on the same eight placement starts. It tests additional training
without changing the reward or adding SGS. Physical checks are evaluation-only;
the progress reward still uses stock success as its task-reward component.

```sh
caffeinate -i ../robosuite/.venv/bin/python -m uniform.test_longer_placement \
  --source-run stack_reward_pilots/20261029_073556_017334 \
  --checkpoint-steps 202752 \
  --additional-steps 1000000 \
  --workers 4 \
  --eval-every-cycles 20 --eval-repeats 2 --final-eval-repeats 5
```

The example reloads policy AND optimizer from 202,752 transitions, starts fresh
worker episodes, and adds 1,001,472 transitions (163 full rollouts), ending at
1,204,224 cumulative training transitions. Actor/critic/loss hyperparameters and
progress-potential parameters are retained. `--workers 2` is available for lower
resource use: n_steps becomes 3,072 per worker so rollout size remains 6,144.
Worker changes also change trajectory segmentation; use four to match the pilot.

There are 10 evaluation checkpoints: before additional training, every 122,880
added transitions, and at the final budget. Each checkpoint evaluates all eight
IDs deterministically once and with two sampled episodes per ID (five at the
final checkpoint). Each report includes:

- Original stock success reached once (kept separately).
- Physical stack sustained for one continuous second.
- Physical stack still sustained at the end of the full 1,000-step episode.
- Longest physical-stack streak, final conditions and failed-frame counts per
  condition, broken down by configuration/stage. There is no early success reset.

`evaluate_stack(..., physical_checks=True)` enables the stricter reports; the
existing default remains unchanged. Evaluation preserves RNG and policy weights.
The geometric/velocity thresholds from `uniform/check.py` are recorded in output.

New outputs live in `longer_placement_runs/<timestamp>/`: frozen pool, source
checkpoint and source copies, initial weight/optimizer fingerprints, per-cycle
logs, training episodes, evaluations, results/status, periodic checkpoints, and
`final_model.zip`. Checkpoints/results are saved before evaluation. Provisional
`best_deterministic.zip` and `best_sampled.zip` rank checkpoints by stable-at-end
rate, then sustained-ever rate; inspect counts in `results.json`. A best file
with zero successes is not evidence of a solved task. Repeated checkpoint
selection is optimistic and does not replace an independent final evaluation.
There is no automatic mid-run crash resume. Original pilot results are untouched.

To start instead from the earlier saved policy, pass `--checkpoint-steps 122880`.
For a short integration check only, use
`--additional-steps 6144 --workers 2 --skip-evaluation`.

## Fresh physical/sustained-placement learning

`uniform/sustained_placement_run.py` constructs a new PPO actor, critic AND
optimizer. It does not load any saved policy. It freezes the same 32-state pool
and uniformly samples only the eight placement-approach starts.

```sh
caffeinate -i ../robosuite/.venv/bin/python -m uniform.sustained_placement_run \
  --pool stack_pools/20261029_012238_498673/pool.json \
  --steps 1000000 --workers 4 \
  --eval-every-cycles 20 --eval-repeats 2 --final-eval-repeats 5
```

The new explicit environment reward mode is `sustained`. It returns:

```text
float(check_stack(current_state)['is_stacked'])
    + weight * (gamma * Phi(next) - Phi(previous))
```

The task reward checks released, supported, aligned, level and low-motion
stacking using `uniform/check.py`; it is NOT robosuite's original success flag.
A physically correct frame earns 1, and maintaining it earns repeated reward.
The reward does not wait on an unobserved history counter. Episode success
requires the valid stack to have persisted for one second AND still be valid at
step 1,000. Stock success-ever, physical sustained-ever, and stable-at-end are
logged separately. Original `stock`/`progress` reward modes retain their earlier
behavior. Observations and the progress potential are unchanged.

The default million-step budget rounds to 1,001,472 transitions (163 cycles).
Four workers use 1,536 transitions each per rollout; `--workers 2` uses 3,072 each
and keeps rollout size 6,144. Networks are 64x64 actor/critic on CPU, batch 256,
ten epochs, gamma .99. The initial checkpoint explicitly records zero training
steps, no loaded checkpoint, and an empty optimizer state. This is not the
`test_longer_placement` continuation experiment.

Evaluation runs before training, every 122,880 transitions, and at the end.
All eight IDs get one deterministic episode and two sampled episodes (five at
final). All episodes run 1,000 steps; evaluation does not update weights, and
both reward and evaluation use matching physical thresholds and duration.
The new success count means stable-at-end. The original flag is separately
reported as `stock_successes` in evaluation JSON. Do not compare the new primary
success counts directly to earlier stock-flag counts without labeling them.

Outputs: `sustained_placement_runs/<timestamp>/`, with frozen pool/source,
settings/status and root results. `sustained/` contains checkpoints (including
initial zero-step), `final_model.zip`, provisional `best_deterministic.zip` and
`best_sampled.zip`, per-cycle progress, training episodes and evaluations. Best
models rank stable-at-end rate first, sustained-ever second. Zero-score best
files do not indicate success, and best-checkpoint selection is not independent
validation. This is one seed on known assisted starts; no whole-task,
generalization or SGS claim is established. No automatic crash resume.

Checks before full training:

```sh
../robosuite/.venv/bin/python -m uniform.check_sustained_reward \
  --pool stack_pools/20261029_012238_498673/pool.json

# One training cycle with full short evaluations, fresh optimizer:
../robosuite/.venv/bin/python -m uniform.sustained_placement_run \
  --pool stack_pools/20261029_012238_498673/pool.json \
  --steps 6144 --workers 2 --eval-repeats 1 --final-eval-repeats 1
```

Reward checks test a held cube, a physically released stable stack, and a
successful placement deliberately disrupted before episode end. They verify
that physics is unchanged from old progress mode, no physical reward is given
for holding, and a later collapse fails episode success despite earlier success.
These checks perform no PPO training and write new `sustained_reward_checks/`.

To watch new checkpoints without recording, use the existing viewer with the
new mode and checkpoint steps actually saved in your run:

```sh
../robosuite/.venv/bin/mjpython -m uniform.watch_stack_policy \
  --run sustained_placement_runs/<timestamp> --mode sustained \
  --checkpoints 122880 1001472 --ids 3 7 --pause
```

## Constrained movement placement pilot

`uniform/constrained_movement.py` maps normalized Panda OSC actions to requests
of at most 2 cm per translation axis and 10 degrees per rotation axis. These are
controller target increments, not guaranteed measured displacements; diagonal
commands can exceed 2 cm in total length. Gripper control retains its full range.
The same mapping is used by workers, evaluation, and saved-policy playback.
PPO retains the original seven-dimensional action space and log probabilities.
Earlier runs used 5 mm and 3 degrees. Playback reads each run's saved limits,
so these new defaults do not change how earlier checkpoints are displayed.

This is a fresh actor, critic and optimizer, uniformly sampling the same eight
placement-approach starts. It reuses the sustained physical-frame reward and
strict stability evaluations. It does not continue the previous policy or use
SGS. Smaller exploratory motions may help precision, but can slow transport.

From `hep/outputs`, run the bounded pilot:

```sh
caffeinate -i ../robosuite/.venv/bin/python -m uniform.constrained_movement \
  --pool stack_pools/20261029_012238_498673/pool.json \
  --steps 200000 --workers 4 --eval-every-cycles 10 \
  --eval-repeats 2 --final-eval-repeats 5
```

The rollout rounds this budget to 202,752 training transitions. Evaluations
occur at zero, every 61,440 training transitions, and the final checkpoint.
Each evaluates all eight starts with mean actions and sampled actions; final
sampled evaluation uses five attempts per start. Evaluation steps are additional
to the training budget. Logs, snapshots, checkpoints and evaluations go into
`constrained_movement_runs/<timestamp>/`, with the same `sustained/` layout as
the unconstrained experiment. Saved settings record the action constraints.

```sh
../robosuite/.venv/bin/mjpython -m uniform.watch_stack_policy \
  --run constrained_movement_runs/<timestamp> --mode sustained \
  --checkpoints 202752 --ids 3 7 --pause
```

## Placement feasibility and published-baseline audit (2026-10-29)

Before another PPO run, use `uniform/check_scripted_placement.py` to complete
alignment, lowering, opening and withdrawal from the eight frozen placement
starts. It also checks two controls: hold forever and open immediately. These
are oracle scripted controllers, not learned policies. No training or simulator
state editing occurs after reset. The saved run supplies the exact action limits
and reward parameters; this does not alter its results or checkpoints.

```sh
# Headless audit: 24 episodes of 1,000 transitions, no PPO training.
../robosuite/.venv/bin/python -m uniform.check_scripted_placement \
  --run constrained_movement_runs/20261029_114520_061880

# Watch one complete scripted placement, no video recording.
../robosuite/.venv/bin/mjpython -m uniform.check_scripted_placement \
  --run constrained_movement_runs/20261029_114520_061880 \
  --ids 3 --behaviors scripted --render
```

New outputs go to `scripted_placement_checks/<timestamp>/`: `steps.jsonl` logs
actions, stage, distance, physical conditions, task reward and shaping reward;
`report.json` records end stability, longest stable duration, failed-condition
counts, stage timeouts, total and gamma-discounted returns. A completed audit
does not itself mean every episode passed; read each episode's `success` and
`scripted_stage_timeouts`. Rendered full episodes last 50 seconds each.

Published references checked:

| Reference | What it establishes | Difference from our pilot |
| --- | --- | --- |
| [robosuite benchmark](https://robosuite.ai/docs/algorithms/benchmarking.html) | Official v1.0 experiments use SAC with physical-state observations and compare controllers | Does not validate SB3 PPO on robosuite 1.5.2 or our reward |
| [ManiSkill PPO tested examples](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/examples.sh) | State PickCube: 10M steps; StackCube: 25M; 1,024 environments and 8 update epochs | Different simulator and start distribution; much larger data budget |
| [ManiSkill official PPO benchmark](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/baselines.sh) | StackCube uses 50M steps, 4,096 environments and three training seeds | GPU/CUDA implementation, not a Mac benchmark |
| [ManiSkill StackCube task](https://github.com/haosulab/ManiSkill/blob/main/mani_skill/envs/tasks/tabletop/stack_cube.py) | Staged dense reward includes approach, grasp/placement, release and static-object feedback | Our physical task reward is zero until all physical checks pass, plus a discounted distance potential; 1,000-step episodes versus its 50-step horizon |

The [reference PPO implementation](https://github.com/haosulab/ManiSkill/blob/main/examples/baselines/ppo/ppo.py)
uses three 256-unit hidden layers in each network, initial log standard deviation
-0.5, and its own defaults for discounting/GAE and KL stopping. Our pilot uses
two 64-unit layers, SB3 initial log standard deviation 0, gamma .99 and GAE .95,
with no target-KL stopping. Copying a few parameters is an adaptation, not a
reproduction. These source links track upstream main and should be pinned to
a commit before a formal reproduction.

The reference PPO reward is not potential-only shaping; it explicitly supplies
intermediate release and stability incentives. Our potential's discounted sum
telescopes to `-Phi(start) + gamma**T * Phi(end)` before task rewards/value
bootstrapping. Local progress can aid value learning, but this does not create
an explicit release signal. This is a learning-signal hypothesis, not proof of
the cause of PPO failure.

Recommended order: verify scripted success and reward ordering on all starts;
inspect learned actions against that successful sequence; then choose a labeled
local adaptation or an exact external baseline reproduction. Do not change
action scales, reward, horizon and optimizer together. A known-working PPO
reference exists, but no directly matching current-robosuite/Mac PPO placement
recipe was established by this audit. Do not present our custom pilot as one.

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
