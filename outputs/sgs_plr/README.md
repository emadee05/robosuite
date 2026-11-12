# SGS vs PLR vs uniform on Stack

Goal: a benchmark, built in steps, that tests whether Success-Guided Sampling (SGS,
arXiv 2610.12465) beats PLR and uniform start sampling. First on robosuite Stack;
then, only if it holds there, on a second robosuite task. Metrics and seed counts
are fixed before each comparison, and every run is reported, whichever sampler wins.

## 1. What was wrong with the previous comparison

The previous round (`outputs/README.md`, sections "Full-task Stack ladder" and
"Stack with paper-style resets") found:

| setup | uniform | SGS | PLR |
|---|---|---|---|
| v2 scripted ladder, 300k, 3 seeds | 0.06 | **0.46** | not run |
| o1 OmniReset-style pool, 1M, 3 seeds | 0.02 | 0.08 (per cell, κ=1) | **0.20** |

Problems found afterwards (2026-11-07):

1. **Near-Goal can be solved by letting go.** A fixed policy that holds the arm still
   and opens the gripper stacks 18/32 o1 Near-Goal and 4/32 Grasp evaluation starts
   (0.17 overall, close to PLR's 0.20). On generated Near-Goal starts with real grasps,
   64% are solved this way. The o1 metric could not tell learning apart from letting
   go. (Re-scoring in step 0.4 showed that PLR did learn more than letting go; see Results.)
2. **Untrained policies hide the shortcut by chance.** The Panda gripper moves on
   the sign of its command, so an untrained policy whose gripper output is slightly
   negative opens and scores about 20/32 on Near-Goal. On the
   `next_steps/pool` probe, 2 of 3 random inits did this and one scored 0. On o1, all
   three inits happened to output a positive command and scored 0, which made o1
   look clean.
3. **The headline metric was wrong.** The paper headlines success from Reaching. No run
   on o1 succeeded from Reaching or from `home`, so every difference came from
   Near-Goal, which is exactly where the shortcut is.
4. **Too few visits per start for SGS.** 384 training starts with about 300 episodes per
   100k steps gives about 3 visits per start in 1M steps, too few for a success
   estimate. Per-start SGS sampled almost uniformly (entropy 5.84 vs ln 384 = 5.95).
5. **PLR was never run on the ladder** where SGS clearly beat uniform, so the two
   setups compared different pairs of samplers.
6. **Unequal tuning.** SGS got a κ sweep (κ=5 was worse: a sharper kernel gives
   almost no extra weight to cells at 1–10% success). PLR ran at one setting.
7. **Smaller issues.** o1 Grasp starts teleport the fingers closed; there were 32
   evaluation starts per family; there were 3 seeds; SFL (the paper's other
   baseline) is not implemented; nothing measured the gap between training and
   fresh-start success, which is where SGS's argument (keep coverage) and PLR's
   weakness (it concentrates about 90% of episodes on a few starts) should show.

## 2. Plan

### Phase 0: fix the benchmark (this folder, done 2026-11-07)

| step | what | why |
|---|---|---|
| 0.1 | `pool.py`: new pool `f1`. Real OSC grasps; every start tagged with whether "hold still + open" solves it; such starts **rejected from the evaluation split**, kept and tagged in training | Removes problem 1 from the metric. Easy training starts stay because they are a legitimate stepping stone |
| 0.2 | `floor.py`: score "still + open", "still + closed" and 3 untrained inits on every pool | A run counts as learning only above these floors (problem 2) |
| 0.3 | 32 training starts per family (96) and 64 evaluation starts per family (256) | About 10 visits per start in 1M steps (problem 4); a more precise evaluation (problem 7) |
| 0.4 | `rescore.py`: re-evaluate the 13 finished o1 checkpoints on `f1`'s evaluation split; report training success, fresh-start success and the gap | Shows how much of the o1 result survives without the shortcut (problem 3) |
| 0.5 | Fixed metrics for every later comparison: **Reaching success** (headline), fresh-start success on all non-trivial starts, area under the learning curve, train-vs-fresh gap; 5 seeds for the final comparison | Settled before the runs (problems 3, 7) |

### Phase 1: show it (or not) on Stack

| step | what | why |
|---|---|---|
| 1.4 | PLR on the v2 ladder, 3 seeds, 300k, same settings as `v2_uniform_*`/`v2_sgs_*` (**done**: `run_v2_plr.sh`) | Cheapest test of SGS vs PLR on a setup where SGS already beats uniform (problem 5) |
| 1.5 | On `f1`: uniform, PLR, per-start SGS and per-cell SGS, 3 seeds × 1M | With about 10 visits per start, SGS's success estimates are usable |
| 1.6 | Intermediate starts between Near-Goal/Grasp and Reaching (gripper around the cube and closed; cube lifted 1–2 cm; finer Reaching height bins) | No sampler ever succeeded from Reaching. SGS follows the frontier only if difficulty is dense, which is what made v2 work |
| 1.7 | Equal tuning: SGS κ∈{1,3}, T∈{1,2}, history∈{10,20}; PLR β∈{0.1,0.3}, ρ∈{0.1,0.3}; same number of runs each; then 5 seeds × 3M at the best settings | Removes problem 6 |
| 1.8 | Decision: if SGS beats PLR on Reaching or fresh-start success, write it up with all runs. If not, add SFL before concluding | Prevents cherry-picking |

### Phase 2: a second robosuite task (only after Phase 1 holds)

- **PickPlaceCan** first: the same families (reach the can, grasp, near the bin)
  reuse most of `resets.py`. Needs its own success check and its own let-go filter.
- Then **NutAssemblySquare** (long, precise insertion: a frontier-following test)
  or **Door** (articulated object).
- First refactor to a generic start generator (families plus success check per task)
  so the trainer, samplers, floors and comparisons stay shared.

## 3. Setup

Run everything from `hep/outputs` with the robosuite venv. The trainer is the
unchanged `sgs/train_stack_ladder.py` (samplers `uniform`, `sgs` [+`--sgs-cells`,
`--sgs-kappa`], `plr`); this folder adds pools, floors, re-scoring and launch scripts.

| file | purpose |
|---|---|
| `resets.py` | Reaching (on table / airborne) / Grasp / Near-Goal start generator: real OSC grasps, random aperture, SGS cells |
| `pool.py` | builds a pool; tags and filters "let go" starts |
| `common.py` | reference policies (still + open/closed, untrained, checkpoint) and per-family evaluation |
| `floor.py` | success floors on a pool's evaluation split |
| `rescore.py` | evaluates finished checkpoints on another pool; training vs fresh gap |
| `run_v2_plr.sh` | step 1.4 launcher |
| `run_f2.sh` | step 1.5 launcher (pool f2) |
| `run_f3.sh` | fresh-start-per-reset launcher (pool f3) |
| `results/` | JSON outputs of `floor.py` and `rescore.py` |

```sh
cd hep/outputs
PY="env PYTHONPATH=. ../robosuite/.venv/bin/python"
$PY -m sgs_plr.pool --out stack_omnireset_pools/f2          # 352 starts, about 2 min (f1 was built with the older resets.py)
$PY -m sgs_plr.floor --pool stack_omnireset_pools/f1
$PY -m sgs_plr.rescore --pool stack_omnireset_pools/f1 o1_uniform_s0 o1_plr_s0 o1_sgscells_s0
sgs_plr/run_v2_plr.sh                                       # 3 × 300k PLR runs in parallel
$PY -m sgs.compare_stack_ladder v2_uniform_s0 v2_uniform_s1 v2_uniform_s2 \
    v2_sgs_s0 v2_sgs_s1 v2_sgs_s2 v2_plr_s0 v2_plr_s1 v2_plr_s2
```

Training runs go to `stack_task_ladder_runs/<name>/` (`args.json`, `log.csv`,
`evals.json`, `ckpt.pt`). Pools go to `stack_omnireset_pools/<name>/`.

## 4. Results

### Pool f1 (step 0.1)

| split | family | kept | let-go starts |
|---|---|---|---|
| train | Reaching | 32 | 0 |
| train | Grasp | 32 | 2 kept, tagged |
| train | Near-Goal | 32 | 21 kept, tagged |
| eval | home | 64 | 0 |
| eval | Reaching | 64 | 0 |
| eval | Grasp | 64 | 5 rejected |
| eval | Near-Goal | 64 | 115 rejected (64% of generated) |

### Floors on f1's evaluation split (step 0.2)

| policy | all | home | Reaching | Grasp | Near-Goal |
|---|---|---|---|---|---|
| still + open | 0.008 | 0 | 0 | 0 | 0.031 |
| still + closed | 0 | 0 | 0 | 0 | 0 |
| untrained, 3 inits | 0 | 0 | 0 | 0 | 0 |

The filter works: a run only needs to beat about 0.01 to show learning. Still + open
gets 2/64 Near-Goal starts in `StackLadderEnv`, whose restore differs slightly from
the generator's state.

### o1 checkpoints re-scored on f1 (step 0.4)

Final o1 checkpoints evaluated on f1's 256 evaluation starts (real grasps, let-go
starts removed). "train" is training success in the last log row; "gap" = train − fresh.

| sampler | fresh (seeds 0,1,2) | mean fresh | Near-Goal | Grasp | train | gap |
|---|---|---|---|---|---|---|
| uniform | 0.008, 0.004, 0.020 | 0.011 | 0.02–0.08 | 0 | 0.01 | 0.00 |
| SGS per start (seed 0) | 0.000 | 0.000 | 0 | 0 | 0 | 0 |
| SGS per cell, κ=1 | 0.090, 0.004, 0.035 | 0.043 | 0.02–0.33 | 0–0.06 | 0.13 | 0.08 |
| SGS per cell, κ=5 | 0.031, 0.000, 0.000 | 0.010 | 0–0.11 | 0–0.02 | 0.04 | 0.03 |
| PLR | 0.121, 0.172, 0.117 | **0.137** | **0.44–0.64** | 0.02–0.05 | 0.50 | **0.37** |

- PLR **did learn more than letting go**: on non-trivial Near-Goal starts it scores
  0.44–0.64, against a floor of 0.03. The o1 ranking (PLR > SGS per cell > uniform)
  survives the filter.
- PLR also has by far the largest train-vs-fresh gap (training success 0.36–0.70,
  fresh 0.12–0.17). It overfits its few favourite starts, as expected, but at this
  scale it is still best on fresh starts.
- Reaching and `home` remain 0 for every run.
- Caveat: o1 runs trained on teleported-grasp starts and are evaluated here on real
  grasps; the shift is the same for every sampler.

### v2 ladder with PLR (step 1.4)

Final evaluation success on the 128 ladder starts, 300k transitions:

| sampler | finals (seeds 0,1,2) | mean | last 3 evals | AUC |
|---|---|---|---|---|
| uniform | 0.172, 0.000, 0.008 | 0.060 | 0.049 | 0.031 |
| PLR | 0.305, 0.344, 0.352 | 0.333 | 0.319 | 0.231 |
| SGS | 0.461, 0.359, 0.562 | **0.461** | **0.416** | **0.248** |

- SGS ends above PLR in all three seeds (0.36–0.56 vs 0.31–0.35), but the AUC is
  almost tied (0.248 vs 0.231). PLR learns as fast early and then stalls.
- The difference is at the hard rungs. SGS reaches 0.1–0.5 on r36–r64 in every seed;
  PLR is near 0 there in seeds 0 and 2. Seed 1 is a flat 0.25 that looks like one scene
  solved across all rungs.
- Sampling: PLR puts 56–65% of late episodes on a single rung (r22 or r36). SGS
  spreads episodes over r15–r64 (no rung above 14%).
- With 3 seeds this is suggestive, not conclusive. The final comparison (step 1.7) uses
  5 seeds.

### Pool f2: closer to the paper's resets (2026-11-07)

Two changes to `resets.py` bring the start families closer to the paper's Appendix A.2:

- **Random aperture in Grasp and Near-Goal** (aperture = how far open the fingers are at
  the start). Before, the gripper was always clamped shut on the cube (Grasp, and 70% of
  Near-Goal). Now, after a real grasp and carry, the gripper opens for 0–9 steps. The
  carry leaves the command at fully closed, so 0–4 steps vary the squeeze, 5 is a light
  grip, and 6–9 open the fingers (gap 0.045–0.068 m vs a 0.04 m cube), so the cube drops.
  About 60% of these starts hold the cube. *Why:* the policy must also learn to re-grip or
  recover, not only to finish from a perfect grasp.
- **Reaching includes a cube off the table.** Half of the Reaching starts put cube A
  between the open fingers 3–15 cm above the table, hold it in place while the gripper
  closes for 0–12 steps, then release it. Few close steps: the cube falls somewhere and
  must be fetched. Many: it is held. The other half keep the cube resting on the table
  with the gripper near it. *Why:* this matches the paper's Reaching ("object anywhere in
  the workspace ... the object can fall anywhere, requiring the robot to go fetch it").
- The evaluation filter now also rejects starts solved by holding still with the gripper
  closed (none so far). Each start stores `finger_gap`, `cube_held`, `airborne` and its
  SGS `cell`; `--sgs-cells` uses the stored cell (21 non-empty training cells).

| split | family | kept | cube held | airborne | trivial (rejected in eval / tagged in train) |
|---|---|---|---|---|---|
| train | Reaching | 32 | 11 | 13 | 0 |
| train | Grasp | 32 | 17 | – | 3 tagged |
| train | Near-Goal | 32 | 27 | – | 17 tagged |
| eval | home | 64 | – | – | 0 |
| eval | Reaching | 64 | 16 | 26 | 0 |
| eval | Grasp | 64 | 39 | – | 3 rejected |
| eval | Near-Goal | 64 | 35 | – | 62 rejected |

Floors on f2's evaluation split: 0.000 for still + open, still + closed and all 3
untrained inits. A short 20k cell-SGS sanity run already reaches 0.035 (Near-Goal 0.12).

### Step 1.5 on f2 (done 2026-11-08)

`sgs_plr/run_f2.sh`: uniform, PLR, SGS per start and SGS per cell; seeds 0–2; 1M
transitions each; one batch per seed with all 4 samplers running together. Runs are
`stack_task_ladder_runs/f2_<variant>_s<seed>`; the train-vs-fresh numbers are in
`results/rescore_f2_eval.json`.

Success on f2's 256 fresh evaluation starts (floors are 0) and on the training starts:

| sampler | fresh (seeds 0,1,2) | mean fresh | AUC | Near-Goal fresh | training success | gap | sampler entropy (ln 96 = 4.56) |
|---|---|---|---|---|---|---|---|
| uniform | 0.012, 0.004, 0.012 | 0.009 | 0.012 | 0.02–0.05 | 0.02–0.05 | 0.02 | 4.56 |
| SGS per cell | 0.012, 0.016, 0.023 | 0.017 | 0.015 | 0.05–0.08 | 0.12–0.22 | 0.13 | about 3.9 |
| SGS per start | 0.023, 0.031, 0.027 | 0.027 | 0.023 | 0.08–0.11 | 0.12–0.26 | 0.17 | about 4.2 |
| PLR | 0.023, 0.039, 0.059 | **0.040** | **0.031** | 0.08–0.19 | **0.38–0.67** | **0.53** | about 0.8 |

- **Everything is low.** f2 is much harder than o1. The let-go starts are gone from
  evaluation, apertures are random, and there are only 96 training starts. The best
  fresh score is 15/256 starts. Reaching and `home` are 0 for every run.
- **Ranking: PLR > SGS per start > SGS per cell > uniform**, but PLR and SGS per start
  overlap: they tie on seed 0, and SGS's spread (0.023–0.031) sits inside PLR's
  (0.023–0.059). With 3 seeds and differences of 2–8 starts out of 256, PLR vs SGS is
  not resolved. Both beat uniform in every seed.
- **Per-start SGS now behaves as intended.** With about 10 visits per start it samples
  non-uniformly (entropy about 4.2) and puts 41–59% of episodes on Near-Goal, against
  about 3 visits per start and uniform-like sampling on o1. Per-cell SGS was worse than
  per-start here; with 96 starts the cells (1–12 starts each) average over starts
  that differ too much.
- **PLR memorises.** Its entropy of about 0.8 is roughly 2 starts' worth of mass. It
  solves 38–67% of its training episodes (90% on its Grasp/Near-Goal favourites) but
  only 2–6% of fresh starts. SGS's gap is a third of PLR's.
- **The key trade-off at laptop scale:** with many starts (o1, 384) SGS gets too few
  visits per start to estimate success; with few starts (f2, 96) every sampler, and PLR
  most, overfits to the specific starts and generalises poorly. The paper avoids both
  with 32,768 starts *and* thousands of parallel environments.

### Fresh start at every reset: pool f3 (launched 2026-11-08)

*Why:* step 1.5 showed a trade-off. With 384 starts, SGS gets too few visits per start;
with 96, every sampler memorises its starts (PLR most: 0.57 training vs 0.04 fresh).
Drawing a fresh start at every reset removes both problems: nothing can be memorised,
and SGS only needs success estimates per cell, which get hundreds of visits each.

*How:* rather than generating starts inside the training loop (similar compute, but
samplers would not see the same starts), pool `f3` holds **10,368 training starts**
(3,456 per family, generated by the same `resets.py` as f2), so 1M transitions (about
10k episodes) use each start about once. The evaluation split is generated separately as
before: 64 per family, with starts solved by holding still removed.

| sampler | how it picks the next start |
|---|---|
| uniform `--fresh` | least-used start in the whole pool (no repeats until all are used) |
| PLR | unchanged: half the draws go to an unseen start, half replay seen starts by \|GAE\| rank + staleness. This is standard PLR over a large level buffer |
| SGS per cell `--sgs-cells --fresh` | samples a cell with the paper's kernel on the cell's success rate (last 20 outcomes), then the least-used start in that cell. Coarse cells: 22 after folding |
| SGS fine cells `--cell-field cell_fine` | same, with about 60 finer cells (8 path/depth bins × 3 aperture levels; 4 height bins for Reaching) |

Cells with fewer than 20 training starts are folded into their family's largest cell
(`--min-cell-size`; only the rare "cube held on the table" Reaching cell). Without this,
SGS would replay that one start hundreds of times.

```sh
$PY -m sgs_plr.pool --train-per-family 3456 --eval-per-family 64 --chunk 128 --procs 8 \
    --no-tag-train --seed 13 --out stack_omnireset_pools/f3     # about 20 min
sgs_plr/run_f3.sh                                                # 4 variants x 3 seeds x 1M
```

Pool stats: training starts with the cube held: Reaching 1,047/3,456 (1,629 airborne),
Grasp 2,298, Near-Goal 2,799. Evaluation: 76 Near-Goal starts rejected as trivial.

**Results (done 2026-11-08; 3 seeds × 1M transitions; success on the 256 fresh
evaluation starts).** Seed 1 first crashed at about 105k steps because the disk filled
(caused by something else on the machine) and was rerun from scratch.

| variant | s0 | s1 | s2 | final mean ± sd | last 3 evals | AUC |
|---|---|---|---|---|---|---|
| uniform `--fresh` | 0.016 | 0.000 | 0.016 | 0.010 ± 0.007 | 0.007 | 0.010 |
| PLR | 0.031 | 0.031 | 0.000 | 0.021 ± 0.015 | 0.016 | 0.017 |
| SGS coarse cells | 0.012 | 0.039 | 0.020 | 0.023 ± 0.011 | 0.026 | 0.025 |
| SGS fine cells | 0.023 | 0.082 | 0.008 | 0.038 ± 0.032 | 0.034 | 0.037 |

- All of the success is on Near-Goal starts. Reaching and home are 0 for every run, and
  Grasp is 0.00–0.01.
- The best single run, SGS fine s1, reaches 0.28 on Near-Goal.
- SGS puts 60–74% of draws on Near-Goal and 6–12% on Reaching, its current frontier
  (success there is about 0).
- On means, SGS fine > SGS coarse ≈ PLR > uniform, and SGS has the highest AUC. But the
  seed spread is as large as the gaps. Within the same seed, SGS coarse loses to uniform
  on s0, and SGS fine loses to PLR on s0 and to uniform on s2. Every number is under 0.1.
  This is a weak trend, not evidence.
- Several curves peak around 100–300k transitions and then decline.
- PLR still overfits to the starts it replays: 0.31–0.42 success on training episodes
  vs 0–0.03 on fresh starts (gap 0.30–0.39). The gap is 0.03–0.13 for SGS and
  0.02–0.05 for uniform (`results/rescore_f3_eval.json`).

*Why the curves peak and then decline* (diagnosed 2026-11-08 from `log.csv` and the final checkpoints):

1. **Exploration collapses.**
   - `ent_coef` is 0, so the policy std falls from 0.22 to about 0.13 by 1M transitions.
   - The logged entropy goes from −0.4 to below −2.0.
2. **Action means run past the ±1 clip.**
   - The Gaussian mean is unbounded, and the env clips actions to ±1. On Grasp and
     Near-Goal starts, 20–40% of the final policies' arm outputs are beyond ±1 (max 3.6),
     so the control becomes bang-bang. Untrained, the means are about 0.
   - The worst runs saturate most: uniform s0 and SGS coarse s0 at 35–39%. The best,
     PLR s1 and SGS fine s1, saturate least, at 7–20%.
   - Placing a cube needs small motions, so this works directly against stacking.
3. **Updates are large and noisy.**
   - Each update uses 512 transitions (4 envs × 128) in minibatches of 64.
   - For the first half of training, KL hits the 0.05 early-stop after about 3 of the
     10 epochs on every update.
   - Training success itself crashes mid-run in some seeds: PLR s2 goes 0.35 → 0.07 at
     300–500k, and SGS coarse s0 goes 0.17 → 0.01 at 200–400k.
   - The lr anneals linearly to 0, so late training cannot recover.
4. **Much of the training reward comes from starts that are trivial in evaluation.**
   - f3's training split was not filtered. On a sample of 120 Near-Goal training starts,
     58% are solved by holding still with the gripper open or closed. The evaluation
     split contains none.
   - Final policies succeed 2–5× more often on those starts. For example, PLR s1 scores
     0.83 on them vs 0.20 on the rest.
   - So training success overstates the skill, and much of the success signal rewards
     finishing a nearly done stack.
5. **Evaluation noise.** 64 starts per family, so a peak of 0.10 vs a later 0.06 is
   about 10 starts.

**Longer uniform run** (`f3_uniform_long3m_s0`, 3M transitions, same settings, done
2026-11-09):
- Fresh success ends at 0.004 (one start), peaks at 0.039 at 700k, and averages about
  0.01 throughout.
- Training success never rises above about 0.05.
- Entropy falls to −3.2 as the lr anneals.

So triple the budget does not help under these PPO settings, and the bottleneck is the
learner, not the training length.

Candidate fixes for a short ablation:
- squash or clip the action mean (tanh), or penalise means beyond ±1
- an entropy bonus (`--ent-coef 0.005`) or a floor on log_std
- a bigger batch (`--num-envs 8 --num-steps 256`, i.e. 2,048 transitions per update)
- tag the training starts and drop or down-weight the trivial ones

*Interpretation:* fresh starts removed memorisation and the visits-per-start problem.
But at this budget (4 envs, 1M transitions) the policy learns very little that carries
over to new starts, so the samplers can barely be told apart. Before another sampler
comparison, learning itself has to work better: a larger budget or more envs, or
easier intermediate starts.

### Where this leaves the plan

- **Ladder (dense difficulty):** SGS > PLR > uniform at the end of training, through
  better coverage of the hard rungs. This is the first SGS > PLR evidence.
- **OmniReset pool (sparse difficulty, 384 starts):** PLR > SGS > uniform, even
  without the shortcut.
- **f2 (paper-style resets with random aperture and airborne Reaching, 96 starts):**
  PLR ≳ SGS per start > SGS per cell > uniform, with PLR and SGS overlapping. PLR wins
  on fresh starts despite memorising.
- **f3 (a fresh start at every reset, 10,368 starts):** SGS fine ≳ SGS coarse ≈ PLR >
  uniform on means, but every score is under 0.1 and the seeds overlap. PLR's
  memorisation persists on the starts it replays.
- Suggested next steps:
  1. **Make learning work first.** Run one longer uniform run (about 3M transitions, or
     more envs) to check that a bigger budget raises fresh success at all. Also find out
     why the curves peak early and then decline (entropy collapse, learning rate,
     value-function drift).
  2. Step 1.6: intermediate starts toward Reaching. Still no sampler succeeds there,
     and that is where the paper's headline metric lives.
  3. Then rerun f3 with 5 seeds and bootstrap confidence intervals before claiming
     any PLR vs SGS ordering.
