# Stack runs: SGS vs PLR vs uniform

Results for every Stack start-sampling run in this comparison: the scripted ladder
(`v2`), the first paper-style pool (`o1`), and the two paper-style pools with random
gripper opening (`f2`, `f3`). Each run folder holds:
- `args.json`: the full training configuration
- `evals.json`: every periodic evaluation, with per-family and per-start success
- `log.csv`: per-update training log: sampler shares, entropy, PPO statistics

Checkpoints are not included. The step-by-step lab notes, with the diagnosis behind
each change, are in `../README.md`.

All runs use the same PPO learner (`sgs/train_stack_ladder.py`): Panda, OSC position
control (4-d action), sparse reward on a strict physical stack check (cube released,
centred, level, slow, supported, held 5 steps), 4 envs × 128 steps per update, lr
3e-4 annealed to 0, no entropy bonus. Only the **start sampler** and the **start pool**
change between runs.

## 1. The samplers

| name in run IDs | sampler | how it picks the next start |
|---|---|---|
| `uniform` | uniform | random start from the pool; with `--fresh` (f3), the least-used start |
| `plr` | PLR (Jiang et al. 2021) | 50% of draws go to an unseen start. The other 50% replay seen starts, ranked by mean \|GAE\| (β = 0.1), mixed with staleness (ρ = 0.1) |
| `sgs` | SGS per start (arXiv 2610.12465) | weight each start by the paper's kernel on its success over the last 20 episodes (κ = 1, T = 2), which favours ~50% success |
| `sgscells` | SGS by group | same kernel, but success is tracked per **group** of similar starts (about 21 coarse groups). Draw a group, then a start inside it |
| `sgscellsk5` | SGS by group, κ = 5 | sharper kernel (the paper's locomotion setting) |
| `sgsfine` | SGS by finer group | about 60 groups (8 bins of progress × 3 grip levels, 4 height bins for Reaching) |

## 2. How the starts were sampled

### v2: scripted ladder (128 starts)

- A privileged script stacks the cube from home: reach, descend, close, lift, carry,
  lower, open, retreat.
- **8 cube layouts × 16 rungs.** Rung `rN` is the saved state N steps before the
  script's stack, with N = 2, 8, 15, 22, 28, 32, 36, 40, 44, 48, 53, 58, 64, 72, 85.
  `far` is the home pose.
- Difficulty rises smoothly along a single feasible path.
- **No separate test split:** runs are evaluated on their own 128 training starts.

### o1: first paper-style pool (384 training + 128 test starts)

Three families, following the paper's Appendix A.2. Every start has its own random
cube layout (A and B within ±10 cm, at least 9 cm apart, A rotated ±0.2 rad).

- **Reaching:** cube A on the table, gripper near its grasp point (±1.5 cm sideways,
  0–8 cm above), closed for 0–5 steps.
- **Grasp:** A held on the line from its table spot to B (`path_fraction` 0–1),
  3–12 cm above stacking height.
- **Near-Goal:** A held directly above B, `depth_fraction` 0–1.1, i.e. 0.3–5.8 cm
  above B.
- **Grip:** A is teleported between the fingers and the finger joints are set
  closed, so the grip is always firm. The exception: 30% of Near-Goal starts open
  1–5 steps.
- **Test split:** 32 starts per family, plus home. **Not filtered:** holding still
  and opening the gripper stacks 18/32 Near-Goal and 4/32 Grasp test starts.

### f1: o1-style starts with a filtered test split (test only, no training)

- Real grasps, and the test split excludes starts that holding still solves (gripper
  open or closed).
- Used only to re-score the o1 checkpoints.

### f2: paper-style starts with random grip (96 training + 256 test starts)

Built by `sgs_plr/pool.py` with `sgs_plr/resets.py`, seed 11. Every value is drawn
uniformly at random.

- **Reaching**, chosen 50/50:
  - **On the table:** as in o1.
  - **In the air:** the open gripper goes 3–15 cm above A's spot and A is
    teleported between the fingers. It closes 0–12 steps while A is pinned, then
    A is released, and stays held or falls.
- **Grasp / Near-Goal:**
  - Same positions as o1, but **reached with a real grasp:** the gripper descends
    and closes on A on the table, then carries it. No teleporting.
  - Then the gripper **opens 0–9 steps**: 0–4 firm to light squeeze, 5–6 loose,
    7–9 apart (the cube usually drops). Dropped cubes are kept, as in the paper.
- **Hold-still check:** every start is tried with the arm held still and the
  gripper open, then closed.
  - Test starts that either policy solves are rejected.
  - Training starts are kept but tagged: 17/32 Near-Goal, 3/32 Grasp.
- **Test split:** 64 per family, plus 64 home.
- **During training:** every start is reused dozens of times over 1M transitions.

### f3: same recipe, fresh start at each reset (10,368 training + 256 test starts)

- **Pool:** the same generator as f2, 3,456 starts per family, seed 13. Training
  starts were not hold-still checked; about 58% of Near-Goal training starts are
  solvable that way.
- **Fresh starts (`--fresh`):** each start has a use counter.
  - **Uniform:** always takes the least-used start, so it uses all 10,368 before any
    repeat.
  - **SGS:** draws a group with the kernel, then the least-used start inside it.
  - Over 1M transitions (about 10k episodes), each start is used about once, so
    neither sampler can memorise.
- **PLR is unchanged:** it still replays seen starts.
- Groups with fewer than 20 starts are merged into their family's largest group.
- The pool is not included (11 MB). Rebuild it with
  `python -m sgs_plr.pool --train-per-family 3456 --eval-per-family 64 --no-tag-train --seed 13 --out stack_omnireset_pools/f3`.

### Summary

| | v2 ladder | o1 | f2 | f3 |
|---|---|---|---|---|
| training starts | 128 (16 rungs × 8 layouts) | 384 | 96 | 10,368 |
| test starts | the training starts | 128, unfiltered | 256, filtered | 256, filtered |
| how A gets in the gripper | scripted | teleported, fingers set | real grasp | real grasp |
| grip | as scripted | firm (30% of Near-Goal loosened) | random, 0–9 open steps | random, 0–9 open steps |
| Reaching cube | n/a (rungs r72–r85) | on the table | 50% on the table, 50% in the air | same as f2 |
| uses per start | many | about 3 | dozens | about 1 |
| training length | 300k | 1M | 1M | 1M (plus one 3M uniform run) |

## 3. Results

Success is the fraction of test starts stacked by the deterministic policy. "Final" is
the last evaluation; AUC is the mean over the whole learning curve. 3 seeds unless
noted.

### v2 ladder (300k, success on the training starts)

| sampler | s0 | s1 | s2 | final mean ± sd | AUC |
|---|---|---|---|---|---|
| uniform | 0.172 | 0.000 | 0.008 | 0.060 ± 0.079 | 0.031 |
| PLR | 0.305 | 0.344 | 0.352 | 0.333 ± 0.021 | 0.231 |
| **SGS per start** | 0.461 | 0.359 | 0.562 | **0.461 ± 0.083** | **0.248** |

PLR put 56–65% of its draws on a single rung. SGS spread its draws over the hard
rungs.

### o1 (1M, original unfiltered test set)

| sampler | s0 | s1 | s2 | final mean ± sd | AUC |
|---|---|---|---|---|---|
| uniform | 0.008 | 0.023 | 0.031 | 0.021 ± 0.010 | 0.014 |
| SGS per start (s0 only) | 0.000 | | | 0.000 | 0.004 |
| SGS by group, κ = 1 | 0.180 | 0.000 | 0.062 | 0.081 ± 0.074 | 0.050 |
| SGS by group, κ = 5 | 0.016 | 0.000 | 0.008 | 0.008 ± 0.006 | 0.011 |
| **PLR** | 0.188 | 0.227 | 0.195 | **0.203 ± 0.017** | **0.143** |
| *hold still, open gripper* | | | | *0.17* | |

The same checkpoints re-scored on f1's filtered test set
(`results/rescore_f1_eval.json`):

| sampler | success on new starts | Near-Goal | success on training starts | gap |
|---|---|---|---|---|
| uniform | 0.011 | 0.02–0.08 | 0.01 | 0.00 |
| SGS by group, κ = 1 | 0.043 | 0.02–0.33 | 0.13 | 0.08 |
| SGS by group, κ = 5 | 0.010 | 0–0.11 | 0.04 | 0.03 |
| **PLR** | **0.137** | **0.44–0.64** | 0.50 | **0.37** |

### f2 (1M, filtered test set)

| sampler | s0 | s1 | s2 | final mean ± sd | AUC | success on training starts |
|---|---|---|---|---|---|---|
| uniform | 0.012 | 0.004 | 0.012 | 0.009 ± 0.004 | 0.012 | 0.02–0.05 |
| SGS by group | 0.012 | 0.016 | 0.023 | 0.017 ± 0.005 | 0.015 | |
| SGS per start | 0.023 | 0.031 | 0.027 | 0.027 ± 0.003 | 0.023 | |
| **PLR** | 0.023 | 0.039 | 0.059 | **0.040 ± 0.014** | **0.031** | **0.38–0.67** |

Sampler entropy at the end of training: PLR about 0.8 (effectively 2–3 starts); SGS
per start about 4.2 (about 65 starts).

### f3 (1M, fresh start at each reset, filtered test set)

| sampler | s0 | s1 | s2 | final mean ± sd | AUC | training vs new-start success |
|---|---|---|---|---|---|---|
| uniform | 0.016 | 0.000 | 0.016 | 0.010 ± 0.007 | 0.010 | 0.02–0.07 vs 0–0.02 |
| PLR | 0.031 | 0.031 | 0.000 | 0.021 ± 0.015 | 0.017 | **0.31–0.42** vs 0–0.03 |
| SGS by group | 0.012 | 0.039 | 0.020 | 0.023 ± 0.011 | 0.025 | 0.04–0.12 vs 0.01–0.04 |
| **SGS by finer group** | 0.023 | 0.082 | 0.008 | **0.038 ± 0.032** | **0.037** | 0.07–0.20 vs 0.01–0.07 |
| uniform, 3M (`f3_uniform_long3m_s0`) | 0.004 | | | | | |

- All success is on Near-Goal and Grasp starts. The best run, SGS by finer group s1,
  reaches 0.30 on Near-Goal.
- Reaching and home are 0 for every sampler, in every pool except the ladder.
- SGS spent 60–74% of its draws on Near-Goal.

## 4. PLR vs SGS: where each won and why

| setting | uses per start | winner | why |
|---|---|---|---|
| v2 ladder | many | **SGS** (0.46 vs 0.33) | Smooth difficulty, so there is always a band of rungs near 50% success for the kernel to target, and per-start estimates are reliable. PLR fixed on one rung (56–65% of draws); SGS covered all the hard rungs. |
| o1 | about 3 | **PLR** (0.20 vs 0.08; 0.14 vs 0.04 re-scored) | Three visits per start are too few for SGS to estimate success; per-start SGS was nearly uniform (entropy 5.84 vs a maximum of 5.95). PLR drilled a few starts and learned real Near-Goal skill (0.44–0.64), but with a 0.37 gap between training and new starts. |
| f2 | dozens, 96 starts | **PLR** (0.040 vs 0.027), seeds overlap | With so few starts, PLR drilled 2–3 of them, memorised them (0.38–0.67 on its training starts) and some of that skill transferred. SGS spread its draws across about 65 starts, which gave less depth at this budget. SGS by group did worse, because hold-still training starts inflated its group success estimates. |
| f3 | about 1 | **SGS** on average (0.038 / 0.023 vs 0.021), within noise | Memorising is impossible for SGS and uniform. PLR still spends half its draws replaying and memorising (0.31–0.42 on training starts vs 0–0.03 on new ones). SGS trains Near-Goal on new starts, so what it learns has to generalise. By seed: PLR beat both SGS versions on s0, and SGS beat PLR on s1 and s2. |

**The pattern:** SGS does better the more reliable its success estimates are, and the
less a sampler can gain by memorising. PLR wins when there are few starts to drill.

## 5. Caveats

- **The ladder is evaluated on its own training starts,** so memorising counts as
  success there, and its r2/r8 rungs were not hold-still checked. Its SGS win shows
  better coverage, not better generalisation.
- **o1's test set includes hold-still starts.** Holding still and opening the
  gripper scores 0.17 there, close to PLR's 0.20. Use the f1 re-score for o1.
- **f2/f3 scores are all below 0.1, and the spread between seeds is as large as the
  gaps between samplers.** No ordering there is significant with 3 seeds.
- **The PPO setup is the bottleneck on paper-style starts.** The diagnosis is in
  `../README.md`:
  - 3× more training did not help (the 3M run ends at 0.004).
  - Updates are small (512 transitions) and hit the KL early-stop after about 3 of
    10 epochs.
  - 20–40% of final action means are beyond the ±1 limit.
  - Exploration collapses with no entropy bonus.
  - Much of the training reward comes from hold-still Near-Goal starts.
  - The planned next step is to fix PPO with uniform only, then rerun f3 with 5 seeds.

## 6. Files

```
outputs/sgs_plr/
  README.md            lab notes: problems found, plan, diagnosis
  runs/README.md       this file
  runs/<run id>/       args.json, evals.json, log.csv
  results/             floor_*.json (fixed-policy baselines), rescore_*.json (train vs new starts)
  pools/{v2,o1,f1,f2}/ pool.json (metadata) and pool.npz (simulator states)
```

Run modules from `outputs/` with `PYTHONPATH=.`:
- `python -m sgs.compare_stack_ladder <run ids>`: summary tables
- `python -m sgs_plr.rescore --pool <pool> <run ids>`: training vs new-start success
  (needs the checkpoints)
- launch scripts: `sgs_plr/run_f2.sh`, `sgs_plr/run_f3.sh`, `sgs_plr/run_v2_plr.sh`
