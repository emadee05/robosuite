# Small-scale SGS on robosuite Panda Lift

A minimal, readable testbed for comparing **uniform sampling** against **SGS
(Success Guided Sampling)** when choosing which task configuration each RL
episode starts from.

SGS is from *"A Balanced Data Diet: Addressing the Exploration Bottleneck in
Mega-Scale RL for Robot Control"* (Zhang, Guaman Castro, Yin, Dagnino, Gupta,
Scalise, Boots — CoRL 2026, [arXiv:2610.12465](https://arxiv.org/abs/2610.12465)).

Everything here runs on CPU. No GPU is needed or wanted (see *Why this runs on
CPU*), so it runs the same on a MacBook as on a 256-core server — just slower.
Total install is ~2 GB.

---

## Install

```bash
bash setup.sh        # uses uv if you have it, else python3 -m venv + pip
```

Then confirm the simulator works and see how fast *your* machine is:

```bash
.venv/bin/python scripts/bench.py --envs 1 4 8 --steps 200
```

Two things to know on macOS:

- **`mujoco` must stay below 3.16.** 3.16 changed the joint-address bindings and
  robosuite 1.5.2 then dies on reset with an opaque `AssertionError`.
  `requirements.txt` pins `mujoco==3.3.7`.
- Rendering is off (`MUJOCO_GL=disable`), which is valid on every platform and
  costs nothing because we only use state observations. If you later want
  images or video, set `SGS_MUJOCO_GL=cgl` on macOS (`egl` on Linux).

---

## Quick start

```bash
# one PPO run with uniform sampling (starts in tmux, returns immediately)
bash scripts/launch.sh demo_uniform --sampler uniform

# the same thing with SGS -- identical in every other respect
bash scripts/launch.sh demo_sgs --sampler sgs

# the full comparison: 2 samplers x 3 seeds, all at once
bash scripts/run_comparison.sh 5000000 0.20 64

# how fast is the simulator here? (quick, runs in the foreground)
.venv/bin/python scripts/bench.py

# plots
.venv/bin/python scripts/plot.py runs/demo_uniform runs/demo_sgs
```

`--num-envs` and `--torch-threads` default to values derived from your core
count, so the same command is sensible on a laptop and on a server:

| | `--num-envs` = `cpu_count - 4`, clamped to [4, 64] | `--torch-threads` = `cpu_count // 3`, clamped to [2, 8] |
|---|---|---|
| M2 Pro, 12 cores | 8 | 4 |
| server, 256 cores | 64 | 8 |

Both resolved values are printed at startup and recorded in
`runs/<name>/args.json`, so a run stays reproducible even though the default is
machine-dependent. Pass them explicitly to override.

**Training runs always go in tmux.** `scripts/launch.sh` puts each run in its
own window of a session called `sgs`:

| command | what it does |
|---|---|
| `tmux attach -t sgs` | watch the runs; `Ctrl-b d` detaches again |
| `Ctrl-b n` / `Ctrl-b p` | next / previous run |
| `Ctrl-b w` | pick a run from a list |
| `tmux list-windows -t sgs` | what is running, without attaching |
| `Ctrl-c` inside a window | cancel that one run |
| `tmux kill-session -t sgs` | cancel everything |

Windows stay open after a run exits (`remain-on-exit`), so a crash is still
readable; close a dead one with `Ctrl-b &`. Output is also teed to
`runs/<name>.out`. `scripts/launch.sh` passes every argument after the run name
straight through to `sgs.train`, and refuses to reuse a window name.

To run in the foreground instead (debugging a short run), call the module
directly: `.venv/bin/python -m sgs.train --sampler sgs --run-name scratch`.

Each run writes to `runs/<name>/`: `log.csv` (one row per PPO update),
`args.json`, `configs.npz` (the task pool), `p_history.npy` (per-configuration
success rate over time), `ckpt.pt`.

---

## What is actually being compared

A **task configuration** `tau` is one starting situation. Here that is 10
numbers: cube `x`, cube `y`, cube yaw, and a 7-dim offset on the Panda's initial
joint angles (`sgs/configs.py`). Before training we draw a **fixed pool** of `N`
configurations and never change it. Every episode starts from one member of that
pool; the *only* difference between the two conditions is which one.

- `uniform` — every configuration equally likely, always.
- `sgs` — configurations whose recent success rate is near a target `t` are
  sampled most often.

Same policy, optimiser, reward, pool, and seed otherwise. `--config-seed` is
deliberately separate from `--seed` so that different training seeds all face
the same set of tasks.

### Read `eval/success`, not `train/success`

`train_success` is the success rate on the episodes the sampler actually chose.
**It is not comparable across samplers.** SGS deliberately spends its episodes
on configurations the policy is currently failing, so its training success rate
is *lower by construction* even when its policy is strictly better.

`eval_success` is the honest number: every configuration in the pool, run once,
uniformly, with deterministic (mean) actions. That is the quantity the paper
plots, and the only one worth putting on a slide.

---

## The SGS implementation

`sgs/sampler.py` implements the paper's two equations literally. For each
configuration `i`, `p_i` is the mean of its last `H` boolean episode outcomes:

```
w_i = (p_i + eps)^(kappa*t) * (1 - p_i + eps)^(kappa*(1-t))     # Eq. 1
l_i = log(w_i + eps)
P(i) = softmax(l / T)_i                                          # Eq. 2
```

`t` is the target success rate (the kernel peaks there), `kappa` sharpens the
kernel, `T` sharpens the softmax, `eps` keeps fully-solved and fully-unsolved
configurations at small but non-zero probability.

Sanity check of the shape, with `t=0.5, kappa=1, eps=1e-4, T=2` and five
configurations whose success rates are 0.0 / 0.2 / 0.5 / 0.8 / 1.0:

```
P(i) = [0.046, 0.291, 0.325, 0.291, 0.046]
```

Peaked on the frontier, symmetric, nothing starved. With `kappa=5` it sharpens
to `[0.025, 0.254, 0.442, 0.254, 0.025]`.

### Where we differ from the paper, and why

| | Paper (manipulation) | Here | Why |
|---|---|---|---|
| `N` (pool size) | 32,768 | 64 | At `N=32768` and `H=100` you need **3.3M episodes** just to fill every history once. At 200 steps/episode that is 650M env steps before SGS has a single complete statistic. Out of reach at this scale. |
| `H` (history) | 100 | 50 | Same arithmetic. 64x50 = 3,200 episodes to fill, ~640K env steps — reachable. |
| `t`, `kappa`, `eps`, `T` | 0.5, 1, 1e-4, 2 | identical | The paper reports SGS is insensitive to these; no reason to deviate. |
| Parallel envs | up to 2^20 | 64–128 | See *The scale caveat*. |
| Reward | boolean success + regularisers | robosuite dense shaping, boolean success for SGS | A sparse-reward PPO will not get off the ground at 1M steps. The sampler still only ever sees the boolean. |
| Unseen configurations | unspecified | `p_i = 0` (paper-literal) | At 1M parallel envs every config is hit immediately, so the paper never has to say. `--sgs-warmup-episodes K` draws the first K uniformly if you prefer every history seeded first. |

---

## The scale caveat — read this before interpreting any result

The paper's own numbers say the effect is **scale-dependent**:

- Franka nut-and-bolt at 1M envs: SGS 70%, uniform 6%, PLR 5%.
- UR5e rod-in-hole at 32K envs: SGS succeeds, uniform and PLR reach 0 — but at
  256K envs **all methods reach ~98%**.
- The paper states SGS performs *"comparably to uniform sampling on simpler
  tasks"*, and does not test below 4,000 parallel environments.

We are running 64–128. So the honest expectation for this testbed is **no
significant gap**, and that is a finding, not a failure. What makes the
difference appear at all is the shape of the configuration pool:

1. **There must be a real difficulty gradient.** robosuite's stock Lift
   randomises the cube over a 6 cm x 6 cm box — every configuration is about
   equally hard, so there is no frontier and SGS provably cannot help. The
   default here is `--cube-range 0.20` (a 40 cm x 40 cm box), which spans
   "directly under the gripper" to "at the edge of the Panda's reach".
2. **A sizeable fraction of the pool must be unsolved at any given moment.**
   If uniform sampling already solves everything, there is nothing to reallocate.
   Watch the `frontier_frac` column (fraction of configurations with
   `0.2 <= p_i <= 0.8`). If it is pinned at 0.0 or 1.0 for the whole run, the
   experiment has no headroom and the pool needs widening or narrowing.
3. **Dense shaping shrinks the gap.** SGS targets the exploration bottleneck;
   dense reward *is* an exploration aid. `--sparse-reward` makes the task much
   harder and the comparison much more paper-like — try it once the dense
   baseline works.

If you want the paper's regime rather than an analogy to it, robosuite is the
wrong simulator: it is CPU MuJoCo, one env per process. You would need a
GPU-parallel simulator (MuJoCo MJX, ManiSkill3, Isaac Lab) to reach thousands
of environments.

---

## Measured throughput (Linux server, 256 cores)

These are the numbers from the Linux box. **Run `scripts/bench.py` to get your
own** — on a 12-core M2 Pro expect roughly 8 parallel environments and, by
extrapolation from the single-env rate, somewhere near 850 env-steps/s for the
full training loop, i.e. about 20 minutes per 1M environment steps. That is an
estimate, not a measurement.

"env-steps/s" below means robosuite control steps per second; each one is 25
MuJoCo physics steps at 20 Hz control.

Raw simulator, no learning (`scripts/bench.py`):

| parallel envs | env-steps/s | 1M steps |
|---|---|---|
| 1 | 137 | 122 min |
| 8 | 921 | 18 min |
| 32 | 2,126 | 8 min |
| 64 | 3,543 | 5 min |
| 128 | 5,062 | 3 min |

Full training loop, including PPO updates (`python -m sgs.train`):

| parallel envs | env-steps/s | 1M steps | 5M steps |
|---|---|---|---|
| 16 | 650 | 26 min | 2.1 h |
| 32 | 975 | 17 min | 1.4 h |
| 64 | 1,503 | 11 min | 0.9 h |
| 128 | 2,142 | 8 min | 0.6 h |

Scaling is sublinear because the trainer's main process is a serial bottleneck
(one synchronous pipe round-trip to every worker per step). Running several
independent runs concurrently scales much better than growing one run, which is
exactly what a seeds-and-hyperparameters sweep wants anyway.

### Why this runs on CPU

Measured here, on an A100, for the default 2x256 MLP:

| | policy forward, 64 steps | PPO update (10 epochs) |
|---|---|---|
| CPU | 0.03 s | 0.37 s |
| CUDA | 0.94 s | 2.04 s |

The network is far too small to amortise kernel-launch latency, so the GPU is
roughly **5x slower end-to-end**. `--device cpu` is the default. Also note torch
defaults to one thread per core (128 here), which is pure overhead at this size;
`--torch-threads 8` is the default.

The A100s only start to matter if the network grows a lot or the simulator moves
onto the GPU.

---

## Knobs that actually matter

| flag | default | note |
|---|---|---|
| `--cube-range` | 0.20 | half-width of the cube's xy box, metres. **The single most important knob.** 0.03 reproduces stock robosuite (no difficulty gradient). |
| `--n-configs` | 64 | pool size `N`. |
| `--sgs-history` | 50 | `H`. Lower = faster adaptation, noisier `p_i`. |
| `--sgs-target` | 0.5 | `t`. 0.66 is the paper's locomotion value. |
| `--sgs-kappa` | 1.0 | kernel sharpness. |
| `--sparse-reward` | off | turns off dense shaping. Much harder, much more paper-like. |
| `--num-envs` | auto | `cpu_count - 4`, clamped to [4, 64]. See the throughput table. |
| `--terminate-on-success` | — | not exposed as a flag; set in `sgs/env.py` if you want shorter episodes. |

---

## Files

| file | what it holds |
|---|---|
| `sgs/configs.py` | the task-configuration space and the fixed pool |
| `sgs/sampler.py` | `UniformSampler` and `SGSSampler` (paper Eqs. 1 and 2) |
| `sgs/env.py` | robosuite Lift wrapped so it can be reset to a given configuration |
| `sgs/vec.py` | subprocess vector env that does **not** auto-reset |
| `sgs/ppo.py` | PPO (CleanRL-style), GAE with explicit truncation bootstrapping |
| `sgs/train.py` | the training loop that ties sampler + env + PPO together |
| `setup.sh` | create the virtualenv (macOS or Linux) |
| `scripts/launch.sh` | start one run in a tmux window |
| `scripts/bench.py` | simulator throughput vs worker count |
| `scripts/plot.py` | learning curves and the per-configuration success heatmap |
| `scripts/run_comparison.sh` | 2 samplers x N seeds, each in its own tmux window |

### Environment notes

- `robosuite 1.5.2` + `mujoco 3.3.7`. **mujoco must stay below 3.16** — 3.16
  breaks robosuite's `get_joint_qpos_addr` bindings.
- `sgs/env.py` forces `MUJOCO_GL=disable` before importing robosuite. This is
  needed on the Linux box (no EGL/OSMesa libraries installed) and harmless
  everywhere else, since state observations never build a GL context. It is
  forced rather than defaulted because importing mujoco *rewrites* `MUJOCO_GL`
  to the backend it picked, and spawned worker processes would inherit that
  rewritten value and die on import. For rendering: `SGS_MUJOCO_GL=cgl` on
  macOS, `egl` on Linux (with GL libraries installed).

## Emily experiments

See [outputs/README.md](outputs/README.md) for the uniform and SGS Lift/Stack
code, run instructions, recorded results, and evaluation limitations.
