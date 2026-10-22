"""Train PPO on robosuite Panda Lift with a chosen sampler over task configurations.

    python -m sgs.train --sampler uniform --total-steps 2000000 --run-name uniform_s0
    python -m sgs.train --sampler sgs     --total-steps 2000000 --run-name sgs_s0

The *only* difference between the two runs is which configuration each episode
resets to.  Everything else -- policy, optimiser, reward, configuration pool,
seed -- is identical, which is what makes the comparison mean anything.

The headline number is `eval/success`: mean success over the configuration pool
under a *uniform* distribution, measured with deterministic actions.  Do not
compare `train/success` across samplers -- SGS deliberately trains on hard
configurations, so its training success rate is lower by construction even when
the policy is better.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .configs import ConfigSpace, difficulty_proxy, sample_config_set
from .ppo import (
    Agent,
    RunningMeanStd,
    RunningReturnScaler,
    compute_gae,
    ppo_update,
)
from .sampler import make_sampler
from .vec import ConfigVecEnv


def default_num_envs() -> int:
    """One worker per core, less 4 for the trainer and the OS, clamped to [4, 64].

    256-core server -> 64 (the clamp; past ~128 the serial main loop dominates).
    12-core M2 Pro   -> 8.
    The resolved value is recorded in args.json, so runs stay reproducible
    across machines even though the default is not.
    """
    return min(64, max(4, (os.cpu_count() or 8) - 4))


def default_torch_threads() -> int:
    """cpu_count // 3, clamped to [2, 8].

    torch otherwise takes one thread per core, which is pure overhead for a
    2x256 MLP. The env workers are idle during the PPO update and torch is idle
    during the rollout, so a few threads cost little contention.
    256-core server -> 8.  12-core M2 Pro -> 4.
    """
    return min(8, max(2, (os.cpu_count() or 8) // 3))


def parse_args():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # --- experiment
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--out-dir", type=str, default="runs")
    p.add_argument("--seed", type=int, default=0)
    # CPU, deliberately.  A 2x256 MLP on batches of a few thousand is launch-latency
    # bound on an A100: measured here, cuda is ~5x SLOWER end-to-end than cpu
    # (see README "Why this runs on CPU").  Switch to cuda only if you grow the
    # network a lot or move to a GPU simulator.
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--torch-threads", type=int, default=None,
                   help="default: cpu_count//3, clamped to [2, 8]. torch "
                        "otherwise takes one thread per core, which is pure "
                        "overhead at this network size")
    # --- sampler
    p.add_argument("--sampler", choices=["uniform", "sgs"], default="uniform")
    p.add_argument("--n-configs", type=int, default=64, help="N: size of the fixed pool")
    p.add_argument("--config-seed", type=int, default=12345,
                   help="Seeds the pool itself. Keep FIXED across samplers and "
                        "across seeds, or you are comparing different task sets.")
    p.add_argument("--sgs-target", type=float, default=0.5, help="t")
    p.add_argument("--sgs-kappa", type=float, default=1.0, help="kappa")
    p.add_argument("--sgs-eps", type=float, default=1e-4, help="eps")
    p.add_argument("--sgs-temperature", type=float, default=2.0, help="T")
    p.add_argument("--sgs-history", type=int, default=50, help="H")
    p.add_argument("--sgs-warmup-episodes", type=int, default=0,
                   help="episodes drawn uniformly before SGS kicks in (0 = paper)")
    # --- configuration space
    p.add_argument("--cube-range", type=float, default=0.20,
                   help="cube x and y are uniform in [-r, +r] metres. robosuite's "
                        "own default is 0.03, which gives no difficulty gradient.")
    p.add_argument("--arm-noise", type=float, default=0.15)
    # --- env
    p.add_argument("--num-envs", type=int, default=None,
                   help="default: cpu_count-4, clamped to [4, 64]")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--control-freq", type=int, default=20)
    p.add_argument("--sparse-reward", action="store_true",
                   help="turn OFF robosuite's dense shaping (much harder)")
    # --- ppo
    p.add_argument("--total-steps", type=int, default=2_000_000)
    p.add_argument("--num-steps", type=int, default=64, help="rollout length per env")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false",
                   default=True, help="by default the learning rate anneals to 0")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip-coef", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--update-epochs", type=int, default=10)
    p.add_argument("--num-minibatches", type=int, default=8)
    p.add_argument("--target-kl", type=float, default=0.05)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--log-std-init", type=float, default=-0.5)
    p.add_argument("--no-reward-scaling", action="store_true")
    # --- eval
    p.add_argument("--eval-every", type=int, default=10, help="in PPO updates")
    p.add_argument("--eval-set", choices=["pool", "holdout"], default="pool")
    p.add_argument("--n-eval-configs", type=int, default=64,
                   help="only used when --eval-set holdout")
    return p.parse_args()


@torch.no_grad()
def evaluate(venv, agent, obs_rms, configs, device, horizon):
    """Deterministic rollout of every configuration exactly once.

    Returns (mean success, per-configuration success array).  An env stops being
    counted once it succeeds, and a chunk exits early when every env in it has,
    which makes eval cheap once the policy works.
    """
    n_env = venv.num_envs
    out = np.zeros(len(configs), dtype=np.float32)
    for start in range(0, len(configs), n_env):
        chunk = configs[start : start + n_env]
        k = len(chunk)
        pad = np.concatenate([chunk, np.repeat(chunk[-1:], n_env - k, axis=0)]) if k < n_env else chunk
        obs = venv.reset_to(pad)
        succeeded = np.zeros(n_env, dtype=bool)
        for _ in range(horizon):
            t = torch.as_tensor(obs_rms.normalize(obs), device=device)
            action, _, _, _ = agent.act(t, deterministic=True)
            obs, _, term, trunc, infos = venv.step(action.cpu().numpy())
            for i, info in enumerate(infos):
                if info:
                    succeeded[i] |= bool(info["is_success"])
            if (term | trunc).all():
                break
        out[start : start + k] = succeeded[:k]
    return float(out.mean()), out


def main():
    args = parse_args()
    # Resolve the machine-dependent defaults before anything records them.
    if args.num_envs is None:
        args.num_envs = default_num_envs()
    if args.torch_threads is None:
        args.torch_threads = default_torch_threads()
    print(f"[setup] {os.cpu_count()} cores -> --num-envs {args.num_envs} "
          f"--torch-threads {args.torch_threads} --device {args.device}")
    run_name = args.run_name or f"{args.sampler}_s{args.seed}_{int(time.time())}"
    out = Path(args.out_dir) / run_name
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")

    space = ConfigSpace(
        cube_x_range=(-args.cube_range, args.cube_range),
        cube_y_range=(-args.cube_range, args.cube_range),
        arm_dq_scale=args.arm_noise,
    )
    # The pool is seeded independently of --seed so that different training
    # seeds all face the same set of tasks.
    pool = sample_config_set(args.n_configs, space, seed=args.config_seed)
    eval_pool = (
        pool
        if args.eval_set == "pool"
        else sample_config_set(args.n_eval_configs, space, seed=args.config_seed + 777)
    )
    np.savez(out / "configs.npz", pool=pool, eval_pool=eval_pool,
             difficulty=difficulty_proxy(pool))

    sampler_kw = {}
    if args.sampler == "sgs":
        sampler_kw = dict(
            target=args.sgs_target, kappa=args.sgs_kappa, eps=args.sgs_eps,
            temperature=args.sgs_temperature, history=args.sgs_history,
            warmup_episodes=args.sgs_warmup_episodes,
        )
    sampler = make_sampler(args.sampler, args.n_configs, seed=args.seed, **sampler_kw)

    venv = ConfigVecEnv(
        args.num_envs,
        dict(horizon=args.horizon, control_freq=args.control_freq,
             reward_shaping=not args.sparse_reward),
        seed=args.seed * 1000,
    )
    obs_dim, act_dim = venv.obs_dim, venv.act_dim
    agent = Agent(obs_dim, act_dim, hidden=args.hidden,
                  log_std_init=args.log_std_init).to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    obs_rms = RunningMeanStd((obs_dim,))
    rew_scaler = None if args.no_reward_scaling else RunningReturnScaler(
        args.num_envs, args.gamma)

    T, N = args.num_steps, args.num_envs
    batch_size = T * N
    num_updates = max(1, args.total_steps // batch_size)

    obs_b = np.zeros((T, N, obs_dim), dtype=np.float32)
    act_b = np.zeros((T, N, act_dim), dtype=np.float32)
    logp_b = np.zeros((T, N), dtype=np.float32)
    rew_b = np.zeros((T, N), dtype=np.float32)
    val_b = np.zeros((T, N), dtype=np.float32)
    done_b = np.zeros((T, N), dtype=np.float32)
    term_b = np.zeros((T, N), dtype=np.float32)
    finalv_b = np.zeros((T, N), dtype=np.float32)

    cur_idx = sampler.sample(N)
    obs = venv.reset_to(pool[cur_idx])
    ep_returns, ep_successes = [], []
    log_path = out / "log.csv"
    log_f = open(log_path, "w", buffering=1)
    header = ("update,global_step,time_s,sps,train_success,train_return,"
              "eval_success,sampler_entropy,frontier_frac,mean_p,"
              "policy_loss,value_loss,entropy,approx_kl")
    log_f.write(header + "\n")
    print(header.replace(",", " | "))

    running_ret = np.zeros(N, dtype=np.float64)
    p_history = []
    start_time = time.time()
    global_step = 0
    eval_success = float("nan")

    for update in range(1, num_updates + 1):
        if args.anneal_lr:
            frac = 1.0 - (update - 1.0) / num_updates
            optimizer.param_groups[0]["lr"] = frac * args.lr

        raw_obs_batch = np.zeros((T, N, obs_dim), dtype=np.float32)
        for t in range(T):
            raw_obs_batch[t] = obs
            obs_n = obs_rms.normalize(obs)
            obs_b[t] = obs_n
            with torch.no_grad():
                a, lp, _, v = agent.act(torch.as_tensor(obs_n, device=device))
            act_np = a.cpu().numpy()
            act_b[t], logp_b[t], val_b[t] = act_np, lp.cpu().numpy(), v.cpu().numpy()

            next_obs, rew, term, trunc, infos = venv.step(act_np)
            global_step += N
            done = term | trunc
            running_ret += rew
            rew_b[t] = rew_scaler(rew, done) if rew_scaler is not None else rew
            done_b[t], term_b[t] = done.astype(np.float32), term.astype(np.float32)

            done_ids = np.nonzero(done)[0]
            if len(done_ids):
                # V(final_obs) must be taken BEFORE the env is reset.
                with torch.no_grad():
                    fv = agent.value(
                        torch.as_tensor(obs_rms.normalize(next_obs[done_ids]),
                                        device=device))
                finalv_b[t, done_ids] = fv.cpu().numpy()
                for i in done_ids:
                    sampler.update(int(cur_idx[i]), infos[i]["is_success"])
                    ep_successes.append(float(infos[i]["is_success"]))
                    ep_returns.append(float(running_ret[i]))
                    running_ret[i] = 0.0
                new_idx = sampler.sample(len(done_ids))
                cur_idx[done_ids] = new_idx
                next_obs[done_ids] = venv.reset_to(pool[new_idx], done_ids)
            obs = next_obs

        obs_rms.update(raw_obs_batch.reshape(-1, obs_dim).astype(np.float64))
        with torch.no_grad():
            last_v = agent.value(
                torch.as_tensor(obs_rms.normalize(obs), device=device)
            ).cpu().numpy()

        next_vals = np.concatenate([val_b[1:], last_v[None]], axis=0)
        bootstrap = np.where(done_b > 0, finalv_b, next_vals) * (1.0 - term_b)
        adv, ret = compute_gae(rew_b, val_b, bootstrap, done_b,
                               args.gamma, args.gae_lambda)

        tt = lambda x: torch.as_tensor(x, device=device)
        stats = ppo_update(
            agent, optimizer,
            (tt(obs_b.reshape(-1, obs_dim)), tt(act_b.reshape(-1, act_dim)),
             tt(logp_b.reshape(-1)), tt(adv.reshape(-1)), tt(ret.reshape(-1)),
             tt(val_b.reshape(-1))),
            clip_coef=args.clip_coef, ent_coef=args.ent_coef, vf_coef=args.vf_coef,
            update_epochs=args.update_epochs, num_minibatches=args.num_minibatches,
            target_kl=args.target_kl,
        )
        finalv_b[:] = 0.0

        p = sampler.success_rates()
        probs = sampler.probs()
        entropy = float(-(probs * np.log(probs + 1e-12)).sum())
        # "Frontier" = configurations the policy solves between 20% and 80% of
        # the time. This is the mass SGS is trying to put its samples on.
        frontier = float(np.mean((p >= 0.2) & (p <= 0.8)))

        if update % args.eval_every == 0 or update == num_updates:
            eval_success, per_cfg = evaluate(
                venv, agent, obs_rms, eval_pool, device, args.horizon)
            np.save(out / "eval_per_config_latest.npy", per_cfg)
            cur_idx = sampler.sample(N)
            obs = venv.reset_to(pool[cur_idx])
            running_ret[:] = 0.0
            if rew_scaler is not None:
                rew_scaler.ret[:] = 0.0

        p_history.append(p.copy())
        elapsed = time.time() - start_time
        row = (
            f"{update},{global_step},{elapsed:.1f},{global_step/elapsed:.0f},"
            f"{np.mean(ep_successes[-200:]) if ep_successes else float('nan'):.4f},"
            f"{np.mean(ep_returns[-200:]) if ep_returns else float('nan'):.2f},"
            f"{eval_success:.4f},{entropy:.4f},{frontier:.4f},{p.mean():.4f},"
            f"{stats['policy_loss']:.4f},{stats['value_loss']:.4f},"
            f"{stats['entropy']:.4f},{stats['approx_kl']:.5f}"
        )
        log_f.write(row + "\n")
        print(row.replace(",", " | "), flush=True)

        if update % (args.eval_every * 5) == 0 or update == num_updates:
            torch.save({"agent": agent.state_dict(), "obs_rms": obs_rms.state_dict(),
                        "args": vars(args)}, out / "ckpt.pt")
            np.save(out / "p_history.npy", np.array(p_history, dtype=np.float32))

    np.save(out / "p_history.npy", np.array(p_history, dtype=np.float32))
    log_f.close()
    venv.close()
    print(f"\ndone -> {out}")


if __name__ == "__main__":
    main()
