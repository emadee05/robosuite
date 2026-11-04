"""Uniform vs SGS PPO on the full-task Stack ladder (sgs.stack_task_ladder).

The only difference between two runs with the same seed is --sampler. The
trainer, not the workers, chooses every reset configuration. PPO is the Lift
ladder's CleanRL-style implementation (uniform.ladder_ppo, from f628446).

    ../robosuite/.venv/bin/python -m sgs.train_stack_ladder --pool stack_task_ladders/<id> \
        --sampler sgs --steps 300000 --seed 0 --name sgs_s0
"""
import os
for _key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
             'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_key] = '1'
import argparse
import csv
import json
import multiprocessing as mp
import time
from collections import deque, defaultdict
from pathlib import Path

import numpy as np
import torch

from uniform.ladder_ppo import Agent, RunningMeanStd, RunningReturnScaler, compute_gae, ppo_update
from sgs.sgs_sampler import SGSSampler

RUNS = Path(__file__).resolve().parent.parent / 'stack_task_ladder_runs'


# ------------------------------------------------------------------ vec env
def _worker(remote, pool, horizon, seed, slack):
    from sgs.stack_task_ladder import StackLadderEnv
    env = StackLadderEnv(pool, horizon=horizon, seed=seed, slack=slack)
    try:
        while True:
            cmd, data = remote.recv()
            if cmd == 'step':
                remote.send(env.step(data))
            elif cmd == 'reset':
                remote.send(env.reset(options={'configuration_id': data})[0])
            elif cmd == 'spec':
                remote.send((env.observation_space.shape[0], env.action_space.shape[0]))
            else:
                break
    except (KeyboardInterrupt, EOFError):
        pass
    finally:
        env.close()


class VecEnv:
    def __init__(self, n, pool, horizon, seed, slack):
        ctx = mp.get_context('spawn')
        self.remotes, work = zip(*[ctx.Pipe() for _ in range(n)])
        self.procs = [ctx.Process(target=_worker, args=(w, str(pool), horizon, seed * 1000 + i, slack),
                                  daemon=True) for i, w in enumerate(work)]
        for p in self.procs:
            p.start()
        self.remotes[0].send(('spec', None))
        self.obs_dim, self.act_dim = self.remotes[0].recv()
        self.n = n

    def reset(self, ids, cfgs):
        for i, c in zip(ids, cfgs):
            self.remotes[i].send(('reset', int(c)))
        return np.stack([self.remotes[i].recv() for i in ids])

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(('step', a))
        obs, rew, term, trunc, _ = zip(*[r.recv() for r in self.remotes])
        return np.stack(obs), np.array(rew, np.float32), np.array(term), np.array(trunc)

    def close(self):
        for r in self.remotes:
            try:
                r.send(('close', None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=5)


# ------------------------------------------------------------------ sampler
class Sampler:
    """Uniform or SGS (paper Eqs. 1-2 via SGSSampler) over N configurations."""

    def __init__(self, mode, n, history, rng, empty_p):
        self.mode, self.n, self.rng = mode, n, rng
        self.sgs = SGSSampler(configurations=n, history_size=history)
        self.empty_p = empty_p

    def probs(self):
        if self.mode == 'uniform':
            return np.full(self.n, 1 / self.n)
        if self.empty_p == .5:
            return np.asarray(self.sgs.probabilities())
        # Same equations with a different value for unvisited configurations.
        p = np.array([sum(h) / len(h) if h else self.empty_p for h in self.sgs.history])
        eps = 1e-4
        w = (p + eps) ** (.5 * self.sgs.kappa) * (1 - p + eps) ** (.5 * self.sgs.kappa)
        logits = np.log(w + eps) / self.sgs.temperature
        e = np.exp(logits - logits.max())
        return e / e.sum()

    def sample(self, k):
        return self.rng.choice(self.n, size=k, p=self.probs())

    def update(self, cid, success):
        self.sgs.update(int(cid), bool(success))

    def rates(self):
        return np.array([sum(h) / len(h) if h else np.nan for h in self.sgs.history])


# --------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(venv, agent, obs_rms, n_configs, horizon):
    """Each configuration once, deterministic actions; returns success per config."""
    success = np.zeros(n_configs, bool)
    queue = list(range(n_configs))
    active = {}
    for i in range(venv.n):
        if queue:
            active[i] = queue.pop(0)
    obs = np.zeros((venv.n, venv.obs_dim), np.float32)
    ids = list(active)
    obs[ids] = venv.reset(ids, [active[i] for i in ids])
    while active:
        a = agent.act(torch.as_tensor(obs_rms.normalize(obs)), deterministic=True)[0].numpy()
        nobs, _, term, trunc = venv.step(a)  # idle workers step harmlessly
        for i in list(active):
            if term[i] or trunc[i]:
                success[active[i]] = bool(term[i])
                if queue:
                    active[i] = queue.pop(0)
                    nobs[i] = venv.reset([i], [active[i]])[0]
                else:
                    del active[i]
        obs = nobs
    return success


# --------------------------------------------------------------------- main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pool', type=Path, required=True)
    p.add_argument('--sampler', choices=['uniform', 'sgs'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--steps', type=int, default=300_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-envs', type=int, default=4)
    p.add_argument('--num-steps', type=int, default=128, help='rollout per env; batch = envs x steps')
    p.add_argument('--horizon', type=int, default=200, help='cap on the per-start time limit')
    p.add_argument('--slack', type=int, default=60, help='time limit = script steps remaining + slack')
    p.add_argument('--sgs-history', type=int, default=20)
    p.add_argument('--sgs-empty-p', type=float, default=0., help='p for unvisited configs (paper: 0)')
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--update-epochs', type=int, default=10)
    p.add_argument('--num-minibatches', type=int, default=8)
    p.add_argument('--target-kl', type=float, default=.05)
    p.add_argument('--gamma', type=float, default=.99)
    p.add_argument('--ent-coef', type=float, default=0.)
    p.add_argument('--log-std-init', type=float, default=-1.5)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--eval-every', type=int, default=25_000, help='transitions between full evals')
    args = p.parse_args()

    out = RUNS / args.name
    out.mkdir(parents=True, exist_ok=False)
    manifest = json.loads((args.pool / 'pool.json').read_text())
    meta = manifest['configurations']
    N = len(meta)
    labels = manifest['labels']
    label_of = np.array([m['label'] for m in meta])
    (out / 'args.json').write_text(json.dumps(dict(vars(args), pool=str(args.pool.resolve()),
                                                   n_configs=N, labels=labels), indent=1) + '\n')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    sampler = Sampler(args.sampler, N, args.sgs_history, rng, args.sgs_empty_p)

    venv = VecEnv(args.num_envs, args.pool.resolve(), args.horizon, args.seed, args.slack)
    E, T = args.num_envs, args.num_steps
    agent = Agent(venv.obs_dim, venv.act_dim, args.hidden, args.log_std_init)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    obs_rms = RunningMeanStd((venv.obs_dim,))
    scaler = RunningReturnScaler(E, args.gamma)
    n_updates = args.steps // (E * T)

    cur = sampler.sample(E)
    obs = venv.reset(range(E), cur)
    recent = deque(maxlen=200)          # (label, success)
    log_f = open(out / 'log.csv', 'w', newline='')
    log = None
    evals = []
    next_eval, step, t0 = 0, 0, time.time()
    try:
        for update in range(1, n_updates + 1):
            for g in opt.param_groups:
                g['lr'] = args.lr * (1 - (update - 1) / n_updates)
            if step >= next_eval:
                succ = evaluate(venv, agent, obs_rms, N, args.horizon)
                by = {l: float(succ[label_of == l].mean()) for l in labels}
                evals.append(dict(step=step, all=float(succ.mean()), by_label=by,
                                  per_config=succ.astype(int).tolist()))
                print(f'[{args.name}] eval @ {step:>7}: all={succ.mean():.3f}  ' +
                      ' '.join(f'{l}={v:.2f}' for l, v in by.items()), flush=True)
                next_eval += args.eval_every
                cur = sampler.sample(E)
                obs = venv.reset(range(E), cur)
                scaler.ret[:] = 0
            B = dict(obs=np.zeros((T, E, venv.obs_dim), np.float32), act=np.zeros((T, E, venv.act_dim), np.float32),
                     logp=np.zeros((T, E), np.float32), val=np.zeros((T, E), np.float32),
                     rew=np.zeros((T, E), np.float32), done=np.zeros((T, E), np.float32),
                     term=np.zeros((T, E), np.float32), finalv=np.zeros((T, E), np.float32))
            for t in range(T):
                obs_rms.update(obs)
                o = obs_rms.normalize(obs)
                with torch.no_grad():
                    a, lp, _, v = agent.act(torch.as_tensor(o))
                nobs, rew, term, trunc = venv.step(a.numpy())
                done = term | trunc
                B['obs'][t], B['act'][t], B['logp'][t], B['val'][t] = o, a.numpy(), lp.numpy(), v.numpy()
                B['rew'][t] = scaler(rew, done)
                B['done'][t], B['term'][t] = done, term
                ids = np.flatnonzero(done)
                if len(ids):
                    with torch.no_grad():
                        B['finalv'][t, ids] = agent.value(torch.as_tensor(obs_rms.normalize(nobs[ids]))).numpy()
                    for i in ids:
                        sampler.update(cur[i], term[i])
                        recent.append((label_of[cur[i]], bool(term[i])))
                    cur[ids] = sampler.sample(len(ids))
                    nobs[ids] = venv.reset(ids, cur[ids])
                obs = nobs
            step += E * T
            with torch.no_grad():
                next_v = agent.value(torch.as_tensor(obs_rms.normalize(obs))).numpy()
            nv = np.vstack([B['val'][1:], next_v[None]])
            boot = np.where(B['done'] > 0, B['finalv'], nv) * (1 - B['term'])
            adv, ret = compute_gae(B['rew'], B['val'], boot, B['done'], args.gamma, .95)
            f = lambda x, *s: torch.as_tensor(x.reshape(E * T, *s))
            stats = ppo_update(agent, opt, (f(B['obs'], venv.obs_dim), f(B['act'], venv.act_dim), f(B['logp']),
                                            f(adv), f(ret), f(B['val'])),
                               update_epochs=args.update_epochs, num_minibatches=args.num_minibatches,
                               target_kl=args.target_kl, ent_coef=args.ent_coef)
            probs = sampler.probs()
            rates = sampler.rates()
            share = defaultdict(int); tsucc = defaultdict(list)
            for l, s in recent:
                share[l] += 1; tsucc[l].append(s)
            row = dict(update=update, step=step, time_s=round(time.time() - t0, 1),
                       sps=round(step / (time.time() - t0)),
                       train_success=np.mean([s for _, s in recent]) if recent else 0,
                       eval_all=evals[-1]['all'],
                       sampler_entropy=float(-(probs * np.log(probs)).sum()),
                       frontier_frac=float(np.mean((rates >= .2) & (rates <= .8))),
                       **{k: round(v, 5) for k, v in stats.items()},
                       **{f'share_{l}': share[l] / max(1, len(recent)) for l in labels},
                       **{f'train_{l}': np.mean(tsucc[l]) if tsucc[l] else np.nan for l in labels},
                       **{f'prob_{l}': float(probs[label_of == l].sum()) for l in labels})
            if log is None:
                log = csv.DictWriter(log_f, fieldnames=list(row))
                log.writeheader()
            log.writerow(row)
            log_f.flush()
            if update % 20 == 0:
                print(f'[{args.name}] {step:>7} sps={row["sps"]} train={row["train_success"]:.2f} '
                      f'frontier={row["frontier_frac"]:.2f} H={row["sampler_entropy"]:.2f}', flush=True)
        succ = evaluate(venv, agent, obs_rms, N, args.horizon)
        by = {l: float(succ[label_of == l].mean()) for l in labels}
        evals.append(dict(step=step, all=float(succ.mean()), by_label=by, per_config=succ.astype(int).tolist()))
        print(f'[{args.name}] FINAL eval @ {step}: all={succ.mean():.3f}  ' +
              ' '.join(f'{l}={v:.2f}' for l, v in by.items()), flush=True)
        torch.save(dict(agent=agent.state_dict(), obs_rms=obs_rms.state_dict(), step=step,
                        sampler_history=[list(h) for h in sampler.sgs.history]), out / 'ckpt.pt')
    finally:
        (out / 'evals.json').write_text(json.dumps(evals) + '\n')
        log_f.close()
        venv.close()


if __name__ == '__main__':
    main()
