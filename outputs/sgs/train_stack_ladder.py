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

    def __init__(self, mode, n, history, rng, empty_p, cells=None, kappa=1., fresh=False):
        self.mode, self.n, self.rng = mode, n, rng
        # fresh: draw the least-used start (uniform: over all starts; SGS with cells: inside
        # the chosen cell), so a large pool approximates a fresh start at every reset.
        self.fresh, self.uses = fresh, np.zeros(n, int)
        # cells: optional int array (n,) mapping configs to cells; SGS then keeps one
        # success history per cell, samples a cell, then a config uniformly inside it.
        self.cells = None if cells is None else np.asarray(cells)
        n_hist = n if self.cells is None else int(self.cells.max()) + 1
        self.sgs = SGSSampler(configurations=n_hist, history_size=history, kappa=kappa)
        self.empty_p = empty_p
        # PLR (Jiang et al. 2021) state: score = mean |GAE| of the config's latest transitions.
        self.score, self.last = np.zeros(n), np.zeros(n)
        self.seen, self.clock = np.zeros(n, bool), 0

    def plr_replay_probs(self, beta=.1, rho=.1):
        idx = np.flatnonzero(self.seen)
        ranks = np.empty(len(idx))
        ranks[np.argsort(-self.score[idx])] = np.arange(1, len(idx) + 1)
        ps = (1 / ranks) ** (1 / beta); ps /= ps.sum()
        stale = self.clock - self.last[idx]
        pc = stale / stale.sum() if stale.sum() > 0 else np.full(len(idx), 1 / len(idx))
        out = np.zeros(self.n)
        out[idx] = (1 - rho) * ps + rho * pc
        return out

    def update_scores(self, cfg, abs_adv):
        for c in np.unique(cfg):
            self.score[c] = abs_adv[cfg == c].mean()

    def probs(self):
        if self.mode == 'uniform':
            return np.full(self.n, 1 / self.n)
        if self.mode == 'plr':
            if not self.seen.any():
                return np.full(self.n, 1 / self.n)
            unseen = ~self.seen
            q = .5 if unseen.any() else 0.     # replay vs. visit an unseen config
            out = (1 - q) * self.plr_replay_probs()
            if unseen.any():
                out[unseen] += q / unseen.sum()
            return out
        if self.empty_p == .5:
            q = np.asarray(self.sgs.probabilities())
        else:
            # Same equations with a different value for unvisited configurations.
            p = np.array([sum(h) / len(h) if h else self.empty_p for h in self.sgs.history])
            eps = 1e-4
            w = (p + eps) ** (.5 * self.sgs.kappa) * (1 - p + eps) ** (.5 * self.sgs.kappa)
            logits = np.log(w + eps) / self.sgs.temperature
            e = np.exp(logits - logits.max())
            q = e / e.sum()
        if self.cells is None:
            return q
        size = np.bincount(self.cells, minlength=len(q))
        return q[self.cells] / size[self.cells]

    def _least_used(self, members):
        u = self.uses[members]
        c = int(self.rng.choice(members[u == u.min()]))
        self.uses[c] += 1
        return c

    def sample(self, k):
        if self.fresh and self.mode == 'uniform':
            return np.array([self._least_used(np.arange(self.n)) for _ in range(k)])
        if self.fresh and self.mode == 'sgs' and self.cells is not None:
            p = self.probs()
            q = np.bincount(self.cells, weights=p)          # probability of each cell
            return np.array([self._least_used(np.flatnonzero(self.cells == self.rng.choice(len(q), p=q / q.sum())))
                             for _ in range(k)])
        if self.mode != 'plr':
            return self.rng.choice(self.n, size=k, p=self.probs())
        out = []
        for _ in range(k):
            c = int(self.rng.choice(self.n, p=self.probs()))
            self.clock += 1
            self.last[c], self.seen[c] = self.clock, True
            self.uses[c] += 1
            out.append(c)
        return np.array(out)

    def update(self, cid, success):
        self.sgs.update(int(cid if self.cells is None else self.cells[cid]), bool(success))

    def rates(self):
        r = np.array([sum(h) / len(h) if h else np.nan for h in self.sgs.history])
        return r if self.cells is None else r[self.cells]


# --------------------------------------------------------------------- eval
@torch.no_grad()
def evaluate(venv, agent, obs_rms, eval_ids):
    """Each listed configuration once, deterministic actions; success aligned with eval_ids."""
    pos = {c: k for k, c in enumerate(eval_ids)}
    success = np.zeros(len(eval_ids), bool)
    queue = list(eval_ids)
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
                success[pos[active[i]]] = bool(term[i])
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
    p.add_argument('--sampler', choices=['uniform', 'sgs', 'plr'], required=True)
    p.add_argument('--name', required=True)
    p.add_argument('--steps', type=int, default=300_000)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-envs', type=int, default=4)
    p.add_argument('--num-steps', type=int, default=128, help='rollout per env; batch = envs x steps')
    p.add_argument('--horizon', type=int, default=200, help='cap on the per-start time limit')
    p.add_argument('--slack', type=int, default=60, help='time limit = script steps remaining + slack')
    p.add_argument('--sgs-history', type=int, default=20)
    p.add_argument('--sgs-empty-p', type=float, default=0., help='p for unvisited configs (paper: 0)')
    p.add_argument('--sgs-kappa', type=float, default=1., help='Beta-kernel concentration (paper: 1 manip, 5 loco)')
    p.add_argument('--sgs-cells', action='store_true',
                   help='SGS success estimates per cell (sgs.stack_omnireset.cell_key), not per config')
    p.add_argument('--cell-field', default='cell', help="pool metadata field with each start's cell")
    p.add_argument('--min-cell-size', type=int, default=20,
                   help='cells with fewer training starts are folded into their family\'s largest cell')
    p.add_argument('--fresh', action='store_true',
                   help='uniform / cell-SGS draw the least-used start (fresh starts from a large pool)')
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
    labels = manifest['labels']
    label_all = np.array([m['label'] for m in meta])
    split = np.array([m.get('split', 'train') for m in meta])
    train_ids = np.flatnonzero(split == 'train')
    # Pools with an 'eval' split evaluate on those fresh configurations only.
    eval_ids = np.flatnonzero(split == 'eval') if (split == 'eval').any() else train_ids
    N = len(train_ids)
    label_of = label_all[train_ids]
    eval_labels = label_all[eval_ids]
    train_labels = [l for l in labels if (label_of == l).any()]
    (out / 'args.json').write_text(json.dumps(dict(vars(args), pool=str(args.pool.resolve()),
                                                   n_configs=N, n_eval=len(eval_ids), labels=labels), indent=1) + '\n')
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(args.seed)
    cells = None
    if args.sgs_cells:
        from sgs.stack_omnireset import cell_key
        keys = [meta[i][args.cell_field] if args.cell_field in meta[i] else cell_key(meta[i]) for i in train_ids]
        # Fold cells with too few starts into the largest cell of the same family; otherwise
        # SGS would replay a tiny cell's few starts many times (memorisation).
        count = {k: keys.count(k) for k in set(keys)}
        fam = {i: meta[train_ids[j]]['label'] for j, i in enumerate(keys)}
        for small in [k for k, c in count.items() if c < args.min_cell_size]:
            same = [k for k in count if fam[k] == fam[small] and count[k] >= args.min_cell_size]
            if same:
                big = max(same, key=count.get)
                keys = [big if k == small else k for k in keys]
                print(f'[{args.name}] cell {small} ({count[small]} starts) folded into {big}', flush=True)
        names = sorted(set(keys))
        cells = np.array([names.index(k) for k in keys])
        print(f'[{args.name}] {len(names)} SGS cells, sizes {np.bincount(cells).tolist()}', flush=True)
    sampler = Sampler(args.sampler, N, args.sgs_history, rng, args.sgs_empty_p, cells, args.sgs_kappa, args.fresh)

    venv = VecEnv(args.num_envs, args.pool.resolve(), args.horizon, args.seed, args.slack)
    E, T = args.num_envs, args.num_steps
    agent = Agent(venv.obs_dim, venv.act_dim, args.hidden, args.log_std_init)
    opt = torch.optim.Adam(agent.parameters(), lr=args.lr, eps=1e-5)
    obs_rms = RunningMeanStd((venv.obs_dim,))
    scaler = RunningReturnScaler(E, args.gamma)
    n_updates = args.steps // (E * T)

    cur = sampler.sample(E)
    obs = venv.reset(range(E), train_ids[cur])
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
                succ = evaluate(venv, agent, obs_rms, eval_ids)
                by = {l: float(succ[eval_labels == l].mean()) for l in labels if (eval_labels == l).any()}
                evals.append(dict(step=step, all=float(succ.mean()), by_label=by,
                                  per_config=succ.astype(int).tolist()))
                print(f'[{args.name}] eval @ {step:>7}: all={succ.mean():.3f}  ' +
                      ' '.join(f'{l}={v:.2f}' for l, v in by.items()), flush=True)
                next_eval += args.eval_every
                cur = sampler.sample(E)
                obs = venv.reset(range(E), train_ids[cur])
                scaler.ret[:] = 0
            B = dict(obs=np.zeros((T, E, venv.obs_dim), np.float32), act=np.zeros((T, E, venv.act_dim), np.float32),
                     logp=np.zeros((T, E), np.float32), val=np.zeros((T, E), np.float32),
                     rew=np.zeros((T, E), np.float32), done=np.zeros((T, E), np.float32),
                     term=np.zeros((T, E), np.float32), finalv=np.zeros((T, E), np.float32),
                     cfg=np.zeros((T, E), int))
            for t in range(T):
                obs_rms.update(obs)
                o = obs_rms.normalize(obs)
                with torch.no_grad():
                    a, lp, _, v = agent.act(torch.as_tensor(o))
                nobs, rew, term, trunc = venv.step(a.numpy())
                done = term | trunc
                B['obs'][t], B['act'][t], B['logp'][t], B['val'][t] = o, a.numpy(), lp.numpy(), v.numpy()
                B['cfg'][t] = cur
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
                    nobs[ids] = venv.reset(ids, train_ids[cur[ids]])
                obs = nobs
            step += E * T
            with torch.no_grad():
                next_v = agent.value(torch.as_tensor(obs_rms.normalize(obs))).numpy()
            nv = np.vstack([B['val'][1:], next_v[None]])
            boot = np.where(B['done'] > 0, B['finalv'], nv) * (1 - B['term'])
            adv, ret = compute_gae(B['rew'], B['val'], boot, B['done'], args.gamma, .95)
            if args.sampler == 'plr':
                sampler.update_scores(B['cfg'], np.abs(adv))
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
                       **{f'share_{l}': share[l] / max(1, len(recent)) for l in train_labels},
                       **{f'train_{l}': np.mean(tsucc[l]) if tsucc[l] else np.nan for l in train_labels},
                       **{f'prob_{l}': float(probs[label_of == l].sum()) for l in train_labels})
            if log is None:
                log = csv.DictWriter(log_f, fieldnames=list(row))
                log.writeheader()
            log.writerow(row)
            log_f.flush()
            if update % 20 == 0:
                print(f'[{args.name}] {step:>7} sps={row["sps"]} train={row["train_success"]:.2f} '
                      f'frontier={row["frontier_frac"]:.2f} H={row["sampler_entropy"]:.2f}', flush=True)
        succ = evaluate(venv, agent, obs_rms, eval_ids)
        by = {l: float(succ[eval_labels == l].mean()) for l in labels if (eval_labels == l).any()}
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
