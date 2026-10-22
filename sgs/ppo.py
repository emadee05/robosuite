"""A small, readable PPO (CleanRL-style) for continuous control.

Nothing here knows about SGS.  The only non-textbook detail is that the trainer
hands us an explicit bootstrap value per timestep, because our vector env does
*not* auto-reset: episodes end by hitting the horizon (truncation), and the
return of a truncated episode has to be bootstrapped with V(final_obs) or the
critic learns that time running out is a catastrophe.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer


class RunningMeanStd:
    """Welford-style running statistics, used to normalise observations."""

    def __init__(self, shape, epsilon=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x):
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + delta**2 * self.count * batch_count / tot) / tot
        self.count = tot

    def normalize(self, x, clip=10.0):
        return np.clip(
            (x - self.mean) / np.sqrt(self.var + 1e-8), -clip, clip
        ).astype(np.float32)

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}


class RunningReturnScaler:
    """Divides rewards by the running std of the discounted return.

    Dense Lift rewards are ~1 per step, so an undiscounted 200-step return is
    ~200 and the critic spends its first thousand updates just learning the
    scale.  This keeps value targets near unit variance.
    """

    def __init__(self, num_envs, gamma=0.99):
        self.ret = np.zeros(num_envs, dtype=np.float64)
        self.rms = RunningMeanStd(())
        self.gamma = gamma

    def __call__(self, rewards, dones):
        self.ret = self.ret * self.gamma + rewards
        self.rms.update(self.ret)
        scaled = rewards / np.sqrt(self.rms.var + 1e-8)
        self.ret[dones] = 0.0
        return scaled.astype(np.float32)


class Agent(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256, log_std_init=-0.5):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        # State-independent action std, as is standard for PPO on MuJoCo.
        self.actor_logstd = nn.Parameter(torch.full((1, act_dim), float(log_std_init)))

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    def act(self, obs, action=None, deterministic=False):
        mean = self.actor_mean(obs)
        if deterministic:
            return mean, None, None, self.value(obs)
        std = self.actor_logstd.expand_as(mean).exp()
        dist = Normal(mean, std)
        if action is None:
            action = dist.sample()
        return (
            action,
            dist.log_prob(action).sum(-1),
            dist.entropy().sum(-1),
            self.value(obs),
        )


def compute_gae(rewards, values, bootstrap_values, dones, gamma, gae_lambda):
    """Standard GAE, with an explicit per-step bootstrap value.

    bootstrap_values[t] must already be 0 for genuinely terminal steps and
    V(final_obs) for truncated ones.  `dones` only controls where the advantage
    recursion is cut.
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float32)
    lastgaelam = np.zeros(N, dtype=np.float32)
    for t in reversed(range(T)):
        delta = rewards[t] + gamma * bootstrap_values[t] - values[t]
        lastgaelam = delta + gamma * gae_lambda * (1.0 - dones[t]) * lastgaelam
        advantages[t] = lastgaelam
    return advantages, advantages + values


def ppo_update(
    agent,
    optimizer,
    batch,
    clip_coef=0.2,
    ent_coef=0.0,
    vf_coef=0.5,
    max_grad_norm=0.5,
    update_epochs=10,
    num_minibatches=32,
    target_kl=0.02,
):
    obs, actions, logprobs, advantages, returns, values = batch
    n = obs.shape[0]
    mb_size = max(1, n // num_minibatches)
    idx = np.arange(n)
    stats = {}
    for epoch in range(update_epochs):
        np.random.shuffle(idx)
        approx_kls = []
        for start in range(0, n, mb_size):
            mb = idx[start : start + mb_size]
            _, newlogprob, entropy, newvalue = agent.act(obs[mb], actions[mb])
            logratio = newlogprob - logprobs[mb]
            ratio = logratio.exp()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - logratio).mean()
                approx_kls.append(approx_kl.item())

            mb_adv = advantages[mb]
            mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
            pg_loss = torch.max(
                -mb_adv * ratio,
                -mb_adv * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef),
            ).mean()
            v_loss = 0.5 * ((newvalue - returns[mb]) ** 2).mean()
            entropy_loss = entropy.mean()
            loss = pg_loss - ent_coef * entropy_loss + vf_coef * v_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

        stats = {
            "policy_loss": pg_loss.item(),
            "value_loss": v_loss.item(),
            "entropy": entropy_loss.item(),
            "approx_kl": float(np.mean(approx_kls)),
            "epochs_run": epoch + 1,
        }
        if target_kl is not None and np.mean(approx_kls) > target_kl:
            break
    return stats
