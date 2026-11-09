"""CrossQ (Bhatt et al., ICLR 2024) on stacked parameters, in the same form as StackedSAC.

CrossQ removes the target network, puts batch renormalization in the critics and the actor, widens
the critics, and evaluates (s, a) and (s', a') in one joint forward pass, so the normalization sees
the mixture of both.

The update order follows sb3_contrib/crossq/crossq.py. Test T12 checks the match on identical data.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .optim import make_adam
from .sac import HALF_LOG_2PI, LOG_2

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0


@dataclass
class CrossQConfig:
    # CrossQ paper and its official train.py. The batch size matches the study's Table 5 and the paper.
    actor_lr: float = 1e-3
    critic_lr: float = 1e-3
    alpha_lr: float = 1e-3
    adam_beta1: float = 0.5  # actor and critic only, the temperature uses the Adam default of 0.9
    gamma: float = 0.99
    batch_size: int = 256
    policy_delay: int = 3
    actor_hidden: int = 256
    # The paper uses 2048. This study uses 1024, the sb3-contrib default, to fit its compute budget. It was checked
    # against sb3-contrib CrossQ at the same width.
    critic_hidden: int = 1024
    bn_momentum: float = 0.01  # 1 - 0.99 in the Flax convention of the official code
    bn_eps: float = 1e-3
    renorm_warmup: int = 100_000
    r_max: float = 3.0
    d_max: float = 5.0


class BatchRenormStats:
    """Running statistics of one batch renormalization layer, one row per network."""

    def __init__(self, N: int, features: int, device, dtype):
        self.mean = torch.zeros(N, 1, features, device=device, dtype=dtype)
        self.var = torch.ones(N, 1, features, device=device, dtype=dtype)
        self.steps = 0

    def state_dict(self) -> dict:
        return {"mean": self.mean.cpu(), "var": self.var.cpu(), "steps": self.steps}

    def load_state_dict(self, state: dict) -> None:
        self.mean.copy_(state["mean"].to(self.mean.device))
        self.var.copy_(state["var"].to(self.var.device))
        self.steps = state["steps"]


def _linear_params(in_dim: int, out_dim: int):
    layer = torch.nn.Linear(in_dim, out_dim)
    return layer.weight.detach().T.clone(), layer.bias.detach().unsqueeze(0).clone()


def init_networks(init_seed: int, obs_dim: int, act_dim: int, cfg: CrossQConfig):
    """PyTorch default initial weights, as in sb3-contrib, on the CPU, under one seed's torch seed.

    Returns (W, b) pairs in the shapes that torch.bmm expects: (in, out) and (1, out).
    """
    torch.manual_seed(init_seed)
    h = cfg.actor_hidden
    actor = [_linear_params(obs_dim, h), _linear_params(h, h), _linear_params(h, act_dim), _linear_params(h, act_dim)]
    q = cfg.critic_hidden
    in_dim = obs_dim + act_dim
    q1 = [_linear_params(in_dim, q), _linear_params(q, q), _linear_params(q, 1)]
    q2 = [_linear_params(in_dim, q), _linear_params(q, q), _linear_params(q, 1)]
    return actor, q1, q2


class StackedCrossQ:
    name = "crossq"

    def __init__(self, init_seeds: list[int], obs_dim: int, act_dim: int, cfg: CrossQConfig, device, dtype):
        self.S = len(init_seeds)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.cfg = cfg
        self.device, self.dtype = torch.device(device), dtype
        self.target_entropy = -act_dim
        S = self.S

        states = [init_networks(seed, obs_dim, act_dim, cfg) for seed in init_seeds]
        h, q = cfg.actor_hidden, cfg.critic_hidden

        # Actor: BRN, linear, ReLU, BRN, linear, ReLU, BRN, then the two output layers.
        # The list holds: s0 t0 W0 b0, s1 t1 W1 b1, s2 t2 Wm bm, Ws bs.
        self.actor_stats = [BatchRenormStats(S, f, self.device, dtype) for f in (obs_dim, h, h)]
        self.actor = []
        for n_layer, features in enumerate((obs_dim, h, h, None)):
            weight, bias = self._stack([st[0][n_layer] for st in states])
            if features is not None:
                self.actor += [self._ones(S, features), self._zeros(S, features)]
            self.actor += [weight, bias]

        # Critics: one stack of 2S networks, Q1 of every seed and then Q2, as in StackedSAC.
        self.critic_stats = [BatchRenormStats(2 * S, f, self.device, dtype) for f in (obs_dim + act_dim, q, q)]
        self.critic = []
        for n_layer, features in enumerate((obs_dim + act_dim, q, q)):
            pairs = [st[1][n_layer] for st in states] + [st[2][n_layer] for st in states]
            weight, bias = self._stack(pairs)
            self.critic += [self._ones(2 * S, features), self._zeros(2 * S, features), weight, bias]
        self.critic_frozen = [p.detach() for p in self.critic]  # shares storage with self.critic

        self.log_alpha = torch.zeros(S, device=self.device, dtype=dtype, requires_grad=True)

        betas = (cfg.adam_beta1, 0.999)
        self.opt_critic = make_adam(self.critic, cfg.critic_lr, self.device, betas=betas)
        self.opt_actor = make_adam(self.actor, cfg.actor_lr, self.device, betas=betas)
        self.opt_alpha = make_adam([self.log_alpha], cfg.alpha_lr, self.device)
        self.updates = 0

    def _stack(self, pairs):
        W = torch.stack([p[0] for p in pairs]).to(device=self.device, dtype=self.dtype).requires_grad_(True)
        b = torch.stack([p[1] for p in pairs]).to(device=self.device, dtype=self.dtype).requires_grad_(True)
        return W, b

    def _ones(self, N: int, features: int) -> torch.Tensor:
        return torch.ones(N, 1, features, device=self.device, dtype=self.dtype, requires_grad=True)

    def _zeros(self, N: int, features: int) -> torch.Tensor:
        return torch.zeros(N, 1, features, device=self.device, dtype=self.dtype, requires_grad=True)

    # ---------------- batch renormalization ----------------

    def _brn(self, x, scale, shift, stats: BatchRenormStats, training: bool):
        """Same maths as sb3_contrib.common.torch_layers.BatchRenorm, one statistic per network."""
        cfg = self.cfg
        if training:
            batch_mean = x.mean(dim=1, keepdim=True)
            batch_var = x.var(dim=1, keepdim=True)
            if stats.steps > cfg.renorm_warmup:
                with torch.no_grad():
                    batch_std = (batch_var + cfg.bn_eps).sqrt()
                    running_std = (stats.var + cfg.bn_eps).sqrt()
                    r = (batch_std / running_std).clamp(1 / cfg.r_max, cfg.r_max)
                    d = ((batch_mean - stats.mean) / running_std).clamp(-cfg.d_max, cfg.d_max)
                # The floor only acts on a feature with no variance at all, such as a ReLU that is off
                # for the whole batch. There the square root's gradient would turn the update to NaN.
                mean = batch_mean - d * batch_var.clamp_min(1e-12).sqrt() / r
                var = batch_var / (r * r)
            else:
                mean, var = batch_mean, batch_var
            with torch.no_grad():
                stats.mean.add_((batch_mean.detach() - stats.mean).mul(cfg.bn_momentum))
                stats.var.add_((batch_var.detach() - stats.var).mul(cfg.bn_momentum))
            stats.steps += 1
        else:
            mean, var = stats.mean, stats.var
        return scale * ((x - mean) / (var + cfg.bn_eps).sqrt()) + shift

    # ---------------- forward passes ----------------

    def _actor_body(self, obs, training: bool):
        s0, t0, W0, b0, s1, t1, W1, b1, s2, t2, Wm, bm, Ws, bs = self.actor
        st0, st1, st2 = self.actor_stats
        h = self._brn(obs, s0, t0, st0, training)
        h = F.relu(torch.bmm(h, W0) + b0)
        h = self._brn(h, s1, t1, st1, training)
        h = F.relu(torch.bmm(h, W1) + b1)
        h = self._brn(h, s2, t2, st2, training)
        return torch.bmm(h, Wm) + bm, (torch.bmm(h, Ws) + bs).clamp(LOG_STD_MIN, LOG_STD_MAX)

    def _policy(self, obs, eps, training: bool):
        """Reparameterized sample and exact log probability of the squashed Gaussian."""
        mu, log_std = self._actor_body(obs, training)
        x = mu + eps * log_std.exp()
        base = -0.5 * eps * eps - log_std - HALF_LOG_2PI
        log_det = 2.0 * (LOG_2 - x - F.softplus(-2.0 * x))
        return torch.tanh(x), (base - log_det).sum(-1, keepdim=True)

    def _q(self, params, obs, act, training: bool):
        """Q values of all 2S critics, shape (2S, B, 1)."""
        s0, t0, W0, b0, s1, t1, W1, b1, s2, t2, W2, b2 = params
        st0, st1, st2 = self.critic_stats
        x = torch.cat([obs, act], dim=-1).repeat(2, 1, 1)
        h = self._brn(x, s0, t0, st0, training)
        h = F.relu(torch.bmm(h, W0) + b0)
        h = self._brn(h, s1, t1, st1, training)
        h = F.relu(torch.bmm(h, W1) + b1)
        h = self._brn(h, s2, t2, st2, training)
        return torch.bmm(h, W2) + b2

    # ---------------- acting ----------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device, dtype=self.dtype).unsqueeze(1)
        eps_t = torch.as_tensor(eps, device=self.device, dtype=self.dtype).unsqueeze(1)
        mu, log_std = self._actor_body(obs_t, training=False)
        return torch.tanh(mu + eps_t * log_std.exp()).squeeze(1).cpu().numpy()

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device, dtype=self.dtype).unsqueeze(1)
        mu, _ = self._actor_body(obs_t, training=False)
        return torch.tanh(mu).squeeze(1).cpu().numpy()

    # ---------------- learning ----------------

    def update(self, rows: torch.Tensor, noise: np.ndarray, record: bool = False):
        """One update for every seed. rows: (S, B, D). noise: (S, 2, B, A)."""
        S, O, A = self.S, self.obs_dim, self.act_dim
        cfg = self.cfg
        B = rows.shape[1]
        obs = rows[..., :O]
        act = rows[..., O:O + A]
        rew = rows[..., O + A:O + A + 1]
        nxt = rows[..., O + A + 1:O + A + 1 + O]
        done = rows[..., -1:]
        eps = torch.as_tensor(noise, device=self.device, dtype=self.dtype)
        eps_next, eps_pi = eps[:, 0], eps[:, 1]

        self.updates += 1
        alpha = self.log_alpha.detach().exp().view(S, 1, 1)

        with torch.no_grad():
            next_act, next_logp = self._policy(nxt, eps_next, training=False)

        # The joint forward pass, with batch statistics taken over the mixture of both batches.
        q_all = self._q(self.critic, torch.cat([obs, nxt], dim=1), torch.cat([act, next_act], dim=1), training=True)
        q_now, q_next = q_all[:, :B], q_all[:, B:].detach()
        with torch.no_grad():
            soft = torch.minimum(q_next[:S], q_next[S:]) - alpha * next_logp
            target = rew + cfg.gamma * (1 - done) * soft
        critic_each = 0.5 * ((q_now - target.repeat(2, 1, 1)) ** 2).mean(dim=(1, 2))
        self.opt_critic.zero_grad(set_to_none=True)
        critic_each.sum().backward()
        self.opt_critic.step()

        actor_each = alpha_each = None
        if self.updates % cfg.policy_delay == 0:
            pi, logp = self._policy(obs, eps_pi, training=True)
            with torch.no_grad():
                entropy_gap = (logp.detach() + self.target_entropy).mean(dim=(1, 2))
                alpha_each = -self.log_alpha.detach() * entropy_gap if record else None
            # Gradient of -(log_alpha * gap).mean() with respect to log_alpha.
            self.log_alpha.grad = -entropy_gap
            self.opt_alpha.step()

            q_pi = self._q(self.critic_frozen, obs, pi, training=False)
            actor_each = (alpha * logp - torch.minimum(q_pi[:S], q_pi[S:])).mean(dim=(1, 2))
            self.opt_actor.zero_grad(set_to_none=True)
            actor_each.sum().backward()
            self.opt_actor.step()

        if record:
            critic_total = (critic_each[:S] + critic_each[S:]).detach()
            zero = torch.zeros_like(critic_total)
            actor_out = actor_each.detach() if actor_each is not None else zero
            alpha_out = alpha_each if alpha_each is not None else zero
            return torch.stack([critic_total, actor_out, alpha_out], dim=1).cpu().numpy()
        return None

    # ---------------- state ----------------

    def seed_is_finite(self) -> np.ndarray:
        """Per-seed flag: False once any tracked actor weight, or a critic's stats, is NaN or Inf."""
        with torch.no_grad():
            S = self.S
            ok = torch.isfinite(self.log_alpha.detach())
            ok = ok & torch.isfinite(self.actor[2].detach()).flatten(1).all(dim=1)
            stats = torch.isfinite(self.critic_stats[1].mean).flatten(1).all(dim=1)
            ok = ok & stats[:S] & stats[S:]
        return ok.cpu().numpy()

    def state_dict(self) -> dict:
        return {
            "actor": [p.detach().cpu() for p in self.actor],
            "critic": [p.detach().cpu() for p in self.critic],
            "actor_stats": [s.state_dict() for s in self.actor_stats],
            "critic_stats": [s.state_dict() for s in self.critic_stats],
            "log_alpha": self.log_alpha.detach().cpu(),
            "opt_critic": self.opt_critic.state_dict(),
            "opt_actor": self.opt_actor.state_dict(),
            "opt_alpha": self.opt_alpha.state_dict(),
            "updates": self.updates,
        }

    def load_state_dict(self, state: dict) -> None:
        with torch.no_grad():
            for ours, saved in zip(self.actor, state["actor"]):
                ours.copy_(saved.to(self.device))
            for ours, saved in zip(self.critic, state["critic"]):
                ours.copy_(saved.to(self.device))
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        for ours, saved in zip(self.actor_stats, state["actor_stats"]):
            ours.load_state_dict(saved)
        for ours, saved in zip(self.critic_stats, state["critic_stats"]):
            ours.load_state_dict(saved)
        self.opt_critic.load_state_dict(state["opt_critic"])
        self.opt_actor.load_state_dict(state["opt_actor"])
        self.opt_alpha.load_state_dict(state["opt_alpha"])
        self.updates = state["updates"]
