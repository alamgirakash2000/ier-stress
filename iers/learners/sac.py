"""SAC on stacked parameters.

The update follows third_party/ier/algorithms/base/sac.py, step for step: critic, then actor,
then temperature, then the target network. Test T12 checks the match on identical data.
Initial weights come from the upstream Actor and Critic classes. Stacked parameters share one
Adam per network, which is the same as one Adam per seed because Adam works element by element
and the gradient of a sum of per-seed losses is the per-seed gradient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .. import upstream
from ..batched import stack_linear
from .optim import make_adam

LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0  # upstream Actor.log_std_bounds
LOG_2 = math.log(2.0)
HALF_LOG_2PI = math.log(math.sqrt(2 * math.pi))
_ACTOR_LAYERS = ("act_net.0", "act_net.2", "mean_linear", "log_std_linear")
_CRITIC_LAYERS = ("0", "2", "4")


@dataclass
class SACConfig:
    # Table 5 of the IER paper and the upstream SACConfig.
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    gamma: float = 0.99
    tau: float = 0.005
    reward_scale: float = 1.0
    batch_size: int = 256


def init_networks(init_seed: int, obs_dim: int, act_dim: int):
    """Build the upstream networks under one seed's own torch seed, on the CPU."""
    Actor, Critic = upstream.sac_network_classes()
    torch.manual_seed(init_seed)
    actor = Actor(obs_dim, act_dim)
    critic = Critic(obs_dim, act_dim)
    return actor.state_dict(), critic.state_dict()


class StackedSAC:
    name = "sac"

    def __init__(self, init_seeds: list[int], obs_dim: int, act_dim: int, cfg: SACConfig, device, dtype):
        self.S = len(init_seeds)
        self.obs_dim, self.act_dim = obs_dim, act_dim
        self.cfg = cfg
        self.device, self.dtype = torch.device(device), dtype
        self.target_entropy = -act_dim

        states = [init_networks(seed, obs_dim, act_dim) for seed in init_seeds]
        actors = [s[0] for s in states]
        critics = [s[1] for s in states]

        self.actor = []
        for layer in _ACTOR_LAYERS:
            W, b = stack_linear([a[f"{layer}.weight"] for a in actors], [a[f"{layer}.bias"] for a in actors])
            self.actor += [self._leaf(W), self._leaf(b)]

        # The twin critics sit in one stack of 2S networks: Q1 of every seed, then Q2.
        self.critic = []
        for layer in _CRITIC_LAYERS:
            weights = [c[f"Q1.{layer}.weight"] for c in critics] + [c[f"Q2.{layer}.weight"] for c in critics]
            biases = [c[f"Q1.{layer}.bias"] for c in critics] + [c[f"Q2.{layer}.bias"] for c in critics]
            W, b = stack_linear(weights, biases)
            self.critic += [self._leaf(W), self._leaf(b)]
        self.critic_frozen = [p.detach() for p in self.critic]  # shares storage with self.critic
        self.target = [p.detach().clone() for p in self.critic]

        # Upstream starts the temperature at 1.0, so log_alpha starts at 0.
        self.log_alpha = torch.zeros(self.S, device=self.device, dtype=dtype, requires_grad=True)

        self.opt_critic = make_adam(self.critic, cfg.critic_lr, self.device)
        self.opt_actor = make_adam(self.actor, cfg.actor_lr, self.device)
        self.opt_alpha = make_adam([self.log_alpha], cfg.alpha_lr, self.device)
        self.updates = 0

    def _leaf(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(device=self.device, dtype=self.dtype).clone().requires_grad_(True)

    # ---------------- forward passes ----------------

    @staticmethod
    def _actor_body(params, obs):
        W0, b0, W1, b1, Wm, bm, Ws, bs = params
        h = F.relu(torch.bmm(obs, W0) + b0)
        h = F.relu(torch.bmm(h, W1) + b1)
        mu = torch.bmm(h, Wm) + bm
        log_std = torch.tanh(torch.bmm(h, Ws) + bs)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
        return mu, log_std

    def _policy(self, obs, eps):
        """Reparameterized sample and log probability, as in upstream SquashedNormal."""
        mu, log_std = self._actor_body(self.actor, obs)
        std = log_std.exp()
        x = mu + eps * std
        base = -((x - mu) ** 2) / (2 * std**2) - log_std - HALF_LOG_2PI
        log_det = 2.0 * (LOG_2 - x - F.softplus(-2.0 * x))
        return torch.tanh(x), (base - log_det).sum(-1, keepdim=True)

    def _q(self, params, obs, act):
        """Q values of all 2S critics, shape (2S, B, 1)."""
        W0, b0, W1, b1, W2, b2 = params
        x = torch.cat([obs, act], dim=-1).repeat(2, 1, 1)
        h = F.relu(torch.bmm(x, W0) + b0)
        h = F.relu(torch.bmm(h, W1) + b1)
        return torch.bmm(h, W2) + b2

    # ---------------- acting ----------------

    @torch.no_grad()
    def act(self, obs: np.ndarray, eps: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device, dtype=self.dtype).unsqueeze(1)
        eps_t = torch.as_tensor(eps, device=self.device, dtype=self.dtype).unsqueeze(1)
        mu, log_std = self._actor_body(self.actor, obs_t)
        return torch.tanh(mu + eps_t * log_std.exp()).squeeze(1).cpu().numpy()

    @torch.no_grad()
    def act_deterministic(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, device=self.device, dtype=self.dtype).unsqueeze(1)
        mu, _ = self._actor_body(self.actor, obs_t)
        return torch.tanh(mu).squeeze(1).cpu().numpy()

    # ---------------- learning ----------------

    def update(self, rows: torch.Tensor, noise: np.ndarray, record: bool = False):
        """One update for every seed. rows: (S, B, D). noise: (S, 2, B, A)."""
        S, O, A = self.S, self.obs_dim, self.act_dim
        cfg = self.cfg
        obs = rows[..., :O]
        act = rows[..., O:O + A]
        rew = rows[..., O + A:O + A + 1]
        nxt = rows[..., O + A + 1:O + A + 1 + O]
        done = rows[..., -1:]
        eps = torch.as_tensor(noise, device=self.device, dtype=self.dtype)
        eps_next, eps_pi = eps[:, 0], eps[:, 1]

        alpha = self.log_alpha.detach().exp().view(S, 1, 1)

        with torch.no_grad():
            next_act, next_logp = self._policy(nxt, eps_next)
            tq = self._q(self.target, nxt, next_act)
            soft = torch.minimum(tq[:S], tq[S:]) - alpha * next_logp
            target = rew * cfg.reward_scale + cfg.gamma * (1 - done) * soft

        q = self._q(self.critic, obs, act)
        # Per-critic mean squared error. Upstream sums the two critics' losses.
        critic_each = ((q - target.repeat(2, 1, 1)) ** 2).mean(dim=(1, 2))
        self.opt_critic.zero_grad(set_to_none=True)
        critic_each.sum().backward()
        self.opt_critic.step()

        pi, logp = self._policy(obs, eps_pi)
        q_pi = self._q(self.critic_frozen, obs, pi)
        actor_each = (alpha * logp - torch.minimum(q_pi[:S], q_pi[S:])).mean(dim=(1, 2))
        self.opt_actor.zero_grad(set_to_none=True)
        actor_each.sum().backward()
        self.opt_actor.step()

        with torch.no_grad():
            entropy_gap = (logp.detach() + self.target_entropy).mean(dim=(1, 2))
            alpha_each = -self.log_alpha.detach() * entropy_gap if record else None
        # Gradient of -(log_alpha * gap).mean() with respect to log_alpha.
        self.log_alpha.grad = -entropy_gap
        self.opt_alpha.step()

        tau = cfg.tau
        with torch.no_grad():
            if self.device.type == "cuda":
                for t, p in zip(self.target, self.critic_frozen):
                    t.mul_(1.0 - tau).add_(p, alpha=tau)
            else:  # Same result. Plain operations round the same at any shape, see optim.py.
                for t, p in zip(self.target, self.critic_frozen):
                    t.mul_(1.0 - tau).add_(p.mul(tau))
        self.updates += 1

        if record:
            critic_total = critic_each[:S] + critic_each[S:]
            return torch.stack([critic_total.detach(), actor_each.detach(), alpha_each], dim=1).cpu().numpy()
        return None

    # ---------------- state ----------------

    def seed_is_finite(self) -> np.ndarray:
        """Per-seed flag: False once any actor weight or temperature of that seed is NaN or Inf."""
        with torch.no_grad():
            ok = torch.isfinite(self.log_alpha.detach())
            for p in self.actor[:2]:
                ok = ok & torch.isfinite(p.detach()).flatten(1).all(dim=1)
        return ok.cpu().numpy()

    def state_dict(self) -> dict:
        return {
            "actor": [p.detach().cpu() for p in self.actor],
            "critic": [p.detach().cpu() for p in self.critic],
            "target": [p.cpu() for p in self.target],
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
            for ours, saved in zip(self.target, state["target"]):
                ours.copy_(saved.to(self.device))
            self.log_alpha.copy_(state["log_alpha"].to(self.device))
        self.opt_critic.load_state_dict(state["opt_critic"])
        self.opt_actor.load_state_dict(state["opt_actor"])
        self.opt_alpha.load_state_dict(state["opt_alpha"])
        self.updates = state["updates"]
