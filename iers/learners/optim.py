"""Adam for stacked parameters, with results that do not depend on tensor shape on the CPU.

torch.optim.Adam uses combined kernels (addcmul, addcdiv, lerp). On the CPU these round differently
in the SIMD body of a loop and in its scalar remainder. So a seed's 6-element bias gets different
last bits when it sits in a 6-element tensor (2 remainder elements) or a 12-element one (none).
With the same maths written as plain add, multiply, divide and square root, which round the same
everywhere, separate and batched runs stay bit-identical on the CPU. Test T10 relies on this.
Test test_plain_adam_matches_torch checks that it matches torch.optim.Adam up to rounding.

On CUDA the fused torch.optim.Adam is used, since it has no such remainder and costs fewer kernels.
"""

from __future__ import annotations

import math

import torch


class PlainAdam:
    def __init__(self, params, lr: float, betas=(0.9, 0.999), eps: float = 1e-8):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.t = 0
        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.params:
            p.grad = None

    @torch.no_grad()
    def step(self) -> None:
        self.t += 1
        step_size = self.lr / (1.0 - self.beta1**self.t)
        sqrt_bias2 = math.sqrt(1.0 - self.beta2**self.t)
        keep1, keep2 = 1.0 - self.beta1, 1.0 - self.beta2
        for p, m, v in zip(self.params, self.m, self.v):
            g = p.grad
            m.mul_(self.beta1).add_(g.mul(keep1))
            v.mul_(self.beta2).add_(g.mul(g).mul(keep2))
            denom = v.sqrt().div(sqrt_bias2).add(self.eps)
            p.sub_(m.div(denom).mul(step_size))

    def state_dict(self) -> dict:
        return {"t": self.t, "m": [x.cpu() for x in self.m], "v": [x.cpu() for x in self.v]}

    def load_state_dict(self, state: dict) -> None:
        self.t = state["t"]
        for ours, saved in zip(self.m, state["m"]):
            ours.copy_(saved)
        for ours, saved in zip(self.v, state["v"]):
            ours.copy_(saved)


def make_adam(params, lr: float, device: torch.device, betas=(0.9, 0.999)):
    params = list(params)
    if device.type == "cuda":
        return torch.optim.Adam(params, lr=lr, betas=betas, fused=True)
    return PlainAdam(params, lr=lr, betas=betas)
