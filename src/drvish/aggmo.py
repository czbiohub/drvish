# implementation of Aggregated Momentum Gradient Descent, adapted from
# the code at https://github.com/AtheMathmo/AggMo

from typing import Sequence

import torch
from torch.optim.optimizer import Optimizer, required


class AggMo(Optimizer):
    """Implements Aggregated Momentum Gradient Descent"""

    def __init__(
        self,
        params: dict,
        lr: float = required,
        betas: Sequence[float] = (0.0, 0.9, 0.99),
        weight_decay: float = 0.0,
        clip_norm: float = None,
    ):
        defaults = dict(
            lr=lr, betas=betas, weight_decay=weight_decay, clip_norm=clip_norm
        )
        super(AggMo, self).__init__(params, defaults)

    @classmethod
    def from_exp_form(cls, params, lr=required, a=0.1, k=3, weight_decay=0):
        betas = [1 - a ** i for i in range(k)]
        return cls(params, lr, betas, weight_decay)

    def __setstate__(self, state):
        super(AggMo, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            betas = group["betas"]
            total_mom = float(len(betas))

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if group["clip_norm"] is not None:
                    d_p.clamp_(-group["clip_norm"], group["clip_norm"])

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = {}
                    for beta in betas:
                        param_state["momentum_buffer"][beta] = torch.zeros_like(p.data)
                for beta in betas:
                    buf = param_state["momentum_buffer"][beta]
                    buf.mul_(beta).add_(d_p)
                    p.data.sub_(group["lr"] / total_mom, buf)
        return loss

    def zero_momentum_buffers(self):
        for group in self.param_groups:
            betas = group["betas"]
            for p in group["params"]:
                param_state = self.state[p]
                param_state["momentum_buffer"] = {}
                for beta in betas:
                    param_state["momentum_buffer"][beta] = torch.zeros_like(p.data)

    def update_hparam(self, name, value):
        for param_group in self.param_groups:
            param_group[name] = value
