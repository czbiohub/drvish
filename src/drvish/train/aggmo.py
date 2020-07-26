# implementation of Aggregated Momentum Gradient Descent, adapted from
# the code at https://github.com/AtheMathmo/AggMo

from typing import Dict, Sequence

import torch
from torch.optim.optimizer import Optimizer, required


class AggMo(Optimizer):
    """Implements Aggregated Momentum Gradient Descent"""

    def __init__(
        self,
        params: Dict,
        lr: float = required,
        betas: Sequence[float] = (0.0, 0.9, 0.99),
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        defaults = dict(
            lr=lr / len(betas),
            betas=betas,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
        super().__init__(params, defaults)

    @classmethod
    def from_exp_form(
        cls,
        params: Dict,
        lr: float = required,
        a: float = 0.1,
        k: int = 3,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ):
        betas = [1 - a ** i for i in range(k)]
        return cls(params, lr, betas, weight_decay, nesterov)

    @torch.no_grad()
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
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)

                param_state = self.state[p]
                if "momentum_buffer" not in param_state:
                    param_state["momentum_buffer"] = {
                        beta: torch.zeros_like(p) for beta in betas
                    }

                for beta in betas:
                    buf = param_state["momentum_buffer"][beta]
                    buf.mul_(beta).add_(d_p)
                    if nesterov:
                        buf = d_p.add(buf, alpha=beta)

                    p.add_(buf, alpha=-group["lr"])
        return loss
