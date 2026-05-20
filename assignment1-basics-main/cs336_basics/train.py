import torch
import math
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Callable, Iterable
from typing import Optional
from cs336_basics.utils import softmax

def cross_entropy(
    logits: Float[Tensor, "... vocab_size"],
    x: Int[Tensor, "..."]
) -> Float[Tensor, ""]:
    max_logits = logits.max(dim=-1, keepdim=True).values
    shifted = logits - max_logits
    target_logit = shifted.gather(dim=-1,index=x.unsqueeze(-1)).squeeze(-1)
    logsumexp = torch.log(torch.exp(shifted).sum(dim=-1))
    return (-target_logit + logsumexp).mean()

class AdamW(torch.optim.Optimizer):
    def __init__(
        self, 
        params, 
        lr: float = 1e-3, 
        weight_decay: float = 0.0,
        eps: float = 1e-8,
        betas: tuple[float, float] = (0.9, 0.999)
    ):
        if lr < 0:
            raise ValueError(f"Invalid lr: {lr}")
        defaults = {
            "lr" : lr,
            "weight_decay" : weight_decay,
            "eps" : eps,
            "betas" : betas
        }
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            b1, b2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                g = p.grad.data
                t = state.get("t", 0) + 1
                m = state.get("m", torch.zeros_like(p.data))
                v = state.get("v", torch.zeros_like(p.data))
                lr_t = lr * math.sqrt(1.0 - b2 ** t) / (1.0 - b1 ** t)

                p.data -= lr * weight_decay * p.data
                m = b1 * m + (1.0 - b1) * g
                v = b2 * v + (1.0 - b2) * g ** 2
                p.data -= lr_t * m / (torch.sqrt(v) + eps)

                state["t"] = t
                state["m"] = m
                state["v"] = v
        return loss