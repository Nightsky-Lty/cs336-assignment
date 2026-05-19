from jaxtyping import Float
from torch import Tensor
import torch
def silu(x : Float[Tensor,"..."]) -> Float[Tensor,"..."]:
    return x * torch.sigmoid(x)

def softmax(x: Float[Tensor, "..."], dim: int) -> Float[Tensor, "..."]:
    x_max = x.max(dim=dim, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim=dim, keepdim=True)
    out = x_exp / x_sum
    return out
    