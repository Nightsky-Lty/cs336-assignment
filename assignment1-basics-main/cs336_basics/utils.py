from jaxtyping import Float
from torch import Tensor
import torch
def silu(x : Float[Tensor,"..."]) -> Float[Tensor,"..."]:
    return x * torch.sigmoid(x)