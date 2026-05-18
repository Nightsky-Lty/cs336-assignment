import torch
from torch import Tensor
from jaxtyping import Float
from einops import einsum
import math
class Linear(torch.nn.Module):
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device = None, 
        dtype: torch.dtype=None
        ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        w = torch.empty(out_features, in_features, device=device, dtype=dtype)
        sigma = math.sqrt(2.0 / (1.0 * in_features + out_features))
        torch.nn.init.trunc_normal_(w, mean=0.0, std=sigma, a=-3.0 * sigma, b=3.0 * sigma)
        self.weights: Float[Tensor, "d_out d_in"] = torch.nn.Parameter(w)
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        out = einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")
        return out