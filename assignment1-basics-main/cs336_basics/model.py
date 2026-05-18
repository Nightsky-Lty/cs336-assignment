import torch
from torch import Tensor
from jaxtyping import Float, Int
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
        w: Float[Tensor, "d_out d_in"] = torch.empty(out_features, in_features, device=device, dtype=dtype)
        std = math.sqrt(2.0 / (1.0 * in_features + out_features))
        torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)
        self.weight: Float[Tensor, "d_out d_in"] = torch.nn.Parameter(w)
    
    def forward(self, x: Float[Tensor, "... d_in"]) -> Float[Tensor, "... d_out"]:
        out = einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")
        return out
    
class Embedding(torch.nn.Module):
    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device=None,
        dtype: torch.dtype=None
        ):
        super().__init__()
        w: Float[Tensor, "vocab_size d_model"] = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        std = 1.0
        torch.nn.init.trunc_normal_(w, mean=0.0, std=std, a=-3.0, b=3.0)
        self.weight: Float[Tensor, "vocab_size d_model"] = torch.nn.Parameter(w)
    
    def forward(self, x: Int[Tensor, "..."]) -> Float[Tensor, "... d_model"]:
        out = self.weight[x]
        return out

