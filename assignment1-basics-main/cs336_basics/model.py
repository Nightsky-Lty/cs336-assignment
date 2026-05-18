import torch
from torch import Tensor
from jaxtyping import Float, Int
from einops import einsum
from einops import rearrange
from cs336_basics.utils import silu
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

class RMSNorm(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float,
        device: torch.device=None,
        dtype: torch.dtype=None    
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        w: Float[Tensor, "d_model"] = torch.ones(d_model,device=device, dtype=dtype)
        self.weight = torch.nn.Parameter(w)

    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        in_dtype = x.dtype
        x = x.to(torch.float32)

        rms = torch.sqrt((x * x).mean(-1, keepdim=True) + self.eps)
        out = x / rms * self.weight
        return out.to(in_dtype)

class FeedForward(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device=None,
        dtype=None
    ):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device, dtype)
        self.w2 = Linear(d_ff, d_model, device, dtype)
        self.w3 = Linear(d_model, d_ff, device, dtype)
    
    def forward(self, x: Float[Tensor, "... d_model"]) -> Float[Tensor, "... d_model"]:
        out1 = silu(self.w1(x))
        out2 = self.w3(x)
        out = self.w2(out1 * out2)
        return out

class RotaryPositionalEmbedding(torch.nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("d_k must be even")

        positions = torch.arange(max_seq_len, device=device,dtype=torch.float32)
        model_idx = torch.arange(d_k // 2, device=device,dtype=torch.float32)
        freqs = theta ** (-2 * model_idx / d_k)
        angle = positions[:,None] * freqs[None,:]
        cos_value = torch.cos(angle)
        sin_value = torch.sin(angle)
        
        self.register_buffer("cos_value", cos_value)
        self.register_buffer("sin_value", sin_value)
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        self.theta = theta
    
    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_k"], 
        token_positions: Int[Tensor, "... seq_len"]
    ) -> Float[Tensor, "... seq_len d_k"]:
        cos = self.cos_value[token_positions].to(x.dtype)
        sin = self.sin_value[token_positions].to(x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out_even = x_even * cos - x_odd * sin
        out_odd = x_even * sin + x_odd * cos
        out = torch.stack((out_even,out_odd),dim=-1).flatten(-2)
        return out
