import torch
from torch import Tensor
from jaxtyping import Float, Int, Bool
from einops import einsum
from einops import rearrange
from cs336_basics.utils import silu, softmax
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

def scaled_dot_product_attention(
    Q: Float[Tensor, "... queries d_k"],
    K: Float[Tensor, "... keys d_k"],
    V: Float[Tensor, "... keys d_v"],
    mask: Bool[Tensor, "... queries keys"] | None = None,
) -> Float[Tensor, "... queries d_v"]:
    dot = einsum(Q, K, "... queries d_k, ... keys d_k -> ... queries keys")
    d_k = Q.shape[-1]
    dot = dot / math.sqrt(d_k)
    if mask is not None:
        dot = dot.masked_fill(~mask, -torch.inf)
    dot = softmax(dot, dim=-1)
    out = einsum(dot, V, "... queries keys, ... keys d_v -> ... queries d_v")
    return out

class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.head_dim = d_model // num_heads
        self.wq = Linear(d_model, d_model)
        self.wk = Linear(d_model, d_model)
        self.wv = Linear(d_model, d_model)
        self.wo = Linear(d_model, d_model)
        
        if theta is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len)
        else:
            self.rope = None

    def make_causal_mask(seq_len: int, device: torch.device) -> Bool[Tensor, "seq_len seq_len"]:
        return torch.tril(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool))

    def forward(
        self, 
        x: Float[Tensor, "... seq_len d_model"],
        token_positions: Int[Tensor, "... seq_len"] | None = None
    ) -> Float[Tensor, "... seq_len d_model"]:
        Q = self.wq(x)
        K = self.wk(x)
        V = self.wv(x)

        Q = rearrange(Q, "... seq_len (h d_q) -> ... h seq_len d_q",h=self.num_heads)
        K = rearrange(K, "... seq_len (h d_k) -> ... h seq_len d_k", h=self.num_heads)
        V = rearrange(V, "... seq_len (h d_v) -> ... h seq_len d_v", h=self.num_heads)

        if self.rope is not None:
            if token_positions is None:
                seq_len = x.shape[-2]
                token_positions = torch.arange(seq_len, device=x.device)
            Q = self.rope(Q, token_positions)
            K = self.rope(K, token_positions)

        out = scaled_dot_product_attention(Q, K, V)
        out = rearrange(out, "... h seq_len d_v -> ... seq_len (h d_v)")

        return self.wo(out)


