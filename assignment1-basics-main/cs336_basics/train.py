import torch
from torch import Tensor
from jaxtyping import Float, Int
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