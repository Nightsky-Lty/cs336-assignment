import torch
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics import utils

def sample_next_token(
    logits: Float[Tensor, "vocab_size"],
    temperature: float,
    eps: float = 1e-8,
    P: float = 1
) -> int:
    if temperature <= 0 or P < 0 or P > 1:
        raise ValueError("Invalid perameters")
    prob_sum = 0
    prob = utils.softmax(logits / (temperature + eps), -1)
    sorted_prob, sorted_indices = torch.sort(prob, descending=True)
    V = torch.zeros_like(prob,dtype=torch.bool)
    for p, idx in zip(sorted_prob, sorted_indices):
        prob_sum += p
        V[idx] = True
        if prob_sum >= P:
            break
    prob_sum = prob[V].sum()
    prob[~V] = 0
    prob[V] /= prob_sum
    return torch.multinomial(prob, 1)[0].item()

def generate(
    prompt: list[int],
    model: torch.nn.Module,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    eos_token_id: int,
    device: torch.device
) -> list[int]:
    with torch.no_grad():
        out = []
        current_prompt = torch.tensor(prompt + out,dtype=torch.long,device=device)
        while len(out) < max_new_tokens:
            logits = model(current_prompt)[-1,:]
            next_token = sample_next_token(logits=logits, temperature=temperature, P=top_p)
            out.append(next_token)
            if next_token == eos_token_id:
                break
            current_prompt = torch.tensor(prompt + out,dtype=torch.long,device=device)
    return out
        


    