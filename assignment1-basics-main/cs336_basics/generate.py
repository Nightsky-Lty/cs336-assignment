import argparse
import torch
from torch import Tensor
from jaxtyping import Float, Int
from cs336_basics import utils, tokenizer, train
from cs336_basics.model import TransformerLM
import os
from typing import IO, Any, BinaryIO


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

def load_model(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    device: str
):
    params = torch.load(src, map_location=device)
    model.load_state_dict(params["model"])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, default="data/tinystories_train_bpe_vocab_10000.pkl")
    parser.add_argument("--merges_path", type=str, default="data/tinystories_train_bpe_merges_10000.pkl")
    parser.add_argument("--device", type=str, default="cpu")

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--eos_token", type=str, default="<|endoftext|>")

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    args = parser.parse_args()

    tk = tokenizer.Tokenizer.from_files(
        vocab_filepath = args.vocab_path,
        merges_filepath = args.merges_path,
        special_tokens = [args.eos_token]
    ) 
    model = TransformerLM(
        vocab_size = args.vocab_size,
        context_length = args.context_length,
        d_model = args.d_model,
        num_layers = args.num_layers,
        num_heads = args.num_heads,
        d_ff = args.d_ff,
        rope_theta = args.rope_theta
    ).to(args.device)

    load_model(
        src = args.checkpoint_path,
        model = model,
        device = args.device
    )

    prompt_idx = tk.encode(args.prompt)
    eos_token_id = tk.bytes_to_id[args.eos_token.encode("utf-8")]
    out_idx = generate(
        prompt = prompt_idx,
        model = model, 
        max_new_tokens = args.max_new_tokens,
        temperature = args.temperature,
        top_p = args.top_p,
        eos_token_id = eos_token_id,
        device = args.device
    )
    out = tk.decode(out_idx)
    print(args.prompt + out)

if __name__ == "__main__":
    main()

    
