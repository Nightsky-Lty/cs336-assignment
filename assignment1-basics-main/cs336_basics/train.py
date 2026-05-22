import torch
import math
import os
from typing import IO, Any, BinaryIO
from torch import Tensor
from jaxtyping import Float, Int
from collections.abc import Callable, Iterable
from typing import Optional
from cs336_basics.model import TransformerLM
import numpy.typing as npt
import numpy as np
import argparse
import time

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
        betas: tuple[float, float] = (0.9, 0.95)
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
    
def learning_rate_schedule(
    t: int,
    amax: float,
    amin: float,
    Tw: int,
    Tc: int
) -> float:
    if t < Tw:
        return t / Tw * amax
    elif t > Tc:
        return amin
    else:
        return amin + 0.5 * (1 + math.cos((t - Tw) / (Tc - Tw) * math.pi)) * (amax - amin)

def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6
) -> None:
    g = 0.0
    params = list(parameters)
    for param in params:
        if param.grad is None:
            continue
        g += (param.grad ** 2).sum()
    g = torch.sqrt(g)
    if g > max_l2_norm:
        coef = max_l2_norm / (g + eps)
        for param in params:
            if param.grad is None:
                continue
            param.grad *= coef

def data_loading(
        dataset: npt.NDArray, 
        batch_size: int, 
        context_length: int, 
        device: str
):
    starts = np.random.randint(0, len(dataset) - context_length, batch_size)
    inputs = np.stack([dataset[st: st + context_length] for st in starts])
    targets = np.stack([dataset[st + 1: st + 1 + context_length] for st in starts])
    inputs_tensor = torch.from_numpy(inputs).long().to(device=device)
    targets_tensor = torch.from_numpy(targets).long().to(device=device)
    return inputs_tensor, targets_tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes]
):
    params: dict = {}
    params["model"] = model.state_dict()
    params["optimizer"] = optimizer.state_dict()
    params["iteration"] = iteration
    torch.save(params, out)

def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    params = torch.load(src)
    model.load_state_dict(params["model"])
    optimizer.load_state_dict(params["optimizer"])
    return params["iteration"]

def evaluate(
    model: torch.nn.Module,
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str
):
    model.eval()
    with torch.no_grad():
        data, target = data_loading(dataset, batch_size, context_length, device)
        logits = model(data)
        loss = cross_entropy(logits, target)
    model.train()
    return loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)

    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)

    parser.add_argument("--max_iters", type=int, default=40000)
    parser.add_argument("--max_learning_rate", type=float, default=1e-3)
    parser.add_argument("--min_learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--cosine_cycle_iters", type=int, default=10000)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--adamw_beta1", type=float, default=0.9)
    parser.add_argument("--adamw_beta2", type=float, default=0.95)
    parser.add_argument("--adamw_eps", type=float, default=1e-8)
    parser.add_argument("--max_l2_norm", type=float, default=1.0)

    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_path", type=str, default="checkpoint/checkpoint.pt")
    parser.add_argument("--resume_from", type=str, default=None)

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = np.memmap(args.data_path, dtype=np.uint16, mode="r")
    
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta
    ).to(args.device)
    optimizer = AdamW(
        params=model.parameters(),
        weight_decay=args.weight_decay,
        lr=args.max_learning_rate,
        eps=args.adamw_eps,
        betas=(args.adamw_beta1, args.adamw_beta2)
    )

    start_it = 0
    if args.resume_from is not None:
        start_it = load_checkpoint(
            src=args.resume_from, 
            model=model, 
            optimizer=optimizer
        )

    model.train()
    start_time = time.time()

    for step in range(start_it + 1, args.max_iters + 1):
        lr = learning_rate_schedule(
            t=step,
            amax=args.max_learning_rate,
            amin=args.min_learning_rate,
            Tw=args.warmup_iters,
            Tc=args.cosine_cycle_iters
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        data, target = data_loading(
            dataset, 
            args.batch_size, 
            args.context_length, 
            args.device
        )
        optimizer.zero_grad()
        logits = model(data)
        loss = cross_entropy(logits, target)
        loss.backward()
        gradient_clipping(
            parameters=model.parameters(), 
            max_l2_norm=args.max_l2_norm
        )
        optimizer.step()

        if step % args.log_interval == 0:
            tokens_processed = step * args.batch_size * args.context_length
            print(
                f"step={step}, "
                f"loss={loss.item():.4f}, "
                f"lr={lr:.6e}, "
                f"time={(time.time() - start_time):.2f}s, "
                f"tokens={tokens_processed}"
            )
        
        if step % args.save_interval == 0:
            checkpoint_dir = os.path.dirname(args.checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                iteration=step,
                out=args.checkpoint_path
            )

if __name__ == "__main__":
    main()