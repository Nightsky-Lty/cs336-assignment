import argparse
import cs336_basics.model, cs336_basics.optimizer, cs336_basics.nn_utils
import torch
from contextlib import nullcontext
from pathlib import Path
from torch import Tensor
import numpy as np
from timeit import default_timer

MODEL_SIZES = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("forward", "forward_backward", "train_step"),
        default="train_step"
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--model-size",
        choices=tuple(MODEL_SIZES.keys()),
        default="small"
    )
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--d_model", type=int, default=None)
    parser.add_argument("--d_ff", type=int, default=None)
    parser.add_argument("--num_layers", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--measure_steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_mixed_precision", action="store_true")
    parser.add_argument("--use_memory_profile", action="store_true")

    args = parser.parse_args()

    defaults = MODEL_SIZES[args.model_size]
    if args.d_model is None:
        args.d_model = defaults["d_model"]
    if args.d_ff is None:
        args.d_ff = defaults["d_ff"]
    if args.num_layers is None:
        args.num_layers = defaults["num_layers"]
    if args.num_heads is None:
        args.num_heads = defaults["num_heads"]

    if args.d_model % args.num_heads != 0:
        raise ValueError(f"d_model ({args.d_model}) must be divisible by num_heads ({args.num_heads}).")

    return args

def print_result(times: list[float]):
    if len(times) == 0:
        return
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"mean_time_s: {mean_time:.6f}")
    print(f"std_time_s: {std_time:.6f}")
    print(f"times_s: {[round(t, 6) for t in times]}")

def get_ctx(use_mixed_precision: bool):
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_mixed_precision else nullcontext()


def get_unique_snapshot_path(snapshot_stem: str) -> str:
    candidate = Path(f"{snapshot_stem}.pickle")
    if not candidate.exists():
        return str(candidate)

    index = 1
    while True:
        candidate = Path(f"{snapshot_stem}_{index}.pickle")
        if not candidate.exists():
            return str(candidate)
        index += 1


def run_single_step(
    mode: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: Tensor,
    targets: Tensor,
    use_mixed_precision: bool,
):
    if mode == "forward":
        with torch.no_grad():
            ctx = get_ctx(use_mixed_precision)
            with ctx:
                return model(inputs)

    if mode == "forward_backward":
        model.zero_grad()
        ctx = get_ctx(use_mixed_precision)
        with ctx:
            out = model(inputs)
            loss = cs336_basics.nn_utils.cross_entropy(out, targets)
        loss.backward()
        return loss

    if mode == "train_step":
        optimizer.zero_grad()
        ctx = get_ctx(use_mixed_precision)
        with ctx:
            out = model(inputs)
            loss = cs336_basics.nn_utils.cross_entropy(out, targets)
        loss.backward()
        optimizer.step()
        return loss

    raise ValueError(f"Invalid mode: {mode}")

def measure_mode(
    mode: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    inputs: Tensor,
    targets: Tensor | None,
    warmup_steps: int,
    measure_steps: int,
    use_mixed_precision: bool,
    use_memory_profile: bool,
    snapshot_stem: str,
):
    for _ in range(warmup_steps):
        run_single_step(
            mode=mode,
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            use_mixed_precision=use_mixed_precision,
        )
        torch.cuda.synchronize()
    elapsed_times = []
    if use_memory_profile:
        torch.cuda.memory._record_memory_history(max_entries=1000000)
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        start_time = default_timer()
        run_single_step(
            mode=mode,
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            use_mixed_precision=use_mixed_precision,
        )
        torch.cuda.synchronize()
        end_time = default_timer()
        elapsed_times.append(end_time - start_time)
    print_result(elapsed_times)
    if use_memory_profile:
        snapshot_path = get_unique_snapshot_path(snapshot_stem)
        torch.cuda.memory._dump_snapshot(snapshot_path)
        print(f"memory_snapshot: {snapshot_path}")
        torch.cuda.memory._record_memory_history(enabled=None)

def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    model = cs336_basics.model.BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.seq_len,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
    ).to(args.device)
    optimizer = cs336_basics.optimizer.AdamW(
        model.parameters(),
        lr = args.lr
    )
    inputs = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.seq_len),
        device=args.device
    )
    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.seq_len),
        device=args.device
    )
    measure_mode(
        mode=args.mode,
        model=model,
        optimizer=optimizer,
        inputs=inputs,
        targets=targets,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        use_mixed_precision=args.use_mixed_precision,
        use_memory_profile=args.use_memory_profile,
        snapshot_stem=f"memory_snapshot_{args.mode}_{args.model_size}_seq{args.seq_len}",
    )

if __name__ == "__main__":
    main()
