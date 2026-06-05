from cs336_basics.model import scaled_dot_product_attention
import torch
from torch import Tensor
from timeit import default_timer
import numpy as np
import argparse

def print_result(times: list[float]):
    if len(times) == 0:
        return
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"mean_time_s: {mean_time:.6f}")
    print(f"std_time_s: {std_time:.6f}")
    print(f"times_s: {[round(t, 6) for t in times]}")

def measure_attn(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    warmup_steps: int,
    measure_steps: int,
    use_torch_compile: bool
):
    if use_torch_compile:
        attn = torch.compile(scaled_dot_product_attention)
    else:
        attn = scaled_dot_product_attention
    forward_times = []
    for _ in range(warmup_steps):
        out = attn(Q, K, V)
        loss = out.sum()
        torch.cuda.synchronize()
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        start_time = default_timer()
        out = attn(Q, K, V)
        torch.cuda.synchronize()
        end_time = default_timer()
        forward_times.append(end_time - start_time)
    print("forward result:")
    print_result(forward_times)

    def grad_clear():
        Q.grad = None
        K.grad = None
        V.grad = None

    backward_times = []
    for _ in range(warmup_steps):
        grad_clear()
        out = attn(Q, K, V)
        loss = out.sum()
        torch.cuda.synchronize()
        loss.backward()
        torch.cuda.synchronize()
    for _ in range(measure_steps):
        torch.cuda.synchronize()
        grad_clear()
        out = attn(Q, K, V)
        loss = out.sum()
        torch.cuda.synchronize()
        start_time = default_timer()
        loss.backward()
        torch.cuda.synchronize()
        end_time = default_timer()
        backward_times.append(end_time - start_time)

    print("backward result:")
    print_result(backward_times)

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_torch_compile", action="store_true")
    args = parser.parse_args()
    return args

def main():
    args = parser_args()
    batch_size = 8
    seed = 42
    warm_steps = 10
    measure_steps = 100
    device = "cuda"
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    d_models = [16, 32, 64, 128]
    seq_lengths = [256, 1024, 4096, 8192, 16384]
    for d_model in d_models:
        for seq_length in seq_lengths:
            Q = torch.randn((batch_size, seq_length, d_model), device=device, requires_grad=True)
            K = torch.randn((batch_size, seq_length, d_model), device=device, requires_grad=True)
            V = torch.randn((batch_size, seq_length, d_model), device=device, requires_grad=True)
            measure_attn(Q, K, V, warm_steps, measure_steps, args.use_torch_compile)
            

if __name__ == "__main__":
    main()