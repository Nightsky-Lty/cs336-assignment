from cs336_basics.model import scaled_dot_product_attention
import torch
from torch import Tensor
from timeit import default_timer
import numpy as np

def print_result(times: list[float]):
    if len(times) == 0:
        return
    mean_time = np.mean(times)
    std_time = np.std(times)
    print(f"mean_time_s: {mean_time:.6f}")
    print(f"std_time_s: {std_time:.6f}")
    print(f"times_s: {[round(t, 6) for t in times]}")

def measure_attn(Q: Tensor, K: Tensor, V: Tensor, warmup_steps: int, measure_steps: int):
    forward_times = []
    for _ in warmup_steps:
        out = scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        torch.cuda.synchronize()
    for _ in measure_steps:
        torch.cuda.synchronize()
        start_time = default_timer()
        out = scaled_dot_product_attention(Q, K, V)
        end_time = default_timer()
        forward_times.append(end_time - start_time)
        torch.cuda.synchronize()
    print("forward result:")
    print_result(forward_times)

    backward_times = []
    for _ in warmup_steps:
        out = scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        loss.backward()
        torch.cuda.synchronize()
    for _ in measure_steps:
        torch.cuda.synchronize()
        start_time = default_timer()
        out = scaled_dot_product_attention(Q, K, V)
        loss = out.sum()
        loss.backward()
        end_time = default_timer()
        backward_times.append(end_time - start_time)
        torch.cuda.synchronize()
    print("backward result:")
    print_result(backward_times)

def main():
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
            measure_attn(Q, K, V, warm_steps, measure_steps)
            

if __name__ == "__main__":
    main()