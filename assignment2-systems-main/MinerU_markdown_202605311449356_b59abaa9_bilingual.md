# CS336 Assignment 2 (systems): Systems and Parallelism

## CS336 作业 2（系统）：系统与并行

Version 26.1.3

版本 26.1.3

CS336 Staff

CS336 课程组

Spring 2026

2026 年春季

# 1 Assignment Overview

## 1 作业概览

In this assignment, you will gain some hands-on experience with improving single-GPU training speed and scaling training to multiple GPUs.

在本次作业中，你将通过实践了解如何提升单 GPU 训练速度，以及如何将训练扩展到多 GPU。

# What you will implement

## 你将实现的内容

1. Benchmarking and profiling harness

   基准测试与性能分析框架

2. Activation checkpointing

   激活检查点

3. Flash Attention 2 Triton kernel

   FlashAttention-2 的 Triton 内核

4. Distributed data parallel training

   分布式数据并行训练

5. Optimizer state sharding

   优化器状态分片

6. Fully sharded data parallel training

   全分片数据并行训练

# What the code looks like

## 代码结构

The assignment code and this write-up are available on GitHub at:

本次作业的代码和说明文档可在 GitHub 上获取：

`github.com/stanford-cs336/assignment2-systems`

Please clone the repository using Git. If there are any updates, we will notify you and you can git pull to get the latest.

请使用 Git 克隆仓库。如果有更新，课程组会通知你，你可以通过 `git pull` 获取最新内容。

1. `cs336-basics/`: In this assignment, you’ll be profiling some of the components that we built in assignment 1. This folder contains the staff solution code for assignment 1, so you will find `cs336-basics/pyproject.toml` and the `cs336-basics/cs336_basics` package here. If you want to use your own implementation of the model, you can modify the `pyproject.toml` file in the base directory to point to your own package.

   `cs336-basics/`：本次作业中，你将对作业 1 中实现的一些组件做性能分析。这个文件夹包含作业 1 的课程组参考实现，因此你会在这里看到 `cs336-basics/pyproject.toml` 和 `cs336-basics/cs336_basics` 包。如果你想使用自己实现的模型，可以修改根目录中的 `pyproject.toml`，让它指向你自己的包。

2. `/`: The `cs336-systems` base directory. We created an empty module named `cs336_systems`. Note that there’s no code in here, so you should be able to do whatever you want from scratch.

   `/`：`cs336-systems` 根目录。这里提供了一个名为 `cs336_systems` 的空模块。注意其中没有现成代码，因此你可以从零开始自由实现。

3. `tests/*.py`: This directory contains all the tests you must pass. These tests invoke the hooks defined in `tests/adapters.py`. You’ll implement the adapters to connect your code to the tests. Writing more tests and/or modifying the test code can be helpful for debugging your code, but your implementation is expected to pass the original provided test suite.

   `tests/*.py`：该目录包含你必须通过的全部测试。这些测试会调用 `tests/adapters.py` 中定义的接口。你需要实现这些 adapter，把你的代码接到测试上。你可以自行增加测试或修改测试代码来帮助调试，但最终实现必须通过原始提供的测试集。

4. `README.md`: This file contains more details about the expected directory structure, as well as some basic instructions on setting up your environment.

   `README.md`：该文件给出了期望的目录结构细节，以及一些基础环境配置说明。

# How to submit

## 提交方式

You will submit the following files to Gradescope:

你需要向 Gradescope 提交以下文件：

- `writeup.pdf`: Answer all the written questions. Please typeset your responses.

  `writeup.pdf`：回答所有书面问题，请使用排版良好的格式。

- `code.zip`: Contains all the code you’ve written.

  `code.zip`：包含你编写的全部代码。

Run the script in `test_and_make_submission.sh` to create the `code.zip` file.

运行 `test_and_make_submission.sh` 中的脚本来生成 `code.zip`。

# 2 Profiling and Benchmarking

## 2 性能分析与基准测试

In the first part of the assignment, we will explore how to optimize the performance of our Transformer model to make the most efficient use of the GPU. We will profile our model to understand where it spends time and memory during the forward and backward passes, then optimize the self-attention operation with custom GPU kernels, making it faster than is possible with regular PyTorch. In the subsequent parts of the assignment, we will leverage multiple GPUs and understand how to train a model across a cluster.

在作业的第一部分，我们将探索如何优化 Transformer 模型的性能，以更高效地利用 GPU。我们会对模型做性能分析，理解前向与反向传播中时间和显存的主要开销，然后用自定义 GPU 内核优化自注意力操作，使其比普通 PyTorch 实现更快。后续部分则会进一步利用多 GPU，并理解如何在集群上训练模型。

# 2.1 Profiling

## 2.1 性能分析

Before implementing any optimizations, it is helpful to first profile our program to understand where it spends resources (e.g., time and memory). Otherwise, we risk optimizing parts of the model that don’t account for significant time or memory, and therefore not seeing measurable end-to-end improvements.

在进行任何优化之前，先对程序做性能分析很有帮助，这样可以弄清资源究竟消耗在什么地方（例如时间和显存）。否则，我们可能会把精力花在并非主要瓶颈的部分，最终看不到可测量的端到端提升。

We will implement three performance evaluation paths:

我们将实现三种性能评估方式：

1. Simple end-to-end benchmarking using the Python standard library to time our forward and backward passes

   使用 Python 标准库做简单的端到端基准测试，测量前向与反向传播时间

2. Compute profiling with the NVIDIA Nsight Systems tool to understand how that time is distributed across operations on both the CPU and GPU

   使用 NVIDIA Nsight Systems 做计算性能分析，理解时间如何分布在 CPU 与 GPU 的各类操作上

3. Memory profiling

   显存分析

# 2.1.1 Setup - Importing your basics Transformer Model

## 2.1.1 环境准备：导入你在 basics 作业中的 Transformer 模型

Let’s start by making sure that you can load the model from the previous assignment. In the previous assignment, we set up our model in a Python package, so that it could be easily imported later. We have added a reference implementation of the model in the `./cs336-basics` folder, and have pointed to it in the `pyproject.toml` file. By calling `uv run [command]` as usual, `uv` will automatically locate this local `cs336-basics` package. If you would like to use your own implementation of the model, you can modify the `pyproject.toml` file to point to your own package.

先确保你能够加载上一次作业中的模型。上一份作业中，我们将模型组织成了一个 Python 包，方便后续直接导入。课程组已经在 `./cs336-basics` 文件夹中加入了模型的参考实现，并在 `pyproject.toml` 中配置了它。像平时一样运行 `uv run [command]` 时，`uv` 会自动找到这个本地 `cs336-basics` 包。如果你想使用自己实现的模型，可以修改 `pyproject.toml`，让它指向你的包。

You can test that you can import your model with:

你可以用下面的方式测试模型是否能成功导入：

```txt
$ uv run python
Using CPython 3.13.13
Creating virtual environment at: /path/to/uv/env/dir
    Built cs336-systems @ file:///path/to/systems/dir
    Built cs336-basics @ file:///path/to/basics/dir
Installed 78 packages in 168ms
Python 3.13.13 (main, Apr 7 2026, 20:49:46) [Clang 22.1.1] on linux Type "help", "copyright", "credits" or "license" for more information.
>>> import cs336_basics
...
```

The relevant modules from assignment 1 should now be available (e.g., for `model.py`, you can import it with `import cs336_basics.model`).

此时，作业 1 中相关模块应该都可以使用了（例如对于 `model.py`，你可以通过 `import cs336_basics.model` 导入）。

# 2.1.2 Model Sizing

## 2.1.2 模型规模设置

Throughout this assignment, we will be benchmarking and profiling models to better understand their performance. To get a sense of how things change at scale, we will work with and refer to the following model configurations. For all models except in the leaderboard, we’ll use a vocabulary size of 10,000 and a batch size of 4, with varying context lengths.

在整个作业中，我们会不断对模型做基准测试和性能分析，以更好地理解其性能表现。为了观察规模变化带来的影响，我们将使用并反复引用如下模型配置。除排行榜部分外，所有模型都使用 10,000 的词表大小和 batch size 4，context length 则会变化。

This assignment (and later ones) will require a lot of results to be presented in tables and plots. We strongly recommend that you automate constructing tables for your writeup in code, since formatting tables in LaTeX or Typst can be very tedious. See `pandas.DataFrame.to_latex()` and `pandas.DataFrame.to_typst()` or write your own function to generate them from your preferred tabular representation.

本次作业（以及后续作业）会要求你在表格和图中展示大量结果。我们强烈建议你用代码自动生成写作报告中的表格，因为手工在 LaTeX 或 Typst 中排版表格通常非常繁琐。可以参考 `pandas.DataFrame.to_latex()`、`pandas.DataFrame.to_typst()`，或者自己写函数从喜欢的表格表示中生成。

<table><tr><td>Size</td><td>d_model</td><td>d_ff</td><td>num_layers</td><td>num_heads</td></tr><tr><td>small</td><td>768</td><td>3072</td><td>12</td><td>12</td></tr><tr><td>medium</td><td>1024</td><td>4096</td><td>24</td><td>16</td></tr><tr><td>large</td><td>1280</td><td>5120</td><td>36</td><td>20</td></tr><tr><td>xl</td><td>2560</td><td>10240</td><td>32</td><td>32</td></tr><tr><td>10B</td><td>4608</td><td>12288</td><td>50</td><td>36</td></tr></table>

表 1：不同模型规模的规格，这些配置大多参考 GPT-2。

Use context length 512 unless otherwise specified.

除非另有说明，否则使用 context length 512。

# 2.1.3 End-to-End Benchmarking

## 2.1.3 端到端基准测试

We will now implement a simple performance evaluation script. We will be testing many variations of our model (changing precision, swapping layers, etc.), so it will pay off to have your script enable these variations via command-line arguments to make them easy to run later on.

现在我们要实现一个简单的性能评估脚本。由于后续会测试很多模型变体（如修改精度、更换层实现等），因此最好让脚本通过命令行参数支持这些变化，方便后续复用。

To start off, let’s do the simplest possible profiling of our model by timing the forward pass, backward pass, and optimizer step. Since we will only be measuring speed and memory, it’s fine to use random weights and data.

作为开始，我们先做最简单的性能分析：测量模型的前向传播、反向传播以及优化器步进的时间。由于这里只关心速度和显存，用随机权重和随机数据即可。

Measuring performance is subtle. One common pitfall in benchmarking GPU code is that CUDA calls are asynchronous. When you launch a CUDA kernel, such as with `torch.matmul`, the PyTorch call usually returns immediately without等待矩阵乘真正完成。这意味着 CPU 可以继续往前调度更多操作，而 GPU 在后台完成运算，这是性能优化的一部分；但也意味着如果你只是测量 `torch.matmul` 这行 Python 代码返回的时间，并不能反映 GPU 实际执行矩阵乘的耗时。In PyTorch, we can call `torch.cuda.synchronize()` to wait for all scheduled GPU kernels to complete, which allows more accurate timing. 这里的同步指的是 CPU 运行时与 GPU 运行时之间的同步。

# Problem (benchmarking_script): Benchmarking Script (4 points)

## 题目（benchmarking_script）：基准测试脚本（4 分）

(a) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model.

（a）编写一个脚本，对模型的前向传播、反向传播和优化器步进做基础的端到端基准测试。

Your script should support:

脚本应支持以下功能：

- Given hyperparameters (e.g., number of layers), initialize a model.

  给定超参数（例如层数）后初始化模型。

- Generate a random batch of data.

  生成一批随机数据。

- Run warm-up steps before timing, then time a chosen number of measurement steps.

  在正式计时前先运行若干个预热步骤，然后对指定数量的步骤进行计时。

- Support timing forward-only, forward+backward, or full training steps including optimizer step.

  支持分别测量仅前向、前向加反向、以及包含优化器步进的完整训练步骤。

- Use the Python `timeit` module or `timeit.default_timer()`.

  使用 Python 的 `timeit` 模块或 `timeit.default_timer()`。

- Call `torch.cuda.synchronize()` after each step.

  每一步之后调用 `torch.cuda.synchronize()`。

Deliverable:

提交内容：

A script that initializes a basics Transformer model with the given hyperparameters, creates a random batch of data, and times forward-only, forward-and-backward, and full training steps.

一个脚本：能够根据给定超参数初始化 basics Transformer 模型，构造随机 batch，并对仅前向、前向加反向、以及完整训练步骤进行计时。

(b) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and report average and standard deviation over 10 measurement steps.

（b）对第 2.1.2 节列出的模型规模，测量前向、反向和优化器步进时间。使用 5 次预热，并在 10 次正式测量上报告平均值和标准差。

(c) Repeat the analysis without warm-up steps, and also try 1 or 2 warm-up steps. Explain how and why the results differ.

（c）在不做预热的情况下重复上述分析，并尝试只做 1 次或 2 次预热。说明结果如何变化，以及为什么会发生这种变化。

# 2.1.4 Nsight Systems Profiler

## 2.1.4 Nsight Systems 分析器

End-to-end benchmarking tells us total runtime, but not where time and memory are spent. For that, we need a profiler.

端到端基准测试能告诉我们总耗时，但无法说明时间和显存具体耗费在哪些部分。为此，我们需要分析器。

Standard Python profilers such as `cProfile` cannot correctly profile CUDA kernels because those kernels execute asynchronously on the GPU. NVIDIA provides `nsys`, which can be used from the command line.

标准 Python 性能分析工具（例如 `cProfile`）无法正确分析 CUDA kernel，因为它们在 GPU 上异步执行。NVIDIA 提供了命令行工具 `nsys` 来解决这个问题。

Example:

示例：

```txt
$ uv run nsys profile -- python benchmark.py
```

You can then inspect the trace in the NVIDIA Nsight Systems desktop app.

之后你可以在 NVIDIA Nsight Systems 桌面应用中查看分析结果。

More complete profiling command:

更完整的分析命令示例：

```txt
$ uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autogradshapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python benchmark.py
```

These flags control traced APIs, PyTorch NVTX labels, backtraces, and GPU utilization metrics.

这些参数分别控制需要追踪的 API、PyTorch 的 NVTX 标记、回溯信息以及 GPU 利用率指标。

Profiling has overhead. In particular, `--cudabacktrace=all` and `--python-backtrace=cuda` are expensive when you do not need backtraces.

性能分析本身会引入额外开销。特别是当你不需要回溯信息时，`--cudabacktrace=all` 和 `--python-backtrace=cuda` 会显著拖慢运行。

The assignment encourages you to use NVTX ranges to isolate warm-up, forward, backward, and even sub-steps within attention.

作业鼓励你使用 NVTX range 来区分预热、前向、反向，甚至区分注意力内部的不同子步骤。

# Problem (nsys_profile): Nsight Systems Profiling (5 points)

## 题目（nsys_profile）：Nsight Systems 性能分析（5 分）

Choose two model sizes from Table 1 and three power-of-two context lengths larger than 128, where the largest length is the longest you can fit in memory. Use `nsys` profiles to answer:

从表 1 中选择两个模型规模，再选择三个大于 128 的 2 的幂次 context length，其中最大的长度应当是你在显存中能放下的最大值。使用 `nsys` 分析并回答以下问题：

- Total forward-pass time, and whether it matches Python timing.

  前向传播总时间，以及它是否与 Python 测量结果一致。

- Which CUDA kernel dominates cumulative GPU time in forward, how often it is invoked, and whether backward changes that answer.

  哪个 CUDA kernel 在前向中占据最多累计 GPU 时间、它在单次前向中被调用多少次，以及在前向加反向时这个结论是否变化。

- Which non-matmul kernels still contribute noticeable runtime.

  除矩阵乘外，还有哪些 kernel 也占据了可观的运行时间。

- In a full training step, how the fraction of time spent on matmuls changes relative to inference.

  在完整训练步骤中，相比仅推理，矩阵乘所占时间比例如何变化。

- Compare softmax runtime with matmul runtime inside self-attention.

  比较 self-attention 内 softmax 与矩阵乘的运行时间差异。

# 2.1.5 Mixed Precision

## 2.1.5 混合精度

Up to this point, we have been using FP32 for parameters and activations. Modern NVIDIA GPUs have Tensor Cores that greatly accelerate lower-precision matrix multiplications, especially FP16 and BF16.

到目前为止，我们一直使用 FP32 保存参数和激活。现代 NVIDIA GPU 具备 Tensor Core，可显著加速低精度矩阵乘，尤其是 FP16 和 BF16。

Naively converting everything to lower precision can hurt numerical stability. FP16 may underflow small gradients to zero and can overflow due to its smaller dynamic range.

如果简单粗暴地把所有内容都转成低精度，数值稳定性可能变差。FP16 可能会让较小梯度直接下溢为零，也可能因为动态范围更小而产生溢出。

Mixed precision uses `torch.autocast` so that some ops run in lower precision while numerically sensitive ops stay in higher precision.

混合精度通过 `torch.autocast` 让部分操作在低精度下执行，同时把数值敏感操作保留在更高精度下。

```python
model: torch.nn.Module = ...
dtype: torch.dtype = ...
x: torch.Tensor = ...

with torch.autocast(device_type="cuda", dtype=dtype):
    y = model(x)
```

# Problem (mixed_precision_accumulation): Mixed-Precision Accumulation (1 point)

## 题目（mixed_precision_accumulation）：混合精度累加（1 分）

Run the provided code and comment on the accuracy of the results.

运行题目给出的代码，并评论结果的精度表现。

核心观察：

- FP32 累加通常最接近理论值。
- FP16 直接累加会出现更明显的舍入误差。
- 将 FP16 值提升到 FP32 后再累加，结果会显著更准确。

This exercise builds intuition for why reductions and accumulations often stay in FP32 even when the model otherwise uses lower precision.

这个练习帮助你理解：即使模型大部分计算采用低精度，归约和累加操作通常仍然要保留在 FP32 中。

# Problem (benchmarking_mixed_precision): Benchmarking Mixed Precision (2 points)

## 题目（benchmarking_mixed_precision）：混合精度基准测试（2 分）

(a) Consider the provided `ToyModel`. Suppose parameters start in FP32 and we use autocast with FP16 on GPU. State the dtypes of:

（a）考虑题目给出的 `ToyModel`。假设参数初始为 FP32，并在 GPU 上使用 FP16 autocast。请说明以下各项的数据类型：

- model parameters inside autocast
- output of `fc1`
- output of `ln`
- predicted logits
- loss
- gradients

对应中文：

- autocast 上下文中的模型参数
- 第一层前馈层 `fc1` 的输出
- 层归一化 `ln` 的输出
- 模型预测 logits
- loss
- 梯度

(b) Explain why layer norm is treated differently from feed-forward layers under FP16 autocast, and whether BF16 changes that story.

（b）解释为什么在 FP16 autocast 下，layer norm 与前馈层的处理方式不同；如果换成 BF16，是否仍需区别对待，并说明原因。

(c) Modify your benchmark script to optionally run mixed precision with BF16. Compare full precision and mixed precision timings across model sizes.

（c）修改你的基准测试脚本，使其可选地使用 BF16 混合精度运行。比较不同模型规模下全精度和混合精度的前向、反向时间，并总结趋势。

# 2.1.6 Profiling Memory

## 2.1.6 显存分析

So far we focused on compute. We now turn to memory, which is another major bottleneck in language model training and inference.

目前为止我们主要关注计算性能。接下来转向显存，它同样是语言模型训练与推理中的主要瓶颈之一。

PyTorch provides a memory profiler that can record allocations over time:

PyTorch 提供了显存分析工具，可以记录随时间变化的分配情况：

```python
torch.cuda.memory._record_memory_history(max_entries=1000000)
...
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
torch.cuda.memory._record_memory_history(enabled=None)
```

The resulting `memory_snapshot.pickle` can be loaded into `pytorch.org/memory_viz`.

生成的 `memory_snapshot.pickle` 可以拖入 `pytorch.org/memory_viz` 查看。

# Problem (memory_profiling): Memory Profiling (4 points)

## 题目（memory_profiling）：显存分析（4 分）

Profile a complete training step for the `xl` model at context lengths 128 and 2048.

对 `xl` 模型在 context length 128 和 2048 下的完整训练步骤做显存分析。

Tasks include:

任务包括：

- Add memory-profiling support to your script.

  为脚本加入显存分析选项。

- Produce active-memory timeline screenshots for forward-only and full training.

  生成仅前向和完整训练步骤的 active memory timeline 截图。

- Report peak memory for both context lengths, in forward-only and full training.

  报告两种 context length 下，仅前向与完整训练的峰值显存。

- Compare mixed precision peak memory.

  比较混合精度下的峰值显存变化。

- Compute the residual-stream activation tensor size in MiB for `xl`.

  计算 `xl` 模型残差流激活张量在单精度下的大小（MiB）。

- Use `memory_viz` and Nsight to identify largest allocations and saved-for-backward tensors.

  使用 `memory_viz` 和 Nsight 识别最大的分配项，以及为反向传播保存的张量。

# 3 Single-GPU Memory

## 3 单 GPU 显存

Later sections explore sharding across GPUs, but there are also memory-saving techniques that help even on a single GPU. The most common is gradient checkpointing, also called activation checkpointing.

后续章节会讨论跨 GPU 的分片方法，但也有一些技巧即使在单 GPU 上也能节省显存。其中最常见的是梯度检查点，也叫激活检查点。

# 3.1 Autograd Residuals

## 3.1 Autograd 残留张量

To do backward propagation, PyTorch needs to save certain tensors from the forward pass. These are often called residuals or saved tensors.

为了执行反向传播，PyTorch 需要保存前向传播中的某些张量。这些张量通常被称为 residuals 或 saved tensors。

The assignment walks through an `RMSNorm` example and uses hooks to inspect which tensors are saved and later reloaded.

作业通过 `RMSNorm` 示例，并借助 hook 来查看哪些张量在前向时被保存、在反向时又被重新读取。

# 3.1.1 Operator Fusion

## 3.1.1 算子融合

The example shows that the operator granularity is too fine, causing too many large saved tensors. One motivation for fusion is to package the whole RMSNorm computation as a single operation so that backward can save less state.

这个例子说明当前算子粒度过细，导致保存了过多大张量。算子融合的一个核心动机就是把整个 RMSNorm 计算打包成单个操作，从而在反向传播时保存更少状态。

`torch.compile` can automatically fuse RMSNorm well enough that only the truly necessary tensors remain.

`torch.compile` 往往可以自动把 RMSNorm 融合得足够好，只留下真正必要的保存项。

# 3.2 Activation Checkpointing

## 3.2 激活检查点

Even with fusion, activation memory can remain huge. The assignment demonstrates that a single `xl` Transformer block can save several GiB of tensors for backward.

即便使用算子融合，激活显存仍可能极大。作业中演示了一个 `xl` Transformer block 就可能为反向传播保存数 GiB 的张量。

Checkpointing changes the tradeoff: instead of storing all intermediates, you store only inputs at selected checkpoints and recompute the interior during backward.

检查点机制改变了这种权衡：不再保存所有中间结果，而是在若干检查点处只保存输入，在反向时重新计算中间过程。

The example uses `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)` and shows that long-term saved memory can drop drastically.

示例使用 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`，展示了长期保存的显存如何显著下降。

# Problem (gradient_checkpointing): Memory-Optimal Gradient Checkpointing (4 points)

## 题目（gradient_checkpointing）：最优显存的梯度检查点（4 分）

(a) For a Transformer with `n` identical sequential blocks, describe the checkpointing strategy that minimizes peak activation memory if compute cost is ignored. You may nest checkpoint calls.

（a）对于一个由 `n` 个相同顺序 block 构成的 Transformer，在不考虑计算开销的前提下，描述能够最小化激活峰值显存的检查点策略。你可以嵌套 checkpoint。

(b) For the `xl` model with batch size 4 and sequence length 2048, if you are only allowed one level of recomputation (no nesting), determine the best checkpoint block size, validate it with profiles, and compare nearby choices.

（b）对于 batch size 4、sequence length 2048 的 `xl` 模型，如果你只允许一层重计算（不能嵌套 checkpoint），请找出最佳 checkpoint block 大小，并用 profile 结果验证，同时与更小和更大的相邻选择进行比较。

# 4 GPU Kernels

## 4 GPU 内核

# 4.1 Optimizing Attention with FlashAttention-2

## 4.1 用 FlashAttention-2 优化注意力

The attention operation is:

注意力操作为：

$$
\operatorname{Attention} (Q, K, V) = \operatorname{softmax} \left(\operatorname{mask} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right)\right) V
$$

Naive attention materializes a `seq_len x seq_len` score matrix for each batch/head, which causes quadratic memory growth in sequence length.

朴素注意力会为每个 batch/head 显式构造一个 `seq_len x seq_len` 的分数矩阵，因此显存会随序列长度二次增长。

FlashAttention-2 avoids materializing this matrix in HBM by computing attention in tiles.

FlashAttention-2 通过分块计算注意力，避免在 HBM 中显式存储整个注意力矩阵。

# 4.1.1 Benchmarking PyTorch Attention

## 4.1.1 基准测试 PyTorch 注意力

You are asked to benchmark your existing attention implementation over a grid of sequence lengths and hidden sizes, using batch size 8 and no multi-head dimension. Some settings are expected to OOM.

你需要在一组 sequence length 和隐藏维度配置上，对现有注意力实现做基准测试，使用 batch size 8 且不启用多头维度。一些配置预计会直接 OOM。

Deliverable highlights:

提交重点：

- Time 100 forward passes and 100 backward passes.

  计时 100 次前向和 100 次反向传播。

- Measure memory before backward starts.

  在反向开始前测量显存占用。

- Record OOM thresholds and explain memory growth with sequence length.

  记录发生 OOM 的规模阈值，并解释显存为何随序列长度增长。

# 4.2 Benchmarking JIT-Compiled Attention

## 4.2 基准测试 JIT 编译的注意力

Since PyTorch 2.0, `torch.compile` can automatically apply graph-level optimizations and often generate fused Triton kernels.

从 PyTorch 2.0 起，`torch.compile` 可以自动进行图级优化，并且常常会生成融合后的 Triton kernel。

```python
layer = SomePyTorchModule(...)
compiled_layer = torch.compile(layer)
```

# Problem (torch_compile): Torch Compile (2 points)

## 题目（torch_compile）：Torch Compile（2 分）

(a) Extend the attention benchmark to compare compiled vs uncompiled attention.

（a）扩展注意力基准测试，对比编译版与未编译版注意力实现。

(b) Compile the whole Transformer model in your end-to-end benchmark and compare forward / forward+backward / optimizer performance.

（b）在端到端基准测试中编译整个 Transformer 模型，并比较前向、前向加反向、以及含优化器更新的性能变化。

# 4.2.1 Example - Weighted Sum

## 4.2.1 示例：加权求和

This section introduces Triton through a simpler example before you implement FlashAttention. The operation multiplies each row of a matrix by a weight vector and sums over the last dimension.

这一节先通过一个更简单的例子介绍 Triton，再让你去实现 FlashAttention。这个操作会把矩阵的每一行与一个权重向量逐元素相乘，并在最后一维求和。

The write-up walks through:

文档会逐步讲解：

- A Triton forward kernel using block pointers
- A PyTorch `autograd.Function` wrapper
- A Triton backward kernel
- How to interoperate with PyTorch autograd

对应中文：

- 如何使用 block pointer 写 Triton 前向 kernel
- 如何封装成 PyTorch `autograd.Function`
- 如何写 Triton 反向 kernel
- 如何与 PyTorch autograd 协同工作

# 4.2.2 FlashAttention-2 Forward Pass

## 4.2.2 FlashAttention-2 前向传播

This is the core implementation section. You will replace your PyTorch attention with a Triton implementation following FlashAttention-2.

这是实现部分的核心章节。你将用遵循 FlashAttention-2 思路的 Triton 实现替换原有的 PyTorch 注意力实现。

Three core ideas are emphasized:

这里强调三个核心思想：

1. Tiling

   分块计算

2. Recomputation

   重计算

3. Operator fusion

   算子融合

The forward pass avoids storing the full attention matrix by computing softmax online across key tiles, while tracking running row-wise maxima and normalization terms.

前向传播通过跨 key tile 在线计算 softmax，避免存储完整注意力矩阵，同时维护按行的运行最大值和归一化量。

The algorithm also stores `L` (logsumexp), which will later simplify the backward pass.

算法还会保存 `L`（logsumexp），以简化后续反向传播。

# Problem (flash_forward): FlashAttention-2 Forward Pass (15 points)

## 题目（flash_forward）：FlashAttention-2 前向传播（15 分）

(a) First write a pure PyTorch `autograd.Function` implementation of FlashAttention-2 forward, ignoring causal masking. Save the necessary tensors for backward and allow the backward method to raise `NotImplementedError` for now.

（a）先写一个纯 PyTorch 的 `autograd.Function` 版本，实现 FlashAttention-2 的前向传播，暂时忽略 causal mask。保存反向所需张量，`backward` 目前可以直接抛出 `NotImplementedError`。

(b) Then write a Triton forward kernel that follows Algorithm 1 and wrap it in another `autograd.Function`.

（b）随后按算法 1 编写 Triton 前向 kernel，并再封装成一个 `autograd.Function`。

(c) Finally, add an optional causal-masking flag. When enabled, use query/key index comparison to construct the mask and add `-1e6` to masked attention scores.

（c）最后加入一个可选的因果掩码标志。启用时，使用 query/key 索引比较构造 mask，并对被遮蔽的位置在注意力分数上加 `-1e6`。

# Problem (flash_backward): FlashAttention-2 Backward Pass (5 points)

## 题目（flash_backward）：FlashAttention-2 反向传播（5 分）

Implement the backward pass for your FlashAttention-2 `autograd.Function` using regular PyTorch plus `torch.compile`, not Triton. Follow Equations 13 to 19 and use the precomputed `D = rowsum(O * dO)`.

使用普通 PyTorch 加 `torch.compile`（而不是 Triton）为你的 FlashAttention-2 `autograd.Function` 实现反向传播。请遵循公式 13 到 19，并使用预先计算的 `D = rowsum(O * dO)`。

# Problem (flash_benchmarking): FlashAttention-2 Benchmarking (5 points)

## 题目（flash_benchmarking）：FlashAttention-2 基准测试（5 分）

Write a benchmark using `triton.testing.do_bench` to compare your FlashAttention-2 implementation with regular PyTorch attention on a single B200.

编写一个使用 `triton.testing.do_bench` 的基准测试，在单张 B200 上比较你的 FlashAttention-2 实现与普通 PyTorch 注意力实现。

Sweep over:

扫描范围：

- Sequence lengths: powers of 2 from 128 to 65536
- Embedding dims: powers of 2 from 16 to 128
- Precisions: `torch.bfloat16` and `torch.float32`
- Batch size fixed to 1
- Causal masking enabled

对应中文：

- 序列长度：从 128 到 65536 的 2 的幂
- 嵌入维度：从 16 到 128 的 2 的幂
- 精度：`torch.bfloat16` 与 `torch.float32`
- batch size 固定为 1
- 启用 causal mask

# 5 Distributed Data Parallel Training

## 5 分布式数据并行训练

This part studies how to use multiple GPUs for training, starting from communication primitives and moving to naive and improved forms of DDP.

这一部分研究如何用多张 GPU 做训练：先从通信原语讲起，再实现朴素版和改进版 DDP。

# 5.1 Single-Node Distributed Communication in PyTorch

## 5.1 PyTorch 中的单机分布式通信

The write-up introduces `torch.distributed`, process groups, ranks, `all_reduce`, and `mp.spawn`.

文档介绍了 `torch.distributed`、进程组、rank、`all_reduce` 以及 `mp.spawn` 的基本用法。

Key points:

要点：

- Each worker process has a rank.
- `world_size` is the total number of processes.
- Collective ops act over the process group.
- Use `nccl` for GPU jobs and `gloo` mainly for CPU/local debugging.

对应中文：

- 每个 worker 进程都有一个 rank。
- `world_size` 是总进程数。
- 集体通信操作作用于整个进程组。
- GPU 训练使用 `nccl`，`gloo` 主要用于 CPU 或本地调试。

# Problem (distributed_communication_single_node): Distributed Communication (Single Node) (5 points)

## 题目（distributed_communication_single_node）：分布式通信（单机）（5 分）

This problem asks you to implement and understand basic single-node collective communication behavior in PyTorch.

这道题要求你实现并理解 PyTorch 中最基础的单机集体通信行为。

# 5.2 A Naïve Implementation of Distributed Data Parallel Training

## 5.2 朴素版分布式数据并行训练

The naive DDP approach all-reduces each parameter gradient across ranks after the backward pass.

朴素版 DDP 的做法是在反向传播结束后，对每个参数梯度分别做一次跨 rank 的 all-reduce。

# Problem (naive_ddp): Naïve DDP (5 points)

## 题目（naive_ddp）：朴素 DDP（5 分）

Implement a naive DDP wrapper that all-reduces individual parameter gradients after backward.

实现一个朴素 DDP 包装器：在反向传播之后，对每个参数的梯度单独做 all-reduce。

# Problem (naive_ddp_benchmarking): Naïve DDP Benchmarking (3 points)

## 题目（naive_ddp_benchmarking）：朴素 DDP 基准测试（3 分）

Benchmark total training-step time and the fraction spent communicating gradients for the `xl` model on 1 node x 2 GPUs.

在 1 节点 2 GPU 的设置下，对 `xl` 模型测量每个训练步骤的总时间，以及其中用于梯度通信的时间占比。

# 5.3 Improving Upon the Minimal DDP Implementation

## 5.3 改进最小化 DDP 实现

Two limitations of the naive version are highlighted:

朴素实现有两个主要问题：

1. It issues one all-reduce per parameter tensor.

   每个参数张量都单独触发一次 all-reduce。

2. It waits until backward fully finishes before starting communication.

   它要等整个 backward 完成之后才开始通信。

# 5.3.1 Reducing the Number of Communication Calls

## 5.3.1 减少通信调用次数

Flatten all gradients into one tensor, all-reduce once, then unflatten back.

把所有梯度展平拼成一个大张量，只做一次 all-reduce，再拆回各参数。

# Problem (minimal_ddp_flat_benchmarking): Minimal DDP with Flat Gradients Benchmarking (2 points)

## 题目（minimal_ddp_flat_benchmarking）：扁平化梯度的最小 DDP 基准测试（2 分）

Modify minimal DDP to communicate a single flattened gradient tensor and compare against the per-parameter all-reduce version.

把最小 DDP 修改为传输一个扁平化后的大梯度张量，并与逐参数 all-reduce 的版本做对比。

# 5.3.2 Overlapping Computation with Communication of Individual Parameter Gradients

## 5.3.2 将反向计算与逐参数梯度通信重叠

Because backward computes gradients layer by layer, parameter gradients can be communicated as soon as they are ready.

由于 backward 是逐层产生梯度的，所以某个参数梯度一旦准备好，就可以立刻启动通信，而不必等待全部梯度都算完。

This uses:

这通常需要：

- `register_post_accumulate_grad_hook`
- asynchronous collectives with `async_op=True`
- a final wait step before `optimizer.step()`

对应中文：

- `register_post_accumulate_grad_hook`
- 通过 `async_op=True` 启动异步通信
- 在 `optimizer.step()` 之前统一等待通信完成

# Problem (ddp_overlap_individual_parameters): DDP with Overlapping Individual Parameters (5 points)

## 题目（ddp_overlap_individual_parameters）：逐参数通信重叠的 DDP（5 分）

Implement a DDP container that:

实现一个 DDP 容器，要求：

- broadcasts initial weights
- wraps arbitrary `nn.Module`
- asynchronously all-reduces gradients as they become ready
- provides `finish_gradient_synchronization()`

对应中文：

- 在训练前广播初始权重
- 可包装任意 `nn.Module`
- 梯度一准备好就异步 all-reduce
- 提供 `finish_gradient_synchronization()` 接口

# Problem (ddp_overlap_individual_parameters_benchmarking): DDP Overlapping Individual Parameters Benchmarking (1 point)

## 题目（ddp_overlap_individual_parameters_benchmarking）：逐参数通信重叠 DDP 的基准测试（1 分）

Benchmark this overlapped DDP against the earlier naive and flat-gradient variants, and use Nsight to show visually whether communication overlaps with backward computation.

将这种重叠通信的 DDP 与之前的朴素版和扁平梯度版进行基准对比，并使用 Nsight 从可视化上展示通信是否与 backward 计算重叠。

# 6 Optimizer State Sharding

## 6 优化器状态分片

DDP replicates full parameters and optimizer state on every rank. For AdamW, optimizer state can take roughly 2x parameter memory.

DDP 会在每个 rank 上复制完整参数和优化器状态。以 AdamW 为例，优化器状态大约还会额外占用 2 倍参数量的显存。

Optimizer state sharding assigns each rank only a subset of parameters to optimize locally, then broadcasts updated parameters after the step.

优化器状态分片的思想是：每个 rank 只负责一部分参数的优化器状态和更新，然后在 step 之后把更新后的参数广播给其他 rank。

# Problem (optimizer_state_sharding): Optimizer State Sharding (15 points)

## 题目（optimizer_state_sharding）：优化器状态分片（15 分）

Implement a wrapper around an arbitrary PyTorch optimizer that shards parameters across ranks, performs local optimizer steps on each shard, and then synchronizes updated parameters.

实现一个针对任意 PyTorch 优化器的包装器：把参数分配到不同 rank，在本地仅更新自己的 shard，然后同步更新后的参数。

# Problem (optimizer_state_sharding_accounting): Optimizer State Sharding Accounting (5 points)

## 题目（optimizer_state_sharding_accounting）：优化器状态分片的开销分析（5 分）

You need to measure:

你需要测量：

- peak memory after initialization
- peak memory just before optimizer step
- peak memory right after optimizer step
- runtime per iteration with and without sharding
- differences vs ZeRO stage 1

对应中文：

- 初始化后的峰值显存
- optimizer step 前的峰值显存
- optimizer step 后的峰值显存
- 有无分片时的单步训练耗时
- 与 ZeRO stage 1 的差异

# 7 Fully-Sharded Data Parallel

## 7 全分片数据并行（FSDP）

Optimizer-state sharding plus DP still replicates full model weights. FSDP also shards the weights themselves.

仅做优化器状态分片加 DP 仍然会复制完整模型权重。FSDP 则进一步对权重本身也做分片。

Each GPU stores only its shard and must all-gather full weights before a layer’s forward or backward use, then free them afterward.

每张 GPU 只保存自己的权重 shard，但在某层前向或反向真正使用该层权重之前，需要先 all-gather 出完整权重，使用后再释放。

Norm-like small layers usually should not be sharded because communication overhead can dominate their compute.

像 norm 这类较小的层通常不值得分片，因为通信延迟可能会盖过其计算收益。

# Problem (fsdp): Fully-Sharded Data Parallel (15 points)

## 题目（fsdp）：全分片数据并行（15 分）

Implement an FSDP wrapper that:

实现一个 FSDP 包装器，要求：

- wraps a full model
- shreds Linear / Embedding weights across ranks
- all-gathers weights ahead of use
- reduce-scatters gradients
- optionally uses lower `compute_dtype` for communication and compute while keeping master weights in FP32

对应中文：

- 包装整个模型
- 对其中的 Linear / Embedding 权重做跨 rank 分片
- 在真正使用前提前 all-gather 权重
- 对梯度做 reduce-scatter
- 可选地在通信和计算时使用更低的 `compute_dtype`，同时保留 FP32 主权重

# Problem (fsdp_accounting): FSDP Accounting (5 points)

## 题目（fsdp_accounting）：FSDP 开销分析（5 分）

Estimate expected peak-memory savings from FSDP and profile whether the all-gathers finish early enough not to block forward execution.

估算 FSDP 可节省的峰值显存，并通过 profile 验证权重 all-gather 是否能及时完成，从而不阻塞前向执行。

# 8 Analyzing Parallelism Strategies

## 8 并行策略分析

This section asks you to do analytical communication/computation tradeoff calculations for several strategies:

这一节要求你对多种并行策略做通信与计算的权衡分析：

- Data Parallelism (DP)
- Fully-Sharded Data Parallelism (FSDP)
- Tensor Parallelism (TP)
- 2D Parallelism (FSDP + TP)

# 8.1 Communication Primitives

## 8.1 通信原语

The section derives idealized runtimes for ring all-gather, ring reduce-scatter, and ring all-reduce in terms of tensor size `S`, number of devices `N`, and per-device egress bandwidth `W`.

这一节推导了在理想化模型下，ring all-gather、ring reduce-scatter 和 ring all-reduce 的时间表达式，变量包括张量大小 `S`、设备数 `N` 和每台设备的出站带宽 `W`。

# Problem (alternate_ring_all_reduce): Alternate ring all-reduce (1 point)

## 题目（alternate_ring_all_reduce）：另一种 ring all-reduce（1 分）

Analyze the alternative ring all-reduce algorithm described in the text and derive its total runtime.

分析文中给出的另一种 ring all-reduce 算法，并推导其总运行时间。

# 8.2 Analyzing Data Parallel

## 8.2 分析数据并行

This section focuses on a single FFN layer and asks you to compute backward FLOPs, communication time, and the maximum DP degree before communication becomes the bottleneck.

这一节以单个 FFN 层为对象，要求你计算 backward 的 FLOPs、通信时间，以及在通信成为瓶颈之前 DP 最大能扩展到多少设备。

# Problem (data_parallel_calcs): Data parallel calculations (3 points)

## 题目（data_parallel_calcs）：数据并行计算题（3 分）

For DP, derive:

对于 DP，请推导：

- backward FLOPs
- backward communication time
- maximum `N_DP` before communication bottlenecks

对应中文：

- 反向传播 FLOPs
- 反向传播通信时间
- 在通信成为瓶颈前，`N_DP` 能达到的最大规模

# 8.3 Analyzing Fully Sharded Data Parallel

## 8.3 分析 FSDP

This section asks for analogous calculations for FSDP, including all-gathers in forward/backward and reduce-scatters of gradients.

这一节要求你对 FSDP 做类似推导，包括前向/反向中的 all-gather，以及梯度的 reduce-scatter。

# Problem (fsdp_calcs): Fully sharded data parallel calculations (3 points)

## 题目（fsdp_calcs）：全分片数据并行计算题（3 分）

Derive forward/backward FLOPs, communication time, and scaling limits for FSDP.

推导 FSDP 的前向/反向 FLOPs、通信时间以及可扩展上限。

# 8.4 Analyzing Tensor Parallel

## 8.4 分析张量并行

This section studies a tensor-parallel FFN where `W1` and `W2` are column-parallel and `W3` is row-parallel.

这一节研究一种张量并行的 FFN：`W1` 和 `W2` 使用列并行，`W3` 使用行并行。

# Problem (tp_calcs): Tensor parallel calculations (4 points)

## 题目（tp_calcs）：张量并行计算题（4 分）

You need to:

你需要：

- write out the backward pass equations for the sharded FFN
- compute forward/backward FLOPs
- compute forward/backward communication time
- derive scaling limits before communication bottlenecks

对应中文：

- 写出该分片 FFN 的反向传播方程
- 计算前向/反向 FLOPs
- 计算前向/反向通信时间
- 推导在通信成为瓶颈前的扩展上限

# 8.5 2D Parallelism (FSDP + TP)

## 8.5 二维并行（FSDP + TP）

This final analytical section combines FSDP and TP on a 2D device grid, with separate TP and FSDP ranks.

这一节把 FSDP 和 TP 组合在二维设备网格上，每个设备同时具有 TP rank 和 FSDP rank。

# Problem (fsdp_tp_calcs): 2D parallelism calculations (6 points)

## 题目（fsdp_tp_calcs）：二维并行计算题（6 分）

Derive forward FLOPs, communication time, and the maximum total number of devices `N = N_TP * N_FSDP` under optimal balancing, both with and without overlap between FSDP-axis and TP-axis collectives.

推导二维并行下的前向 FLOPs、通信时间，以及在最优划分下总设备数 `N = N_TP * N_FSDP` 的上限；同时分别讨论 FSDP 轴与 TP 轴通信可以重叠和不能重叠两种情况。

# 9 Leaderboard

## 9 排行榜

The leaderboard benchmark measures a full training step for an 8B model on two B200 GPUs, with batch size 2 and sequence length 32768.

排行榜基准会在两张 B200 GPU 上测量一个 8B 模型的完整训练步骤，batch size 为 2，sequence length 为 32768。

Restrictions:

限制条件：

- You may optimize aggressively, but cannot change the model’s input/output behavior.
- Your implementation must remain correct under BF16 with causal masking.
- It must pass the same tests as your normal implementation.
- It must be your own implementation.

对应中文：

- 可以激进优化，但不能改变模型输入输出行为。
- 你的实现必须在 BF16 和 causal mask 下保持正确。
- 必须通过与常规实现相同的测试。
- 必须是你自己的实现。

The benchmark times forward, loss, backward, and AdamW update together.

基准测试会把前向、loss、反向传播和 AdamW 更新整体一起计时。

The assignment suggests possible optimizations such as:

文档给出了一些可能的优化方向：

- tuning Triton tile sizes / autotune
- tuning `torch.compile`
- fused AdamW
- fused LM head + cross-entropy
- improved FlashAttention backward
- skipping guaranteed-zero causal tiles
- activation checkpointing if memory is tight

对应中文：

- 调整 Triton tile size / autotune
- 调整 `torch.compile`
- 融合版 AdamW
- 融合 LM head 与 cross-entropy
- 进一步优化 FlashAttention backward
- 跳过因果 mask 下必为零的 tile
- 当显存紧张时使用激活检查点

# Problem (leaderboard): Leaderboard: fastest training step (10 points)

## 题目（leaderboard）：排行榜，最快训练步（10 分）

Submit your best wall-clock time for a full forward-and-backward training step with AdamW.

提交你在包含 AdamW 更新的完整前向加反向训练步骤上的最佳实际墙钟时间。

The baseline is about 10 seconds, and leaderboard solutions are expected to beat it.

朴素基线大约是 10 秒，排行榜提交应当优于这一基线。

# Bibliography

## 参考文献

[1] T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.” Available: https://arxiv.org/abs/2307.08691

[2] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” NeurIPS 2022. Available: https://openreview.net/forum?id=H4DqfPSibmx

[3] M. Milakov and N. Gimelshein, “Online normalizer calculation for softmax.” Available: https://arxiv.org/abs/1805.02867

[4] H. He, “Making Deep Learning Go Brrrr From First Principles,” 2022. Available: https://horace.io/brrr_intro.html

[5] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.” 2020.

[6] J. Austin et al., “How to Scale Your Model,” 2025.

[7] H. Z. P. N. M. M. L. W. T. W. Nouamane Tazi Ferdinand Mom, “The Ultra-Scale Playbook: Training LLMs on GPU Clusters.” 2025.

---

说明：

这个文件保留了原作业的核心英文内容，并补充了逐节中文翻译与要点说明，代码块、公式、表格和术语名称尽量保持原样，便于你对照阅读与继续完成作业。
