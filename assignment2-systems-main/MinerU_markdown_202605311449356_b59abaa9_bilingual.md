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

在本次作业中，你将通过动手实践，学习如何提升单 GPU 训练速度，以及如何把训练扩展到多 GPU。

# What you will implement

## 你将实现的内容

1. Benchmarking and profiling harness 

1. 基准测试与性能分析框架

2. Activation checkpointing 

2. 激活检查点

3. Flash Attention 2 Triton kernel 

3. Flash Attention 2 Triton 内核

4. Distributed data parallel training 

4. 分布式数据并行训练

5. Optimizer state sharding 

5. 优化器状态分片

6. Fully sharded data parallel training 

6. 全分片数据并行训练

# What the code looks like

## 代码结构

The assignment code and this write-up are available on GitHub at: 

本次作业的代码和说明文档可在 GitHub 获取：

github.com/stanford-cs336/assignment2-systems 

请使用 Git 克隆该仓库。如果后续有更新，我们会通知你；届时你可以通过 `git pull` 获取最新版本。

Please clone the repository using Git. If there are any updates, we will notify you and you can git pull to get the latest. 

1. cs336-basics/: In this assignment, you’ll be profiling some of the components that we built in assignment 1. This folder contains the staff solution code for assignment 1, so you will find cs336- basics/pyproject.toml and the cs336-basics/cs336_basics package here. If you want to use your own implementation of the model, you can modify the pyproject.toml file in the base directory to point to your own package. 

1. `cs336-basics/`：在本次作业中，你会对作业 1 中构建的一些组件做性能分析。该文件夹中包含作业 1 的课程组参考解，因此你会在这里看到 `cs336-basics/pyproject.toml` 和 `cs336-basics/cs336_basics` 包。如果你希望使用自己实现的模型，可以修改根目录中的 `pyproject.toml`，使其指向你自己的包。

2. /: The cs336-systems base directory. We created an empty module named cs336_systems. Note that there’s no code in here, so you should be able to do whatever you want from scratch. 

2. `/`：`cs336-systems` 根目录。我们创建了一个名为 `cs336_systems` 的空模块。注意这里没有现成代码，因此你应该可以完全从零开始，自由实现。

3. tests/*.py: This directory contains all the tests you must pass. These tests invoke the hooks defined in tests/adapters.py. You’ll implement the adapters to connect your code to the tests. Writing more tests and/or modifying the test code can be helpful for debugging your code, but your implementation is expected to pass the original provided test suite. 

3. `tests/*.py`：这个目录中包含你必须通过的全部测试。这些测试会调用 `tests/adapters.py` 中定义的接口。你将实现这些 adapter，把你的代码接到测试上。你也可以自行编写更多测试，或者修改测试代码来帮助调试，但最终你的实现必须通过原始提供的测试集。

4. README.md: This file contains more details about the expected directory structure, as well as some basic instructions on setting up your environment. 

4. `README.md`：该文件包含预期目录结构的更多细节，以及一些基础环境配置说明。

# How to submit

## 提交方式

You will submit the following files to Gradescope: 

你需要向 Gradescope 提交以下文件：

• writeup.pdf: Answer all the written questions. Please typeset your responses. 

• `writeup.pdf`：回答所有书面问题。请将你的答案排版整理后提交。

• code.zip: Contains all the code you’ve written. 

• `code.zip`：包含你编写的全部代码。

Run the script in test_and_make_submission.sh to create the code.zip file. 

运行 `test_and_make_submission.sh` 中的脚本来生成 `code.zip` 文件。

# 2 Profiling and Benchmarking

## 2 性能分析与基准测试

In the first part of the assignment, we will explore how to optimize the performance of our Transformer model to make the most efficient use of the GPU. We will profile our model to understand where it spends time and memory during the forward and backward passes, then optimize the self-attention operation with custom GPU kernels, making it faster than is possible with regular PyTorch. In the subsequent parts of the assignment, we will leverage multiple GPUs and understand how to train a model across a cluster. 

在作业的第一部分中，我们将探索如何优化 Transformer 模型的性能，以便最高效地利用 GPU。我们会对模型进行性能分析，理解它在前向和反向传播过程中把时间和显存耗费在了哪里；随后我们会使用自定义 GPU 内核来优化自注意力操作，使其速度超过常规 PyTorch 实现。作业后续部分则会进一步利用多张 GPU，并理解如何在一个集群上训练模型。

# 2.1 Profiling

## 2.1 性能分析

Before implementing any optimizations, it is helpful to first profile our program to understand where it spends resources (e.g., time and memory). Otherwise, we risk optimizing parts of the model that don’t account for significant time or memory, and therefore not seeing measurable end-to-end improvements. 

在实现任何优化之前，先分析程序把资源花在了哪里（例如时间与显存）会很有帮助。否则，我们可能会去优化那些并不占主要时间或显存的部分，最终看不到可测量的端到端收益。

We will implement three performance evaluation paths: 

我们将实现三种性能评估路径：

1. Simple end-to-end benchmarking using the Python standard library to time our forward and backward passes 

1. 使用 Python 标准库做简单的端到端基准测试，测量前向和反向传播耗时。

2. Compute profiling with the NVIDIA Nsight Systems tool to understand how that time is distributed across operations on both the CPU and GPU 

2. 使用 NVIDIA Nsight Systems 工具做计算性能分析，理解这些时间是如何分布到 CPU 与 GPU 上各类操作中的。

3. Memory profiling 

3. 显存分析。

# 2.1.1 Setup - Importing your basics Transformer Model

## 2.1.1 环境准备：导入你在 basics 作业中的 Transformer 模型

Let’s start by making sure that you can load the model from the previous assignment. In the previous assignment, we set up our model in a Python package, so that it could be easily imported later. We have added a reference implementation of the model in the ./cs336-basics folder, and have pointed to it in the pyproject.toml file. By calling uv run [command] as usual, uv will automatically locate this local cs336-basics package. If you would like to use your own implementation of the model, you can modify the pyproject.toml file to point to your own package. 

先确保你能够加载上一份作业中的模型。在上一份作业中，我们把模型组织成了一个 Python 包，这样它以后就能很方便地被导入。我们已经在 `./cs336-basics` 文件夹中加入了模型的参考实现，并且在 `pyproject.toml` 中把它配置好了。像平常一样调用 `uv run [command]` 时，`uv` 会自动定位这个本地的 `cs336-basics` 包。如果你想使用自己实现的模型，可以修改 `pyproject.toml`，让它指向你自己的包。

You can test that you can import your model with: 

你可以用下面的方式测试是否能够导入模型：

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

The relevant modules from assignment 1 should now be available (e.g., for model.py, you can import it with import cs336_basics.model). 

此时，作业 1 中的相关模块都应该已经可用了（例如，对于 `model.py`，你可以通过 `import cs336_basics.model` 导入它）。

# 2.1.2 Model Sizing

## 2.1.2 模型规模设置

Throughout this assignment, we will be benchmarking and profiling models to better understand their performance. To get a sense of how things change at scale, we will work with and refer to the following model configurations. For all models except in the leaderboard, we’ll use a vocabulary size of 10,000 and a batch size of 4, with varying context lengths. This assignment (and later ones) will require a lot of results to be presented in tables and plots. We strongly recommend that you automate constructing tables for 

在整个作业中，我们会不断对模型做基准测试和性能分析，以便更好地理解它们的性能表现。为了感受规模变化带来的影响，我们将使用并反复引用下面这些模型配置。除排行榜部分外，所有模型都使用 10,000 的词表大小和 batch size 4，而上下文长度则会变化。

your writeup in code, since formatting tables in LaTeX or Typst can be very tedious. See pandas.DataFrame.to_latex() and pandas.DataFrame.to_typst() or write your own function to generate them from your preferred tabular representation. 

本次作业（以及后续作业）会要求你用大量表格和图来展示结果。我们强烈建议你用代码自动生成报告中的表格，因为手工在 LaTeX 或 Typst 中排表通常会非常繁琐。可以参考 `pandas.DataFrame.to_latex()`、`pandas.DataFrame.to_typst()`，或者编写你自己的函数，从偏好的表格数据表示中自动生成。

<table><tr><td>Size</td><td>d_model</td><td>d_ff</td><td>num_layers</td><td>num_heads</td></tr><tr><td>small</td><td>768</td><td>3072</td><td>12</td><td>12</td></tr><tr><td>medium</td><td>1024</td><td>4096</td><td>24</td><td>16</td></tr><tr><td>large</td><td>1280</td><td>5120</td><td>36</td><td>20</td></tr><tr><td>xl</td><td>2560</td><td>10240</td><td>32</td><td>32</td></tr><tr><td>10B</td><td>4608</td><td>12288</td><td>50</td><td>36</td></tr></table>

Table 1: Specifications of different model sizes. These are mostly based on GPT-2 configs.

表 1：不同模型规模的配置规格。这些配置大多参考 GPT-2。

Use context length 512 unless otherwise specified. 

除非另有说明，否则使用 context length 512。

# 2.1.3 End-to-End Benchmarking

## 2.1.3 端到端基准测试

We will now implement a simple performance evaluation script. We will be testing many variations of our model (changing precision, swapping layers, etc.), so it will pay off to have your script enable these variations via command-line arguments to make them easy to run later on. 

现在我们要实现一个简单的性能评估脚本。后面我们会测试模型的许多不同变体（例如修改精度、更换层实现等），因此如果你的脚本能通过命令行参数启用这些变体，后续使用起来会方便很多。

To start off, let’s do the simplest possible profiling of our model by timing the forward pass, backward pass, and optimizer step. Since we will only be measuring speed and memory, it’s fine to use random weights and data. 

作为第一步，我们先做最简单的模型性能分析：测量前向传播、反向传播和优化器步进的耗时。由于我们这里只测速度和显存，因此使用随机权重和随机数据就足够了。

Measuring performance is subtle — some common traps can cause us to not measure what we want. For benchmarking GPU code, one caveat is that CUDA calls are asynchronous. When you call a CUDA kernel, such as when you invoke torch.matmul, the PyTorch function call returns control to your code without waiting for the matrix multiplication to finish. In this way, the CPU can continue running ahead and scheduling new operations while the GPU finishes the matrix multiplication, which is a major performance win. On the other hand, this means that naïvely measuring how long the torch.matmul call takes to return does not tell us how long the GPU takes to actually run the matrix multiplication. In PyTorch, we can call torch.cuda.synchronize() to wait for all scheduled GPU kernels to complete, allowing us to get more accurate measurements of CUDA kernel runtime. The synchronization in this operation refers to synchronizing the CPU runtime with the GPU runtime. With this in mind, let’s write our basic profiling infrastructure. 

性能测量其实有很多细节陷阱，很容易测不到自己真正关心的东西。对于 GPU 代码，最常见的一个注意点是 CUDA 调用是异步的。当你调用一个 CUDA kernel，比如执行 `torch.matmul` 时，PyTorch 这个函数调用会在矩阵乘法真正完成之前就把控制权交还给你的代码。这样 CPU 可以继续向前运行，调度新的操作，而 GPU 同时在后台完成矩阵乘法，这是性能上的重要收益。但这也意味着，如果你只是天真地测量 `torch.matmul` 这一行 Python 代码返回所花的时间，并不能得到 GPU 真正执行矩阵乘法所需的时间。在 PyTorch 中，我们可以调用 `torch.cuda.synchronize()` 等待所有已调度的 GPU kernel 完成，从而更准确地测量 CUDA kernel 的运行时间。这里所说的同步，是指让 CPU 运行时与 GPU 运行时对齐。带着这一点，我们来搭建基础的 profiling 框架。

# Problem (benchmarking_script):  Benchmarking Script (4 points)

### 题目（benchmarking_script）：基准测试脚本（4 分）

(a) Write a script to perform basic end-to-end benchmarking of the forward pass, backward pass, and optimizer step in your model. Specifically, your script should support the following: 

(a) 编写一个脚本，对模型的前向传播、反向传播和优化器步进做基础的端到端基准测试。具体来说，你的脚本应支持以下内容：

• Given hyperparameters (e.g., number of layers), initialize a model. 

• 给定超参数（例如层数）后初始化模型。

• Generate a random batch of data. 

• 生成一批随机数据。

• Run `w` warm-up steps (before you start measuring time), then time the execution of `n` steps (either only forward, forward and backward, or forward and backward with optimizer step, depending on an argument). For timing, you can use the Python `timeit` module (e.g., either using the `timeit` function, or using `timeit.default_timer()`, which gives you the system’s highest resolution clock, thus a better default for benchmarking than `time.time()`). 

• 先运行若干次预热步骤，再对若干次执行做计时；根据命令行参数，计时对象可以是仅前向、前向加反向、或者前向加反向再加优化器步进。计时时，你可以使用 Python 的 `timeit` 模块（例如 `timeit` 函数本身，或者 `timeit.default_timer()`；后者会使用系统上分辨率最高的时钟，因此比 `time.time()` 更适合做基准测试）。

• Call torch.cuda.synchronize() after each step. 

• 在每一步之后调用 `torch.cuda.synchronize()`。

Deliverable: A script that will initialize a basics Transformer model with the given hyperparameters, create a random batch of data, and time forward-only, forward-andbackward, and full training steps that include the optimizer step. 

提交内容：一个脚本，它能够根据给定超参数初始化 basics Transformer 模型，生成随机 batch，并对仅前向、前向加反向、以及包含优化器步进的完整训练步骤进行计时。

(b) Time the forward, backward, and optimizer step for the model sizes described in Section 2.1.2. Use 5 warmup steps and compute the average and standard deviation of timings over 10 measurement steps. How long does a forward pass take? How about a backward pass? Do you see high variability across measurements, or is the standard deviation small? 

(b) 对第 2.1.2 节中列出的模型规模，测量前向、反向以及优化器步进的时间。使用 5 次 warmup，并在 10 次正式测量上计算时间的平均值和标准差。前向传播需要多久？反向传播呢？不同测量之间的波动大吗，还是标准差很小？

Deliverable: A 1-2 sentence response with your timings. 

提交内容：用 1 至 2 句话给出你的计时结果。

(c) One caveat of benchmarking is not performing the warm-up steps. Repeat your analysis without the warm-up steps. How does this affect your results? Why do you think this happens? Also try to run the script with 1 or 2 warm-up steps. Why might the result still be different? 

(c) 做基准测试时，一个常见问题是不进行 warm-up。请在不做 warm-up 的情况下重复你的分析。这样会如何影响结果？你认为为什么会这样？另外再尝试只做 1 次或 2 次 warm-up。为什么结果仍然可能不同？

Deliverable: A 2-3 sentence response. 

提交内容：2 至 3 句话说明。

# 2.1.4 Nsight Systems Profiler

## 2.1.4 Nsight Systems 分析器

End-to-end benchmarking does not tell us where our model spends time and memory during forward and backward passes, and so does not expose specific optimization opportunities. To know how much time our program spends in each component (e.g., function), we can use a profiler. An execution profiler instruments the code by inserting guards when functions begin and finish running, and thus can give detailed execution statistics at the function level (such as number of calls, how long they take on average, cumulative time spent on this function, etc). 

端到端基准测试并不能告诉我们模型在前向和反向传播中具体把时间和显存花在了哪里，因此也就不能直接暴露出明确的优化机会。为了知道程序在各个组成部分（例如函数）中花了多少时间，我们需要用 profiler。执行型 profiler 会在函数开始和结束时插入监测逻辑，因此可以给出函数粒度上的详细统计信息（比如调用次数、平均耗时、在该函数上累计花费的时间等）。

Standard Python profilers (e.g., CProfile) are not able to profile CUDA kernels since these kernels are executed asynchronously on the GPU. Fortunately, NVIDIA ships a profiler that we can use via the CLI nsys. We recommend that you get an up-to-date version either from your package manager, or using the installers from their download page. In this part of the assignment, you will use nsys to analyze the runtime of your Transformer model. 

标准的 Python profiler（例如 `cProfile`）无法分析 CUDA kernel，因为这些 kernel 在 GPU 上是异步执行的。幸运的是，NVIDIA 提供了一个可以通过命令行使用的 profiler：`nsys`。我们建议你通过包管理器或官方下载页面安装一个较新的版本。在这部分作业中，你将使用 `nsys` 来分析 Transformer 模型的运行时表现。

Using nsys is straightforward: run your Python script from the previous section with nsys profile prepended. For example, you can run a basic profile for the script benchmark.py with: 

`nsys` 的使用方式很直接：在上一节里的 Python 脚本前面加上 `nsys profile` 即可。例如，如果你的脚本名叫 `benchmark.py`，你可以像下面这样执行一个基础 profile：

$ uv run nsys profile -- python benchmark.py 

然后，你可以在本地机器上使用 NVIDIA Nsight Systems 桌面应用打开生成的 profile。若你在 profile 的 CUDA API 行（CPU 侧）中选中某个具体的 CUDA API 调用，CUDA HW 行（GPU 侧）中所有对应的 kernel 执行就会被高亮显示出来。

You can then view the profile on your local machine with the 

NVIDIA Nsight Systems desktop application. Selecting a particular CUDA API call (on the CPU) in the CUDA API row of the profile will highlight all corresponding kernel executions (on the GPU) in the CUDA HW row. 

更完整的一次 profile 命令可能长这样：

A more comprehensive profiling run may look like: 

$ uv run nsys profile --trace=cuda,cudnn,cublas,osrt,nvtx --pytorch=functions-trace,autogradshapes-nvtx --cudabacktrace=all --python-backtrace=cuda --gpu-metrics-devices=0 -- python benchmark.py 

In this example, --trace specifies which APIs to log, --pytorch inserts nvtx labels during module calls and autograd, --cudabacktrace and --python-backtrace give better backtraces to understand where in your code a given kernel was invoked from, and --gpu-metrics-devices specifies which GPU’s utilization to measure. 

在这个例子中，`--trace` 指定了需要记录哪些 API；`--pytorch` 会在模块调用与 autograd 过程中插入 NVTX 标签；`--cudabacktrace` 和 `--python-backtrace` 会提供更好的回溯信息，帮助你理解某个 kernel 是从你代码中的哪里被调用的；`--gpu-metrics-devices` 用来指定要测量哪张 GPU 的利用率。

Adding profiling to a run is not free, and will overall slow down your runs. It’s often worth only enabling features you’re looking for in a given run. Specifically, you might want to remove --cudabacktrace=all and --python-backtrace=cuda when tracebacks aren’t needed, since they have outsize overhead. 

给一次运行加入 profiling 并不是免费的，它会整体拖慢你的程序。通常只打开你当前需要的特性会更划算。特别是，当你不需要回溯信息时，可以去掉 `--cudabacktrace=all` 和 `--python-backtrace=cuda`，因为它们的额外开销非常大。

We encourage you to experiment with various command-line options for nsys profile to get a sense of what it can do. You can also annotate your code with NVTX ranges, which will appear as blocks in the NVTX row of the profile capturing all CUDA API calls and associated kernel executions. In particular, you should use NVTX ranges to ignore the warm-up steps in your benchmarking script (by applying an --nvtx-capture filter on the nvtx label in the profile). You can also isolate which kernels are responsible for the forward and backward passes of your model, and you can even isolate which kernels are responsible for different parts of a self-attention layer by annotating your implementation as follows: 

我们鼓励你多尝试不同的 `nsys profile` 命令行选项，以了解它能做什么。你也可以在代码中自行加入 NVTX range 标记，它们会出现在 profile 的 NVTX 行中，以块的形式包住相关的 CUDA API 调用和相应的 kernel 执行。尤其是，你应当使用 NVTX ranges 在 benchmarking 脚本中屏蔽 warm-up 步骤（通过 profile 时对 NVTX 标签应用 `--nvtx-capture` 过滤）。你也可以借此分离出模型前向与反向传播对应的 kernel，甚至进一步隔离自注意力层中不同部分各自对应的 kernel，例如按下面这种方式给实现加标注：

```python
import torch.cuda.nvtx as nvtx

@nvtx.range("scaled dot product attention")
def annotated_scaled_dot_product_attention(
    ... # Q, K, V, mask
):
    ...
    with nvtx.range("computing attention scores"):
    ... # compute attention scores between Q and K

    with nvtx.range("computing softmax"):
    ... # compute softmax of attention scores

    with nvtx.range("final matmul"):
    ... # compute output projection 
```

You can swap your original implementation with the annotated version in your benchmarking script via: 

cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention 

你可以在 benchmarking 脚本中用下面这行代码，把原始实现替换成带注释版本：

Finally, it’s worth noting that torch.compile can make it hard to attribute time and resources to specific parts of your code. You will likely have to wrap and strip various parts of your code in torch.compile and nvtx annotations to correctly attribute time and resource usage to various parts of your source. 

最后还需要注意，`torch.compile` 会让时间与资源归因变得更困难。为了正确判断源码中各部分各自消耗了多少时间和资源，你很可能需要反复对不同代码片段加上或移除 `torch.compile` 与 NVTX 标注。

你可以在 benchmarking 脚本中用下面这种方式，把原始实现替换成这个带标注的版本：

最后需要注意的是，`torch.compile` 会让你更难把时间和资源准确归因到具体源代码片段上。为了正确归因，你很可能需要对代码的不同部分分别做 `torch.compile` 包裹或剥离，并适当加上 NVTX 标注。

# Problem (nsys_profile):  Nsight Systems Profiling (5 points)

### 题目（nsys_profile）：Nsight Systems 性能分析（5 分）

Profile your forward pass, backward pass, and optimizer step using nsys with two model sizes from Table 1 of your choice as well as three power-of-two context lengths larger than 128, where the largest available size should be the longest context length you can fit in memory. Pick the combinations you think would be the most interesting to look at. For each profile answer the following questions: 

使用 `nsys` 对你的前向传播、反向传播和优化器步进做性能分析。模型规模从表 1 中任选两个；同时再选三个大于 128 的 2 的幂作为 context length，其中最大值应当是你显存里实际能放下的最大长度。请选择你认为最值得观察的组合。对每个 profile，回答以下问题：

(a) What is the total time spent on your forward pass? Does it match what we had measured before with the Python standard library? 

(a) 你的前向传播总共花了多少时间？这个结果和之前使用 Python 标准库测得的时间一致吗？

Deliverable: A 1-2 sentence response. 

提交内容：1 至 2 句话说明。

(b) What CUDA kernel takes the most cumulative GPU time during the forward pass? How many times is this kernel invoked during a single forward pass of your model? Is it the same kernel that takes the most runtime when you do both forward and backward passes? (Hint: look at the “CUDA GPU Kernel Summary” under “Stats System View”, and filter using NVTX ranges to identify which parts of the model are responsible for which kernels.) 

(b) 在前向传播中，哪个 CUDA kernel 占据了最多累计 GPU 时间？在单次前向传播中，这个 kernel 被调用了多少次？如果你同时做前向和反向传播，耗时最多的 kernel 还是它吗？提示：请查看 “Stats System View” 里的 “CUDA GPU Kernel Summary”，并结合 NVTX range 过滤，识别模型中哪些部分对应哪些 kernel。

Deliverable: A 1-2 sentence response. 

提交内容：1 至 2 句话说明。

(c) Although the vast majority of FLOPs take place in matrix multiplications, you will notice that several other kernels still take a non-trivial amount of the overall runtime. What other kernels besides matrix multiplies do you see accounting for non-trivial CUDA runtime in the forward pass? 

(c) 虽然绝大部分 FLOPs 都发生在矩阵乘法中，但你会注意到还有一些其他 kernel 也占据了不可忽略的运行时间。在前向传播中，除了矩阵乘法外，你还看到哪些 kernel 占据了比较可观的 CUDA 运行时间？

Deliverable: A 1-2 sentence response. 

提交内容：1 至 2 句话说明。

(d) Profile running one complete training step with your implementation of AdamW (i.e., the forward pass, computing the loss and running a backward pass, and finally an optimizer step, as you’d do during training). How does the fraction of time spent on matrix multiplication change, compared to doing inference (forward pass only)? How about other kernels? 

(d) 用你实现的 AdamW 跑一个完整训练步骤做 profile（也就是：前向、计算 loss、反向传播、最后优化器更新，就像训练时那样）。与仅做推理（只前向）相比，矩阵乘法所占时间比例发生了怎样的变化？其他 kernel 呢？

Deliverable: A 1-2 sentence response. 

提交内容：1 至 2 句话说明。

(e) Compare the runtime of the softmax operation versus the matrix multiplication operations within the self-attention layer of your model during a forward pass. How does the difference in runtimes compare to the difference in FLOPs? 

(e) 在模型前向传播中，比较 self-attention 层里 softmax 操作与矩阵乘法操作的运行时间。它们在运行时间上的差异，与 FLOPs 上的差异相比如何？

Deliverable: A 1-2 sentence response. 

提交内容：1 至 2 句话说明。

# 2.1.5 Mixed Precision

## 2.1.5 混合精度

Up to this point in the assignment, we’ve been running with FP32 precision—all model parameters and activations have the torch.float32 datatype. However, modern NVIDIA GPUs contain specialized GPU cores (Tensor Cores) for accelerating matrix multiplies at lower precisions. For example, the NVIDIA B200 spec sheet says that its maximum throughput with FP32 is 80 TFLOPS, while its maximum throughput with FP16 (half-precision floats) or BF16 (bfloat16) is significantly higher at a whopping 2500 TFLOPS. As a result, using lower-precision datatypes should help us speed up training and inference. 

到目前为止，本作业中我们一直都在使用 FP32 精度，也就是所有模型参数和激活都具有 `torch.float32` 数据类型。然而，现代 NVIDIA GPU 内置了专门用于低精度矩阵乘法加速的 Tensor Cores。例如，NVIDIA B200 的规格表显示，其 FP32 的最大吞吐量为 80 TFLOPS，而 FP16（半精度浮点）或 BF16（bfloat16）的最大吞吐量则高得多，达到惊人的 2500 TFLOPS。因此，使用更低精度的数据类型应当有助于加快训练和推理速度。

However, naïvely casting our model into a lower-precision format may come with reduced model accuracy. For example, many gradient values in practice are often too small to be representable in FP16, and thus become zero when naïvely training with FP16 precision. To combat this, it’s common to use loss scaling when training with FP16—the loss is simply multiplied by a scaling factor, increasing gradient magnitudes so they don’t flush to zero. Furthermore, FP16 has a lower dynamic range than FP32, which can lead to overflows that manifest as a NaN loss. Full bfloat16 training is generally more stable (since BF16 has the same dynamic range as FP32), but can still affect final model performance compared to FP32. 

不过，如果只是把模型天真地整体转换成低精度格式，可能会损失模型精度。例如，在实际中，许多梯度值往往小到 FP16 无法表示，于是在直接用 FP16 训练时会被冲刷成 0。为了解决这个问题，人们在 FP16 训练时常常会使用 loss scaling，也就是把 loss 乘上一个缩放因子，使梯度的数值范围扩大，不至于直接下溢为 0。除此之外，FP16 的动态范围也比 FP32 小，因此更容易发生溢出，并体现为 loss 变成 NaN。完全使用 bfloat16 训练通常会更稳定一些（因为 BF16 与 FP32 具有相同的动态范围），但与 FP32 相比，它仍然可能影响模型的最终性能。

To take advantage of the speedups from lower-precision datatypes, it’s common to use mixed-precision training. In PyTorch, this is implemented with the torch.autocast context manager. In this case, certain operations (e.g., matrix multiplies) are performed in lower-precision datatypes, while other operations that require the full dynamic range of FP32 (e.g., accumulations and reductions) are kept as-is. For example, the following code will automatically identify which operations to perform in lower-precision during the forward pass and cast these operations to the specified data type: 

为了既利用低精度带来的速度提升，又避免完全低精度训练的数值问题，实践中常会使用混合精度训练。在 PyTorch 中，这通常通过 `torch.autocast` 上下文管理器实现。这样一来，一部分操作（例如矩阵乘法）会在更低精度下执行，而另外一些需要 FP32 全动态范围的操作（例如累加和归约）则保持高精度。例如，下面这段代码会在前向传播时自动识别哪些操作适合以低精度执行，并把这些操作转成指定的数据类型：

```python
model : torch.nn.Module = ... # e.g. your Transformer model
dtype : torch.dtype = ... # e.g. torch.bfloat16
x : torch.Tensor = ... # input data

with torch.autocast(device_type="cuda", dtype=dtype):
    y = model(x) 
```

As alluded to above, it is generally a good idea to keep accumulations in higher precision even if the tensors themselves being accumulated have been downcast. The following exercise will help build your intuition as to why this is the case. 

正如上面提到的那样，即使被累加的张量本身已经被降为低精度，通常也仍然最好让“累加”本身在更高精度下进行。下面这个练习会帮助你建立直觉，理解为什么要这样做。

# Problem (mixed_precision_accumulation):  Mixed-Precision Accumulation (1 point)

### 题目（mixed_precision_accumulation）：混合精度累加（1 分）

Run the following code and comment on the accuracy of the results. 

运行下面的代码，并评论结果的准确性。

```python
s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float32)
print(s)

s = torch.tensor(0, dtype=torch.float16)
for i in range(1000):
    s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000): 
```

```python
s += torch.tensor(0.01, dtype=torch.float16)
print(s)

s = torch.tensor(0, dtype=torch.float32)
for i in range(1000):
    x = torch.tensor(0.01, dtype=torch.float16)
    s += x.type(torch.float32)
print(s) 
```

Deliverable: A 2-3 sentence response. 

提交内容：2 至 3 句话说明。

We will now apply mixed precision first to a toy model to build intuition and then to our benchmarking script. 

接下来，我们会先在一个玩具模型上应用混合精度，以建立直觉，然后再把它用到 benchmarking 脚本中。

Problem (benchmarking_mixed_precision):  Benchmarking Mixed Precision (2 points)

### 题目（benchmarking_mixed_precision）：混合精度基准测试（2 分）

(a) Consider the following model:

(a) 考虑下面这个模型：

```python
class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
    super().__init__()
    self.fc1 = nn.Linear(in_features, 10, bias=False)
    self.ln = nn.LayerNorm(10)
    self.fc2 = nn.Linear(10, out_features, bias=False)
    self.relu = nn.ReLU()

    def forward(self, x):
    x = self.relu(self.fc1(x))
    x = self.ln(x)
    x = self.fc2(x)
    return x 
```

Suppose we are training the model on a GPU and that the model parameters are originally in FP32. We’d like to use autocasting mixed precision with FP16. What are the data types of: 

假设我们正在 GPU 上训练这个模型，并且模型参数最初是 FP32。我们想使用带 FP16 的 autocast 混合精度。请说明以下各项的数据类型：

• the model parameters within the autocast context? 

• autocast 上下文中的模型参数？

• the output of the first feed-forward layer (ToyModel.fc1)? 

• 第一层前馈层（ToyModel.fc1）的输出？

• the output of layer norm (ToyModel.ln)? 

• layer norm（ToyModel.ln）的输出？

• the model’s predicted logits? 

• 模型预测得到的 logits？

• the loss? 

• loss？

• the model’s gradients? 

• 模型的梯度？

Deliverable: The data types for each of the components listed above. 

提交内容：分别给出以上各项的数据类型。

(b) You should have seen that FP16 mixed precision autocasting treats the layer normalization layer differently than the feed-forward layers. What parts of layer normalization are sensitive to mixed precision? If we use BF16 instead of FP16, do we still need to treat layer normalization differently? Why or why not? 

(b) 你应该已经看到，在 FP16 混合精度 autocast 下，layer normalization 层与前馈层的处理方式不同。layer normalization 的哪些部分对混合精度更敏感？如果我们把 FP16 换成 BF16，是否仍然需要区别对待 layer normalization？为什么？

Deliverable: A 2-3 sentence response. 

提交内容：2 至 3 句话说明。

(c) Modify your benchmarking script to optionally run the model using mixed precision with BF16. Time the forward and backward passes with and without mixed-precision for each language model size described in Section 2.1.2. Compare the results of using full precision versus mixed precision, and comment on any trends as model size changes. You may find the nullcontext no-op context manager to be useful. 

(c) 修改你的 benchmarking 脚本，使模型可以选择用 BF16 混合精度运行。对第 2.1.2 节中列出的每种语言模型规模，分别在使用和不使用混合精度的情况下测量前向与反向传播的时间。比较全精度与混合精度的结果，并评论随着模型规模变化，你观察到的趋势。你可能会发现 `nullcontext` 这个空上下文管理器会很有用。

Deliverable: A 2-3 sentence response with your timings and commentary. 

提交内容：2 至 3 句话，给出你的计时结果和观察。

# 2.1.6 Profiling Memory

## 2.1.6 显存分析

So far, we have been looking at compute performance. We’ll now shift our attention to memory, another major resource in language model training and inference. PyTorch also ships with a powerful memory profiler, which can keep track of allocations over time. 

到目前为止，我们主要关注的是计算性能。现在我们把注意力转向显存，它也是语言模型训练和推理中的一种关键资源。PyTorch 还提供了一个很强大的显存 profiler，它可以记录显存分配随时间的变化。

To use the memory profiler, you can modify your benchmarking script as follows: 

要使用显存 profiler，你可以像下面这样修改 benchmarking 脚本：

```python
... # warm-up phase in your benchmarking script

# Start recording memory history.

torch.cuda.memory._record_memory_history(max_entries=1000000)
... # what you want to profile in your benchmarking script

# Save a pickle file to be loaded by PyTorch's online tool.

torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")

# Stop recording history.

torch.cuda.memory._record_memory_history(enabled=None) 

```

This will output a file memory_snapshot.pickle that you can load into the following online tool: pytorch.org/memory_viz. This tool will let you see the overall memory usage timeline as well as each individual allocation that was made, with its size and a stack trace leading to the code where it originates. To use this tool, you should open the link above in a web browser, and then drag and drop your pickle file onto the page. 

这会输出一个 `memory_snapshot.pickle` 文件，你可以把它加载到这个在线工具中：`pytorch.org/memory_viz`。这个工具会显示整体显存使用时间线，以及每一笔具体分配的大小，并附带一个堆栈回溯，指出这笔分配来自代码中的哪个位置。使用方法是：在浏览器中打开上面的链接，然后把你的 pickle 文件拖放到页面上。

You will now use the PyTorch profiler to analyze the memory usage of your model. 

接下来，你将使用 PyTorch profiler 来分析模型的显存使用情况。

# Problem (memory_profiling):  Memory Profiling (4 points)

### 题目（memory_profiling）：显存分析（4 分）

Profile your complete training step of forward pass, backward pass, and optimizer step of the xl model from Table 1 with context lengths of 128 and 2048. 

对表 1 中的 `xl` 模型，在 context length 分别为 128 和 2048 时，分析完整训练步骤（前向、反向、优化器步进）的显存使用。

(a) Add an option to your profiling script to run your model through the memory profiler. 

(a) 给你的 profiling 脚本增加一个选项，使模型可以通过显存 profiler 运行。

It may be helpful to reuse some of your previous infrastructure (e.g., to activate mixed-precision, load specific model sizes, etc). Then, run your script to get a memory profile of the xl model when either doing inference only (just forward pass) or a full training step. What do your memory timelines look like? Can you tell which stage is running based on the peaks you see? 

复用你之前的基础设施会很有帮助（例如启用混合精度、加载指定模型大小等）。然后运行脚本，得到 `xl` 模型在只做推理（仅前向）和完整训练步骤下的显存 profile。它们的显存时间线是什么样的？你能否仅根据图中的峰值判断当前运行处于哪个阶段？

Deliverable: Two images of the “Active memory timeline” of an xl model, from the memory_viz tool: one for the forward pass, and one for running a full training step (forward and backward passes, then optimizer step), and a 2-3 sentence response. 

提交内容：从 `memory_viz` 工具中导出两张 `xl` 模型的 “Active memory timeline” 图片：一张对应仅前向，一张对应完整训练步骤（前向、反向，然后优化器步进），并附带 2 至 3 句话说明。

(b) What is the peak memory usage of each context length when doing a forward pass? What about when doing a full training step? 

(b) 在仅做前向传播时，两种 context length 的峰值显存分别是多少？做完整训练步骤时又分别是多少？

Deliverable: A table with two numbers per context length. 

提交内容：一个表格，每个 context length 对应两个数字。

(c) Find the peak memory usage of the xl model when using mixed-precision, for both a forward pass and a full training step. Does mixed-precision significantly affect memory usage? 

(c) 在使用混合精度时，求出 `xl` 模型仅前向和完整训练步骤下的峰值显存。混合精度是否会显著影响显存使用？

Deliverable: A 2-3 sentence response. 

提交内容：2 至 3 句话说明。

(d) Consider the xl model. Given our reference hyperparameters, what is the size of a tensor of activations in the Transformer residual stream, in single-precision? Give this size in MiB (i.e., divide the number of bytes by 10242). 

(d) 考虑 `xl` 模型。使用我们的参考超参数设置，在单精度下，Transformer residual stream 中一个激活张量的大小是多少？请以 MiB 为单位给出（即用字节数除以 `1024^2`）。

Deliverable: A 1-2 sentence response with your derivation. 

提交内容：1 至 2 句话，附带推导过程。

(e) Now look closely at the “Active Memory Timeline” from pytorch.org/memory_viz of a memory snapshot of the xl model doing a forward pass. When you reduce the “Detail” level, the tool hides the smallest allocations to the corresponding level (e.g., putting “Detail” at 10% only shows the 10% largest allocations). What is the size of the largest allocations shown? Looking through the stack trace, can you tell where those allocations come from? 

(e) 仔细查看 `pytorch.org/memory_viz` 中 `xl` 模型做一次前向传播时生成的内存快照的 “Active Memory Timeline”。当你降低 “Detail” 级别时，工具会隐藏最小的那部分分配（例如把 “Detail” 调成 10%，就只显示最大的 10% 分配）。此时显示出的最大分配有多大？结合堆栈回溯，你能看出这些分配来自哪里吗？

Deliverable: A 1-2 sentence response. 

提交内容：1 至 2 句话说明。

(f) Nsight Systems also has flags for memory profiling. You can combine these with the Nsight flags from before to understand what allocations are happening at different steps in your model’s lifespan. Use the PyTorch-provided NVTX labels to determine how much memory is saved for backward (these tensors are often called residuals) by a single TransformerBlock in your model. Note the 5 largest contributing operations, and what percentage of the overall memory they contribute. 

(f) Nsight Systems 也支持显存 profiling。你可以把这些选项和之前的 Nsight 标志组合使用，从而理解模型不同生命周期阶段到底发生了哪些分配。利用 PyTorch 提供的 NVTX 标签，确定在你的模型中，单个 `TransformerBlock` 为 backward 保存了多少显存（这些张量通常被称为 residuals）。指出贡献最大的 5 个操作，以及它们分别占总体显存的百分比。

During the backward pass, all these tensors will be freed, but new gradient tensors are emitted at the same time. Based on your profiles showing how much memory was allocated during the forward pass, and how much memory usage changes for every TransformerBlock in the backward pass, calculate how much memory the produced gradient tensors for a TransformerBlock take. Does the result match what you expect? 

在 backward 阶段，这些张量会被释放，但与此同时新的梯度张量也会被生成。根据你的 profile 中前向传播期间分配了多少显存，以及 backward 过程中每经过一个 `TransformerBlock` 显存使用如何变化，计算一个 `TransformerBlock` 产生的梯度张量占用了多少显存。这个结果是否符合你的预期？

Deliverable: Screenshots from Nsight Systems and a 1-2 paragraph response. 

提交内容：Nsight Systems 截图，以及 1 至 2 段说明。

# 3 Single-GPU Memory

## 3 单 GPU 显存

The later parts of this assignment will explore tricks to shard your tensors across multiple GPUs, but there are also tricks that can be applied even to single-GPU training. The most common of these is gradient checkpointing (also known as activation checkpointing). 

作业后面的部分会探索如何把张量分片到多张 GPU 上，但也有一些技巧即使在单 GPU 训练中同样适用。其中最常见的一种就是梯度检查点（也叫激活检查点）。

# 3.1 Autograd Residuals

## 3.1 Autograd 残留张量

Recall that in order to perform a backward pass through your model, we need to save the activations that were produced in the forward pass. While this is obviously the case for some operations, by default it’ll happen for many more than you might expect. The tensors saved for the backward pass are called “residuals”, or simply “saved tensors”. 

回忆一下：为了对模型执行反向传播，我们必须保存前向传播中产生的激活。虽然对某些操作来说这一点显而易见，但默认情况下，实际被保存的张量会比你预想的多得多。那些为 backward 而保存的张量被称为 “residuals”，也常直接叫 “saved tensors”。

Let’s build some understanding of what’s being saved in our network. Starting with our unassuming RMSNorm function (pure FP32 for simplicity), let’s add some hooks for when tensors are being saved or retrieved by autograd. 

让我们先建立对“网络里到底保存了哪些东西”的直觉。从一个看起来很普通的 `RMSNorm` 函数开始（为简单起见，全用 FP32），我们给 autograd 加一些 hook，观察张量在什么时候被保存、又在什么时候被取回。

```python
import torch
from torch import nn

x = torch.randn((4, 512, 2560), requires_grad=True)

class RMSNorm(nn.Module):
    def __init__()
    self,
    hidden_size: int,
    eps: float = 1e-5,
    device=None,
    ):
    super().__init__()
    self.weight = nn.Parameter(torch.ones(hidden_size, device=device))
    self.eps = eps

    def forward(self, x):
    rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    x = x * rms
    return self.weight * x

def pack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

def unpack_hook(t):
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    print(f"Loading residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

ln = RMSNorm(x.shape[-1])

with torch.autograd.graph_saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward() 
```

The output shows a worrying amount of tensors being written out, several of them at full activation size! 

`autograd_experiment.py` 的输出显示出了一个令人担忧的现象：有很多张量被写出保存，其中好几个的大小都与完整激活一样大。

```txt
$ uv run scripts/autograd_experiment.py
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0 object at 0x7f7dd319b5e0>
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None

Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=<MulBackward0 object at 0x7f7cf14e6740> 
```

```txt
Loading residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=<RsqrtBackward0 object at 0x7f7cf14e6740>
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None 
```

# 3.1.1 Operator Fusion

## 3.1.1 算子融合

In this case, it’s clear that the granularity of the operations used is too high. We want a single op that takes in the RMSNorm weights and the activation, and spits out the output, as well as for that operation to be unitary in the backward pass. This is one motivation for kernel fusion. Since the RMSNorm is fairly well behaved, we can even automatically fuse it using torch.compile. 

在这个例子里，很明显当前使用的操作粒度太细了。我们希望有一个单一操作，它接收 RMSNorm 的权重和输入激活，并输出结果；同时在 backward 中也把这个操作视作一个整体。这正是 kernel fusion 的动机之一。由于 RMSNorm 的行为相对规整，我们甚至可以用 `torch.compile` 自动把它融合起来。

```python
...
ln = torch.compile(RMSNorm(x.shape[-1]))
with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = ln(x)
    y.sum().backward() 
```

The new output is significantly better: 

融合后的输出明显好多了：现在我们只需要为 backward 保存一个完整尺寸的激活张量，也就是 RMSNorm 的输入。你还会注意到，加载这些 residual 的顺序不再是保存顺序的简单逆序，并且每个 residual 都不再带有 `grad_fn` 依赖了。这意味着 PyTorch 现在把整个 RMSNorm 当成了一个单一函数来处理。

```txt
Saving residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([2560]), dtype=torch.float32, grad_fn=None
Loading residual: shape=torch.Size([4, 512, 1]), dtype=torch.float32, grad_fn=None 
```

We only need to save a single full-size activation tensor for the backward pass — namely, the input to the RMSNorm function. Notice also how the order of loading is no longer the reverse of saving, and each residual no longer has a grad_fn dependency — PyTorch is treating the entirety of our RMSNorm as a single function. 

现在我们只需要为 backward 保存一个完整尺寸的激活张量，也就是 RMSNorm 函数的输入。你还会注意到，加载这些 residual 的顺序不再是保存顺序的简单逆序，并且每个 residual 都不再带有 `grad_fn` 依赖了。这意味着 PyTorch 现在把整个 RMSNorm 当成了一个单一函数来处理。

# 3.2 Activation Checkpointing

## 3.2 激活检查点

While fusion is undoubtedly useful, it can only get us so far in saving memory. For instance, let’s fuse a single TransformerBlock at size xl. 

虽然算子融合很有帮助，但它在节省显存方面只能做到一定程度。例如，让我们把一个 `xl` 尺寸的 `TransformerBlock` 融合起来。

```python
import torch
from cs336_basics.model import RotaryEmbedding, TransformerBlock

# num_layers for this model is 32

d_model, d_ff, num_heads, context_length = 2560, 10240, 16, 2048
block = TransformerBlock(d_model=d_model, d_ff=d_ff, num_heads=num_heads,
positional_encoder=RotaryEmbedding(dim=d_model // num_heads, context_length=context_length))

# Fuse as much torch.compile will allow

block = torch.compile(block, fullgraph=True)
x = torch.randn((4, context_length, d_model), requires_grad=True) 

```

```python

...

# Now logs the number of bytes saved

total_size_bytes = 0
def pack_hook(t):
    if isinstance(t, torch.nn.Parameter):  # Skip logging parameters to avoid double counting
    return t
    global total_size_bytes
    shape, dtype, grad_fn = t.shape, t.dtype, t.grad_fn
    total_size_bytes += t.numel() * t.element_size()
    print(f"Saving residual: {shape=}, {dtype=}, {grad_fn=}")
    return t

...

# Run forward pass, saving for backward

with torch.autograd.graph_saved_tensors_hooks(pack_hook, unpack_hook):
    y = block(x)

print(f"Total size of saved tensors in single TransformerBlock: {total_size_bytes / (1024**2):.2f} MiB") 

```

The script shows us how much memory we’re saving for backward: 

这个脚本会展示：为了 backward，我们到底保存了多少显存：

```txt

...
Total size of saved tensors in single TransformerBlock: 3651.31 MiB 

```

3.6 GiB for every layer. If we do this for all layers, we get 114 GiB of activations, just saved for backward! There’s a nontrivial amount of waste in the attention operation’s residuals, which we will fix in Section 4, but even with this fix, the memory use will grow linearly with batch size, sequence length and embedding size. 

每一层就要 3.6 GiB。如果对所有层都这样做，我们会得到 114 GiB 的激活，仅仅是为了 backward 而保存！注意力操作里的 residual 还存在不少浪费，我们会在第 4 节修复这一点；但即便修复之后，显存使用仍然会随着 batch size、sequence length 和 embedding size 线性增长。

# 3.2.1 Recomputation

## 3.2.1 重计算

Instead of holding on to every tensor we generate, it’s possible to save only periodic checkpoints of our results, and recompute the values in-between. PyTorch has an interface we can call to handle this in a simple fashion. torch.utils.checkpoint.checkpoint takes in a function, and arguments to that function. It then modifies the behavior of the function passed by: 

与其保留生成出的每一个张量，不如只保留若干周期性的检查点结果，并在 backward 时重新计算中间部分。PyTorch 提供了一个简单接口来支持这种做法：`torch.utils.checkpoint.checkpoint` 接收一个函数和它的参数，并通过以下方式改变该函数的行为：

1. In the forward pass: 

1. 在前向传播中：

1. Saving the input values to the function 

1. 保存该函数的输入值。

2. Suppressing the saving of tensors in the forward pass 

2. 抑制该函数内部张量在前向阶段的保存。

2. In the backward pass: 

2. 在反向传播中：

1. Prepending a recomputation step where the forward pass is recomputed from the previously saved inputs, and values are saved for backward 

1. 在 backward 之前插入一步重计算：用先前保存的输入重新执行 forward，并在这次重算中为 backward 保存所需值。

2. The backward pass is run and all tensors can be freed 

2. 随后执行 backward，并允许这些张量在使用后被释放。

In the simple case of running through 4 transformer blocks, we see that our memory adds up as we would expect. 

在一个简单例子中，如果串行执行 4 个 transformer blocks，不做 checkpoint 时，显存会如预期那样累加起来。但如果我们对它们做梯度检查点，保存的长期显存就会大幅下降。

```python
...
def four_blocks(x):
    x = block(x) 
```

```python
x = block(x)
x = block(x)
x = block(x)
return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks(x)

print(f"Total size of saved tensors in four TransformerBlocks: {total_size_bytes / (1024**2):.2f} MiB") 
```

Total size of saved tensors in four TransformerBlocks: 14605.25 MiB 

请记住，这并不是消除了显存使用，而是把显存开销分成了两类：一类是为了在 checkpoint 入口处保存信息，以便之后重计算而长期保留的检查点显存；另一类是在 checkpoint 内部重新执行 forward 时短期产生的显存，用来支撑该块的 backward。由于我们真正关心的是峰值显存，因此我们要在“保存检查点本身的代价”与“物化完整 block residual 的代价”之间取得平衡。checkpoint 越多，单个块内部需要物化的显存就越少，但保存 checkpoint 本身所需的显存就越多。

But we can employ gradient checkpointing as follows: 

但我们可以像下面这样使用梯度检查点：

在上面的例子中，很明显重算阶段产生的 residual 显存是主导项（部分原因是 checkpoint 本身很小），因此更有利的策略是把平衡点往 checkpoint 显存这一侧移动，也就是说缩小每个 checkpoint block 的范围。

```python
from torch.utils.checkpoint import checkpoint
def two_blocks(x):
    x = block(x)
    x = block(x)
    return x

def four_blocks_checkpoint(x):
    # checkpoint throws out all the saved tensors until the backward pass
    # when getting to the checkpointed block in the backward pass,
    # it reruns a forward pass to produce the saved tensors,
    # then completes normal backward pass.
    x = checkpoint(two_blocks, x, use_reentrant=False)
    x = checkpoint(two_blocks, x, use_reentrant=False)
    return x

with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
    y = four_blocks_checkpoint(x)

print(f"Total size of saved tensors in four TransformerBlocks with checkpointing: {total_size_bytes / (1024**2):.2f} MiB") 
```

```txt
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([0]), dtype=torch.float32, grad_fn=None
Saving residual: shape=torch.Size([4, 2048, 2560]), dtype=torch.float32, grad_fn=<torch.autograd.function.)CompiledFunctionBackward object at 0x7aa0657a19d0>
Total size of saved tensors in four TransformerBlocks with checkpointing: 160.00 MiB 
```

Keep in mind this hasn’t eliminated the memory use. Rather, it’s factored our memory use into two categories: the longer term storage we save to prepare for recomputation at the entry point of each checkpoint call (the checkpoint itself), and the short term memory generated in the recomputation pass within the checkpointed block to facilitate a backward pass through it. Since our main consideration is the peak memory usage, we want to balance the memory cost of the saved checkpoints with the memory cost of materializing a full block worth of residuals. The more checkpoints we use, the smaller the amount of materialized memory within a single block, but the more memory we need for the checkpoints themselves. 

请记住，这并不是消除了显存使用，而是把显存开销分成了两类：一类是为了在 checkpoint 入口处保存信息，以便之后重计算而长期保留的检查点显存；另一类是在 checkpoint 内部重新执行 forward 时短期产生的显存，用来支撑该块的 backward。由于我们真正关心的是峰值显存，因此我们要在“保存检查点本身的代价”与“物化完整 block residual 的代价”之间取得平衡。checkpoint 越多，单个块内部需要物化的显存就越少，但保存 checkpoint 本身所需的显存就越多。

checkpoint block 做大或做小，并不会影响“重计算总共需要做多少工作”这一点；但如果你愿意继续用更多计算换更少显存，还可以使用递归 checkpointing，也就是在一个 checkpoint 内部再嵌套另一个 checkpoint。

In our example above, it’s clear that the recomputed residuals’ memory is dominating (in part because our checkpoints are tiny), so it would be beneficial to shift the balance more toward checkpointed memory. Thus we would want to decrease the scope of a checkpointed block. 

在上面的例子中，很明显重算阶段产生的 residual 显存是主导项（部分原因是 checkpoint 本身很小），因此更有利的策略是把平衡点往 checkpoint 显存这一侧移动，也就是说缩小每个 checkpoint block 的范围。

Having larger or smaller checkpointed blocks doesn’t affect the computational cost of recomputation, but we can continue to reduce the required memory at the cost of compute by using recursive checkpointing, meaning we nest checkpoint calls inside other checkpoint calls. 

checkpoint block 做大或做小，并不会影响“重计算总共需要做多少工作”这一点；但如果你愿意继续用更多计算换更少显存，还可以使用递归 checkpointing，也就是在一个 checkpoint 内部再嵌套另一个 checkpoint。

# Problem (gradient_checkpointing):  Memory-Optimal Gradient Checkpointing (4 points)

### 题目（gradient_checkpointing）：最优显存的梯度检查点（4 分）

Consider a Transformer with `n` identical blocks stacked sequentially. Without any checkpointing, all `n` blocks’ worth of residuals are kept alive simultaneously, giving `Theta(n)` peak activation memory. We have a free hand to wrap any subset of the forward pass in checkpoint, including nesting checkpoint calls inside one another. 

考虑一个由 `n` 个相同 block 顺序堆叠而成的 Transformer。若不使用 checkpointing，则全部 `n` 个 block 的 residual 会同时存活，峰值激活显存为 `Θ(n)` 个 block 的量级。现在你可以自由决定对 forward 的哪些部分使用 `checkpoint`，甚至可以嵌套 checkpoint。

(a) What checkpointing strategy minimizes peak activation memory, ignoring the compute cost? Describe how you would arrange the checkpoint calls (a code sketch is fine), and give the asymptotic peak activation memory and compute of your strategy as a function of `n`. Assume the residuals saved by a single block dominate any per-checkpoint bookkeeping. 

(a) 如果忽略计算代价，哪种 checkpoint 策略能使峰值激活显存最小？请描述你会如何放置这些 checkpoint 调用（给个代码草图即可），并给出这种策略的渐近峰值激活显存与计算代价，作为 `n` 的函数。可以假设单个 block 的 residual 占主导，而每个 checkpoint 的元数据开销可以忽略。

Deliverable: A 3-5 sentence description of the strategy and its asymptotic peak memory, plus a short code sketch. 

提交内容：用 3 至 5 句话说明策略与其渐近峰值显存，并附一个简短代码草图。

(b) Consider the xl model config with batch size 4 and sequence length 2048 as above. If you only have the time/compute budget to run one step of recomputation (meaning you may not nest checkpoint calls), what is the best checkpointing strategy to reduce peak memory? Profile your run’s peak memory to validate your hypothesis. Compare the peak memory of the next smaller and larger checkpointing block sizes to be sure. 

(b) 仍以 batch size 4、sequence length 2048 的 `xl` 模型为例。如果你的时间/计算预算只允许“一层重计算”（也就是不允许嵌套 checkpoint），那么降低峰值显存的最佳 checkpoint 策略是什么？请用 peak memory profile 验证你的判断，并比较相邻更小与更大的 checkpoint block size，确认你的选择最优。

Deliverable: A 3-5 sentence description of your reasoning along with the measured peak memory for your strategy. 

提交内容：用 3 至 5 句话说明你的推理，并给出该策略下测得的峰值显存。

# 4 GPU Kernels

## 4 GPU 内核

# 4.1 Optimizing Attention with FlashAttention-2

## 4.1 用 FlashAttention-2 优化注意力

# 4.1.1 Benchmarking PyTorch Attention

### 4.1.1 基准测试 PyTorch 注意力

Your profiling likely suggests that there is an opportunity for optimization, both in terms of memory and compute, in your attention layers. At a high level, the attention operation consists of a matrix multiplication followed by softmax, then another matrix multiplication: 

你的 profiling 很可能已经表明：在注意力层中，无论从显存还是计算角度看，都有较大的优化空间。从高层来看，注意力操作由一次矩阵乘法、一次 softmax，以及再一次矩阵乘法构成：

$$
\operatorname{Attention} (Q, K, V) = \operatorname{softmax} \left(\operatorname{mask} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right)\right) V \tag {1}
$$

The naïve attention implementation needs to save attention score matrices of shape seq_len × seq_len for each batch/head element, which can grow very large with long sequence lengths, causing out-ofmemory errors for any tasks with long inputs or outputs. We will implement an attention kernel following the FlashAttention-2 paper, which computes attention by tiles and avoids ever explicitly materializing the seq_len × seq_len attention score matrices, enabling scaling to much longer sequence lengths. 

朴素注意力实现需要为每个 batch/head 元素保存形状为 `seq_len × seq_len` 的 attention score 矩阵；在长序列时，这个矩阵会变得非常大，并导致 out-of-memory。我们将实现一个遵循 FlashAttention-2 论文思路的注意力内核，它通过按 tile 计算注意力，避免在全局显存中显式物化整个 `seq_len × seq_len` 注意力分数矩阵，从而支持更长的序列长度。

(a) Benchmark your attention implementation at different scales. Write a script that will: 

(a) 在不同规模下对你的注意力实现做基准测试。编写一个脚本，完成以下事情：

(i) Fix the batch size to 8 and don’t use multihead attention (i.e. remove the head dimension). 

1. 将 batch size 固定为 8，并且不使用多头注意力（即移除 head 维度）。

(ii) Iterate through the cartesian product of [16, 32, 64, 128] for the head embedding dimension $d _ { \mathrm { m o d e l } }$ , and [256, 1024, 4096, 8192, 16384] for the sequence length. 

2. 遍历 head embedding 维度 `d_model ∈ [16, 32, 64, 128]` 与 sequence length `∈ [256, 1024, 4096, 8192, 16384]` 的笛卡尔积。

(iii) Create random inputs `Q`, `K`, `V` for the appropriate size. 

3. 为合适尺寸创建随机输入 `Q`、`K`、`V`。

(iv) Time 100 forward passes through attention using the inputs. 

4. 对注意力前向传播计时 100 次。

(v) Measure how much memory is in use before the backward pass starts, and time 100 backward passes. 

5. 在 backward 开始前测量显存占用，并对 backward 计时 100 次。

(vi) Make sure to warm up, and to call torch.cuda.synchronize() after each forward/ backward pass. 

6. 注意先做 warm-up，并在每次 forward / backward 后调用 `torch.cuda.synchronize()`。

Depending on your GPU, some of these configurations are expected to run out of memory. Report the timings (or out-of-memory errors) you get for these configurations. At what size do you get out-of-memory errors? Do the accounting for the memory usage of attention in one of the smallest configurations you find that runs out of memory (you can use the equations for memory usage of Transformers from Assignment 1). How does the memory saved for backward change with the sequence length? What would you do to eliminate this memory cost? 

根据你的 GPU，其中一些配置预计会 OOM。请汇报这些配置下的计时结果（或 OOM 错误）。你会在什么规模下遇到 OOM？请对你找到的一个“最小但仍会 OOM”的配置，依据作业 1 中 Transformer 显存开销的公式，对注意力的显存占用做一次估算。为 backward 保存的显存会如何随着 sequence length 改变？如果要消除这部分显存开销，你会怎么做？

Deliverable: A table with your timings, your calculations for the memory usage, and a 1-2 paragraph response. 

提交内容：一个包含计时结果的表格、你的显存估算，以及 1 至 2 段说明。

# 4.2 Benchmarking JIT-Compiled Attention

## 4.2 基准测试 JIT 编译的注意力

Since version 2.0, PyTorch also ships with a powerful just-in-time compiler that automatically tries to apply a number of optimizations to PyTorch functions: see 

从 2.0 版开始，PyTorch 也提供了一个很强大的即时编译器，可以自动尝试对 PyTorch 函数应用多种优化，介绍见：

https://pytorch.org/tutorials/intermediate/torch_compile_tutorial.html

In particular, it will try to automatically generate fused Triton kernels by dynamically analyzing your computation graph. The interface for using the PyTorch compiler is very simple. For instance, if we wanted to apply it to a single layer of our model, we can use: 

尤其重要的是，它会通过动态分析计算图，尽量自动生成融合后的 Triton kernel。使用接口非常简单。比如，如果我们要把它应用到模型中的某一层，可以这样写：

```python
layer = SomePyTorchModule(...)
compiled_layer = torch.compile(layer) 
```

Now, compiled_layer functionally behaves just like layer (e.g., with its forward and backward passes). We can also compile our entire PyTorch model with torch.compile(model), or even a Python function that calls PyTorch operations. 

这样，`compiled_layer` 在功能上就和 `layer` 一样（包括 forward 与 backward）。我们也可以对整个 PyTorch 模型使用 `torch.compile(model)`，甚至可以直接编译一个调用 PyTorch 操作的 Python 函数。

# Problem (torch_compile):  Torch Compile (2 points)

### 题目（torch_compile）：Torch Compile（2 分）

(a) Extend your attention benchmarking script to include a compiled version of your PyTorch implementation of attention, and compare its performance to the uncompiled version with the same configuration as the pytorch_attention problem above. 

(a) 扩展你的注意力 benchmarking 脚本，加入一个对你现有 PyTorch 注意力实现进行 `torch.compile` 编译后的版本，并在与 `pytorch_attention` 题目相同的配置下，把它和未编译版本进行比较。

Deliverable: A table comparing your forward and backward pass timings for your compiled attention module with the uncompiled version from the pytorch_attention problem above. 

提交内容：一个表格，比较你在 `pytorch_attention` 题目中得到的未编译注意力实现，与编译版注意力模块在 forward / backward 时间上的差异。

(b) Now, compile your entire Transformer model in your end-to-end benchmarking script. How does the performance of the forward pass change? What about the combined forward and backward passes and optimizer steps? 

(b) 现在，把整个 Transformer 模型在你的端到端 benchmarking 脚本中编译起来。前向传播性能变化如何？前向加反向、以及再加优化器步进的整体性能又如何？

Deliverable: A table comparing your vanilla and compiled Transformer model. 

提交内容：一个表格，对比普通版和编译版 Transformer 模型。

Given the scaling behaviors we’ve seen with respect to the sequence length, we need significant improvements to handle large sequences. Even with torch.compile, the current implementation suffers from very poor memory access patterns at long sequence length. For that, we will write a Triton implementation of FlashAttention-2, where we’ll have significantly more control over how memory is accessed and when to compute what. 

结合我们此前观察到的 sequence length 扩展行为，要处理长序列，我们还需要显著更强的优化。即便使用了 `torch.compile`，当前实现仍然在长序列下具有非常糟糕的显存访问模式。因此，我们将编写一个 Triton 版本的 FlashAttention-2，从而能更精细地控制显存访问方式以及计算时机。

# 4.2.1 Example - Weighted Sum

## 4.2.1 示例：加权求和

To introduce what you’ll need to know about Triton and how it interoperates with PyTorch, we will work through an example kernel for a “weighted sum” operation. For further resources on getting up to speed with Triton, see Triton’s tutorials. We note that these tutorials do not use the new, convenient block pointer abstraction, which we will walk through below. 

为了介绍你在 Triton 中需要掌握的内容，以及 Triton 如何与 PyTorch 协作，我们先通过一个 “weighted sum” 示例 kernel 来入门。若想进一步学习 Triton，可以查看 Triton 的官方教程。需要注意的是，这些教程没有使用我们下面会讲到的、更方便的 block pointer 抽象。

Given an input matrix `x`, we’ll multiply its entries by a column-wise weight vector `weight`, and sum each row, giving us the matrix-vector product of `x` and `weight`. We are going to work through the forward pass of this operation first, and then write the Triton kernel for the backward pass. 

给定一个输入矩阵 `x`，我们将它的每个元素与一个按列给出的权重向量 `weight` 相乘，再对每一行求和，得到矩阵 `x` 与向量 `weight` 的矩阵向量积。我们会先推导这个操作的 forward，再实现它的 Triton backward kernel。

# Forward pass

The forward pass of our kernel is just the following broadcasted inner product. 

这个 kernel 的 forward 本质上就是下面这个带广播的内积。

```python
def weighted_sum(x, weight):
    # Here, assume that x has n - dim shape [... , D], and weight has 1D shape [D]
    return (weight * x).sum(axis=-1) 
```

When writing our Triton kernel, we’ll have each program instance (potentially running in parallel) compute the weighted sum of a tile of rows of `x`, and write the corresponding scalar outputs to the output tensor. In Triton, a program instance is a block of threads all running the same program, and these thread blocks can be run in parallel on the GPU. Instead of taking tensors as arguments, we take pointers to their first elements, as well as strides for each tensor that tell us how to move along axes. 

在编写 Triton kernel 时，我们会让每个 program instance（可能并行运行）负责计算输入矩阵若干行 tile 的加权和，并把对应的标量输出写入结果张量。在 Triton 中，一个 program instance 可以理解为一组执行相同程序的线程块，这些线程块会在 GPU 上并行运行。与直接把张量作为参数不同，我们传入的是指向张量首元素的指针，以及每个张量的 stride，用来描述如何沿各维移动。

We can use the strides to load a tensor corresponding to the tile of rows of `x` that we’re summing in the running instance, using the program ID to divide up the work (i.e., instance `i` will process the `i`-th tile of rows of `x`). The main difference between the forward pass in Triton and PyTorch in this simple case is the need to do pointer arithmetic and explicit loads/stores. We will use the block pointer abstraction with `tl.make_block_ptr` to greatly simplify the pointer arithmetic, although this means we need to do some setup to prepare the block pointers. 

我们可以利用这些 stride，根据 program ID 来装载当前实例负责处理的那块行 tile。也就是说，第 `i` 个 instance 会处理输入矩阵的第 `i` 块行。在这个简单例子中，Triton forward 与 PyTorch forward 最大的区别，是你必须自己做指针运算，并显式地 `load/store` 数据。为了大幅简化这些指针运算，我们会使用 `tl.make_block_ptr` 提供的 block pointer 抽象，不过这也意味着我们需要先做一些准备工作来定义这些 block pointer。

Refer to Figure 2 for a schematic of tiling and how block pointers are advanced. The weighted sum function from above looks like the following: 

可以参考图 2，理解 tiling 的示意以及 block pointer 是如何推进的。上面的 weighted sum 在 Triton 中大致写成如下形式：

```python
import triton
import triton.language as tl

@triton.jit
def weighted_sum_fwd(
    x_ptr, weight_ptr, # Input pointers
    output_ptr, # Output pointer
    x_stride_row, x_stride_dim, # Strides tell us how to move one element in each axis of a tensor
    weight_stride_dim, # Likely 1
    output_stride_row, # Likely 1
    NUM_ROWS, D,
    ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr, # Tile shapes must be known at compile time
):
    # Each instance will compute the weighted sum of a tile of rows of x.
    # `tl.program_id` gives us a way to check which thread block we're running in
    row_tile_idx = tl.program_id(0)

    # Block pointers give us a way to select from an ND region of memory
    # and move our selection around.
    # The block pointer must know:
    # - The pointer to the first element of the tensor
    # - The overall shape of the tensor to handle out-of-bounds access
    # - The strides of each dimension to use the memory layout properly
    # - The ND coordinates of the starting block, i.e., "offsets"
    # - The block shape to load/store at a time
    # - The order of the dimensions in memory from major to minor
    # axes (= np.argsort(strides)) for optimizations, needed for
    # TMA support on >=Hopper 
```

```python
x_block_ptr = tl.make_block_ptr(
    x_ptr,
    shape=(NUM_ROWS, D,)
    strides=(x_stride_row, x_stride_dim),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0),
)

weight_block_ptr = tl.make_block_ptr(
    weight_ptr,
    shape=(D,)
    strides=(weight_stride_dim,)
    offsets=(0,)
    block_shape=(D_TILE_SIZE,)
    order=(0,)
)

output_block_ptr = tl.make_block_ptr(
    output_ptr,
    shape=(NUM_ROWS,)
    strides=(output_stride_row,)
    offsets=(row_tile_idx * ROWS_TILE_SIZE,)
    block_shape=(ROWS_TILE_SIZE,)
    order=(0,)
)

# Initialize a buffer to write to

output = tl.zeros((ROWS_TILE_SIZE,), dtype=tl.float32)

for i in range(tl.cdiv(D, D_TILE_SIZE)):
    # Load the current block pointer
    # Since ROWS_TILE_SIZE might not divide NUM_ROWS, and D_TILE_SIZE might not divide D,
    # we need boundary checks for both dimensions
    row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero")  # (ROWS_TILE_SIZE, D_TILE_SIZE)
    weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero")  # (D_TILE_SIZE,)
    # Compute the weighted sum of the row.
    output += tl.sum(row * weight[None, :], axis=1)

    # Move the pointers to the next tile.
    # These are (rows, columns) coordinate deltas
    x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))  # Move by D_TILE_SIZE in the last dimension
    weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE, ))  # Move by D_TILE_SIZE

# Write output to the output block pointer (a single scalar per row).

# Since ROWS_TILE_SIZE might not divide NUM_ROWS, we need boundary checks

tl.store(output_block_ptr, output, boundary_check=(0, )) 

```

Let’s now wrap this kernel in a PyTorch Autograd function that will interoperate with PyTorch (i.e., take Tensors as inputs, output a Tensor, and later also work with the autograd engine during the backward pass): 

接下来，我们把这个 kernel 封装进一个 PyTorch Autograd function，使它能够与 PyTorch 协作（也就是接收 Tensor 作为输入，输出一个 Tensor，并且在后续 backward 时也能与 autograd 引擎正常配合）：

```python

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
    # Cache x and weight to be used in the backward pass, when we
    # only receive the gradient wrt. the output tensor, and
    # need to compute the gradients wrt. x and weight.
    D, output_dims = x.shape[-1], x.shape[:-1]

    # Reshape input tensor to 2D
    input_shape = x.shape
    x = rearrange(x, "... d -> (...) d")

    ctx.save_for_backward(x, weight) 

```

```python

assert len(weight.shape) == 1 and weight.shape[0] == D, "Dimension mismatch"
assert x.is_cuda and weight.is_cuda, "Expected CUDA tensors"
assert x.is_contiguous(), "Our pointer arithmetic will assume contiguous x"

ctx.D_TILE_SIZE = triton.next_power_of_2(D) // 16 # Roughly 16 loops through the embedding dimension
ctx.ROWS_TILE_SIZE = 16 # Each thread processes 16 batch elements at a time
ctx.input_shape = input_shape

# Need to initialize empty result tensor. Note that these elements are not necessarily 0!

y = torch.empty(output_dims, device=x.device)

# Launch our kernel with n instances in our 1D grid.

n_rows = y.numel()
weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)](
    x, weight,
    y,
    x.stride(0), x.stride(1),
    weight.stride(0),
    y.stride(0),
    NUM_ROWS=n_rows, D=D,
    ROWS_TILE_SIZE=ctx.ROWS_TILE_SIZE, D_TILE_SIZE=ctx.D_TILE_SIZE,
)
return y.view(input_shape[:-1]) 

```

Notice that when we invoke the Triton kernel with weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)], we define a so-called “launch grid” of thread blocks by passing the tuple (triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),). Then, we can access the thread block index with tl.program_id(0) in our kernel. 

注意，当我们以 `weighted_sum_fwd[(triton.cdiv(n_rows, ctx.ROWS_TILE_SIZE),)]` 这种形式调用 Triton kernel 时，我们实际上是通过这个元组定义了 thread block 的 “launch grid”。随后，在 kernel 内部就可以通过 `tl.program_id(0)` 访问当前线程块的索引。

# Backward pass

Since we are defining our own kernel, we will also need to write our own backward function. 

既然我们是在自己定义 kernel，那么 backward 也需要自己实现。

In the forward pass, we were given the inputs to our layer, and needed to compute its outputs. In the backward pass, recall that we will be given the gradients of the objective with respect to our outputs, and need to compute the gradient with respect to each of our inputs. In our case, our operation has as inputs a matrix $\boldsymbol { x } : \mathbb { R } ^ { n \times h }$ and a weight vector $w : \mathbb { R } ^ { h }$ . For brevity, let’s call our operation $f ( x , w )$ , whose range is $\mathbb { R } ^ { n }$ . Then, assuming we are given $\nabla _ { f ( x , w ) } \mathcal { L }$ , the gradient of the loss $\mathcal { L }$ with respect to the output of our layer, we can apply the multivariate chain rule to obtain the following expressions for the gradients with respect to $x$ and $w$: 

在 forward 中，我们已知层的输入，并需要计算输出。到了 backward，回忆一下：我们会拿到目标函数对输出的梯度，然后需要进一步计算它对每个输入的梯度。在这个例子里，我们的操作输入是一个矩阵 $\boldsymbol{x} : \mathbb{R}^{n \times h}$ 和一个权重向量 $w : \mathbb{R}^{h}$。为简洁起见，把这个操作记作 $f(x, w)$。假设我们已知损失函数对输出的梯度 $\nabla_{f(x,w)} \mathcal{L}$，那么就可以利用多元链式法则，推出对 $x$ 和 $w$ 的梯度表达式如下：

$$
\left(\nabla_ {x} \mathcal {L}\right) _ {i j} = \sum_ {k = 1} ^ {n} \frac {\partial f (x , w) _ {k}}{\partial x _ {i j}} \left(\nabla_ {f (x, w)} \mathcal {L}\right) _ {k} = w _ {j} \cdot \left(\nabla_ {f (x, w)} \mathcal {L}\right) _ {i} \tag {2}
$$

$$
\left(\nabla_ {w} \mathcal {L}\right) _ {j} = \sum_ {i = 1} ^ {n} \frac {\partial f (x , w) _ {i}}{\partial w _ {j}} \left(\nabla_ {f (x, w)} \mathcal {L}\right) _ {i} = \sum_ {i = 1} ^ {n} x _ {i j} \cdot \left(\nabla_ {f (x, w)} \mathcal {L}\right) _ {i} \tag {3}
$$

This gives a simple formula for computing the backward pass. To obtain the backward step with respect to $x$, we apply Equation 2 and take the outer product of $w$ and $\nabla _ { f ( x , w ) }$ . To compute the backward step with respect to $w$ $( \mathrm { i . e . } \ ( \nabla _ { w } \mathcal { L } ) _ { j } )$ , we must multiply our input gradient by the corresponding output row. 

这样我们就得到了一个相对直接的 backward 计算公式。对于 $x$ 的梯度，可以依据式 (2)，把相应量与 $\nabla_{f(x,w)}$ 做外积；对于 $w$ 的梯度，也就是 $(\nabla_w \mathcal{L})_j$，则需要把输入与对应输出行的梯度相乘并求和。

Our kernel for the backward pass will start by defining all the block pointers and then computing $\nabla _ { x } { \mathcal { L } } \colon$ 

我们的 backward kernel 会先定义所有 block pointer，然后从计算 $\nabla_x \mathcal{L}$ 开始：

```txt
@triton.jit
def weighted_sum_backward(
    x_ptr, weight_ptr,  # Input
    grad_output_ptr,  # Grad input
    grad_x_ptr, partial_grad_weight_ptr,  # Grad outputs
    stride_xr, stride_xd, 
```

```python
stride_wd,
stride_gr,
stride_gxr, stride_gxd,
stride_gwb, stride_gwd,
NUM_ROWS, D,
ROWS_TILE_SIZE: tl.constexpr, D_TILE_SIZE: tl.constexpr,
):
row_tile_idx = tl.program_id(0)
n_row_tiles = tl.num_programs(0)

# Inputs

grad_output_block_ptr = tl.make_block_ptr(
    grad_output_ptr,
    shape=(NUM_ROWS,), strides=(stride_gr,)
    offsets=(row_tile_idx * ROWS_TILE_SIZE,)
    block_shape=(ROWS_TILE_SIZE,)
    order=(0,)
)

x_block_ptr = tl.make_block_ptr(
    x_ptr,
    shape=(NUM_ROWS, D,), strides=(stride_xr, stride_xd),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0),
)

weight_block_ptr = tl.make_block_ptr(
    weight_ptr,
    shape=(D,), strides=(stride_wd,)
    offsets=(0,), block_shape=(D_TILE_SIZE,)
    order=(0,)
)

grad_x_block_ptr = tl.make_block_ptr(
    grad_x_ptr,
    shape=(NUM_ROWS, D,), strides=(stride_gxr, stride_gxd),
    offsets=(row_tile_idx * ROWS_TILE_SIZE, 0),
    block_shape=(ROWS_TILE_SIZE, D_TILE_SIZE),
    order=(1, 0),
)

partial_grad_weight_block_ptr = tl.make_block_ptr(
    partial_grad_weight_ptr,
    shape=(n_row_tiles, D,), strides=(stride_gwb, stride_gwd),
    offsets=(row_tile_idx, 0),
    block_shape=(1, D_TILE_SIZE),
    order=(1, 0),
)

for i in range(tl.cdiv(D, D_TILE_SIZE)):
    grad_output = tl.load(grad_output_block_ptr, boundary_check=(0,), padding_option="zero") # (ROWS_TILE_SIZE,)

    # Outer product for grad_x
    weight = tl.load(weight_block_ptr, boundary_check=(0,), padding_option="zero") # (D_TILE_SIZE,)
    grad_x_row = grad_output[:, None] * weight[None, :]
    tl.store(grad_x_block_ptr, grad_x_row, boundary_check=(0, 1))

    # Reduce as many rows as possible for the grad_weight result
    row = tl.load(x_block_ptr, boundary_check=(0, 1), padding_option="zero") # (ROWS_TILE_SIZE, D_TILE_SIZE)
    grad_weight_row = tl.sum(row * grad_output[:, None], axis=0, keep_dims=True)
    tl.store(partial_grad_weight_block_ptr, grad_weight_row, boundary_check=(1,)) # Never out of bounds for dim

    # Move the pointers to the next tile along D
    x_block_ptr = x_block_ptr.advance((0, D_TILE_SIZE))
    weight_block_ptr = weight_block_ptr.advance((D_TILE_SIZE,))
    partial_grad_weight_block_ptr = partial_grad_weight_block_ptr.advance((0, D_TILE_SIZE))
    grad_x_block_ptr = grad_x_block_ptr.advance((0, D_TILE_SIZE)) 

```

Computing the gradient $\nabla _ { x }$ is simple, and we write the result to the appropriate tile of the output tensor. However, computing $\nabla _ { w }$ is a bit more challenging. Each kernel instance is responsible for one row tile of $x$, but we now need to sum across rows of $x$. Instead of doing this sum directly in our backward pass, we will assume that `partial_grad_weight_ptr` contains an `n_row_tiles x D` matrix, where the first dimension is only reduced within a row tile from $x$. We reduce within the current row tile before writing to this tensor. Outside of the kernel, we reduce $\nabla _ { w }$ using `torch.sum` to sum up the results from each row tile.1 

计算 $\nabla_x$ 相对简单，我们直接把结果写入输出张量对应的 tile 即可。但计算 $\nabla_w$ 稍微麻烦一些。每个 kernel instance 只负责输入矩阵的一块行 tile，可是这里我们需要沿所有行进行求和。因此，我们不直接在 backward kernel 中完成这个全局求和，而是假定 `partial_grad_weight_ptr` 保存了一个 `n_row_tiles × D` 的矩阵，其中第一维只在各自的行 tile 内部完成归约。每个 instance 先把自己那部分行 tile 的局部和写到这个矩阵中，随后在 kernel 外部再使用 `torch.sum` 把这些局部结果加总起来，得到最终的 $\nabla_w$。

The final part of the autograd.Function is then relatively simple: 

这样一来，`autograd.Function` 剩下的部分就相对直接了：

```python

class WeightedSumFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
    # ... (defined earlier)

    @staticmethod
    def backward(ctx, grad_out):
    x, weight = ctx_saved_tensors
    ROWS_TILE_SIZE, D_TILE_SIZE = ctx.ROWS_TILE_SIZE, ctx.D_TILE_SIZE  # These don't have to be the same
    n_rows, D = x.shape

    # Our strategy is for each thread block to first write to a partial buffer,
    # then we reduce over this buffer to get the final gradient.
    partial_grad_weight = torch.empty((triton.cdiv(n_rows, ROWS_TILE_SIZE), D), device=x.device, dtype=x.dtype)
    grad_x = torch.empty_like(x)

    weighted_sum_backward[(triton.cdiv(n_rows, ROWS_TILE_SIZE),)](
    x, weight,
    grad_out,
    grad_x, partial_grad_weight,
    x.stride(0), x.stride(1),
    weight.stride(0),
    grad_out.stride(0),
    grad_x.stride(0), grad_x.stride(1),
    partial_grad_weight.stride(0), partial_grad_weight.stride(1),
    NUM_ROWS=n_rows, D=D,
    ROWS_TILE_SIZE=ROWS_TILE_SIZE, D_TILE_SIZE=D_TILE_SIZE,
    )
    grad_weight = partial_grad_weight.sum(axis=0)
    return grad_x, grad_weight 

```

Finally, we can now obtain a function that works much like those implemented in torch.nn.functional: 

最后，我们就能得到一个行为上很像 `torch.nn.functional` 中函数的接口：

```python

f_weightedsum = WeightedSumFunc.apply 

```

Now, calling `f_weightedsum` on two PyTorch tensors `x` and `weight` will give a tensor such as the following: 

现在，如果你对两个 PyTorch 张量调用 `f_weightedsum`，会得到类似下面这样的结果：

```txt

tensor([ 90.8563, -93.6815, -80.8884, ..., 103.4840, -21.4634, -24.0192], device='cuda:0', grad_fn=<WeightedSumFuncBackward>) 

```

Note the grad_fn attached to the tensor — this shows that $\mathrm { P y }$ Torch knows what to call in the backward pass when this tensor appears in the computation graph. This completes our Triton implementation of the weighted sum operation. 

注意输出张量上附带的 `grad_fn`。这表明，当这个张量出现在计算图中时，PyTorch 已经知道在 backward 时应该调用什么逻辑。至此，我们就完成了 weighted sum 操作的 Triton 实现。

# 4.2.2 FlashAttention-2 Forward Pass

## 4.2.2 FlashAttention-2 前向传播

You will replace your PyTorch attention implementation with a significantly improved Triton implementation following FlashAttention-2 [T. Dao, 2023]. FlashAttention-2 employs some tricks to compute the forward pass in tiles, which allows for efficient memory access patterns and avoids the need to materialize the full attention matrix on global memory. 

你将把当前的 PyTorch 注意力实现替换成一个显著更高效的 Triton 实现，它遵循 FlashAttention-2 [T. Dao, 2023] 的思路。FlashAttention-2 使用若干技巧，以 tile 为单位计算 forward，从而拥有更高效的显存访问模式，并避免把完整注意力矩阵写到全局显存中。

Before jumping into this section, we highly recommend reading at least the original FlashAttention paper [T. Dao et al., 2022], which will give you intuition for the core technique that enables efficient attention with FlashAttention: computing the softmax in an online fashion across tiles (a technique proposed in [M. Milakov et al., 2018]). We also recommend checking out H. He [4] for some more intuition on how GPUs actually execute PyTorch code. 

在进入本节前，我们强烈建议你至少阅读原始 FlashAttention 论文 [T. Dao et al., 2022]，这会帮助你理解 FlashAttention 能高效执行注意力计算的核心技巧：跨 tile 以在线方式计算 softmax（这一技巧最早可追溯到 [M. Milakov et al., 2018]）。此外，我们也推荐 H. He [4]，它能帮助你从 GPU 实际如何执行 PyTorch 代码的角度建立更多直觉。

# Understanding inefficiencies in vanilla attention

### 理解朴素注意力中的低效之处

Recall that the forward pass for attention (ignoring masking for now) can be written as: 

忽略 masking 时，注意力前向可写为：

$$
\boldsymbol {S} = \boldsymbol {Q} \boldsymbol {K} ^ {\top} / \sqrt {d} \tag {4}
$$

$$
P _ {i j} = \operatorname{softmax} _ {j} (\boldsymbol {S}) _ {i j} \tag {5}
$$

$$
O = P V \tag {6}
$$

The standard backward pass is 

标准 backward 为：

$$
d V = P ^ {\top} d O \tag {7}
$$

$$
d P = d O V ^ {\top} \tag {8}
$$

$$
\boldsymbol {d} \boldsymbol {S} _ {i} = \operatorname{dsoftmax} (\boldsymbol {d} \boldsymbol {P} _ {i}) = \left(\operatorname{diag} (\boldsymbol {P} _ {i}) - \boldsymbol {P} _ {i} \boldsymbol {P} _ {i} ^ {\top}\right) \boldsymbol {d} \boldsymbol {P} _ {i} \tag {9}
$$

$$
\boldsymbol {d} \boldsymbol {Q} = \boldsymbol {d} \boldsymbol {S} \boldsymbol {K} / \sqrt {d} \tag {10}
$$

$$
\boldsymbol {d} \boldsymbol {K} = \boldsymbol {d} \boldsymbol {S} ^ {\top} \boldsymbol {Q} / \sqrt {d} \tag {11}
$$

As we can see, the backward pass depends on some very large activations from the forward pass. For example, computing $\boldsymbol { d } \boldsymbol { V }$ in Equation 7 requires $\boldsymbol { P }$, which are the attention scores of shape `(batch_size, n_heads, seq_len, seq_len)`; the size of this activation matrix depends quadratically on the sequence length, explaining the memory issues we encountered above when benchmarking attention at large sequence lengths. During both the forward and backward pass of vanilla attention, we pay significant memory IO costs to transfer $\boldsymbol { P }$ and other large activations between on-chip SRAM and GPU HBM. There are several such transfers made in standard implementations: for example, a standard backward pass implementation would read $\boldsymbol { P }$ from HBM in the computations of both Equation 7 and Equation 9. 

从这里可以看出，backward 依赖于 forward 中一些非常大的激活。例如，式 (7) 需要依赖注意力概率矩阵，而它的形状是 `(batch_size, n_heads, seq_len, seq_len)`，大小会随 sequence length 二次增长，这也解释了你之前在大序列长度注意力基准中遇到的显存问题。无论 forward 还是 backward，朴素注意力都会为在片上 SRAM 与 GPU HBM 之间来回搬运这些大张量付出很高的 IO 代价，例如 standard backward 里会在多个地方反复读取这些中间结果。

The main goal of FlashAttention is to avoid reading and writing the attention matrix to and from HBM, to reduce IO and peak memory costs. We accomplish this using three techniques: tiling, recomputation, and operator fusion. 

FlashAttention 的核心目标，就是避免把注意力矩阵写入并再读回 HBM，从而降低 IO 和峰值显存。为此，我们将同时使用三种技术：tiling、recomputation 和 operator fusion。

# Tiling

To avoid reading and writing the attention matrix to and from HBM, we compute the softmax reduction without access to the whole input. Specifically, we restructure the attention computation to split the input into tiles and make several passes over input tiles, thus incrementally performing the softmax reduction. 

为了避免把注意力矩阵写入再读回 HBM，我们需要在看不到完整输入的前提下完成 softmax 归约。具体地，我们把输入切成多个 tile，并对这些 tile 做多次遍历，从而以增量方式完成 softmax 归约。

# Recomputation

We avoid storing the large intermediate attention matrices of shape `(batch_size, n_heads, seq_len, seq_len)` in HBM. Instead, we will save certain “activation checkpoints” in HBM and then recompute part of the forward pass during the backward pass, to get the other activations we need for computing gradients. FlashAttention-2 also stores the logsumexp of the attention scores, $L$, which will be used to simplify the backward pass computation. The expression for $L$ is: 

我们不再把形状为 `(batch_size, n_heads, seq_len, seq_len)` 的大中间注意力矩阵直接存到 HBM，而是只保存某些“激活检查点”，并在 backward 时重算一部分 forward，以恢复求梯度所需的其他激活。FlashAttention-2 还会保存注意力分数的 logsumexp，也就是 $L$，它能让 backward 的计算更简洁。

$$
L _ {i} = \log \left(\sum_ {j} \exp \left(\boldsymbol {S} _ {i j}\right)\right) \tag {12}
$$

In our final kernel we will compute this in an online manner, but the final result should be the same. With tiling and recomputation together, our memory IO and peak usage no longer depend on sequence_length2 and therefore we may use larger sequence lengths. 

在最终 kernel 中，我们会用在线方式计算这个量，但最终结果应保持一致。把 tiling 和 recomputation 结合起来之后，显存 IO 与峰值占用将不再依赖 `sequence_length^2`，因此我们就能处理更长的序列。

# Operator fusion

Lastly, we avoid repeated memory IO for the attention matrix and other intermediate activations by performing all our operations in a single kernel—this is referred to as operator or kernel fusion. We will write a single Triton kernel for the forward pass that performs all the operations involved in attention with limited data transfer between HBM and SRAM. Operator fusion is partly enabled by recomputation, since we can avoid the usual memory IO we would pay to store every intermediate activation to HBM. 

最后，我们把原本分散的操作尽量塞进同一个 kernel 中，避免对注意力矩阵和其他中间激活反复做显存 IO；这就是所谓的 operator fusion 或 kernel fusion。我们会编写一个单独的 Triton forward kernel，在 HBM 与 SRAM 之间只做有限的数据搬运。recomputation 也为这种融合提供了帮助，因为它减少了“把每个中间激活都写回 HBM”的需要。

For more intuition on these techniques, check out the FlashAttention papers [T. Dao et al., 2022; T. Dao, 2023]. 

如果你想对这些技巧形成更直观的理解，可以参考 FlashAttention 的两篇论文 [T. Dao et al., 2022; T. Dao, 2023]。

# Backward pass with recomputation

### 使用重计算的 backward

Using $L$, we can do the appropriate recomputation and compute the backward pass efficiently. Before we start the backward pass, we precompute the value $D = \operatorname { r o w s u m } ( O \circ d O )$ in global memory (where $\circ$ is element-wise multiplication), which is equal to rowsum $( P \circ d P )$ since $P d P ^ { \top } = P { \left( d O V ^ { \top } \right) } ^ { \top } = ( P V ) d O ^ { \top } = O d O ^ { \top }$ (and rowsum $( A \circ B ) = \mathrm { d i a g } ( A B ^ { \top } )$ for any matrices $A$ and $B$). With the $L$ and $D$ vectors, the backward pass can be computed without softmax. The full calculation for the backward pass is now: 

借助 `L`，我们可以在 backward 中做相应的重计算，并高效地完成梯度求解。在 backward 开始前，我们先在全局显存中预计算

$$
\boldsymbol {S} = \boldsymbol {Q} \boldsymbol {K} ^ {\top} / \sqrt {d} \tag {13}
$$

$$
P _ {i j} = \exp \left(S _ {i j} - L _ {i}\right) \tag {14}
$$

$$
\boldsymbol {d} \boldsymbol {V} = \boldsymbol {P} ^ {\top} \boldsymbol {d} \boldsymbol {O} \tag {15}
$$

$$
d P = d O V ^ {\top} \tag {16}
$$

$$
d S _ {i j} = P _ {i j} \left(d P _ {i j} - D _ {i}\right) \tag {17}
$$

$$
\boldsymbol {d} \boldsymbol {Q} = \boldsymbol {d} \boldsymbol {S} \boldsymbol {K} / \sqrt {d} \tag {18}
$$

$$
\boldsymbol {d} \boldsymbol {K} = \boldsymbol {d} \boldsymbol {S} ^ {\top} \boldsymbol {Q} / \sqrt {d} \tag {19}
$$

We can see that the sequence of operations does not require us to have stored the attention scores $P$ in HBM during the forward pass, since we recompute them from the activations $Q$, $K$, and $L$ in Equation 13 and Equation 14. 

其中 `∘` 表示逐元素乘法。这个量与 `rowsum(P ∘ dP)` 相等，因为：

# Details of the FlashAttention forward pass

### FlashAttention forward 的细节

Now that we have a high-level idea of the techniques used in FlashAttention-2, we will dive into the details of the FA2 forward pass kernel that you will implement. In order to avoid reading and writing the attention matrix to and from HBM, we wish to use tiling, i.e., computing each tile of the output independently of the others. This requires us to be able to compute tiles of $P$, ideally tiled in both dimensions (for queries and for keys). 

既然已经有了高层思路，下面我们进一步深入到你将实现的 FA2 forward kernel 的细节。为了避免在 HBM 中读写完整注意力矩阵，我们希望使用 tiling，也就是独立地计算输出的每个 tile。这要求我们能构造 `O` 的局部 tile，理想情况下沿 query 和 key 两个维度都做 tiling。

However, when we apply softmax to $S$, we require entire rows of $S$ to be reduced to compute the softmax denominator, meaning we cannot compute $P$ in tiles directly. FlashAttention-2 solves this problem using online softmax. In the following text, we will use subscript index $i$ to denote the current query tile, and superscript index $(j)$ to denote the current key tile. The tiles along the query dimension will be of size $B _ { q }$ and those along the key dimension will be of size $B _ { k }$ . We will not tile along the hidden dimension $d$. 

但 softmax 需要对整行做归约来计算分母，因此我们不能直接把 `P` 简单按 tile 独立算完。FlashAttention-2 通过 online softmax 解决了这个问题。后文中，下标 `i` 表示当前 query tile，上标 `(j)` 表示当前 key tile。沿 query 维度的 tile 大小为 `B_q`，沿 key 维度的 tile 大小为 `B_k`。我们不在 hidden dimension `d` 上切 tile。

We also keep some row-wise running values, $m _ { i } ^ { ( j ) } \in \mathbb { R } ^ { B _ { q } }$ and $\boldsymbol { l } _ { i } ^ { ( j ) } \in \mathbb { R } ^ { B _ { q } }$ . The row-wise $m _ { i } ^ { ( j ) }$ value is a running maximum, which is tracked so we can compute softmax in a numerically stable manner (recall this trick from our softmax implementation in Assignment 1). We will update $m _ { i } ^ { ( j ) }$ with each new row-wise tile of $S$ (when $j$ increases). Using the running maximum, we can compute the unnormalized softmax values (numerators) as $\tilde { P } _ { i } ^ { ( j ) } = \exp \bigl( S _ { i j } - m _ { i } ^ { ( j ) } \bigr)$ . $l _ { i } ^ { ( j ) }$ is a running proxy for the softmax denominator, and will be updated using the unnormalized softmax values as $l _ { i } ^ { ( j ) } = \exp \left( m _ { i } ^ { \left( j - 1 \right) } - m _ { i } ^ { \left( j \right) } \right) \cdot l _ { i } ^ { ( j - 1 ) } + \operatorname { r o w s u m } \left( \tilde { P } _ { i } ^ { ( j ) } \right)$ . When we finally write the output, we will need to finish normalizing it by using $l _ { i } ^ { ( T _ { k } ) }$ , which is the final value of $\boldsymbol { l } _ { i } ^ { ( j ) }$ after processing all key tiles. Algorithm 1 shows the forward pass on the GPU. 

同时，我们会维护两个按行的运行量：`m_i^(j)` 和 `l_i^(j)`，它们都属于 `R^{B_q}`。`m_i^(j)` 是按行的运行最大值，用于数值稳定地计算 softmax；当 `j` 增长时，它会随着新 key tile 的加入而更新。给定这个运行最大值，我们可以计算未归一化的 softmax 分子：

Algorithm 1: FlashAttention-2 forward pass

`l_i^(j)` 是 softmax 分母的运行代理量，它的更新规则为：

1 Require: $Q \in R^{N_{q} \times d}$ , $K, V \in R^{N_{k} \times d}$ , tile sizes $B_{q}, B_{k}$ 2 Split Q into $T_{q} = \left\lceil \frac{N_{q}}{B_{q}} \right\rceil$ tiles $Q_{1}, ..., Q_{T_{q}}$ of size $B_{q} \times d$ 3 Split K, V into $T_{k} = \left\lceil \frac{N_{k}}{B_{k}} \right\rceil$ tiles $K^{(1)}, ..., K^{(T_{k})}$ and $V^{(1)}, ..., V^{(T_{k})}$ of size $B_{k} \times d$ 4 for i = 1, ..., $T_{q}$ do
5 Load $Q_{i}$ from global memory
6 Initialize $O_{i}^{(0)} = 0 \in R^{B_{q} \times d}, l_{i}^{(0)} = 0 \in R^{B_{q}}, m_{i}^{(0)} = -\infty \in R^{B_{q}}$ 7 for j = 1, ..., $T_{k}$ do
8 Load $K^{(j)}, V^{(j)}$ from global memory
9 Compute tile of pre-softmax attention scores $S_{i}^{(j)} = \frac{Q_{i}(K^{(j)})^{\top}}{\sqrt{d}} \in R^{B_{q} \times B_{k}}$ 10 Compute $m_{i}^{(j)} = \max\left(m_{i}^{(j-1)}, \text{rowmax}\left(S_{i}^{(j)}\right)\right) \in R^{B_{q}}$ 11 Compute $\tilde{P}_{i}^{(j)} = \exp\left(S_{i}^{(j)} - m_{i}^{(j)}\right) \in R^{B_{q} \times B_{k}}$ 12Compute $l_{i}^{(j)} = \exp\left(m_{i}^{(j-1)} - m_{i}^{(j)}\right) l_{i}^{(j-1)} + \text{rowsum}\left(\tilde{P}_{i}^{(j)}\right) \in R^{B_{q}}$ 13Compute $O_{i}^{(j)} = \text{diag}\left(\exp\left(m_{i}^{(j-1)} - m_{i}^{(j)}\right)\right) O_{i}^{(j-1)} + \tilde{P}_{i}^{(j)} V^{(j)}$ 14 end for
15 Compute $O_{i} = \text{diag}\left(l_{i}^{(T_{k})}\right)^{-1} O_{i}^{(T_{k})}$ 16Compute $L_{i} = m_{i}^{(T_{k})} + \log\left(l_{i}^{(T_{k})}\right)$ 17 Write $O_{i}$ to global memory as the i-th tile of O.
18 Write $L_{i}$ to global memory as the i-th tile of L.
19 end for
20 Return the output O and the logsumexp L. 

当我们最终把输出写回时，还需要使用最后的 `l_i^(T_k)` 做归一化。算法 1 给出了 GPU 上的完整前向过程。

Before we get into implementing the forward pass in Triton, we collect here a few general tips and tricks for writing Triton kernels. 

在真正开始实现 Triton forward 之前，我们先汇总几条编写 Triton kernel 的通用技巧。

# Triton Tips and Tricks

### Triton 编写技巧

• You can use print statements in Triton with tl.device_print to debug: https://triton-lang.org/main/python-api/generated/triton.language.device_print.html. There is a setting TRITON_INTERPRET=1 to run the Triton interpreter on CPU, though we have found it buggy. 

在真正开始写 Triton forward 前，我们先给出一些通用技巧：

• 可以用 `tl.device_print` 在 Triton 中打印调试信息：https://triton-lang.org/main/python-api/generated/triton.language.device_print.html。另有一个 `TRITON_INTERPRET=1` 选项可在 CPU 上运行 Triton 解释器，不过我们发现它有一些 bug。

• When defining block pointers, make sure they have the correct offsets, and that block offsets are multiplied by the appropriate tile sizes. 

• 可以使用 `tl.device_print` 在 Triton kernel 中打印调试信息。

• The launch grid of thread blocks is set with 

• 定义 block pointer 时要确认 offset 正确，并且 offset 要乘上相应 tile size。

```python
kernel_fn[(launch_grid_d1, launch_grid_d2, ...)](...arguments...) 
```

in the methods of the torch.autograd.Function subclass, as we saw in the weighted sum example. 

• launch grid 的设置方式为：

正如 weighted sum 示例里那样，这个写法会出现在 `torch.autograd.Function` 子类的方法中。

• Perform matrix multiplications with tl.dot. 

• 矩阵乘法使用 `tl.dot`。

• To advance a block pointer, use *_block_ptr = *_block_ptr.advance(...) 

• 推进 block pointer 时，使用 `*_block_ptr = *_block_ptr.advance(...)`。

# Problem (flash_forward):  FlashAttention-2 Forward Pass (15 points)

### 题目（flash_forward）：FlashAttention-2 前向传播（15 分）

(a) Write a pure PyTorch (no Triton) autograd.Function that implements the FlashAttention-2 forward pass. This will be a lot slower than the regular PyTorch implementation, but will help you debug your Triton kernel. 

(a) 编写一个纯 PyTorch（不使用 Triton）的 `autograd.Function`，实现 FlashAttention-2 的 forward。这个版本会比常规 PyTorch 实现更慢，但它很适合作为 Triton kernel 的调试参考。

Your implementation should take input $Q$, $K$, and $V$ as well as a flag `is_causal` and produce the output $O$ and the logsumexp value $L$. You can ignore the `is_causal` flag for this task. The `autograd.Function.forward` should then save $L$, $Q$, $K$, $V$, and $O$ for the backward pass and return $O$. Remember that the implementation of the `forward` method of `autograd.Function` always takes the context as its first parameter. Any `autograd.Function` class needs to implement a `backward` method, but for now you can make it just raise `NotImplementedError`. If you need something to compare against, you can implement Equation 4 to Equation 6 and Equation 12 in PyTorch and compare your outputs. 

你的实现应当接收 `Q`、`K`、`V` 和 `is_causal` 标志，并输出 `O` 和 logsumexp 值 `L`。对于这一小问，可以忽略 `is_causal`。`autograd.Function.forward` 还应把 backward 所需的 `Q`、`K`、`V`、`O` 和 `L` 保存起来，并返回输出。记住，`autograd.Function` 的 `forward` 方法总是把 `ctx` 作为第一个参数。任何 `autograd.Function` 都必须实现一个 `backward` 方法，但暂时你可以让它直接抛出 `NotImplementedError`。如果你需要参考，可以先用 PyTorch 按式 (4) 到式 (6) 以及式 (12) 实现一个版本，对照输出是否一致。

The interface is then def forward(ctx, Q, K, V, is_causal=False). Determine your own tile sizes, but make sure they are at least of size 16 × 16. We will always test your code with dimensions that are powers of 2 and at least 16, so you don’t need to worry about out-ofbounds accesses. 

接口形式应当是：

Deliverable: A torch.autograd.Function subclass that implements FlashAttention-2 in the forward pass. To test your code, implement 

tile size 由你自己决定，但请确保至少为 `16 × 16`。测试时我们只会使用各维度都是 2 的幂且不小于 16 的输入，因此你暂时不需要处理越界访问。

[adapters.get_flashattention_autograd_function_pytorch] . Then, run the test with uv run pytest -k test_flash_forward_pass_pytorch and make sure your implementation passes it. 

提交内容：一个实现了 FlashAttention-2 forward 的 `torch.autograd.Function` 子类。为测试你的代码，请实现 `adapters.get_flashattention_autograd_function_pytorch`，然后运行 `uv run pytest -k test_flash_forward_pass_pytorch`，确保通过。

(b) Write a Triton kernel for the forward pass of FlashAttention-2 following Algorithm 1. Then, write another subclass of torch.autograd.Function that calls this (fused) kernel in the forward pass, instead of computing the result in PyTorch. A few problem-specific tips: 

(b) 按照算法 1，为 FlashAttention-2 的 forward 编写一个 Triton kernel。然后，再写一个 `torch.autograd.Function` 子类，在 forward 中调用这个融合后的 Triton kernel，而不是用 PyTorch 计算结果。下面这些提示会对你有帮助：

• To debug, we suggest comparing the results of each Triton operation you perform with the tiled PyTorch implementation you wrote in part (a). 

• 调试时，建议把 Triton 中的每一步结果与 (a) 中写出的 tiled PyTorch 实现逐步对比。

• Your launch grid should be set as $( T _ { q } , \text { batch_size } )$, meaning each Triton program instance will load only elements from a single batch index, and only read/write to a single query tile of $Q$, $O$, and $L$. 

• 你的 launch grid 应设为 `(num_query_tiles, batch_size)`，也就是说，每个 Triton program instance 只会访问单个 batch index，并且只会读写 `Q`、`O`、`L` 的一个 query tile。

• The kernel should only have a single loop, which will iterate key tiles $1 \leq j \leq T _ { k }$ . 

• kernel 内只应有一个循环，它沿 key tiles `1 ≤ j ≤ T_k` 迭代。

• Advance block pointers at the end of the loop. 

• 在循环末尾推进 block pointers。

• Use the function declaration below (using the block pointer we give you, you should be able to infer the setup of the rest of the pointers): 

• 你可以使用题目给出的函数声明；给定其中的 `Q_block_ptr`，你应当能够推断出其他 pointer 的设置方式。

```txt
@triton.jit
def flash_fwd_kernel( 
```

```python
Q_ptr, K_ptr, V_ptr,
O_ptr, L_ptr,
stride_qb, stride_qq, stride_qd,
stride_kb, stride_kk, stride_kd,
stride_vb, stride_vk, stride_vd,
stride_ob, stride_oq, stride_od,
stride_lb, stride_lq,
N_QUERIES, N_KEYS,
scale,
D: tl.constexpr,
Q_TILE_SIZE: tl.constexpr,
K_TILE_SIZE: tl.constexpr,
):

# Program indices

query_tile_index = tl.program_id(0)
batch_index = tl.program_id(1)

# Offset each pointer with the corresponding batch index

# multiplied with the batch stride for each tensor

Q_block_ptr = tl.make_block_ptr(
    Q_ptr + batch_index * stride_qb,
    shape=(N_QUERIES, D),
    strides=(stride_qq, stride_qd),
    offsets=(query_tile_index * Q_TILE_SIZE, 0),
    block_shape=(Q_TILE_SIZE, D),
    order=(1, 0),
)
... 

```

where `scale` is $\textstyle { \frac { 1 } { \sqrt { d } } }$ and `Q_TILE_SIZE` and `K_TILE_SIZE` are $B _ { q }$ and $B _ { k }$ respectively. You can tune these later. 

These additional guidelines may help you avoid precision issues: 

• The on chip buffers $( O _ { i } , l , m )$ should have dtype tl.float32. If you’re accumulating into an output buffer, use the acc argument (acc = tl.dot(..., acc=acc)). 

• Cast $\tilde { P } _ { i } ^ { ( j ) }$ to the dtype of $V ^ { ( j ) }$ before multiplying them, and cast $O _ { i }$ to the appropriate dtype before writing it to global memory. Casting is done with tensor.to. You can get the dtype of a tensor with tensor.dtype, and the dtype of a block pointer/pointer with *_block_ptr.type.element_ty. 

Deliverable: A torch.autograd.Function subclass that implements FlashAttention-2 in the forward pass using your Triton kernel. Implement 

[adapters.get_flash_autograd_function_triton] . Then, run the test with uv run pytest -k test_flash_forward_pass_triton and make sure your implementation passes it. 

(c) Add a flag as the last argument to your autograd.Function implementation for causal masking. This should be a boolean flag that, when set to True, enables an index comparison for causal masking. Your Triton kernel should have a corresponding additional parameter is_causal: tl.constexpr (this is a required type annotation). In Triton, construct appropriate index vectors for queries and keys, and compare them to form a square mask of size $B _ { q } \times B _ { k }$ . For elements that are masked out, add the constant value of -1e6 to the corresponding elements of the attention score matrix $S _ { i } ^ { ( j ) }$ . Make sure to save the mask flag for backward using ctx.is_causal = is_causal. 

Deliverable: An additional flag for your torch.autograd.Function subclass that implements the FlashAttention-2 forward pass with causal masking using your Triton kernel. Make sure that the flag is optional and defaults to False so the previous tests still pass. 

# Implementing the backward pass with recomputation

### 用重计算实现 backward

Notice that unlike the standard backward pass in Equation 7 to Equation 11, we can use recomputation to avoid the softmax operation in the backward pass shown in Equation 13 to Equation 19. This means that we can compute the backward pass using a trivial kernel, and no online tricks are required. Thus, for this part, you can implement backward by calling torch.compile on a regular PyTorch function (not Triton). 

注意，与式 (7) 到式 (11) 给出的标准 backward 不同，在式 (13) 到式 (19) 中，我们可以借助重计算，在 backward 中绕开 softmax 操作。这意味着你可以用一个普通的 PyTorch 函数配合 `torch.compile` 来实现 backward，而不需要额外的 Triton 技巧。

# Problem (flash_backward):  FlashAttention-2 Backward Pass (5 points)

### 题目（flash_backward）：FlashAttention-2 反向传播（5 分）

Implement the backward pass for your FlashAttention-2 autograd.Function using PyTorch (not Triton) and `torch.compile`. Your implementation should take the $Q$, $K$, $V$, $O$, $dO$, and $L$ tensors as inputs, and return $dQ$, $dK$, and $dV$. Remember to compute and use the $D$ vector. You may follow along the computations of Equation 13 to Equation 19. 

使用 PyTorch（不是 Triton）加 `torch.compile`，为你的 FlashAttention-2 `autograd.Function` 实现 backward。你的实现应当接收 forward 中保存的各个张量，并输出 `dQ`、`dK` 和 `dV`。记得计算并使用向量 `D`。你可以沿着式 (13) 到式 (19) 的计算顺序实现。

Deliverable: To test your implementation, run uv run pytest -k test_flash_backward. 

提交内容：运行 `uv run pytest -k test_flash_backward` 测试你的实现。

Let’s now compare the performance of your (partially) Triton implementation of FlashAttention-2 with your PyTorch implementation of regular Attention. 

接下来，我们来比较你这个“部分使用 Triton”的 FlashAttention-2 实现，与普通 PyTorch 注意力实现之间的性能差异。

# Problem (flash_benchmarking):  FlashAttention-2 Benchmarking (5 points)

### 题目（flash_benchmarking）：FlashAttention-2 基准测试（5 分）

(a) Write a benchmarking script using triton.testing.do_bench that compares the performance of your (partially) Triton implementation of FlashAttention-2 forward and backward passes with a regular PyTorch implementation (i.e., not using FlashAttention). 

(a) 编写一个 benchmarking 脚本，使用 `triton.testing.do_bench`，比较你的“部分 Triton” FlashAttention-2 实现与常规 PyTorch 注意力实现（即不使用 FlashAttention）在 forward、backward 以及端到端 forward-backward 上的性能。

Specifically, you will report a table that includes latencies for forward, backward, and the endto-end forward-backward pass, for both your Triton and PyTorch implementations. Randomly generate any necessary inputs before you start benchmarking, and run the benchmark on a single B200. Always use batch size 1 and causal masking. Sweep over the cartesian product of sequence lengths of various powers of 2 from 128 up to 65536, embedding dimension sizes of various powers of 2 from 16 up to size 128, and precisions of torch.bfloat16 and torch.float32. You will likely need to adjust tile sizes depending on the input sizes. 

你需要给出一个表格，汇报 forward、backward 和完整前向加反向过程的延迟，对比 Triton 实现与 PyTorch 实现。所有输入在 benchmark 开始前随机生成。benchmark 在单张 B200 上执行，始终使用 batch size 1 和 causal masking。请扫描下列笛卡尔积：sequence length 为从 128 到 65536 的 2 的幂；embedding dimension 为从 16 到 128 的 2 的幂；精度为 `torch.bfloat16` 与 `torch.float32`。你很可能需要根据输入尺寸调整 tile size。

Deliverable: A table of results comparing your implementation of FlashAttention-2 with the PyTorch implementation, using the settings above and reporting forward, backward, and endto-end latencies. 

提交内容：一个结果表格，比较你的 FlashAttention-2 实现与 PyTorch 实现在上述设置下 forward、backward 与端到端延迟的差异。

# 4.2.3 OPTIONAL: Triton backward pass

## 4.2.3 可选：Triton backward

If you’re interested in getting more practice with Triton and/or having a fast leaderboard submission, we provide the tiled FlashAttention-2 backward pass below which you can implement in Triton. Algorithm 2 shows the FlashAttention-2 backward pass as it should be implemented in Triton. A key trick here is to compute $P$ twice, once for $dQ$ and again for $dK$ and $dV$. This lets us skip synchronization across thread blocks, meaning we can avoid slow atomics. 

如果你想进一步练习 Triton，或者想冲排行榜，我们还提供了 tiled FlashAttention-2 backward 的思路，你可以自行用 Triton 实现。算法 2 展示了应如何在 Triton 中写出 FlashAttention-2 backward。这里的关键技巧是把某些量计算两次：一遍用于 `dQ`，另一遍用于 `dK` 和 `dV`。这样我们就不需要跨 thread block 同步，也就避免了昂贵的原子操作。

Algorithm 2: Tiled FlashAttention-2 backward pass

Require: $Q, O, dO \in \mathbb{R}^{N_q \times d}$ , $K, V \in \mathbb{R}^{N_k \times d}$ , $L \in \mathbb{R}^{N_q}$ , tile sizes $B_q$ , $B_k$ Compute $D = \text{rowsum}(O \circ dO) \in \mathbb{R}^{N_q}$ Split $Q, O, dO$ into $T_q = \left[\frac{N_q}{B_q}\right]$ tiles $Q_1, ..., Q_{T_q}, O_1, ..., O_{T_q}, dO_1, ..., dO_{T_q}$ , each of size $B_q \times d$ Split $K, V$ into $T_k = \left[\frac{N_k}{B_k}\right]$ tiles $K^{(1)}, ..., K^{(T_k)}$ and $V^{(1)}, ..., V^{(T_k)}$ , each of size $B_k \times d$ Split $L, D$ into $T_q$ tiles $L_1, ..., L_{T_q}$ and $D_1, ..., D_{T_q}$ , each of size $B_q$ for $j = 1, ..., T_k$ do

Load $K^{(j)}, V^{(j)}$ from global memory

Initialize $dK_0^{(j)} = dV_0^{(j)} = 0 \in \mathbb{R}^{B_k \times d}$ for $i = 1, ..., T_q$ do

Load $Q_i, dO_i$ from global memory

Compute tile of attention scores $S_i^{(j)} = \frac{Q_i(K^{(j)})^\top}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k}$ Compute attention probabilities $P_i^{(j)} = \exp(S_i^{(j)} - L_i) \in \mathbb{R}^{B_q \times B_k}$ Compute $dV_i^{(j)} = dV_{i-1}^{(j)} + (P_i^{(j)})^\top dO_i \in \mathbb{R}^{B_k \times d}$ Compute $dP_i^{(j)} = dO_i(V^{(j)})^\top \in \mathbb{R}^{B_q \times B_k}$ Compute $dS_i^{(j)} = P_i^{(j)} \circ (dP_i^{(j)} - D_i) \in \mathbb{R}^{B_q \times B_k}$ Compute $dK_i^{(j)} = dK_{i-1}^{(j)} + (dS_i^{(j)})^\top Q_i / \sqrt{d} \in \mathbb{R}^{B_k \times d}$ .

end for

Write $dK_{T_q}^{(j)}$ and $dV_{T_q}^{(j)}$ to global memory as the $j$ -th tiles of $dK$ and $dV$ .

end for

for $i = 1, ..., T_q$ do

Load $Q_i, dO_i$ from global memory

Initialize $dQ_i^{(0)} = 0 \in \mathbb{R}^{B_q \times d}$ for $j = 1, ..., T_k$ do

Load $K^{(j)}, V^{(j)}$ from global memory

Compute tile of attention scores $S_i^{(j)} = \frac{Q_i(K^{(j)})^\top}{\sqrt{d}} \in \mathbb{R}^{B_q \times B_k}$ Compute attention probabilities $P_i^{(j)} = \exp(S_i^{(j)} - L_i) \in \mathbb{R}^{B_q \times B_k}$ Compute $dP_i^{(j)} = dO_i(V^{(j)})^\top \in \mathbb{R}^{B_q \times B_k}$ Compute $dS_i^{(j)} = P_i^{(j)} \circ (dP_i^{(j)} - D_i) \in \mathbb{R}^{B_q \times B_k}$ Compute $dQ_i^{(j)} = dQ_i^{(j-1)} + dS_i^{(j)} K^{(j)} / \sqrt{d} \in \mathbb{R}^{B_q \times d}$ end for

Write $dQ_i^{(T_k)}$ to global memory as the $i$ -th tile of $dQ$ .

end for

Return $dQ, dK, dV$ . 

# 5 Distributed Data Parallel Training

## 5 分布式数据并行训练

In this next part of the assignment, we’ll explore methods for using multiple GPUs to train our language models, focusing on data parallelism. We’ll start with a primer on distributed communication in $\mathrm { P y }$ Torch. Then, we’ll study a naïve implementation of distributed data parallel training, then implement and benchmark various improvements to communication efficiency. 

在作业的下一部分，我们将探索如何用多张 GPU 来训练语言模型，重点关注数据并行。我们会先从 PyTorch 分布式通信的基础开始，然后研究一个朴素版的数据并行训练实现，再进一步实现和 benchmark 多种提高通信效率的方法。

# 5.1 Single-Node Distributed Communication in PyTorch

## 5.1 PyTorch 中的单机分布式通信

Let’s start by looking at a simple distributed application in PyTorch, where the goal is to generate four random integer tensors and compute their sum. 

我们先来看一个简单的 PyTorch 分布式应用：目标是生成四个随机整数张量，并把它们求和。

In the distributed case below, we will spawn four worker processes, each of which generates a random integer tensor. To sum these tensors across the worker processes, we will call the all-reduce collective communication operation, which replaces the original data tensor on each process with the all-reduced result (i.e., the sum). 

在下面这个分布式例子中，我们会启动四个 worker 进程，每个进程生成一个随机整数张量。为了在这些 worker 之间求和，我们调用 `all-reduce` 这个集体通信操作。它会把每个进程原本持有的数据张量替换成 all-reduce 之后的结果（这里也就是求和结果）。

Now let’s take a look at some code. 

下面来看一段具体代码。

```python
import os
import torch
import torch.distributed as dist
import torch multiprocessing as mp

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def distributed_demo(rank, world_size):
    setup(rank, world_size)
    data = torch.randint(0, 10, (3,))
    print(f"rank {rank} data (before all-reduce): {data}")
    dist.all_reduce(data, async_op=False)
    print(f"rank {rank} data (after all-reduce): {data}")

if __name__ == "__main__":
    world_size = 4
    mp.spawn(fn=distributed_demo, args=(world_size, ), nprocs=world_size, join=True) 
```

After running the script above, we get the output below. As expected, each worker process initially holds different data tensors. After the all-reduce operation, which sums the tensors across all of the worker processes, data is modified in-place on each of the worker processes to hold the all-reduced result.2 

运行上面的脚本后，会得到如下输出。正如预期，每个 worker 进程一开始持有的张量都不相同；而在 all-reduce 之后，每个进程上的数据张量都会被原地改写为所有 worker 张量求和后的结果。

```txt
$ uv run python distributed_hello_world.py
rank 3 data (before all-reduce): tensor([3, 7, 8])
rank 0 data (before all-reduce): tensor([4, 4, 7])
rank 2 data (before all-reduce): tensor([6, 0, 7])
rank 1 data (before all-reduce): tensor([9, 5, 3])
rank 1 data (after all-reduce): tensor([22, 16, 25])
rank 0 data (after all-reduce): tensor([22, 16, 25])
rank 3 data (after all-reduce): tensor([22, 16, 25])
rank 2 data (after all-reduce): tensor([22, 16, 25]) 
```

Let’s now look back more closely at our script above. The command mp.spawn spawns nprocs processes that run fn with the provided args. In addition, the function fn is called as fn(rank, *args), where rank is the index of the worker process (a value between 0 and nprocs-1). Thus, our distributed_demo function must accept this integer rank as its first positional argument. In addition, we pass in the world_size, which refers to the total number of worker processes. 

现在更仔细地回头看看这段脚本。`mp.spawn` 会启动 `nprocs` 个进程去运行函数 `fn`，并传入指定参数。此外，`fn` 实际会被调用成 `fn(rank, *args)`，其中 `rank` 是 worker 进程的编号（从 0 到 `nprocs - 1`）。因此，我们的 `distributed_demo` 函数必须把这个整数 `rank` 作为第一个位置参数接收。除此之外，我们还传入了 `world_size`，也就是 worker 总数。

Each worker process belongs to a process group, which is initialized via dist.init_process_group. The process group represents multiple worker processes that will coordinate and communicate via a shared master. The master is defined by its IP address and port, and the master runs the process with rank 0. Collective communication operations like all-reduce operate on each process in the process group. 

每个 worker 进程都属于一个 process group，它通过 `dist.init_process_group` 初始化。process group 表示一组将通过共享 master 协调与通信的 worker 进程。master 由 IP 地址和端口定义，并由 rank 0 的进程担任。像 all-reduce 这样的集体通信操作，都是在整个 process group 上执行的。

In this case, we initialized our process group with the "gloo" backend, but other backends are available. In particular, the "nccl" backend will use the NVIDIA NCCL collective communications library, which will generally be more performant for CUDA tensors. However, NCCL can only be used on machines with GPUs, while Gloo can be run on CPU-only machines. You should always use NCCL for distributed GPU training, and only use Gloo for local development where you don’t have a GPU available. We used Gloo in this example because it enables local execution and development on CPU-only machines. 

在这个例子中，我们使用 `"gloo"` backend 来初始化 process group，但其实还有其他 backend 可选。尤其是 `"nccl"` backend 会使用 NVIDIA NCCL 集体通信库，在 CUDA 张量场景下一般会更高效。不过，NCCL 只能用于带 GPU 的机器，而 Gloo 则可以在仅 CPU 的机器上运行。因此，对于分布式 GPU 训练，你应始终使用 NCCL；而 Gloo 主要适用于本地开发时没有 GPU 的场景。这个例子之所以使用 Gloo，是因为它允许在纯 CPU 机器上本地运行和调试。

When running multi-GPU jobs, make sure that different ranks use different GPUs. One method for doing this is to call torch.cuda.set_device(rank) in the setup function, so that tensor.to("cuda") will automatically move it to the specified device. Alternatively, you can explicitly create a per-rank device string (e.g., device = f"cuda:{rank}"), and then use this device string as the target device for any data movement (e.g., tensor.to(f"cuda:{rank}")). 

当你运行多 GPU 作业时，要确保不同 rank 使用不同 GPU。一种做法是在 `setup` 函数中调用 `torch.cuda.set_device(rank)`，这样之后 `tensor.to("cuda")` 就会自动移动到对应设备。另一种做法是显式构造每个 rank 对应的设备字符串（例如 `device = f"cuda:{rank}"`），然后在所有数据移动时显式使用它。

# Terminology

## 术语说明

In the rest of the assignment (and various other resources you might see online), you may encounter the following terms in the context of PyTorch distributed communication. Though we will focus on singlenode, multi-process distributed training in this assignment, the terminology is useful for understanding distributed training in general. See Figure 3 for a visual representation. 

在本作业后续部分（以及你在网上可能看到的其他资源中），你会在 PyTorch 分布式通信语境下遇到一些术语。虽然本作业只关注单机、多进程分布式训练，但理解这些术语有助于你更普遍地理解分布式训练。图 3 给出了直观示意。

node a machine on the network. 

`node`：网络中的一台机器。  
`worker`：参与分布式训练的一个程序实例。在本作业中，每个 worker 只有一个进程，因此我们会交替使用 worker、process 和 worker process 这些词。不过在实际中，一个 worker 也可能内部再使用多个进程（例如做数据加载），所以这些词并不总是完全等价。  
`world size`：一个 process group 中 worker 的总数。  
`global rank`：范围在 `0` 到 `world_size-1` 之间的一个整数，用于唯一标识 process group 中的某个 worker。例如在 world size 为 2 时，一个进程的 global rank 是 0（master），另一个则是 1。  
`local world size`：当应用跨多个 node 运行时，某个给定 node 本地运行的 worker 数。例如若一个应用在两个 node 上各启动 4 个 worker，则 world size 为 8，而 local world size 为 4。若只在单机上运行，则 local world size 就等于全局的 world size。  
`local rank`：范围在 `0` 到 `local_world_size-1` 的一个整数，用于标识某个 worker 在本机上的编号。例如若在两个 node 上各跑 4 个进程，则每个 node 上的进程 local rank 分别是 0、1、2、3。在单机多进程场景中，local rank 等同于 global rank。

worker an instance of a program that’s participating in the distributed training. In this assignment, each worker will have a single process, so we’ll use worker, process, and worker process interchangeably. However, a worker may use multiple processes (e.g., to load data for training), so these terms are not always equivalent in practice. 

图 3：一个运行在 2 个 node 上、world size 为 8 的分布式应用示意图。每个 worker 进程都具有一个 global rank（0 到 7）和一个 local rank（0 到 3）。图引自：

world size The number of total workers in a process group. 

`world size`：一个进程组中的 worker 总数。

global rank An integer ID (between 0 and world_size-1) that uniquely identifies a worker in the process group. For example, for world size of two, one process will have global rank 0 (the master process) and the other process will have rank 1. 

`global rank`：一个整数 ID（范围是 `0` 到 `world_size - 1`），用于在进程组中唯一标识某个 worker。比如 world size 为 2 时，一个进程的 global rank 为 0（主进程），另一个则为 1。

local world size When running applications across different nodes, the local world size is the number of workers running locally on a given node. For example, if we have an application that spawns 4 workers on 2 nodes each, the world size would be 8 and the local world size would be 4. Note that when running on a single node, the local world size of a worker is equivalent to the (global) world size. 

`local world size`：当应用跨多个节点运行时，local world size 表示某个节点本地启动的 worker 数。例如，如果一个应用在 2 个节点上、每个节点启动 4 个 worker，那么 world size 是 8，而 local world size 是 4。若只在单节点上运行，则 local world size 与全局 world size 相同。

local rank An integer ID (between 0 and local_world_size-1) that uniquely identifies the index of a local worker on the machine. For example, if we have an application that spawns 4 processes on 2 nodes each, each node would have workers with local ranks 0, 1, 2, and 3. Note that when running a single-node multi-process distributed application, the local rank of a process is equivalent to its global rank. 

`local rank`：一个整数 ID（范围是 `0` 到 `local_world_size - 1`），用于唯一标识某台机器上的本地 worker 编号。比如在 2 个节点上每个节点启动 4 个进程时，每台机器上的 local rank 会是 0、1、2、3。若是单节点多进程分布式应用，则某个进程的 local rank 与它的 global rank 相同。

![](images/e69b9ca9ade3f00e9f6c7b30cc09e56b0c4c146e2fad412d40efd55fa62a6a09.jpg)

MACHINE1

LOCAL RANK GLOBAL RANK NODE RANK

![](images/f30493aebadaabf66940d6402641675645a96261f50c363cbf063296b7fd53af.jpg)

MACHINE2

Figure 3: A schematic representation of a distributed application running on 2 nodes with a world size of 8. Each worker process is identified by a global rank (from 0 to 7) and a local rank (from 0 to 3). Figure taken from https://lightning.ai/docs/fabric/stable/advanced/distributed_communication.html

图 3：一个运行在 2 个节点上、world size 为 8 的分布式应用示意图。每个 worker 进程都由一个 global rank（0 到 7）和一个 local rank（0 到 3）标识。图片引自：https://lightning.ai/docs/fabric/stable/advanced/distributed_communication.html

# 5.1.1 Best Practices for Benchmarking Distributed Applications

## 5.1.1 分布式应用基准测试的最佳实践

Throughout this portion of the assignment you will be benchmarking distributed applications to better understand the overhead from communication. Here are a few best practices: 

在这一部分作业中，你会对分布式应用做 benchmark，以理解通信带来的开销。下面有一些推荐做法：

• Whenever possible, run benchmarks on the same machine to facilitate controlled comparisons. 

• 尽量在同一台机器上做 benchmark，以方便可控比较。

• Perform several warm-up steps before timing the operation of interest. This is especially important for NCCL communication calls. 5 iterations of warmup is generally sufficient. 

• 在计时前先执行若干次 warm-up。对 NCCL 通信尤其重要。一般来说，5 次 warm-up 就足够。

• Call torch.cuda.synchronize() to wait for CUDA operations to complete when benchmarking on GPUs. Note that this is necessary even when calling communication operations with async_op=False, which returns when the operation is queued on the GPU (as opposed to when the communication actually finishes).3 

• 在 GPU 上做 benchmark 时调用 `torch.cuda.synchronize()` 等待 CUDA 操作完成。注意，即使你在通信操作中使用 `async_op=False` 也仍然需要这样做，因为这个返回只保证通信已在 GPU 上排队，而不是说通信已经真的完成。

• Timings may vary slightly across different ranks, so it’s common to aggregate measurements across ranks to improve estimates. You may find the all-gather collective (specifically the dist.all_gather_object function) to be useful for collecting results from all ranks. 

• 不同 rank 的计时可能略有差异，因此通常会聚合来自各 rank 的测量，以提高估计稳定性。你可能会发现 `all-gather`，尤其是 `dist.all_gather_object`，在收集所有 rank 的结果时很有用。

• In general, debug locally with Gloo on CPU, and then as required in a given problem, benchmark with NCCL on GPU. 

• 总体上，先用 CPU + Gloo 本地调试，再在题目要求的场景下改用 GPU + NCCL 做正式 benchmark。

Switching between the backends should be as simple as modifying your init_process_group call and tensor device casts. 

在不同 backend 之间切换，通常只需要改动 `init_process_group` 的 backend 设置，以及张量搬运到设备的方式。

# Problem (distributed_communication_single_node):  Distributed Communication (Single Node) (5 points)

### 题目（distributed_communication_single_node）：分布式通信（单机）（5 分）

Write a script to benchmark the runtime of the all-reduce operation in the single-node multiprocess setup. The example code above may provide a reasonable starting point. Experiment with varying the following settings: 

编写一个脚本，在单机多进程设置下 benchmark `all-reduce` 操作的运行时间。上面的示例代码可以作为一个合理起点。请尝试改变以下设置：

all-reduce data size float32 data tensors ranging over 1MB, 10MB, 100MB, 1GB. 

all-reduce 数据大小：`float32` 数据张量，大小分别为 1MB、10MB、100MB、1GB。  
GPU / 进程数：2、4、6。

Number of GPUs/processes 2, 4, or 6. 

资源要求：最多使用 6 张 GPU。每次 benchmark 运行都应控制在 5 分钟以内。

Resource requirements: Up to 6 GPUs. Each benchmarking run should take less than 5 minutes. 

提交内容：对不同设置做比较的图表与/或表格，并附 2 至 3 句话说明你的观察以及这些因素之间的相互作用。

Deliverable: Plot(s) and/or table(s) comparing the various settings, with 2-3 sentences of commentary about your results and thoughts about how the various factors interact. 

# 5.2 A Naïve Implementation of Distributed Data Parallel Training

## 5.2 朴素版分布式数据并行训练

Now that we’ve seen the basics of writing distributed applications in PyTorch, let’s build a minimal implementation of distributed data parallel (DDP) training. 

既然我们已经看过了在 PyTorch 中编写分布式应用的基础，现在来搭建一个最小化的数据并行训练（DDP）实现。

Data parallelism splits batches across multiple devices (e.g., GPUs), enabling training on large batch sizes that do not fit on a single device. For example, given four devices that can each handle a maximum batch size of 32, data parallel training would enable an effective batch size of 128. 

数据并行会把 batch 划分到多个设备（例如 GPU）上，从而支持单设备无法容纳的大 batch。举例来说，如果有四个设备，每个设备最多只能处理 batch size 32，那么做数据并行训练后，有效 batch size 就可以达到 128。

Here are the steps for naïvely doing distributed data parallel training. Initially, each device constructs a (randomly-initialized) model. We use the broadcast collective communication operation to send the model parameters from rank 0 to all other ranks. At the start of training, each device holds an identical copy of the model parameters and optimizer states (e.g. the accumulated gradient statistics in Adam). 

朴素分布式数据并行训练的步骤如下。起初，每个设备都构造一个随机初始化的模型。我们使用 `broadcast` 集体通信操作，把 rank 0 上的模型参数发送给所有其他 rank。这样在训练开始时，每个设备都持有一份相同的模型参数和优化器状态（例如 Adam 中累积的梯度统计量）。

1. Given a batch with $n$ examples, the batch is sharded and each device receives $n / d$ disjoint examples (where $d$ is the number of devices used for data parallel training). $d$ should divide $n$, otherwise some ranks would do more work than others, and the step is bottlenecked by the slowest. 

1. 给定一个包含 `B` 个样本的 batch，把它切分到多个设备上，每个设备接收 `B / N_DP` 个互不重叠的样本，其中 `N_DP` 是数据并行使用的设备数。`N_DP` 应整除 `B`，否则不同 rank 的工作量会不一致，而整个 step 会被最慢的 rank 拖住。

2. Each device uses its local copy of the model parameters to run a forward pass on its $n / d$ examples and a backward pass to calculate the gradients. Note that at this point, each device holds the gradients computed from the $n / d$ examples it received. 

2. 每个设备用自己的模型参数副本，对本地的 `B / N_DP` 个样本执行 forward 和 backward，得到本地梯度。此时每个设备只持有由自己那部分样本计算出的梯度。

3. We then use the all-reduce collective communication operation to average the gradients across the different devices, so each device holds the gradients averaged across all $n$ examples. 

3. 接着，我们使用 `all-reduce` 把不同设备上的梯度做平均，这样每个设备都会持有基于整个 batch 的平均梯度。

4. Next, each device runs an optimizer step to update its copy of the parameters—from the optimizer’s perspective, it is simply optimizing a local model. The parameters and optimizer states will stay in sync on all of the different devices since they all start from the same initial model and optimizer state and use the same averaged gradients for each iteration. At this point, we’ve completed a single training iteration and can repeat the process. 

4. 最后，每个设备各自执行一次优化器更新。从优化器视角看，它只是优化了一个本地模型。由于所有设备最初参数和优化器状态一致，并且每一步都使用同样的平均梯度，因此更新后参数与优化器状态仍会保持同步。至此，我们完成了一个训练迭代，然后重复这一过程。

# Problem (naive_ddp):  Naïve DDP (5 points)

### 题目（naive_ddp）：朴素 DDP（5 分）

Deliverable: Implement a naïve form of distributed data parallel training that all-reduces individual parameter gradients after the backward pass. To test your implementation, implement [adapters.get_ddp] and (optionally) [adapters.ddp_on_after_backward] , then run uv run pytest tests/test_ddp.py. 

提交内容：实现一种朴素的数据并行训练方式，在 backward 之后对每个参数张量的梯度分别执行 all-reduce。要测试实现，请完成 `adapters.get_ddp`，以及可选的 `adapters.ddp_on_after_backward`，然后运行：

# Problem (naive_ddp_benchmarking):  Naïve DDP Benchmarking (3 points)

### 题目（naive_ddp_benchmarking）：朴素 DDP 基准测试（3 分）

In this naïve DDP implementation, parameter gradients are individually all-reduced across ranks after each backward pass. To better understand the overhead of data parallel training, create a script to benchmark your previously-implemented language model when trained with this naïve implementation of DDP. Measure the total time per training step and the proportion of time spent on communicating gradients. Collect measurements in the single-node setting (1 node x 2 GPUs) for the xl model size described in Section 2.1.2. 

在这种朴素 DDP 实现中，每次 backward 之后都会对每个参数梯度单独做 all-reduce。为了更好地理解数据并行训练的通信开销，请编写一个脚本，benchmark 你先前实现的语言模型在使用这种朴素 DDP 训练时的表现。测量每个训练 step 的总时间，以及其中用于梯度通信的时间占比。在单机 2 GPU（1 node × 2 GPUs）设置下，对第 2.1.2 节中 `xl` 模型做测量。

Deliverable: A description of your benchmarking setup, along with the measured time per training iteration and time spent communicating gradients for each setting. 

提交内容：描述你的 benchmark 设置，并给出每种设置下每次训练迭代的总时间，以及用于梯度通信的时间。

# 5.3 Improving Upon the Minimal DDP Implementation

## 5.3 改进最小 DDP 实现

The minimal DDP implementation that we saw in Section 5.2 has a couple of key limitations: 

第 5.2 节中的最小 DDP 实现有两个关键局限：

1. It conducts a separate all-reduce operation for every parameter tensor. Each communication call incurs overhead, so it may be advantageous to batch communication calls to minimize this overhead. 

1. 它对每个参数张量都单独发起一次 all-reduce。每次通信调用都有固定开销，因此把这些调用打包起来可能更高效。

2. It waits for the backward pass to finish before communicating gradients. However, the backward pass is incrementally computed. Thus, when a parameter gradient is ready, it can immediately be communicated without waiting for the gradients of the other parameters. This allows us to overlap communication of gradients with computation of the backward pass, reducing the overhead of distributed data parallel training. 

2. 它会等到整个 backward 完成后才开始通信梯度。然而 backward 是逐步产生梯度的，因此一旦某个参数梯度准备好，就可以立即开始通信，而不必等待其他参数。这样就能把梯度通信与 backward 计算重叠起来，从而降低数据并行训练的额外开销。

In this part of the assignment, we’ll address each of these limitations in turn and measure the impact on training speed. 

本部分中，我们将依次解决这两个问题，并测量它们对训练速度的影响。

# 5.3.1 Reducing the Number of Communication Calls

## 5.3.1 减少通信调用次数

Rather than issuing a communication call for each parameter tensor, let’s see if we can improve performance by batching the all-reduce. Concretely, we’ll take the gradients that we want to all-reduce, concatenate them into a single tensor, and then all-reduce the combined gradients across all ranks. It might be helpful to use torch._utils._flatten_dense_tensors and torch._utils._unflatten_dense_tensors. 

与其对每个参数张量分别发起一次通信，不如尝试把所有梯度拼接成一个大张量，对它做一次 all-reduce，然后再拆回去。你可能会发现 `torch._utils._flatten_dense_tensors` 和 `torch._utils._unflatten_dense_tensors` 很有帮助。

# Problem (minimal_ddp_flat_benchmarking):  Minimal DDP with Flat Gradients Benchmarking (2 points)

### 题目（minimal_ddp_flat_benchmarking）：扁平梯度的最小 DDP 基准测试（2 分）

Modify your minimal DDP implementation to communicate a tensor with flattened gradients from all parameters. Compare its performance with the minimal DDP implementation that issues an all-reduce for each parameter tensor under the previously-used conditions (1 node x 2 GPUs, xl model size as described in Section 2.1.2). 

修改你的最小 DDP 实现，使其通信的是“由所有参数梯度展平拼接而成的一个张量”。在与前面相同的设置下（1 node × 2 GPUs，`xl` 模型），把它和“逐参数 all-reduce”的最小 DDP 实现进行比较。

Deliverable: The measured time per training iteration and time spent communicating gradients under distributed data parallel training with a single batched all-reduce call. 1-2 sentences comparing the results when batching vs. individually communicating gradients. 

提交内容：在这种分布式训练设置下，给出每次训练迭代的总时间，以及使用“单次 batched all-reduce”时用于梯度通信的时间。再用 1 至 2 句话比较“批量通信梯度”与“逐个参数通信梯度”的差异。

# 5.3.2 Overlapping Computation with Communication of Individual Parameter Gradients

## 5.3.2 将逐参数梯度通信与计算重叠

While batching the communication calls might help lower the overhead associated with issuing a large number of small all-reduce operations, all of the communication time still directly contributes to the overhead. To resolve this, we can take advantage of the observation that the backward pass incrementally computes gradients for each layer (starting from the loss and moving toward the input)—thus, we can allreduce parameter gradients as soon as they’re ready, reducing the overhead of data parallel training by overlapping computation of the backward pass with communication of gradients. 

把通信调用打包固然能减少大量小 all-reduce 带来的固定开销，但所有通信时间仍然会直接叠加到总耗时上。要进一步解决这个问题，我们可以利用 backward 是逐层产生梯度这一事实：只要某一层的参数梯度准备好了，就立即对它做 all-reduce，这样就可以把梯度通信与 backward 计算重叠起来，从而降低训练额外开销。

We’ll start by implementing and benchmarking a distributed data parallel wrapper that asynchronously all-reduces individual parameter tensors as they become ready during the backward pass. The following pointers may be useful: 

我们先实现并 benchmark 一个 DDP wrapper：当单个参数梯度在 backward 中准备就绪时，它会异步地对该参数张量执行 all-reduce。下面这些提示会有帮助：

Backward hooks To automatically call a function on a parameter after its gradient has been accumulated in the backward pass, you can use the register_post_accumulate_grad_hook function.4 

`register_post_accumulate_grad_hook` 可以让你在某个参数的梯度在 backward 中累积完毕后，自动触发一个函数。  
PyTorch 中所有集体通信操作都支持同步（`async_op=False`）和异步（`async_op=True`）两种执行方式。同步调用会阻塞，直到集体通信已经被排入 GPU 队列；但这并不表示通信已经在 GPU 上真正完成，因为 CUDA 操作本来就是异步的。尽管如此，后续依赖输出的函数调用仍会按预期工作。相对地，异步调用会返回一个分布式请求句柄；因此当函数返回时，通信未必已经排入 GPU，更不用说已经完成。若要等待它至少被正确排队，以便后续操作安全使用输出，可以对返回的句柄调用 `handle.wait()`。

Asynchronous communication All PyTorch collective communication operations support synchronous (async_op=False) and asynchronous execution (async_op=True). Synchronous calls will block until the collective operation is queued on the GPU. This does not mean that the CUDA operation is completed since CUDA operations are asynchronous. That being said, later function calls using the output will behave as expected.5 In contrast, asynchronous calls will return a distributed request handle— as a result, when the function returns, the collective communication operation is not guaranteed to have been queued on the GPU, let alone completed. To wait for the operation to be queued on the GPU (and therefore for the output to be usable in later operations), you can call handle.wait() on the returned communication handle. 

例如，下面两段代码分别用同步和异步方式，对一个张量列表中的每个张量执行 all-reduce。

For example, the following two examples all-reduce each tensor in a list of tensors with either a synchronous or an asynchronous call: 

```python
tensors = [torch.rand(5) for _ in range(10)]

# Synchronous, block until operation is queued on the GPU.

for tensor in tensors:
    dist.all_reduce(tensor, async_op=False)

# Asynchronous, return immediately after each call and

# wait on results at the end.

handles = []
for tensor in tensors:
    handle = dist.all_reduce(tensor, async_op=True)
    handles.append(handle)

# ...

# Possibly execute other commands that don't rely on the all_reduce results

# ...

# Ensure that all-reduce calls were queued and

# therefore other operations depending on the

# all-reduce output can be queued.

for handle in handles:
    handle.wait()
handles.clear() 

```

# Problem (ddp_overlap_individual_parameters):  DDP with Overlapping Individual Parameters (5 points)

### 题目（ddp_overlap_individual_parameters）：逐参数通信重叠的 DDP（5 分）

Implement a Python class to handle distributed data parallel training. The class should wrap an arbitrary PyTorch nn.Module and take care of broadcasting the weights before training (so all ranks have the same initial parameters) and issuing communication calls for gradient averaging. We recommend the following public interface: 

实现一个 Python 类，用来处理分布式数据并行训练。该类应包装任意的 PyTorch `nn.Module`，并负责在训练开始前广播权重（确保所有 rank 有相同初始参数），以及负责梯度平均的通信操作。推荐的公开接口如下：

def __init__(self, module: torch.nn.Module): Given an instantiated PyTorch nn.Module to be parallelized, construct a DDP container that will handle gradient synchronization across ranks. 

`__init__(self, module: torch.nn.Module)`：给定一个已经实例化好的 PyTorch `nn.Module`，构造一个 DDP 容器，负责跨 rank 的梯度同步。  
`forward(self, *inputs, **kwargs)`：调用被包装模块的 `forward()`，并传入相应位置参数和关键字参数。  
`finish_gradient_synchronization(self)`：当被调用时，等待异步通信调用在 GPU 上完成。

def forward(self, *inputs, **kwargs): Calls the wrapped module’s forward() method with the provided positional and keyword arguments. 

使用这个类进行分布式训练时，我们会先把模型包起来，并在 `optimizer.step()` 之前调用 `finish_gradient_synchronization()`，确保优化器更新这个依赖梯度的操作可以安全排队：

def finish_gradient_synchronization(self): When called, wait for asynchronous communication calls to finish on the GPU. 

提交内容：实现这个用于分布式数据并行训练的容器类。该类应能够把梯度通信与 backward 计算重叠。为测试你的 DDP 类，请先实现 `adapters.get_ddp` 以及（可选的）`adapters.ddp_on_after_backward`。然后运行：

To use this class to perform distributed training, we’ll pass it a module to wrap, and then add a call to finish_gradient_synchronization() before we run optimizer.step() to ensure that the optimizer step, an operation that depends on the gradients, can be safely queued: 

建议多运行几次（例如 5 次），确保它稳定通过，避免潜在竞态条件。

```python
model = ToyModel().to(device)
ddp_model = DDP(model)

for _ in range(train_steps):
    x, y = get_batch()
    logits = ddp_model(x)
    loss = loss_fn(logits, y)
    loss.backward()
    ddp_model.finish_gradient_synchronization()
    optimizer.step() 
```

Deliverable: Implement a container class to handle distributed data parallel training. This class should overlap gradient communication and the computation of the backward pass. To test your DDP class, first implement the adapters [adapters.get_ddp] and 

[adapters.ddp_on_after_backward] (the latter is optional, depending on your implementation you may not need it). 

Then, to execute the tests, run uv run pytest tests/test_ddp.py. We recommend running the tests multiple times (e.g., 5) to ensure that it passes reliably. 

提交内容：实现这个用于分布式数据并行训练的容器类。该类应能够把梯度通信与 backward 计算重叠。为测试你的 DDP 类，请先实现 `adapters.get_ddp` 以及 `adapters.ddp_on_after_backward`（后者可选，取决于你的实现）。然后运行 `uv run pytest tests/test_ddp.py`。我们建议多运行几次（例如 5 次），确保它能稳定通过。

# Problem (ddp_overlap_individual_parameters_benchmarking):  DDP Overlapping Individual Parameters Benchmarking (1 point)

### 题目（ddp_overlap_individual_parameters_benchmarking）：逐参数通信重叠 DDP 的基准测试（1 分）

(a) Benchmark the performance of your DDP implementation when overlapping backward pass computation with communication of individual parameter gradients. Compare its performance with our previously-studied settings (the minimal DDP implementation that either issues an all-reduce for each parameter tensor, or a single all-reduce on the concatenation of all parameter tensors) with the same setup: 1 node, 2 GPUs, and the xl model size described in Section 2.1.2. 

(a) benchmark 你的 DDP 实现，把 backward 计算与逐参数梯度通信重叠起来。在相同设置下（1 node、2 GPUs、`xl` 模型），把它和我们之前研究过的几种设置进行比较：一种是最初的逐参数 all-reduce 版 minimal DDP，另一种是对所有参数拼接后做单次 all-reduce 的版本。

Deliverable: The measured time per training iteration when overlapping the backward pass with communication of individual parameter gradients, with 1-2 sentences comparing the results. 

提交内容：给出这种“重叠 backward 与逐参数梯度通信”的 DDP 在每次训练迭代上的耗时，并用 1 至 2 句话比较结果。

(b) Instrument your benchmarking code (using the 1 node, 2 GPUs, xl model size setup) with the Nsight profiler, comparing the initial DDP implementation with this overlapped implementation. Visually compare the two traces, and provide a profiler screenshot demonstrating that one implementation overlaps compute with communication while the other doesn’t. 

(b) 在同样的设置（1 node、2 GPUs、`xl` 模型）下，用 Nsight profiler 给你的 benchmark 加上分析标注，比较最初 DDP 实现与这个重叠版实现。请从可视化 trace 上对比二者，并给出截图，证明一个实现确实做到了计算与通信重叠，而另一个没有。

Deliverable: 2 screenshots (one from the initial DDP implementation, and another from this DDP implementation that overlaps compute with communication) that visually show that communication is or isn’t overlapped with the backward pass. 

提交内容：2 张截图（一个对应原始 DDP 实现，一个对应这个重叠版实现），并能直观显示通信是否与 backward 计算发生了重叠。

# 6 Optimizer State Sharding

## 6 优化器状态分片

Distributed data parallel training is conceptually simple and often very effective, but requires each rank to hold a distinct copy of the model parameters and optimizer state. This redundancy can come with significant memory costs. For example, the AdamW optimizer maintains two floats per parameter, meaning that it consumes twice as much memory as the model weights. S. Rajbhandari et al. [5] describe several methods for reducing this redundancy in data-parallel training by partitioning the (1) optimizer state, (2) gradients, and (3) parameters across ranks, communicating them between workers as necessary. 

分布式数据并行训练在概念上很简单，通常也非常有效，但它要求每个 rank 都维护一整份模型参数和优化器状态。这种冗余会带来显著显存成本。例如，AdamW 为每个参数维护两个浮点状态，因此优化器状态本身就要占用约等于模型权重两倍的显存。S. Rajbhandari 等人 [5] 提出了多种方法，通过在不同 rank 之间划分：(1) 优化器状态、(2) 梯度、(3) 参数，来降低数据并行训练中的这类冗余，并在必要时通过通信重新组合这些部分。

In this part of the assignment, we’ll reduce per-rank memory consumption by implementing a simplified version of optimizer state sharding. Rather than keeping the optimizer states for all parameters, each rank’s optimizer instance will only handle a subset of the parameters (approximately 1 / world_size). When each rank’s optimizer takes an optimizer step, it’ll only update the subset of model parameters in its shard. Then, each rank will broadcast its updated parameters to the other ranks to ensure that the model parameters remain synchronized after each optimizer step. 

本部分中，我们会实现一个“简化版的优化器状态分片”，以降低单个 rank 的显存占用。与其让每个 rank 的优化器都保存所有参数的状态，不如让每个 rank 的优化器实例只处理其中一部分参数（大约 `1 / world_size`）。每个 rank 的优化器只会对自己这一份参数执行 step；完成后，再把更新过的参数广播给其他 rank，从而保持模型参数同步。

# Problem (optimizer_state_sharding):  Optimizer State Sharding (15 points)

### 题目（optimizer_state_sharding）：优化器状态分片（15 分）

Implement a Python class to handle optimizer state sharding. The class should wrap an arbitrary input PyTorch optim.Optimizer and take care of synchronizing updated parameters after each optimizer step. We recommend the following public interface: 

实现一个 Python 类，用于处理优化器状态分片。该类应包装任意一个输入的 PyTorch `optim.Optimizer`，并负责在每次优化器更新后同步已更新的参数。推荐的公开接口如下：

def __init__(self, params, optimizer_cls: Type[Optimizer], **kwargs: Any): Initializes the sharded state optimizer. params is a collection of parameters to be optimized (or parameter groups, in case the user wants to use different hyperparameters, such as learning rates, for different parts of the model); these parameters will be sharded across all the ranks. The optimizer_cls parameter specifies the type of optimizer to be wrapped (e.g., optim.AdamW). Finally, any remaining keyword arguments are forwarded to the constructor of the optimizer_cls. Make sure to call the torch.optim.Optimizer super-class constructor in this method. 

def step(self, closure, **kwargs): Calls the wrapped optimizer’s step() method with the provided closure and keyword arguments. After updating the parameters, synchronize with the other ranks. 

def add_param_group(self, param_group: dict[str, Any]): This method should add a parameter group to the sharded optimizer. This is called during construction of the sharded optimizer by the super-class constructor and may also be called during training (e.g., for gradually unfreezing layers in a model). As a result, this method should handle assigning the model’s parameters among the ranks. 

Deliverable: Implement a container class to handle optimizer state sharding. To test your sharded optimizer, first implement the adapter [adapters.get_sharded_optimizer] . Then, to execute the tests, run uv run pytest tests/test_sharded_optimizer.py. We recommend running the tests multiple times (e.g., 5) to ensure that they pass reliably. 

Now that we’ve implemented optimizer state sharding, let’s analyze its effect on the peak memory usage during training and its runtime overhead. 

提交内容：实现这个用于处理优化器状态分片的容器类。为测试你的 sharded optimizer，请先实现 `adapters.get_sharded_optimizer`，然后运行 `uv run pytest tests/test_sharded_optimizer.py`。我们建议多运行几次（例如 5 次），确保测试稳定通过。

现在我们已经实现了优化器状态分片，接下来分析它对训练峰值显存与运行时开销的影响。

# Problem (optimizer_state_sharding_accounting):  Optimizer State Sharding Accounting (5 points)

### 题目（optimizer_state_sharding_accounting）：优化器状态分片的开销分析（5 分）

(a) Create a script to profile the peak memory usage when training language models with and without optimizer state sharding. Using the standard configuration (1 node, 2 GPUs, xl model size), report the peak memory usage after model initialization, directly before the optimizer step, and directly after the optimizer step. Do the results align with your expectations? Break down the memory usage in each setting (e.g., how much memory for parameters, how much for optimizer states, etc.). 

(a) 编写一个脚本，在训练语言模型时分别测量“使用”和“不使用”优化器状态分片时的峰值显存。使用标准配置（1 node、2 GPUs、`xl` 模型），汇报模型初始化后、优化器步进前、以及优化器步进后的峰值显存。这些结果是否符合你的预期？请进一步拆解各设置下的显存构成（例如参数占多少、优化器状态占多少等）。

Deliverable: 2-3 sentence response with peak memory usage results and a breakdown of how the memory is divided between different model and optimizer components. 

提交内容：2 至 3 句话，给出峰值显存结果，并说明显存如何分配到不同模型 / 优化器组成部分上。

(b) How does our implementation of optimizer state sharding affect training speed? Measure the time taken per iteration with and without optimizer state sharding for the standard configuration (1 node, 2 GPUs, xl model size). 

(b) 我们实现的优化器状态分片会如何影响训练速度？请在标准配置（1 node、2 GPUs、`xl` 模型）下，对“使用”和“不使用”优化器状态分片的每次迭代耗时做测量。

Deliverable: 2-3 sentence response with your timings. 

提交内容：2 至 3 句话，并附带你的计时结果。

(c) How does our approach to optimizer state sharding differ from ZeRO stage 1 (described as ZeRO-DP `P_os` in S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He [5])? 

(c) 我们这里的优化器状态分片方案，与 ZeRO stage 1（在 Rajbhandari 等人 [5] 中称为 ZeRO-DP）有何不同？

Deliverable: 2-3 sentence summary of any differences, especially those related to memory and communication volume. 

提交内容：2 至 3 句话，总结两者的差异，尤其是和显存及通信量相关的差异。

# 7 Fully-Sharded Data Parallel

## 7 全分片数据并行（FSDP）

With optimizer state sharding and data parallel, we’re able to split the optimizer state and activations across our data-parallel axis. However, our model weights remain duplicated — we’re storing a full copy of them all on each GPU. 

借助优化器状态分片和数据并行，我们已经能够沿数据并行轴划分优化器状态和激活。但模型权重本身仍然是被完整复制的，也就是说，每张 GPU 仍然保留着一整份权重。

We can solve this by turning our data parallel (DP) axis into a fully-sharded data parallel axis (FSDP). With FSDP, each GPU stores only its own slice of every weight tensor, but has to pull slices from other GPUs to form the full weight tensor using an all-gather to prepare for a forward or backward pass. 

我们可以通过把原先的数据并行轴（DP）升级为全分片数据并行轴（FSDP）来解决这个问题。在 FSDP 中，每张 GPU 只保存每个权重张量属于自己的切片；在 forward 或 backward 真正需要某层权重前，再通过 all-gather 从其他 GPU 拉取对应切片，拼回完整权重。

To avoid keeping GPU compute waiting around for communication to finish, most FSDP implementations schedule the layer’s all-gather in advance of the operation, meaning the relevant weights are ready before they are needed, preventing communication from blocking computation. This keeps weight sharding communication off the critical path, meaning it has no cost as long as communication can keep up with compute and is scheduled well. 

为了避免 GPU 计算空等通信完成，大多数 FSDP 实现都会在真正使用某层之前就提前安排该层的 all-gather。这样，当这层要计算时，相关权重已经就绪，通信就不会阻塞计算。只要通信速度跟得上计算，并且调度得足够早，权重分片带来的通信就可以不落在关键路径上。

Some layers are small enough in memory and compute that the latency overhead of a transfer is not worth it. You should mark these layers not to be sharded by FSDP. In our architecture, this will mostly be the case for norms. This leaves us with the embedding layer and every linear layer. 

有些层在显存和计算上都太小，通信它们的延迟成本反而不值得。因此这些层不应该被 FSDP 分片。在我们的架构里，这类层主要是 norm。也就是说，真正需要分片的基本上是 embedding 层和所有 linear 层。

While it is necessary to store master weights in FP32 (any values that are repeatedly accumulated into are sensitive to precision), the weights do not need to be used in FP32. In mixed precision, we always convert to the low-precision compute datatype before use, so we may as well convert even before the weight is communicated to save on bandwidth. 

虽然必须用 FP32 保存 master weights（凡是会被反复累加更新的值，对精度都敏感），但权重在参与计算时不必一定是 FP32。在 mixed precision 下，我们本来就会在使用前把它们转换成低精度计算 dtype；既然如此，为了节省带宽，也可以在通信前就先把它们 cast 成低精度。

# Problem (fsdp):  Fully-Sharded Data Parallel (15 points)

### 题目（fsdp）：全分片数据并行（15 分）

Implement a Python class for fully-sharded data parallel training. The class should wrap an arbitrary PyTorch nn.Module (your full model) and hook into or wrap any Linear or Embedding layer within it. We recommend the following public interface: 

实现一个 Python 类，用于全分片数据并行训练。该类应包装任意 PyTorch `nn.Module`（也就是你的完整模型），并在其中所有 `Linear` 或 `Embedding` 层上加 hook 或包装器。推荐的公开接口如下：

def _init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None): Given an instantiated PyTorch nn.Module to be parallelized, construct an FSDP module that will handle weight all-gathers and gradient reduce-scatters. Make sure that your hooks or your module wrappers all-gather the weights in time for the forward pass. To limit memory use, only start gathering after the layer two before the current one has completed its forward pass. In the backward pass, your hooks or module wrappers should all-gather to have the weights available for the computation. When the gradients are available, they should be reducescattered to the appropriate ranks. Make sure to free the gathered weights after use. When compute_dtype is provided, cast the weights to that dtype before communicating or using them for compute, while keeping master weights and optimizer updates in FP32. 

`__init__(self, module: torch.nn.Module, compute_dtype: torch.dtype | None = None)`：给定一个已经实例化好的 PyTorch 模块，构造一个 FSDP 模块，负责权重的 all-gather 以及梯度的 reduce-scatter。请确保你的 hook 或包装器能够足够早地 all-gather 权重，使其在 forward 时可用。为控制显存使用，只有在“当前层之前的前两层”已经完成前向后，才开始为当前层 gather 权重。在 backward 中，你的 hook 或包装器也应当在计算需要前 gather 权重；当梯度可用后，再将其 reduce-scatter 到正确 rank。用完后记得释放 gather 得到的完整权重。当提供 `compute_dtype` 时，请在通信和计算前把权重 cast 到该 dtype，同时仍在 FP32 中保留 master weights 并用它们进行优化器更新。  
`forward(self, *inputs, **kwargs)`：调用被包装模块的 `forward()`。  
`finish_gradient_synchronization(self)`：等待所有异步通信在 GPU 上完成。

def forward(self, *inputs, **kwargs): Calls the wrapped module’s forward() method with the provided positional and keyword arguments. 

提交内容：实现这个处理全分片数据并行训练的容器类。容器内的每个 shard 都应当能与作业 1 中标准的 AdamW 实现兼容。为测试你的 FSDP 实现，请实现 `adapters.get_fsdp`，并运行：

def finish_gradient_synchronization(self): When called, wait for asynchronous communication calls to finish on the GPU. 

建议也多运行几次（例如 5 次），以尽量捕捉竞态条件。

Deliverable: Implement a container class to handle fully sharded data parallel training. Each shard of this container should be compatible with the standard AdamW implementation from assignment 1. To test your FSDP implementation, implement the adapter [adapters.get_fsdp] Run the tests with uv run pytest tests/test_fsdp.py. We recommend running the tests multiple times (e.g., 5) to catch any race conditions. 

# Problem (fsdp_accounting):  FSDP Accounting (5 points)

### 题目（fsdp_accounting）：FSDP 开销分析（5 分）

(a) Given your analysis in Section 6, how much memory do you expect to save from the peak by implementing FSDP? You can ignore the size of the preallocated buffers needed to all-gather weights to each GPU in your calculation. 

(a) 结合你在第 6 节中的分析，实施 FSDP 后，你预计峰值显存能减少多少？在估算中，你可以忽略为 all-gather 到每张 GPU 所预先分配的 buffer 大小。

Deliverable: 2-3 sentence response with your findings. 

提交内容：2 至 3 句话说明你的结论。

(b) Profile the xl model on two GPUs and pay attention to the all-gather of weights. Does the communication finish in time for the forward pass? 

(b) 在两张 GPU 上 profile `xl` 模型，重点观察权重的 all-gather。通信是否能及时完成，从而不阻塞 forward？

Deliverable: 2-3 sentence response with your timings. Include screenshots of Nsight to back up your claims. 

提交内容：2 至 3 句话，并给出你的计时结果。请附上 Nsight 截图支持你的结论。

# 8 Analyzing Parallelism Strategies

## 8 并行策略分析

There are more axes along which we can parallelize our training process. Some common strategies include: 

训练过程还有更多维度可以并行化。一些常见策略包括：

• Data parallelism (DP) — Batches of data are split across multiple devices, and each device computes gradients for its own batch. Gradients are then averaged across devices. 

• 数据并行（DP）：把 batch 划分到多个设备，每个设备各自对自己的样本计算梯度，再跨设备平均梯度。

• Fully-Sharded Data Parallelism (FSDP) — On top of data parallelism, we also split optimizer states, gradients, and weights across devices to reduce memory usage. Devices then need to gather weight shards from other devices during the forward and backward passes. 

• 全分片数据并行（FSDP）：在数据并行基础上，进一步把优化器状态、梯度和权重都分片到不同设备，以节省显存；设备会在 forward 和 backward 过程中按需聚合权重切片。

• Tensor Parallelism (TP) — Weight matrices are sharded across the input or output dimension. Devices compute the activations corresponding to their shard, and activations are then reduced or gathered across devices. 

• 张量并行（TP）：沿某个权重矩阵的输入维或输出维对其做切分；各设备计算自己那一份 shard 对应的激活，然后再对激活做归约或 gather。

• Pipeline Parallelism (PP) — The model is split layerwise into multiple stages, where each stage is run on a different device. 

• 流水线并行（PP）：按层把模型拆成多个 stage，每个 stage 放到不同设备执行。

• Expert Parallelism (EP) — The experts in a Mixture-of-Experts model are split onto different devices, and each device computes the output for its own expert. 

• 专家并行（EP）：在 Mixture-of-Experts 模型中，把不同专家分配到不同设备，每个设备只负责自己那部分专家的输出。

In this section, we’ll do some basic math in a simplified setting to choose between parallelism strategies and decide how to combine them. To start, we’ll focus on DP, FSDP, TP, and their combinations. Our approach will be to calculate the communication cost of each strategy and compare it to the computational cost, which tells us how many devices we can scale to before communication cost becomes the bottleneck. 

在这一节，我们将在一个简化设定下做一些基础推导，以帮助在不同并行策略之间做取舍，并决定如何组合它们。我们将重点关注 DP、FSDP、TP 以及它们的组合。思路是：计算每种策略的通信成本，并把它与计算成本比较，由此判断在通信成为瓶颈之前，最多还能扩展到多少设备。

For a more detailed treatment of TPU/GPU topologies and parallelism strategies, the TPU Scaling Book (J. Austin et al. [6]) is an excellent resource. And for a more detailed pipeline parallel discussion, see The Ultra-Scale Playbook Appendix (H. Z. P. N. M. M. L. W. T. W. Nouamane Tazi Ferdinand Mom [7]). The rest of these books also have a lot of other information you might find useful. 

如果你想看更系统的 TPU / GPU 拓扑与并行策略讨论，推荐阅读 TPU Scaling Book（J. Austin 等 [6]）。如果你想看更详细的流水线并行讨论，可参考 The Ultra-Scale Playbook Appendix（Nouamane Tazi Ferdinand Mom 等 [7]）。

# 8.1 Communication Primitives

## 8.1 通信原语

Our first step will be to understand the communication primitives. In our simplified setting, suppose we have $N$ devices numbered $0, \ldots, N - 1$, and each pair of devices is connected by a link. We’ll also assume each device has $W$ egress (i.e. outgoing) bandwidth; in other words, each device can send data to another device at a rate of $W$ bytes per second. How might we implement gather and reduce? 

我们的第一步，是理解通信原语。在一个简化设定下，假设我们有 `N` 个设备，编号为 `0, ..., N-1`，并且每一对设备之间都通过链路相连。再假设每个设备都有 `W` 的 egress（即出站）带宽，也就是说，每个设备都能以 `W` 字节每秒的速率把数据发给另一个设备。那我们该如何实现 gather 和 reduce 呢？

One common way to implement the all-gather operation is the ring all-gather. Recall that in an all-gather, each device $i$ starts with a chunk $x _ { i }$ of size $\frac { S } { N }$, and ends up with the entire $\boldsymbol { x } = [ x _ { 0 } , \ldots , x _ { N - 1 } ]$ of size $S$ (in bytes). In a ring all-gather, we arrange the devices in a circle. In each step, each device sends its current chunk to the next device to its right, and stores the chunk it received from the device to its left. This process repeats, where each device passes the chunk it just received to the right, and receives a new chunk from the left. After $N - 1$ steps, each device has the entire tensor. 

实现 all-gather 的一种常见办法是 ring all-gather。回忆一下，在 all-gather 中，每个设备 `i` 起初持有一个大小为 `S / N` 的切片 `x_i`，最终则要获得整个张量 `x = [x_0, ..., x_{N-1}]`，其总大小为 `S` 字节。在 ring all-gather 中，我们把设备排成一个环。在每一步中，每个设备都把当前持有的那个切片发送给右边的邻居，同时从左边的邻居接收一个切片并存储下来。随后，这个过程重复：每个设备把刚刚收到的切片继续向右传，并从左边继续接收一个新的切片。经过 `N - 1` 步后，每个设备就都拥有了完整张量。

In our idealized setting, each device simultaneously transmits a chunk of size $\frac { S } { N }$ in each step, with egress bandwidth $W$, and there are $N - 1$ steps, so the ring all-gather takes $\frac { N - 1 } { N } \frac { S } { W }$ seconds. 

在我们这个理想化设定中，每一步中每个设备都同时发送一个大小为 `S / N` 的切片，出站带宽为 `W`，而一共需要 `N - 1` 步，因此 ring all-gather 的时间为：

Next, let’s analyze the ring reduce-scatter. In a reduce-scatter, each device $i$ starts with a full tensor $\boldsymbol { x } ^ { ( i ) }$ of size $S$. We then want to compute the reduction $y = \sum _ { i = 0 } ^ { N - 1 } x ^ { ( i ) }$, but where each device $i$ ends up with just a chunk $y _ { i }$ of size $\frac { S } { N }$. Like the ring all-gather, we’ll start by arranging the devices in a circle. Each device will first divide its tensor $\boldsymbol { x } ^ { ( i ) }$ into $N$ chunks $\left[ x _ { 0 } ^ { ( i ) } , \ldots , x _ { N - 1 } ^ { ( i ) } \right]$, each of size $\frac { S } { N }$. We’ll then pass chunks around just like the ring all-gather, except before passing the chunk on, each device adds its contribution to the chunk (which stores a partial sum). Specifically: 

`((N - 1) / N) * (S / W)` 秒。

For step $t = 1, \ldots, N - 1$, device $i$ does the following: 

接下来分析 ring reduce-scatter。在 reduce-scatter 中，每个设备 `i` 起初持有一个完整张量 `x^(i)`，总大小为 `S`，我们要计算它们的和 `y = sum_i x^(i)`，但最终设备 `i` 只保留 `y_i` 这个大小为 `S / N` 的切片。和 ring all-gather 一样，我们先把设备排成环。每个设备先把自己的张量 `x^(i)` 划分成 `N` 个切片 `x_0^(i), ..., x_{N-1}^(i)`，每个切片大小为 `S / N`。然后像 ring all-gather 那样在环上传递切片，但不同的是：在传递之前，每个设备都会先把自己对该切片的贡献加进去，因此被传递的是“部分和”。

• If $t = 1$, initialize $y \gets x ^ { ( i ) }$, which stores the partial sum so far. 

具体而言，在第 `t = 1, ..., N - 1` 步中，设备 `i` 做以下事情：

• Send chunk $y _ { ( i - t ) \mathrm { m o d } N }$ to device $( i + 1 ) \mathrm { mod } N$. 

• 若 `t = 1`，先令 `y <- x^(i)`，它用于保存当前的部分和。

• Receive chunk $z _ { ( i - t - 1 ) \mathrm { m o d } N }$ from device $( i - 1 ) \mathrm { mod } N$. 

• 将切片 `y_{(i - t) mod N}` 发给设备 `(i + 1) mod N`。

• Update your copy of the partial sum: $y _ { ( i - t - 1 ) \mathrm { m o d } N }  y _ { ( i - t - 1 ) \mathrm { m o d } N } + z _ { ( i - t - 1 ) \mathrm { m o d } N }$ 

• 从设备 `(i - 1) mod N` 接收切片 `z_{(i - t - 1) mod N}`。

After $N - 1$ steps, device $i$ then has the full sum for chunk $y _ { i }$, so ring reduce-scatter takes $\frac { N - 1 } { N } \frac { S } { W }$ seconds, just like ring all-gather. 

• 更新自己的部分和副本：

Finsizesca , let’s implement ring all-redu and ends up with the reductionr followed by a ring all-gather duce, each de. We’ll impleml-reduce takes rts with a full tensorreduce as a ring red seconds. $\boldsymbol { x } ^ { ( i ) }$ of $S ,$ $\begin{array} { r } { y = \sum _ { i = 0 } ^ { N - 1 } x ^ { ( i ) } } \end{array}$ $2 \frac { N - 1 } { N } \frac { S } { W }$ 

经过 `N - 1` 步后，设备 `i` 就拥有了切片 `y_i` 的完整和。因此，ring reduce-scatter 与 ring all-gather 一样，也需要：

# Problem (alternate_ring_all_reduce):  Alternate ring all-reduce (1 point)

### 题目（alternate_ring_all_reduce）：另一种 ring all-reduce（1 分）

Instead of implementing all-reduce as a ring reduce-scatter followed by a ring all-gather, let’s use the following algorithm: 

如果我们不把 all-reduce 实现成“ring reduce-scatter + ring all-gather”，而是使用题目中给出的另一种算法，那么在相同设定下（每设备出站带宽为 `W`，每个 `x^(i)` 的大小为 `S`），该算法的运行时间是多少？

For step $t = 1, \ldots, N - 1$, device $i$ does the following: 

提交内容：给出一个关于 `N`、`S` 和 `W` 的表达式，并用一句话说明理由。

• If $t = 1$, initialize $y \gets x ^ { ( i ) }$, which stores the partial sum so far. 

• Send $x ^ { ( ( i - t + 1 ) \mathrm { mod } N ) }$ to device $( i + 1 ) \mathrm { mod } N$. 

• Receive $x ^ { ( ( i - t ) \mathrm { mod } N ) }$ from device $( i - 1 ) \mathrm { mod } N$. 

• Update your copy of the partial sum: $y \gets y + x ^ { ( ( i - t ) \mathrm { mod } N ) }$. 

In the same setting as above ($W$ egress bandwidth per device, each $\boldsymbol { x } ^ { ( i ) }$ is of size $S$), how long does this algorithm take? 

Deliverable: An answer in terms of $N$, $S$, and $W$, along with a one-sentence justification. 

# 8.2 Analyzing Data Parallel

## 8.2 分析数据并行

Given our communication primitives, we are ready to analyze parallelism strategies. We’ll analyze the parallelization of a single FFN layer. Recall that given input $\boldsymbol { x }$, our forward pass is given by the following: 

有了通信原语之后，我们就可以开始分析并行策略。这里我们以单个 FFN 层为对象。回忆一下，给定输入 `x`，其 forward 为：

$$
\boldsymbol {x} _ {1} = \boldsymbol {x} \boldsymbol {W} _ {1} \tag {20}
$$

$$
\boldsymbol {x} _ {2} = \boldsymbol {x} \boldsymbol {W} _ {2} \tag {21}
$$

$$
z = f \left(x _ {1}\right) * x _ {2} \tag {22}
$$

$$
\boldsymbol {y} = z \boldsymbol {W} _ {\mathbf {3}}, \tag {23}
$$

where $\boldsymbol { x }$ has shape $( B , D )$, $W _ { 1 }$ and $W _ { 2 }$ have shape $( D , D _ { \mathrm { F F } } )$, and $W _ { 3 }$ has shape $( D _ { \mathrm { F F } } , D )$. $f$ is our elementwise activation function (e.g. SiLU), and $\ast$ represents elementwise multiplication. 

其中 `x` 形状为 `(B, D)`，`W1` 和 `W2` 形状为 `(D, D_FF)`，`W3` 形状为 `(D_FF, D)`。`f` 是逐元素激活函数（例如 SiLU），`*` 表示逐元素乘法。

It will also be useful to explicitly write out the backward pass. Recall that given $\boldsymbol { d } \boldsymbol { y }$ with shape $( B , D )$, the backward pass is given by the following: 

写出 backward 也会有帮助。给定 `dy`（形状为 `(B, D)`），其 backward 为：

$$
\boldsymbol {d} \boldsymbol {z} = \boldsymbol {d} \boldsymbol {y} \boldsymbol {W} _ {\mathbf {3}} ^ {\top} \tag {24}
$$

$$
d x _ {2} = d z * f (x _ {1}) \tag {25}
$$

$$
d x _ {1} = d z * f ^ {\prime} (x _ {1}) * x _ {2} \tag {26}
$$

$$
\boldsymbol {d} \boldsymbol {x} = \boldsymbol {d} \boldsymbol {x} _ {1} \boldsymbol {W} _ {1} ^ {\top} + \boldsymbol {d} \boldsymbol {x} _ {2} \boldsymbol {W} _ {2} ^ {\top} \tag {27}
$$

$$
d W _ {3} = z ^ {\top} d y \tag {28}
$$

$$
\boldsymbol {d} \boldsymbol {W} _ {\mathbf {2}} = \boldsymbol {x} ^ {\top} \boldsymbol {d x} _ {\mathbf {2}} \tag {29}
$$

$$
\boldsymbol {d} \boldsymbol {W} _ {1} = \boldsymbol {x} ^ {\top} \boldsymbol {d} \boldsymbol {x} _ {1}, \tag {30}
$$

where ∗ represents elementwise multiplication. 

其中 `*` 仍表示逐元素乘法。

Recall that in data parallelism with $N _ { \mathrm { D P } }$ devices, we shard our input $\boldsymbol { x }$ into shards $\pmb { x } ^ { ( i ) }$ of size $\left( \frac { B } { N _ { \mathrm { D P } } } , D \right)$. The forward pass proceeds as usual without any collectives, producing activations $\boldsymbol { y } ^ { ( i ) }$ of size $\left( \frac { B } { N _ { \mathrm { D P } } } , D \right)$. In the backward pass, proceeding as usual with the batch-sharded activations, device $i$ ends up with gradients 

回忆数据并行：若使用 `N_DP` 个设备，我们把输入 `x` 划分成形状为 `(B / N_DP, D)` 的分片 `x^(i)`。前向传播在 batch 分片上照常执行，不需要任何集体通信，并得到同样按 batch 分片的输出 `y^(i)`。在 backward 中，每个设备最后得到的是它那一部分 batch 对应的局部梯度：

$$
\boldsymbol {d} \boldsymbol {W} _ {\mathbf {3}} ^ {(i)} = \boldsymbol {z} ^ {(i) ^ {\top}} \boldsymbol {d} \boldsymbol {y} ^ {(i)} \tag {31}
$$

$$
\boldsymbol {d} \boldsymbol {W} _ {\mathbf {2}} ^ {(i)} = \boldsymbol {x} ^ {(i) ^ {\top}} \boldsymbol {d x} _ {\mathbf {2}} ^ {(i)} \tag {32}
$$

$$
\boldsymbol {d} \boldsymbol {W} _ {\mathbf {1}} ^ {(\boldsymbol {i})} = \boldsymbol {x} ^ {(\boldsymbol {i}) ^ {\top}} \boldsymbol {d x} _ {\mathbf {1}} ^ {(\boldsymbol {i})}, \tag {33}
$$

where instead of summing over all $B$ outer products, we only have the partial sum over the $\frac { B } { N _ { \mathrm { D P } } }$ outer products for our shard of the input. Then, we need to do an all-reduce across devices to get the full gradients $d W _ { 3 } , d W _ { 2 }$, and $d \mathbf { W _ { 1 } }$. 

这些只是基于各自 batch 分片的部分和，而不是全 batch 的完整梯度。因此，随后我们需要跨设备对 `dW3`、`dW2` 和 `dW1` 执行 all-reduce，才能得到完整梯度。

# Problem (data_parallel_calcs):  Data parallel calculations (3 points)

### 题目（data_parallel_calcs）：数据并行计算题（3 分）

We now have everything we need to calculate when data parallelism becomes communication bottlenecked. Let $C$ (in FLOP/s) denote the device accelerator speed, and $W$ (in bytes per second) denote each device’s egress bandwidth. We can then compute the computation time and communication time. Because computation and communication can be overlapped, we are bottlenecked when communication time becomes larger than computation time. We’ll assume that all weights and activations are in FP16 (i.e. two bytes). 

现在，我们已经可以分析数据并行何时会被通信瓶颈卡住。设设备算力为 `C`（单位 FLOP/s），每设备出站带宽为 `W`（单位 bytes/s）。由于计算与通信可以重叠，因此当通信时间大于计算时间时，通信就成为瓶颈。假设所有权重与激活都使用 FP16（即每个元素 2 字节）。

(a) How many FLOPs are required to compute the backward pass, with $N _ { \mathrm { D P } }$ data parallelism? You can ignore all non-matmul operations. Recall that a matmul $( A , B ) ( B , C ) \to ( A , C )$ takes $2 A B C$ FLOPs. 

(a) 在使用 `N_DP` 个数据并行设备时，backward 需要多少 FLOPs？可以忽略所有非 matmul 操作。回忆：一个 `(A, B) (B, C) -> (A, C)` 的矩阵乘法需要 `2ABC` FLOPs。

Deliverable: An answer in terms of $B , D , D _ { \mathrm { F F } }$ , and $N _ { \mathrm { D P } }$ , along with a one-sentence justification. 

提交内容：给出一个关于 `B`、`D`、`D_FF` 和 `N_DP` 的表达式，并用一句话说明理由。

(b) How much communication time is required in the backward pass, with $N _ { \mathrm { D P } }$ data parallelism? 

(b) 在使用 `N_DP` 个数据并行设备时，backward 需要多少通信时间？

Deliverable: An answer in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $N _ { \mathrm { D P } }$, and $W$, along with a one-sentence justification. 

提交内容：给出一个表达式，并用一句话说明理由。

(c) Fixing the other parameters, how large can $N _ { \mathrm { D P } }$ become before we’re communication bottlenecked? 

(c) 固定其他参数不变，`N_DP` 最多可以增长到多大，才不会在 backward 中被通信卡住？

Deliverable: An inequality with $N _ { \mathrm { D P } }$ on one side, and an expression in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $C$, and $W$ on the other, along with a one-sentence justification. 

提交内容：给出一个关于 `N_DP` 的不等式，并附一句话说明。

# 8.3 Analyzing Fully Sharded Data Parallel

## 8.3 分析 FSDP

Next, let’s analyze FSDP. Recall that just like DP, FSDP shards the batch dimension of the inputs and activations. In addition, to save on memory, we also shard the optimizer states, gradients, and weights. We can shard the weights along either dimension, producing shards $W _ { 1 } ^ { ( i ) } , W _ { 2 } ^ { ( i ) }$, and $W _ { 3 } ^ { ( i ) }$ on device $i$, each of size $\frac { D D _ { \mathrm { F F } } } { N _ { \mathrm { F S D P } } }$. 

接下来分析 FSDP。和 DP 一样，FSDP 也会沿 batch 维切分输入与激活；除此之外，为了节省显存，它还会把权重、梯度与优化器状态做分片。

In the forward pass, we will just do the data parallel forward pass on the batch-sharded input. But to do so, recall that we need to first all-gather the weight shards across devices: 

在 forward 中，我们仍然是在 batch 分片上执行与 DP 类似的计算。但在此之前，我们要先通过 all-gather 把各设备上的权重切片拼回完整权重：

$$
\boldsymbol {W} _ {\mathbf {1}} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {\mathbf {1}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {34}
$$

$$
\boldsymbol {W} _ {\mathbf {2}} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {\mathbf {2}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {35}
$$

$$
\boldsymbol {W} _ {\mathbf {3}} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {\mathbf {3}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {36}
$$

$$
(\text { do   batch -sharded   forward   pass }) \tag {37}
$$

Note that we have some freedom in when we want to do the all-gather: we just need an all-gathered weight before it’s used, and we should then discard it to keep memory costs low. For simplicity, in this section we just list the three all-gathers together. 

当然，你可以自由决定这些 all-gather 的时机：只要保证在某个权重真正被使用前，它已经被 gather 完整即可；用完后则应尽快丢弃，以降低显存开销。这里为简化分析，我们把三次 all-gather 写在一起。

Like the forward, in the backward pass we need to first all-gather the weight shards across devices. We can then do the data parallel backward pass, except we no longer need to do an all-reduce because each device only needs the gradient for its shard. Therefore, we do a reduce-scatter instead: 

backward 同理：开始 backward 前，也要先 all-gather 权重切片。随后执行 batch-sharded backward。但与 DP 不同的是，这里不再需要对完整权重梯度做 all-reduce，因为每个设备最终只需要自己那一份权重 shard 的梯度。因此我们用 reduce-scatter 替代 all-reduce：

$$
\boldsymbol {W} _ {\mathbf {1}} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {\mathbf {1}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {38}
$$

$$
\boldsymbol {W} _ {\mathbf {2}} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {\mathbf {2}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {39}
$$

$$
\boldsymbol {W} _ {\mathbf {3}} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {\mathbf {3}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {40}
$$

$$
(\text { do   batch -sharded   backward   pass }) \tag {41}
$$

$$
\boldsymbol {d} \boldsymbol {W} _ {\mathbf {1}} ^ {(i)} = \text { reduce - scatter } \left(\left\{\boldsymbol {d} \boldsymbol {W} _ {\mathbf {1}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {42}
$$

$$
\boldsymbol {d} \boldsymbol {W} _ {\mathbf {2}} ^ {(i)} = \text { reduce - scatter } \left(\left\{\boldsymbol {d} \boldsymbol {W} _ {\mathbf {2}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {43}
$$

$$
\boldsymbol {d} \boldsymbol {W} _ {\mathbf {3}} ^ {(i)} = \text { reduce - scatter } \left(\left\{\boldsymbol {d} \boldsymbol {W} _ {\mathbf {3}} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {44}
$$

Note that the sharding notation (superscript $( i )$) is overloaded: the inputs to the reduce-scatter are partial sums on the full weights, while the outputs are full sums on the sharded weights. 

注意这里的分片记号 `(i)` 语义在输入和输出上略有重载：reduce-scatter 的输入是“完整权重上的部分和”，而输出则是“分片权重上的完整和”。

# Problem (fsdp_calcs):  Fully sharded data parallel calculations (3 points)

### 题目（fsdp_calcs）：全分片数据并行计算题（3 分）

Under the same setting as the data parallel calculations, let’s calculate when FSDP becomes communication bottlenecked. 

在和数据并行题相同的设定下，分析 FSDP 什么时候会被通信瓶颈卡住。

(a) How many FLOPs are required to compute the backward pass, with $N _ { \mathrm { F S D P } }$ FSDP? What about the forward pass? 

(a) 在使用 `N_FSDP` 个 FSDP 设备时，backward 需要多少 FLOPs？forward 又需要多少？

Deliverable: Two answers in terms of $B , D , D _ { \mathrm { F F } }$ , and $N _ { \mathrm { F S D P } } .$ along with two one-sentence justifications. 

提交内容：给出两个关于 `B`、`D`、`D_FF` 和 `N_FSDP` 的表达式，并分别用一句话说明。

(b) How much communication time is required in the backward pass, with $N _ { \mathrm { F S D P } }$ FSDP? What about the forward pass? 

(b) 在使用 `N_FSDP` 个 FSDP 设备时，backward 需要多少通信时间？forward 又需要多少？

Deliverable: Two answers in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $N _ { \mathrm { F S D P } }$, and $W$, along with two one-sentence justifications. 

提交内容：给出两个表达式，并分别用一句话说明。

(c) Fixing the other parameters, how large can $N _ { \mathrm { F S D P } }$ become before the backward pass is communication bottlenecked? What about the forward pass? 

(c) 固定其他参数不变，`N_FSDP` 最多能扩展到多大，才不会在 backward 中被通信卡住？forward 呢？

Deliverable: Two inequalities with $N _ { \mathrm { F S D P } }$ on one side, and an expression in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $C$, and $W$ on the other, along with two one-sentence justifications. 

提交内容：给出两个关于 `N_FSDP` 的不等式，并分别用一句话说明。

# 8.4 Analyzing Tensor Parallel

## 8.4 分析张量并行

In practice, FSDP is often combined with a parallelism strategy called tensor parallelism (TP). In TP, we shard either the input or output dimension of each weight matrix across devices. Input dimension sharding is often called “row parallel,” while output dimension sharding is often called “column parallel.” 

在实践中，FSDP 往往会和另一种并行策略结合使用，这就是张量并行（TP）。在 TP 中，我们沿某个权重矩阵的输入维或输出维做分片。沿输入维分片常被称为 “row parallel”，沿输出维分片则称为 “column parallel”。

Specifically, suppose we want to shard the matmul $\boldsymbol { x } \boldsymbol { W }$, for $\boldsymbol { x }$ with shape $( B , D )$ and $W$ with shape $( D , D _ { \mathrm { F F } } )$. In column parallel we have shards $W ^ { ( i ) }$ of shape $\left( D , \frac { D _ { \mathrm { F F } } } { N _ { \mathrm { T P } } } \right)$, and we have 

具体地，假设我们要对矩阵乘法 `xW` 做分片，其中 `x` 的形状为 `(B, D)`，`W` 的形状为 `(D, D_FF)`。若采用 column parallel，则每个设备持有的 `W^(i)` 形状为 `(D, D_FF / N_TP)`，并且有：

$$
\boldsymbol {x} \boldsymbol {W} = \text { all - gather } \left(\left\{\boldsymbol {x} \boldsymbol {W} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{TP}} - 1}\right). \tag {45}
$$

On the other hand, for row parallel, we have shards $W ^ { ( i ) }$ of shape $\left( \frac { D } { N _ { \mathrm { T P } } } , D _ { \mathrm { F F } } \right)$, and each device also narrows the input $\boldsymbol { x }$ into a slice $\pmb { x } ^ { ( i ) }$ with shape $\left( B , \frac { D } { N _ { \mathrm { T P } } } \right)$ before doing the matmul. Then, we have 

另一方面，若采用 row parallel，则每个设备持有的 `W^(i)` 形状为 `(D / N_TP, D_FF)`，并且每个设备也会把输入 `x` 缩成一个对应切片 `x^(i)`，其形状为 `(B, D / N_TP)`。然后有：

$$
\boldsymbol {x} \boldsymbol {W} = \text { all   -   reduce } \left(\left\{\boldsymbol {x} ^ {(i)} \boldsymbol {W} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{TP}} - 1}\right). \tag {46}
$$

To parallelize our FFN, we’ll use a specific tensor parallel configuration where $W _ { 1 }$ and $W _ { 2 }$ are column parallel (output-dimension-sharded), while $W _ { 3 }$ is row parallel (input-dimension-sharded). Since the row parallel weight only requires a slice of the input, this configuration lets us skip the all-gather after the column parallel weights. This strategy gives the following forward pass, given input $\boldsymbol { x }$ of size $( B , D )$: 

为了并行化我们的 FFN，我们采用一个特定的 TP 配置：`W1` 与 `W2` 做 column parallel，而 `W3` 做 row parallel。由于 row parallel 的权重只需要输入的一部分切片，因此这种配置使我们可以跳过 `W1` 和 `W2` 之后原本可能需要的 all-gather。这样，给定形状为 `(B, D)` 的输入 `x`，forward 可以写成：

$$
\boldsymbol {x} _ {1} ^ {(i)} = \boldsymbol {x} \boldsymbol {W} _ {1} ^ {(i)} \tag {47}
$$

$$
\boldsymbol {x} _ {2} ^ {(i)} = \boldsymbol {x} \boldsymbol {W} _ {2} ^ {(i)} \tag {48}
$$

$$
\boldsymbol {z} ^ {(i)} = f \left(\boldsymbol {x} _ {1} ^ {(i)}\right) * \boldsymbol {x} _ {2} ^ {(i)} \tag {49}
$$

$$
\boldsymbol {y} ^ {(i)} = \boldsymbol {z} ^ {(i)} \boldsymbol {W} _ {\mathbf {3}} ^ {(i)} \tag {50}
$$

$$
\boldsymbol {y} = \text { all - reduce } \left(\left\{\boldsymbol {y} ^ {(i)} \right\} _ {i = 0} ^ {N _ {\mathrm{TP}} - 1}\right), \tag {51}
$$

where $W _ { 1 } ^ { ( i ) }$ and $W _ { 2 } ^ { ( i ) }$ have shape $\begin{array} { r } { \left( D , \frac { D _ { \mathrm { F F } } } { N _ { \mathrm { T P } } } \right) } \end{array}$ , and ${ W } _ { 3 } ^ { ( i ) }$ has shape $\left( \frac { D _ { \mathrm { F F } } } { N _ { \mathrm { T P } } } , D \right)$ 

其中 `W1^(i)` 和 `W2^(i)` 的形状为 `(D, D_FF / N_TP)`，而 `W3^(i)` 的形状为 `(D_FF / N_TP, D)`。

# Problem (tp_calcs):  Tensor parallel calculations (4 points)

### 题目（tp_calcs）：张量并行计算题（4 分）

Under the same setting as the DP and FSDP calculations, let’s calculate when TP becomes communication bottlenecked. 

在与 DP / FSDP 相同的设定下，分析 TP 何时会被通信瓶颈卡住。

(a) Given input $\boldsymbol { d } \boldsymbol { y }$ of size $( B , D )$, write out the backward pass of the tensor parallel strategy described above (where $W _ { 1 } ^ { ( i ) }$ and $W _ { 2 } ^ { ( i ) }$ have shape $\left( D , \frac { D _ { \mathrm { F F } } } { N _ { \mathrm { T P } } } \right)$, and $W _ { 3 } ^ { ( i ) }$ has shape $\left( \frac { D _ { \mathrm { F F } } } { N _ { \mathrm { T P } } } , D \right)$). 

(a) 给定大小为 `(B, D)` 的输入 `x`，请写出上述张量并行 FFN 策略的 backward 方程。你的表达式应当包含：输入 `x`，分片后的权重 `W1^(i)`、`W2^(i)`、`W3^(i)`，forward 中保存的激活 `x`、`x1^(i)`、`x2^(i)`、`z^(i)`、`y^(i)`，必要的通信原语，以及你认为合适的中间变量。最终方程应能产出每个设备上的 `dW1^(i)`、`dW2^(i)`、`dW3^(i)`，以及 backward 的输出 `dx`。你可以参考第 8.2 节中非分片 FFN 的 backward 并在其基础上改写。

Deliverable: A series of equations describing the backward pass, in terms of $\boldsymbol { d } \boldsymbol { y }$, sharded weights $\left( W _ { 1 } ^ { ( i ) } , W _ { 2 } ^ { ( i ) } , W _ { 3 } ^ { ( i ) } \right)$, activations saved from the forward pass $\left( x , x _ { 1 } ^ { ( i ) } , x _ { 2 } ^ { ( i ) } , z ^ { ( i ) } , y ^ { ( i ) } \right)$, communication primitives, and any intermediate variables you’d like to define. The equations should produce each device’s gradients $d W _ { 1 } ^ { ( i ) } , d W _ { 2 } ^ { ( i ) } , d W _ { 3 } ^ { ( i ) }$, and the backward pass output $\boldsymbol { d } \boldsymbol { x }$. Feel free to reference the non-sharded backward pass in Section 8.2 and modify it. 

提交内容：一组描述 backward 的方程。

(b) How many FLOPs are required to compute the forward pass, with $N _ { \mathrm { T P } }$ TP? What about the backward pass? 

(b) 在使用 `N_TP` 个张量并行设备时，forward 需要多少 FLOPs？backward 又需要多少？

Deliverable: Two answers in terms of $B , D , D _ { \mathrm { F F } }$ , and $N _ { \mathrm { T P } }$ , along with two one-sentence justifications. 

提交内容：两个关于 `B`、`D`、`D_FF` 和 `N_TP` 的表达式，并分别用一句话说明。

(c) How much communication time is required in the forward pass, with $N _ { \mathrm { T P } }$ TP? What about the backward pass? 

(c) 在使用 `N_TP` 个张量并行设备时，forward 需要多少通信时间？backward 又需要多少？

Deliverable: Two answers in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $N _ { \mathrm { T P } }$, and $W$, along with two one-sentence justifications. 

提交内容：两个表达式，并分别用一句话说明。

(d) Fixing the other parameters, how large can $N _ { \mathrm { T P } }$ become before the backward pass is communication bottlenecked? What about the forward pass? 

(d) 固定其他参数不变，`N_TP` 最多可以扩展到多大，才不会在 backward 中被通信卡住？forward 呢？

Deliverable: Two inequalities with $N _ { \mathrm { T P } }$ on one side, and an expression in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $C$, and $W$ on the other, along with two one-sentence justifications. 

提交内容：给出两个关于 `N_TP` 的不等式，并分别用一句话说明。

# 8.5 2D Parallelism (FSDP + TP)

## 8.5 二维并行（FSDP + TP）

We’re finally ready to combine parallelism strategies! In this section, we’ll look at how to optimally combine FSDP and TP. Hint: in the previous sections, you should have found that your batch size and model size parameters limit how many devices you can scale to, where making everything larger allows you to keep scaling devices without being communication bottlenecked. Unfortunately, scaling batch size past a certain point starts to degrade performance because gradients noise shrinks significantly, losing the implicit regularization properties of SGD; this point is often called the “critical batch size.” And scaling laws often tell us how large we want the model to be (that will be your job in the next assignment!). 

现在我们终于可以把不同并行策略组合起来了。本节将研究如何最优地结合 FSDP 与 TP。提示：在前几节中，你应该已经发现，batch size 和模型规模会限制你最多能扩展到多少设备；而把这些规模都做大，则能在不被通信卡住的情况下继续扩展设备数。不幸的是，batch size 扩大到一定程度后，会因为梯度噪声显著减小而降低训练效果，这个点常被称为 “critical batch size”。而 scaling law 往往也会告诉我们模型应该有多大（这会是你下一份作业中的任务）。

In this section, we’ll consider a simplified setting where someone comes to you with all of the problem parameters (batch size, model size, bandwidth, accelerator speed). Your job will be to choose a configuration of FSDP and TP that scales to as many devices as possible, while remaining computebound rather than communication-bound. 

在本节中，我们考虑一个简化情形：有人把问题中所有参数（batch size、模型规模、带宽、加速器算力）都给了你，你的任务是选择一个 FSDP 与 TP 的组合配置，使得总设备数尽可能大，同时仍然保持“计算受限”，而不是“通信受限”。

Let’s first walk through the mechanics of combining FSDP with TP. Each device will have a TP rank $i = 0, \ldots, N _ { \mathrm { T P } } - 1$ and an FSDP rank $j = 0, \ldots, N _ { \mathrm { F S D P } } - 1$, forming a 2D grid with $N = N _ { \mathrm { T P } } N _ { \mathrm { F S D P } }$ devices total. Following TP, we’ll first shard $W _ { 1 }$ and $W _ { 2 }$ along the output dimension, and $W _ { 3 }$ along the input dimension. As a result, we’ll have to insert a TP-style all-reduce on the activations. Next, applying FSDP, we’ll shard the batch dimension of the inputs, and we’ll also further shard each weight matrix along whichever dimension wasn’t sharded by TP. Then, we’ll have to insert FSDP-style all-gathers on the weights before doing our TP-style forward/backward passes, and reduce-scatters on the weight gradients after our TP-style backward pass. 

先来看 FSDP 与 TP 结合时的机制。每个设备都有一个 TP rank `i = 0, ..., N_TP - 1` 和一个 FSDP rank `j = 0, ..., N_FSDP - 1`，从而形成一个大小为 `N = N_TP * N_FSDP` 的二维设备网格。沿 TP 方向，我们先把 `W1` 和 `W2` 沿输出维分片，把 `W3` 沿输入维分片；因此激活上会引入 TP 风格的 all-reduce。再沿 FSDP 方向，我们把输入的 batch 维分片，并沿“TP 尚未切分的那个维度”继续把每个权重矩阵再切一刀。于是，在执行 TP 风格的 forward / backward 前，我们需要先做 FSDP 风格的权重 all-gather；在 TP 风格 backward 后，我们还要对权重梯度做 reduce-scatter。

The result is that each device $( i , j )$ holds the weight shards $W _ { 1 } ^ { ( i , j ) }$ and $W _ { 2 } ^ { ( i , j ) }$ with shape $\left( \frac { D } { N _ { \mathrm { F S D P } } } , \frac { D _ { \mathrm { F F } } } { N _ { \mathrm { T P } } } \right)$, and $W _ { 3 } ^ { ( i , j ) }$ with shape $\left( \frac { D _ { \mathrm { F F } } } { N _ { \mathrm { T P } } } , \frac { D } { N _ { \mathrm { F S D P } } } \right)$. We can then write out the forward pass as the following, given batch-sharded input $\pmb { x } ^ { ( j ) }$ of size $\left( \frac { B } { N _ { \mathrm { F S D P } } } , D \right)$: 

最终，每个设备 `(i, j)` 持有的权重切片为：`W1^(i, j)`、`W2^(i, j)`，形状均为 `(D / N_FSDP, D_FF / N_TP)`；以及 `W3^(i, j)`，形状为 `(D_FF / N_TP, D / N_FSDP)`。给定 batch 分片后的输入 `x^(j)`，其形状为 `(B / N_FSDP, D)`，forward 可写成：

$$
\boldsymbol {W} _ {1} ^ {(i)} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {1} ^ {(i, j)} \right\} _ {j = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {52}
$$

$$
\boldsymbol {W} _ {\mathbf {2}} ^ {(i)} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {\mathbf {2}} ^ {(i, j)} \right\} _ {j = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {53}
$$

$$
\boldsymbol {W} _ {\mathbf {3}} ^ {(i)} = \text { all - gather } \left(\left\{\boldsymbol {W} _ {\mathbf {3}} ^ {(i, j)} \right\} _ {j = 0} ^ {N _ {\mathrm{FSDP}} - 1}\right) \tag {54}
$$

$$
\boldsymbol {x} _ {1} ^ {(i, j)} = \boldsymbol {x} ^ {(j)} \boldsymbol {W} _ {1} ^ {(i)} \tag {55}
$$

$$
\boldsymbol {x} _ {\mathbf {2}} ^ {(i, j)} = \boldsymbol {x} ^ {(j)} \boldsymbol {W} _ {\mathbf {2}} ^ {(i)} \tag {56}
$$

$$
\boldsymbol {z} ^ {(i, j)} = f \left(\boldsymbol {x} _ {\mathbf {1}} ^ {(i, j)}\right) * \boldsymbol {x} _ {\mathbf {2}} ^ {(i, j)} \tag {57}
$$

$$
\boldsymbol {y} ^ {(i, j)} = \boldsymbol {z} ^ {(i, j)} \boldsymbol {W} _ {3} ^ {(i)} \tag {58}
$$

$$
\boldsymbol {y} ^ {(j)} = \text { all - reduce } \left(\left\{\boldsymbol {y} ^ {(i, j)} \right\} _ {i = 0} ^ {N _ {\mathrm{TP}} - 1}\right), \tag {59}
$$

ending up with batch-sharded output $\boldsymbol y ^ { ( j ) }$ of size $\left( \frac { B } { N _ { \mathrm { F S D P } } } , D \right)$. We’ll omit the backward pass for brevity in this section, and just focus on the forward pass. But at this point, you should have all the information you need to write it out yourself. 

最终得到的输出 `y^(j)` 仍然按 batch 维分片，形状为 `(B / N_FSDP, D)`。为简洁起见，本节省略 backward，但你现在应该已经有足够信息自己写出它。

# Problem (fsdp_tp_calcs):  2D parallelism calculations (6 points)

### 题目（fsdp_tp_calcs）：二维并行计算题（6 分）

Under the same setting as the calculations so far, let’s calculate when 2D parallelism becomes bottlenecked. 

在与前面相同的设定下，分析二维并行何时会被通信卡住。

(a) How many FLOPs are required to compute the forward pass, with $N _ { \mathrm { F S D P } } \ \mathrm { F S D P } + N _ { \mathrm { T P } }$ TP? 

(a) 在使用 `N_FSDP` 个 FSDP 分片和 `N_TP` 个 TP 分片时，forward 需要多少 FLOPs？

Deliverable: An answer in terms of $B$, $D$, $D _ { \mathrm { F F } }$, $N _ { \mathrm { F S D P } }$, and $N _ { \mathrm { T P } }$, along with a one-sentence justification. 

提交内容：给出一个关于 `B`、`D`、`D_FF`、`N_FSDP` 和 `N_TP` 的表达式，并用一句话说明。

(b) How much communication time is required in the forward pass, with $N _ { \mathrm { F S D P } } \ \mathrm { F S D P } + N _ { \mathrm { T P } }$ TP? Assume that the communication along each axis can be overlapped (in other words, the collectives along the FSDP axis can be overlapped with the collectives along the TP axis). 

(b) 在这种 `FSDP + TP` 设定下，forward 需要多少通信时间？假设沿 FSDP 轴与 TP 轴的通信可以重叠（也就是说，两条轴上的 collective 可以同时进行）。

Deliverable: An answer in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $N _ { \mathrm { F S D P } }$, $N _ { \mathrm { T P } }$, and $W$, along with a one-sentence justification. Hint: The answer should be expressed as a `max(...)` between two quantities (the FSDP and TP collective costs), since the two can be overlapped. 

提交内容：给出一个表达式，并用一句话说明。提示：答案应写成两个量的 `max(...)`，分别对应 FSDP 与 TP 方向上的通信成本，因为二者是可重叠的。

(c) Under the optimal setting of $N _ { \mathrm { T P } }$ and $N _ { \mathrm { F S D P } }$ , how large can $N = N _ { \mathrm { T P } } N _ { \mathrm { F S D P } }$ become before the forward pass is communication bottlenecked? 

(c) 在 `N_TP` 与 `N_FSDP` 取最优平衡时，总设备数 `N = N_TP * N_FSDP` 最多能扩展到多大，forward 才会被通信卡住？

Deliverable: An inequality with $N$ on one side, and an expression in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $C$, and $W$ on the other, along with a few sentences and equations as justification. 

提交内容：给出一个关于 `N` 的不等式，并用几句文字与公式说明理由。

(d) Now suppose the FSDP-axis and TP-axis collectives cannot be overlapped because they share the same network resources. Under the optimal setting of $N _ { \mathrm { T P } }$ and $N _ { \mathrm { F S D P } }$ , how large can $N =$ $N _ { \mathrm { T P } } N _ { \mathrm { F S D P } }$ become before the forward pass is communication bottlenecked? Don’t worry about truncating $N _ { \mathrm { T P } }$ and $N _ { \mathrm { F S D P } }$ to be integers. 

(d) 现在假设 FSDP 轴和 TP 轴上的 collective 不能重叠，因为它们共享相同的网络资源。那么在最优的 `N_TP` 与 `N_FSDP` 设定下，总设备数 `N = N_TP * N_FSDP` 最多能扩展到多大，forward 才会被通信卡住？这里你不必纠结 `N_TP` 和 `N_FSDP` 必须取整数。

Deliverable: An inequality with $N$ on one side, and an expression in terms of a subset of $B$, $D$, $D _ { \mathrm { F F } }$, $C$, and $W$ on the other, along with a few sentences and equations as justification. 

提交内容：给出一个关于 `N` 的不等式，并用几句文字与公式说明理由。

# 9 Leaderboard

## 9 排行榜

Assignment 2′s leaderboard will test the speed of a full training step for an 8B model. We challenge you to benchmark your code and to optimize memory and runtime, using any tricks you can come up with. The key restrictions are that you cannot change the input/output behavior of the model. Your implementation will be tested against the model in the cs336_basics directory. Your inputs will be tested at BF16 with causal masking, and they must pass the same tests as your regular implementation. The implementation must also be your own, and you cannot use or copy pre-existing implementations. Your timing should be measured on two B200 GPUs using a sample with batch size 2 and sequence length 32,768. It is intentionally difficult to fit the model in memory. See the code below for the full config. We will verify the top 5-10 submissions for correctness and performance. The test we will run to time your implementation is the following: 

作业 2 的排行榜会测试一个 8B 模型完成一次完整训练 step 的速度。我们鼓励你对自己的代码做 benchmark，并尽可能地优化显存和运行时间，使用任何你能想到的技巧。核心限制是：你不能改变模型的输入 / 输出行为。你的实现将会对照 `cs336_basics` 目录中的模型进行测试；输入会在 BF16 和 causal masking 条件下测试，并且它必须通过和你常规实现相同的测试。实现必须是你自己的，不能使用或拷贝已有实现。计时会在两张 B200 GPU 上进行，输入样本 batch size 为 2，sequence length 为 32,768。这个设置故意让模型很难塞进显存。题目中给出了完整配置。我们会对排名前 5 到 10 的提交进行正确性和性能验证。用于计时的测试大致如下：

```python
class Config:
    ctx_len = 32768
    vocab_size = 151936
    d_model = 4096
    d_ff = 11008
    num_layers = 34
    num_heads = 32
    torch_dtype = torch.bfloat16
    is_causal = True
    batch_size = 2
cfg = Config()

def test_timing_forward_backward():
    labels, targets = torch.randint(high=cfg produced_size, size=(2, cfg.batch_size, cfg.N) != config()
    model = BasicsTransformerLM(Config())
    optimizer = AdamW(model.parameters())

def train_step():
    optimizer.zero_grad(set_to_none=True)
    res = model(labels)
    loss = cross_entropy(res, targets).sum()
    loss.backward()
    optimizer.step()

timing_results = triton.testing.do_bench(train_step, rep=30_000, warmup=10_000)
print(timing_results) 
```

For testing purposes, you can reduce the repetition and warmup time (given in ms) to something shorter. 

为了测试方便，你可以把重复次数和 warm-up 时间（单位 ms）调小一些。

Some ideas for improvement and for making sure your model fits: 

一些可能有帮助的优化方向包括：

• Tune the tile sizes for your kernel (use Triton autotune for this!) 

• 调整 kernel 的 tile size（推荐用 Triton autotune）。

• Tune additional Triton/torch.compile config parameters 

• 调整更多 Triton / `torch.compile` 配置。

• Implement fused AdamW 

• 实现融合版 AdamW。

• The base implementation is materializing the full logits ([batch, seq_len, vocab_size]). Write a kernel that fuses the LM head and your cross-entropy loss. You can also have it compute the backward pass immediately in a fused manner 

• 当前基础实现会显式物化完整 logits（`[batch, seq_len, vocab_size]`）。你可以编写一个 kernel，把 LM head 与 cross-entropy loss 融合起来，甚至进一步把 backward 也一并融合进去。

• Improve FlashAttention 

• 改进 FlashAttention。

‣ Implement the backward pass in Triton, not just torch.compile (see Section 4.2.3) 

‣ 用 Triton 实现 backward，而不只是依赖 `torch.compile`（见第 4.2.3 节）。

‣ Do two passes over your input for the backward pass, one for dQ and another for dK and dV, to avoid atomics or synchronization between blocks 

‣ 对 backward 输入做两遍遍历：一遍算 `dQ`，另一遍算 `dK` 和 `dV`，以避免原子操作或 block 间同步。

‣ Stop program instances early when doing causal masking, skipping tiles that are guaranteed to be all zero 

‣ 在做 causal masking 时尽早结束某些 program instance，直接跳过那些确定全为 0 的 tile。

‣ Separate the non-masked tiles from the tile diagonals, computing the first without ever comparing indices, and the second with a single comparison 

‣ 把未被 mask 的 tile 与对角线附近的 tile 分开处理：前者完全不做索引比较，后者只做一次比较。

‣ Use TMA (Tensor Memory Accelerator) functionality on architectures later than Hopper, following a similar pattern to our tutorial 

‣ 在 Hopper 之后的架构上使用 TMA（Tensor Memory Accelerator），实现方式可参考教程中的类似模式。

• Use activation checkpointing to trade runtime speed for memory savings only if you need it 

• 只有在确实需要时，再用 activation checkpointing 以运行时间换取显存节省。

# Problem (leaderboard):  Leaderboard: fastest training step (10 points)

### 题目（leaderboard）：排行榜，最快训练步（10 分）

The benchmark will be run at batch size 2 on two B200 GPUs. Your submission will be evaluated on wall-clock time for a complete training step: forward pass, loss, backward pass, and AdamW update. 

benchmark 会在两张 B200 GPU 上、batch size 2 的设置下运行。你的提交将按照一次完整训练 step 的墙钟时间评测：包括 forward、loss、backward 和 AdamW 更新。

From an empty PyTorch/Triton cache, your benchmarking run must complete within 10 minutes, so be careful with overly aggressive torch.compile and Triton autotuning. 

从一个空的 PyTorch / Triton cache 开始，你的 benchmark 运行必须在 10 分钟内完成，因此请小心不要把 `torch.compile` 和 Triton autotuning 调得过于激进。

Deliverable: Your best wall-clock time for a full forward-and-backward training step with AdamW. 

提交内容：给出你实现的“包含 AdamW 的完整 forward + backward 训练 step”的最佳墙钟时间。

We expect leaderboard submissions to beat the naïve baseline of 10 seconds. 

我们预计排行榜提交应当优于约 10 秒的朴素基线。

Submit your result to the leaderboard here: 

请将你的结果提交到这里：

github.com/stanford-cs336/assignment2-systems-leaderboard 

# Bibliography

## 参考文献

[1] T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.” [Online]. Available: https://arxiv.org/abs/2307.08691 

[1] T. Dao, “FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning.” 在线地址：https://arxiv.org/abs/2307.08691

[2] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” in Advances in Neural Information Processing Systems, A. H. Oh, A. Agarwal, D. Belgrave, and K. Cho, Eds., 2022. [Online]. Available: https://openreview.net/forum?id=H4DqfPSibmx 

[2] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Re, “FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness,” NeurIPS 2022. 在线地址：https://openreview.net/forum?id=H4DqfPSibmx

[3] M. Milakov and N. Gimelshein, “Online normalizer calculation for softmax.” [Online]. Available: https://arxiv.org/abs/1805.02867 

[3] M. Milakov and N. Gimelshein, “Online normalizer calculation for softmax.” 在线地址：https://arxiv.org/abs/1805.02867

[4] H. He, “Making Deep Learning Go Brrrr From First Principles,” 2022, [Online]. Available: https://horace.io/brrr_intro.html 

[4] H. He, “Making Deep Learning Go Brrrr From First Principles,” 2022. 在线地址：https://horace.io/brrr_intro.html

[5] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He, “ZeRO: Memory Optimizations Toward Training Trillion Parameter Models.” 2020. 

[6] J. Austin et al., “How to Scale Your Model,” 2025. 

[7] H. Z. P. N. M. M. L. W. T. W. Nouamane Tazi Ferdinand Mom, “The Ultra-Scale Playbook: Training LLMs on GPU Clusters.” 2025. 
