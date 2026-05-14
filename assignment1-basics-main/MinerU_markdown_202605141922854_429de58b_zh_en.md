# CS336 Assignment 1 (basics): Building a Transformer LM
# CS336 作业 1（基础）：构建一个 Transformer 语言模型

Version 26.0.3
版本 26.0.3

CS336 Staff
CS336 课程教学团队

Spring 2026
2026 年春季

# 1 Assignment Overview
# 1 作业概述

In this assignment, you will build all the components needed to train a standard Transformer language model (LM) from scratch and train some models.
在本次作业中，你将从零开始构建训练标准 Transformer 语言模型（LM）所需的全部组件，并实际训练一些模型。

# What you will implement
# 你将实现的内容

1. Byte-pair encoding (BPE) tokenizer (Section 2)
1. 字节对编码（BPE）分词器（第 2 节）

2. Transformer language model (LM) (Section 3)
2. Transformer 语言模型（LM）（第 3 节）

3. The cross-entropy loss function and the AdamW optimizer (Section 4)
3. 交叉熵损失函数和 AdamW 优化器（第 4 节）

4. The training loop, with support for serializing and loading model and optimizer state (Section 5)
4. 训练循环，并支持模型与优化器状态的序列化和加载（第 5 节）

# What you will run
# 你将运行的内容

1. BPE tokenizer training on the TinyStories dataset.
1. 在 TinyStories 数据集上训练 BPE 分词器。

2. Trained tokenizer encoding on the dataset to convert it into a sequence of integer IDs.
2. 使用训练好的分词器对数据集进行编码，将其转换为整数 ID 序列。

3. Transformer LM training on the TinyStories dataset.
3. 在 TinyStories 数据集上训练 Transformer 语言模型。

4. Sample generation and evaluation of perplexity using the trained Transformer LM.
4. 使用训练好的 Transformer 语言模型进行采样生成并评估困惑度。

5. Model training on OpenWebText, and submit your attained perplexities to a leaderboard.
5. 在 OpenWebText 上训练模型，并将你获得的困惑度提交到排行榜。

# What you can use
# 你可以使用的内容

We expect you to build each component from scratch. In particular, you may not use any definitions from torch.nn, torch.nn.functional, or torch.optim except for the following:
我们希望你从零开始构建每个组件。特别地，除以下内容外，你不能使用 torch.nn、torch.nn.functional 或 torch.optim 中的任何现成定义：

• torch.nn.Parameter
• torch.nn.Parameter

• Container classes in torch.nn (e.g., Module, ModuleList, Sequential, etc.).1
• torch.nn 中的容器类（例如 Module、ModuleList、Sequential 等）。1

• The torch.optim.Optimizer base class
• torch.optim.Optimizer 基类

You may use any other PyTorch definitions. If you would like to use a function or class and are not sure whether it is permitted, feel free to ask on Slack. When in doubt, consider if using it compromises the “from-scratch” ethos of the assignment.
你可以使用其他任意 PyTorch 定义。如果你想使用某个函数或类，但不确定是否允许，请随时在 Slack 上提问。拿不准时，请考虑它是否违背了本作业“从零开始”的精神。

# Statement on AI tools
# 关于 AI 工具的说明

AI can solve many parts of the assignments fully autonomously. This makes it harder to deeply engage with, and learn from, the course material.
AI 可以在几乎无需人工参与的情况下完整解决作业中的许多部分。这会让你更难深入参与课程材料并真正从中学习。

The use of AI tools is permitted for answering high-level conceptual questions, or for providing low-level programming documentation like function signatures and library APIs. However, AI tools are not permitted for implementing any part of any assignment. This includes both coding agents (e.g., Cursor Agents, Codex, Claude Code) and AI autocomplete (e.g., Cursor Tab, GitHub Copilot). When using an AI agent, make sure it uses the AGENTS.md file provided. The prompt should also be included when using chatbots.
允许使用 AI 工具来回答高层次的概念性问题，或提供底层编程文档信息，例如函数签名和库 API。然而，AI 工具不得用于实现任何作业的任何部分。这既包括编码代理（例如 Cursor Agents、Codex、Claude Code），也包括 AI 自动补全（例如 Cursor Tab、GitHub Copilot）。在使用 AI agent 时，请确保它使用提供的 AGENTS.md 文件。若使用聊天机器人，也应附上相应提示词。

We strongly encourage you to disable AI autocomplete (e.g., Cursor Tab, GitHub Copilot) in your IDE when completing assignments (though non-AI autocomplete, e.g., autocompleting function names is totally fine). Previous students have highlighted that disabling AI autocomplete made it easier to engage deeply with the material.
我们强烈建议你在完成作业时关闭 IDE 中的 AI 自动补全（例如 Cursor Tab、GitHub Copilot），不过非 AI 的自动补全（例如函数名补全）完全没有问题。以往学生反馈，关闭 AI 自动补全后更容易深入投入到材料本身。

For the full AI Policy, see this document.
完整 AI 政策请参见该文档。

# What the code looks like
# 代码结构说明

The assignment code, as well as this writeup, are available on GitHub at:
本作业代码以及本文档可在 GitHub 上获取：

$\mathfrak { g i t h u b . c o m / s t a n f o r d - c s 3 3 6 / a s s i g n m e n t 1 - b a s i c s }$

Please git clone the repository. If there are any updates, we will notify you so you can git pull to get the latest.
请使用 git clone 克隆该仓库。如果有更新，我们会通知你，以便你通过 git pull 获取最新版。

1. cs336_basics/*: This is where you write your code. Note that there’s no code in here—you can do whatever you want from scratch!
1. cs336_basics/*：这是你编写代码的地方。注意这里面没有现成代码，你可以完全从零开始自由实现。

2. adapters.py: There is a set of functionality that your code must have. For each piece of functionality (e.g., scaled dot product attention), fill out its implementation (e.g., run_scaled_dot_product_attention) by simply invoking your code. Note: your changes to adapters.py should not contain any substantive logic; this is glue code.
2. adapters.py：这里定义了你的代码必须提供的一组功能。对于每个功能点（例如 scaled dot product attention），你只需通过调用你自己的实现来补全对应接口（例如 run_scaled_dot_product_attention）。注意：你对 adapters.py 的改动不应包含实质性逻辑；这里应只是胶水代码。

3. test_*.py: This contains all the tests that you must pass (e.g., test_scaled_dot_product_attention), which will invoke the hooks defined in adapters.py. Don’t edit the test files.
3. test_*.py：这里包含你必须通过的所有测试（例如 test_scaled_dot_product_attention），它们会调用 adapters.py 中定义的接口。不要修改测试文件。

# How to submit
# 如何提交

In order to submit, run make_submission.sh to construct a submission zip file. Make sure to add any additional files to the list of exclusions in the script if you have large data files or checkpoints you don’t want to include in your submission zip.
提交时，请运行 make_submission.sh 来构建提交用 zip 文件。如果你有不希望包含进提交包的大型数据文件或 checkpoint，请确保将其加入脚本中的排除列表。

You will submit the following files to Gradescope:
你需要向 Gradescope 提交以下文件：

• writeup.pdf: Answer all the written questions. Please typeset your responses.
• writeup.pdf：回答所有书面题目。请使用排版后的正式文档提交。

• code.zip: Contains all the code you’ve written.
• code.zip：包含你编写的全部代码。

To submit to the leaderboard, submit a PR to:
若要提交排行榜成绩，请向以下仓库提交 PR：

$9 \mathtt { i t } \mathtt { h u b . c o m / s t a n f o r d - c s 3 3 6 / a s s i g n m e n t 1 - b a s i c s - l e a d e r b o a r d } \mathtt { i n - d e a m e n t - f o r } \mathtt { d e a m \_ s p a m e n t }$

See the README.md in the leaderboard repository for detailed submission instructions.
详细提交说明请参见排行榜仓库中的 README.md。

# Where to get datasets
# 数据集获取方式

This assignment will use two pre-processed datasets: TinyStories [R. Eldan et al., 2023] and OpenWebText [A. Gokaslan et al., 2019]. Both datasets are single, large plaintext files.
本作业会使用两个预处理后的数据集：TinyStories [R. Eldan et al., 2023] 和 OpenWebText [A. Gokaslan et al., 2019]。两者都是单个大型纯文本文件。

If you are doing the assignment with the class, you can find instructions for downloading the dataset in the compute guide.
如果你是跟随课程一起完成作业，可以在 compute guide 中找到下载数据集的说明。

If you are following along at home, you can download these files with the commands inside the README.md.
如果你是在课外自行跟做，可以根据 README.md 中的命令下载这些文件。

# Low-Resource Tip: Init
# 低资源提示：引言

Throughout the course’s assignment handouts, we will give advice for working through parts of the assignment with fewer or no GPU resources. For example, we will sometimes suggest downscaling your dataset or model size, or explain how to run training code on a Mac integrated GPU or CPU. You’ll find these “low-resource tips” in a blue box (like this one). Even if you are an enrolled Stanford student with access to the course machines, these tips may help you iterate faster and save time, so we recommend reading them!
在整门课程的作业说明中，我们会给出一些建议，帮助你在较少甚至没有 GPU 资源的情况下完成部分任务。例如，我们有时会建议你缩小数据集或模型规模，或者说明如何在 Mac 集成 GPU 或 CPU 上运行训练代码。你会在蓝色提示框中看到这些“低资源提示”（就像这里这样）。即使你是有课程机器可用的 Stanford 在读学生，这些提示也可能帮助你更快迭代、节省时间，因此我们建议你阅读它们。

# Low-Resource Tip: Assignment 1 on Apple Silicon or CPU
# 低资源提示：在 Apple Silicon 或 CPU 上完成作业 1

With the staff solution code, we can train an LM to generate reasonably fluent text on an Apple M4 Max chip with 36 GB RAM, in under 5 minutes on Metal GPU (MPS) and about 30 minutes using the CPU. If these words don’t mean much to you, don’t worry! Just know that if you have a reasonably up-to-date laptop and your implementation is correct and efficient, you will be able to train a small LM that generates simple children’s stories with decent fluency.
使用助教提供的参考实现，我们可以在配备 36 GB 内存的 Apple M4 Max 芯片上训练一个能生成较流畅文本的语言模型：在 Metal GPU（MPS）上不到 5 分钟，用 CPU 大约 30 分钟。如果这些术语你不太熟悉，也不用担心。你只需要知道，如果你的笔记本足够新、实现正确且高效，那么你应该能够训练出一个小型语言模型，生成流畅度尚可的简单儿童故事。

Later in the assignment, we will explain what changes to make if you are on CPU or MPS.
在作业后文中，我们会说明如果你使用 CPU 或 MPS，需要做哪些调整。

# 2 Byte-Pair Encoding (BPE) Tokenizer
# 2 字节对编码（BPE）分词器

In the first part of the assignment, we will train and implement a byte-level byte-pair encoding (BPE) tokenizer [R. Sennrich et al., 2016; C. Wang et al., 2019]. In particular, we will represent arbitrary (Unicode) strings as a sequence of bytes and train our BPE tokenizer on this byte sequence. Later, we will use this tokenizer to encode text (a string) into tokens (a sequence of integers) for language modeling.
在作业的第一部分中，我们将训练并实现一个字节级 byte-pair encoding（BPE）分词器 [R. Sennrich et al., 2016; C. Wang et al., 2019]。具体来说，我们会将任意（Unicode）字符串表示为字节序列，并在该字节序列上训练 BPE 分词器。之后，我们会使用该分词器将文本（字符串）编码为 token（整数序列），以用于语言建模。

# 2.1 The Unicode Standard
# 2.1 Unicode 标准

Unicode is a text encoding standard that maps characters to integer code points. As of Unicode 17.0 (released in September 2025), the standard defines 159,801 characters across 172 scripts. For example, the character “s” has the code point 115 (typically notated as U+0073, where U+ is a conventional prefix and 0073 is 115 in hexadecimal), and the character “牛” has the code point 29275. In Python, you can use the ord() function to convert a single Unicode character into its integer representation. The chr() function converts an integer Unicode code point into a string with the corresponding character.
Unicode 是一种文本编码标准，它将字符映射为整数码点。截至 Unicode 17.0（发布于 2025 年 9 月），该标准已定义了 172 种书写系统中的 159,801 个字符。例如，字符 “s” 的码点是 115（通常记作 U+0073，其中 U+ 是惯例前缀，0073 是十六进制表示的 115），而字符 “牛” 的码点是 29275。在 Python 中，你可以使用 ord() 函数将单个 Unicode 字符转换为其整数表示；chr() 函数则将整数 Unicode 码点转换为对应字符组成的字符串。

```txt
>>> ord('牛')
29275
>>> chr(29275)
'牛'
```

# Problem (unicode1): Understanding Unicode (1 point)
# 题目（unicode1）：理解 Unicode（1 分）

(a) What Unicode character does chr(0) return?
（a）chr(0) 返回的是哪个 Unicode 字符？

Deliverable: A one-sentence response.
提交内容：一句话回答。

(b) How does this character’s string representation (__repr__()) differ from its printed representation?
（b）这个字符的字符串表示（__repr__()）与它打印出来的表示有何不同？

Deliverable: A one-sentence response.
提交内容：一句话回答。

(c) What happens when this character occurs in text? It may be helpful to play around with the following in your Python interpreter and see if it matches your expectations:
（c）当这个字符出现在文本中时会发生什么？你可以在 Python 解释器中尝试下面的代码，看看结果是否符合预期：

```txt
>>> chr(0)
>>> print(chr(0))
```

```txt
>>> "this is a test" + chr(0) + "string"
>>> print("this is a test" + chr(0) + "string")
```

Deliverable: A one-sentence response.
提交内容：一句话回答。

# 2.2 Unicode Encodings
# 2.2 Unicode 编码

While the Unicode standard defines a mapping from characters to code points (integers), it’s impractical to train tokenizers directly on Unicode code points, since the vocabulary would be prohibitively large (around 150K items) and sparse (since many characters are quite rare). Instead, we’ll use a Unicode encoding, which converts a Unicode character into a sequence of bytes. The Unicode standard itself defines three encodings: UTF-8, UTF-16, and UTF-32, with UTF-8 being the dominant encoding for the Internet (more than 98% of all webpages).
虽然 Unicode 标准定义了从字符到码点（整数）的映射，但直接在 Unicode 码点上训练分词器并不现实，因为词表会过大（大约 15 万项）且非常稀疏（许多字符都极少见）。因此，我们将使用 Unicode 编码方式，把 Unicode 字符转换为字节序列。Unicode 标准本身定义了三种编码：UTF-8、UTF-16 和 UTF-32，其中 UTF-8 是互联网上的主流编码（超过 98% 的网页使用它）。

To encode a Unicode string into UTF-8, we can use the encode() function in Python. To access the underlying byte values for a Python bytes object, we can iterate over it (e.g., call list()). Finally, we can use the decode() function to decode a UTF-8 byte string into a Unicode string.
要将 Unicode 字符串编码为 UTF-8，可以在 Python 中使用 encode() 函数。要访问 Python bytes 对象底层的字节值，可以对它进行迭代（例如调用 list()）。最后，我们可以使用 decode() 函数将 UTF-8 字节串解码回 Unicode 字符串。

```python
>>> test_string = "hello! こんにちは!"
>>> utf8_encoded = test_string.encode("utf-8")
>>> print(utf8_encoded)
b'hello! \xe3\x81\x93\xe3\x82\x93\xe3\x81\xab\xe3\x81\xa1\xe3\x81\xaf!'
>>> print(type(utf8_encoded))
<class 'bytes'>
>>> # Get the byte values for the encoded string (integers from 0 to 255).
>>> list(utf8_encoded)
[104, 101, 108, 108, 111, 33, 32, 227, 129, 147, 227, 130, 147, 227, 129, 171, 227, 129, 161, 227, 129, 175, 33]
>>> # One byte does not necessarily correspond to one Unicode character!
>>> print(len(test_string))
13
>>> print(len(utf8_encoded))
23
>>> print(utf8_encoded.decode("utf-8"))
hello! こんにちは!
```

By converting our Unicode code points into a sequence of bytes (e.g., via the UTF-8 encoding), we are essentially taking a sequence of code points (21-bit integers with 159,801 valid values) and transforming it into a sequence of byte values (integers in the range 0 to 255). The 256-length byte vocabulary is much more manageable to deal with. When using byte-level tokenization, we do not need to worry about out-of-vocabulary tokens, since we know that any input text can be expressed as a sequence of integers from 0 to 255.
通过把 Unicode 码点转换成字节序列（例如通过 UTF-8 编码），本质上就是把一个码点序列（21 位整数，共有 159,801 个有效取值）转换为一个字节值序列（范围为 0 到 255 的整数）。长度为 256 的字节词表更易处理。采用字节级分词时，我们不必担心词表外 token，因为任何输入文本都可以表示为 0 到 255 之间整数构成的序列。

# Problem (unicode2): Unicode Encodings (3 points)
# 题目（unicode2）：Unicode 编码（3 分）

(a) What are some reasons to prefer training our tokenizer on UTF-8 encoded bytes, rather than UTF-16 or UTF-32? It may be helpful to compare the output of these encodings for various input strings.
（a）为什么我们更倾向于在 UTF-8 编码后的字节上训练分词器，而不是使用 UTF-16 或 UTF-32？你可以比较不同输入字符串在这些编码下的输出。

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

(b) Consider the following (incorrect) function, which is intended to decode a UTF-8 byte string into a Unicode string. Why is this function incorrect? Provide an example of an input byte string that yields incorrect results.
（b）考虑下面这个意在将 UTF-8 字节串解码为 Unicode 字符串的（错误）函数。为什么这个函数是错误的？请给出一个会产生错误结果的输入字节串示例。

```python
def decode_utf8_bytes_to_str_wrong(bytesstring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])
>>> decode_utf8_bytes_to_str_wrong("hello".encode("utf-8"))
'hello'
```

Deliverable: An example input byte string for which decode_utf8_bytes_to_str_wrong produces incorrect output, with a one-sentence explanation of why the function is incorrect.
提交内容：给出一个会让 decode_utf8_bytes_to_str_wrong 产生错误输出的输入字节串示例，并用一句话解释为什么该函数不正确。

(c) Give a two-byte sequence that does not decode to any Unicode character(s).
（c）给出一个无法解码为任何 Unicode 字符的双字节序列。

Deliverable: An example, with a one-sentence explanation.
提交内容：给出一个例子，并用一句话解释。

# 2.3 Subword Tokenization
# 2.3 子词分词

While byte-level tokenization can alleviate the out-of-vocabulary issues faced by word-level tokenizers, tokenizing text into bytes results in extremely long input sequences. This slows down model training, since a sentence with 10 words might only be 10 tokens long in a word-level language model, but could be 50 or more tokens long in a character-level model (depending on the length of the words). Processing these longer sequences requires more computation at each step of the model. Furthermore, language modeling on byte sequences is difficult because the longer input sequences create long-term dependencies in the data.
虽然字节级分词可以缓解词级分词器面临的词表外问题，但把文本分成字节会导致输入序列极长。这会拖慢模型训练，因为一个包含 10 个单词的句子在词级语言模型中可能只有 10 个 token，但在字符级模型中可能会变成 50 个甚至更多 token（取决于单词长度）。处理这些更长的序列需要模型在每一步做更多计算。此外，在字节序列上做语言建模也更困难，因为更长的输入序列会带来更强的长期依赖。

Subword tokenization is a midpoint between word-level tokenizers and byte-level tokenizers. Note that a byte-level tokenizer’s vocabulary has 256 entries (byte values are 0 to 255). A subword tokenizer trades off a larger vocabulary size for better compression of the input byte sequence. For example, if the byte sequence b'the' often occurs in our raw text training data, assigning it an entry in the vocabulary would reduce this 3-token sequence to a single token.
子词分词位于词级分词器与字节级分词器之间。字节级分词器的词表大小只有 256（字节值范围是 0 到 255）。子词分词器通过扩大词表大小，换取对输入字节序列更好的压缩。例如，如果字节序列 b'the' 在原始训练文本中经常出现，那么为它分配一个词表项，就可以把原本 3 个 token 的序列压缩成 1 个 token。

How do we select these subword units to add to our vocabulary? R. Sennrich et al. [3] propose to use byte-pair encoding (BPE; P. Gage [5]), a compression algorithm that iteratively replaces (“merges”) the most frequent pair of bytes with a single, new unused index. Note that this algorithm adds subword tokens to our vocabulary to maximize the compression of our input sequences—if a word occurs in our input text enough times, it’ll be represented as a single subword unit.
那么我们如何选择要加入词表的这些子词单元呢？R. Sennrich 等人 [3] 提出使用 byte-pair encoding（BPE；P. Gage [5]），这是一种压缩算法，它会迭代地将出现频率最高的一对字节替换（“合并”）为一个新的、此前未使用的索引。注意，这种算法是为了最大化输入序列的压缩率而向词表中添加子词 token 的；如果某个词在输入文本中出现得足够频繁，它最终就会被表示为单个子词单元。

Subword tokenizers with vocabularies constructed via BPE are often called BPE tokenizers. In this assignment, we’ll implement a byte-level BPE tokenizer, where the vocabulary items are bytes or merged sequences of bytes, which give us the best of both worlds in terms of out-of-vocabulary handling and manageable input sequence lengths. The process of constructing the BPE tokenizer vocabulary is known as “training” the BPE tokenizer.
通过 BPE 构建词表的子词分词器通常被称为 BPE 分词器。在本作业中，我们将实现一个字节级 BPE 分词器，其中词表项是字节或者字节合并后的序列。这样可以同时兼顾词表外处理能力和可控的输入序列长度。构建 BPE 分词器词表的过程被称为对 BPE 分词器进行“训练”。

# 2.4 BPE Tokenizer Training
# 2.4 BPE 分词器训练

The BPE tokenizer training procedure consists of three main steps.
BPE 分词器训练过程主要包含三个步骤。

# Vocabulary initialization
# 词表初始化

The tokenizer vocabulary is a one-to-one mapping from bytestring token to integer ID. Since we’re training a byte-level BPE tokenizer, our initial vocabulary is simply the set of all bytes. Since there are 256 possible byte values, our initial vocabulary is of size 256.
分词器词表是从字节串 token 到整数 ID 的一一映射。由于我们训练的是字节级 BPE 分词器，初始词表就是所有可能字节的集合。因为字节值一共有 256 种可能，所以初始词表大小为 256。

# Pre-tokenization
# 预分词

Once you have a vocabulary, you could, in principle, count how often bytes occur next to each other in your text and begin merging them starting with the most frequent pair of bytes. However, this is quite computationally expensive, since we’d have to take a full pass over the corpus each time we merge. In addition, directly merging bytes across the corpus may result in tokens that differ only in punctuation (e.g., dog! vs. dog.). These tokens would get completely different token IDs, even though they are likely to have high semantic similarity (since they differ only in punctuation).
一旦有了词表，理论上你就可以统计文本中哪些字节经常相邻出现，并从最频繁的字节对开始做合并。然而，这在计算上代价很高，因为每进行一次合并，我们都需要完整遍历一遍语料。此外，直接在整个语料范围内跨文本合并字节，可能会产生仅在标点上不同的 token（例如 dog! 和 dog.）。这些 token 会得到完全不同的 token ID，尽管它们很可能在语义上非常相近（因为只差标点）。

To avoid this, we pre-tokenize the corpus. You can think of this as a coarse-grained tokenization over the corpus that helps us count how often pairs of characters appear. For example, the word 'text' might be a pre-token that appears 10 times. In this case, when we count how often the characters ‘t’ and ‘e’ appear next to each other, we will see that the word ‘text’ has ‘t’ and ‘e’ adjacent and we can increment their count by 10 instead of looking through the corpus. Since we’re training a byte-level BPE model, each pretoken is represented as a sequence of UTF-8 bytes.
为避免这一点，我们会先对语料做预分词。你可以把它理解为对语料进行一种粗粒度分词，从而帮助我们统计字符对的出现频率。例如，单词 'text' 可能作为一个 pre-token 出现了 10 次。在这种情况下，当我们统计字符 ‘t’ 和 ‘e’ 相邻出现的次数时，会发现单词 'text' 中存在 ‘t’ 与 ‘e’ 的相邻关系，于是就可以直接把该计数加 10，而不用反复扫描整份语料。由于我们训练的是字节级 BPE 模型，每个 pre-token 会表示为 UTF-8 字节序列。

The original BPE implementation of R. Sennrich et al. [3] pre-tokenizes by simply splitting on whitespace (i.e., s.split(" ")). This method is still found in tokenizers based on SentencePiece (for instance the Llama 1 and 2 tokenizer).
R. Sennrich 等人 [3] 最初的 BPE 实现采用的是最简单的按空白分割进行预分词（即 s.split(" ")）。这种方法在一些基于 SentencePiece 的分词器中仍然可见（例如 Llama 1 和 2 的分词器）。

Most modern tokenizers use a regex-based pre-tokenizer, a practice from GPT-2; A. Radford et al. [6]. We’ll use a slightly prettier form of the original regex, fetched from github.com/openai/tiktoken/pull/234/files:
大多数现代分词器使用基于正则表达式的预分词器，这一做法可以追溯到 GPT-2；A. Radford 等人 [6]。我们将使用原始正则表达式的一个稍作美化的版本，取自 github.com/openai/tiktoken/pull/234/files：

```python
>>> PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
```

It may be useful to interactively split some text with this pre-tokenizer to get a better sense of its behavior:
你可以交互式地用这个预分词器切分一些文本，以更直观地理解它的行为：

```python
>>> # requires `regex` package
>>> import regex as re
>>> re.findall(PAT, "some text that i'll pre-tokenize")
['some', 'text', 'that', 'i', "'ll", 'pre', '-', 'tokenize']
```

When using it in your code, however, you should use re.finditer to avoid storing the pre-tokenized words as you construct your mapping from pre-tokens to their counts.
不过在代码实现中，你应当使用 re.finditer，这样在构建从 pre-token 到其计数的映射时，就不必把所有预分词结果都存储下来。

# Compute BPE merges
# 计算 BPE 合并

Now that we’ve converted our input text into pre-tokens and represented each pre-token as a sequence of UTF-8 bytes, we can compute the BPE merges (i.e., train the BPE tokenizer). At a high level, the BPE algorithm iteratively counts every pair of bytes and identifies the pair with the highest frequency (“A”, “B”). Every occurrence of this most frequent pair (“A”, “B”) is then merged, i.e., replaced with a new token “AB”. This new merged token is added to our vocabulary; as a result, the final vocabulary after BPE training is the size of the initial vocabulary (256 in our case), plus the number of BPE merge operations performed during training. For efficiency during BPE training, we do not consider pairs that cross pre-token boundaries.2 When computing merges, deterministically break ties in pair frequency by preferring the lexicographically greater pair. For example, if the pairs (“A”, “B”), (“A”, “C”), (“B”, “ZZ”), and (“BA”, “A”) all have the highest frequency, we’d merge (“BA”, “A”):
现在我们已经把输入文本转换成了 pre-token，并把每个 pre-token 表示为 UTF-8 字节序列，接下来就可以计算 BPE 合并（即训练 BPE 分词器）了。从高层看，BPE 算法会迭代地统计每一对相邻字节，并找出频率最高的那一对（“A”, “B”）。随后，该最频繁字节对（“A”, “B”）的每一次出现都会被合并，也就是替换成一个新 token “AB”。这个新合并 token 会被加入词表；因此，BPE 训练完成后的最终词表大小等于初始词表大小（在这里是 256）加上训练过程中执行的 BPE 合并次数。为了提升 BPE 训练效率，我们不考虑跨 pre-token 边界的字节对。2 在计算合并时，如果多个字节对频率并列，则应通过选择字典序更大的字节对来确定性地打破平局。例如，如果 (“A”, “B”)、(“A”, “C”)、(“B”, “ZZ”) 和 (“BA”, “A”) 这些字节对都具有最高频率，那么我们应合并 (“BA”, “A”)：

```txt
>>> max([("A", "B"), ("A", "C"), ("B", "ZZ"), ("BA", "A")])
('BA', 'A')
```

# Special tokens
# 特殊 token

Often, some strings (e.g., <|endoftext|>) are used to encode metadata (e.g., boundaries between documents). When encoding text, it’s often desirable to treat some strings as “special tokens” that should never be split into multiple tokens (i.e., will always be preserved as a single token). For example, the end-of-sequence string <|endoftext|> should always be preserved as a single token (i.e., a single integer ID), so we know when to stop generating from the language model. These special tokens must be added to the vocabulary, so they have a corresponding fixed token ID.
很多时候，一些特定字符串（例如 <|endoftext|>）会被用来编码元数据（例如文档边界）。在编码文本时，通常希望把某些字符串视为“特殊 token”，即它们永远不应被拆分成多个 token（也就是始终保持为单个 token）。例如，序列结束字符串 <|endoftext|> 应始终被保留为一个 token（也就是一个固定的整数 ID），这样我们才能知道何时停止语言模型的生成。这些特殊 token 必须加入词表，以便拥有各自固定的 token ID。

Algorithm 1 of R. Sennrich et al. [3] contains an inefficient implementation of BPE tokenizer training (essentially following the steps that we outlined above). As a first exercise, it may be useful to implement and test this function to check your understanding.
R. Sennrich 等人 [3] 的算法 1 给出了一个低效的 BPE 分词器训练实现（本质上就是遵循我们上面描述的那些步骤）。作为第一个练习，你可以先实现并测试这个函数，以检查自己是否真正理解了流程。

# Example (bpe_example): BPE training example
# 例子（bpe_example）：BPE 训练示例

Here is a stylized example from R. Sennrich et al. [3]. Consider a corpus consisting of the following text
下面给出一个来自 R. Sennrich 等人 [3] 的风格化示例。假设语料由以下文本组成：

low low low low low lower lower widest widest widest newest newest newest newest newest newest

and the vocabulary has a special token <|endoftext|>.
并且词表中包含一个特殊 token <|endoftext|>。

# Vocabulary
# 词表

We initialize our vocabulary with our special token <|endoftext|> and the 256 byte values.
我们用特殊 token <|endoftext|> 和 256 个字节值来初始化词表。

# Pre-tokenization
# 预分词

For simplicity and to focus on the merge procedure, we assume in this example that pretokenization simply splits on whitespace. When we pre-tokenize and count, we end up with the frequency table.
为简化说明并聚焦于合并过程，本例中假设预分词只是简单地按空白切分。完成预分词和计数后，我们得到如下频率表：

{low: 5, lower: 2, widest: 3, newest: 6}

It is convenient to represent this as a dict[tuple[bytes, ...], int], e.g. {(l,o,w): 5, …}. Note that even a single byte is a bytes object in Python. There is no byte type in Python to represent a single byte, just as there is no char type in Python to represent a single character.
将其表示为 dict[tuple[bytes, ...], int] 会比较方便，例如 {(l,o,w): 5, …}。注意，在 Python 中，即使单个字节也是 bytes 对象。Python 没有专门表示“单个字节”的 byte 类型，正如它也没有表示“单个字符”的 char 类型一样。

# Merges
# 合并

We first look at every successive pair of bytes and sum the frequency of the words where they appear {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}. The pairs ('e', 's') and ('s', 't') are tied, so we take the lexicographically greater pair, ('s', 't'). We would then merge the pre-tokens so that we end up with {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}.
我们先查看每一个相邻字节对，并汇总它们所在单词的频率，得到 {lo: 7, ow: 7, we: 8, er: 2, wi: 3, id: 3, de: 3, es: 9, st: 9, ne: 6, ew: 6}。其中 ('e', 's') 和 ('s', 't') 并列，因此我们选择字典序更大的 ('s', 't')。于是合并后，pre-token 将变为 {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,e,st): 3, (n,e,w,e,st): 6}。

In the second round, we see that (e, st) is the most common pair (with a count of 9) and we would merge into {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}. Continuing this, the sequence of merges we get in the end will be ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r'].
第二轮中，我们会发现 (e, st) 是最常见的字节对（计数为 9），因此合并后得到 {(l,o,w): 5, (l,o,w,e,r): 2, (w,i,d,est): 3, (n,e,w,est): 6}。继续下去，最终得到的合并序列将是 ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e', 'ne west', 'w i', 'wi d', 'wid est', 'low e', 'lowe r']。

If we take 6 merges, we have ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e'] and our vocabulary elements would be [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne].
如果我们只取前 6 次合并，那么会得到 ['s t', 'e st', 'o w', 'l ow', 'w est', 'n e']，此时词表项将是 [<|endoftext|>, [...256 BYTE CHARS], st, est, ow, low, west, ne]。

With this vocabulary and set of merges, the word newest would tokenize as [ne, west].
在这个词表和合并规则下，单词 newest 会被分成 [ne, west]。

# 2.5 Experimenting with BPE Tokenizer Training
# 2.5 BPE 分词器训练实验

Let’s train a byte-level BPE tokenizer on the TinyStories dataset. Instructions to find / download the dataset can be found in Section 1. Before you start, we recommend taking a look at the TinyStories dataset to get a sense of what’s in the data.
现在让我们在 TinyStories 数据集上训练一个字节级 BPE 分词器。数据集的获取或下载说明见第 1 节。开始之前，我们建议你先浏览一下 TinyStories 数据集，感受一下其中的数据内容。

# Parallelizing pre-tokenization
# 并行化预分词

You will find that a major bottleneck is the pre-tokenization step. You can speed up pre-tokenization by parallelizing your code with the built-in library multiprocessing. Concretely, we recommend that in parallel implementations of pre-tokenization, you chunk the corpus while ensuring your chunk boundaries occur at the beginning of a special token. You are free to use the starter code at the following link verbatim to obtain chunk boundaries, which you can then use to distribute work across your processes:
你会发现一个主要瓶颈在于预分词步骤。你可以使用 Python 内置的 multiprocessing 库对代码进行并行化，以加速预分词。具体来说，我们建议你在并行实现预分词时，对语料进行分块，并确保每个块的边界落在特殊 token 的起始处。你可以直接使用以下链接中的 starter code 来获取这些块边界，然后据此将工作分发到多个进程：

https://github.com/stanford-cs336/assignment1-basics/blob/main/cs336_basics/pretokenization_example.py

This chunking will always be valid, since we never want to merge across document boundaries. For the purposes of the assignment, you can always split in this way. Don’t worry about the edge case of receiving a very large corpus that does not contain <|endoftext|>.
这种分块方式始终是有效的，因为我们本来就不希望跨文档边界进行合并。对本作业而言，你始终可以这样切分。不必担心遇到一个非常大的语料且其中完全不包含 <|endoftext|> 这种边界情况。

# Removing special tokens before pre-tokenization
# 在预分词前移除特殊 token

Before running pre-tokenization with the regex pattern (using re.finditer), you should strip out all special tokens from your corpus (or your chunk, if using a parallel implementation). Make sure that you split on your special tokens, so that no merging can occur across the text they delimit. For example, if you have a corpus (or chunk) like [Doc 1]<|endoftext|>[Doc 2], you should split on the special token <|endoftext|>, and pre-tokenize [Doc 1] and [Doc 2] separately, so that no merging can occur across the document boundary. In other words, special tokens define hard segmentation boundaries during training, but they should not themselves contribute to merge counts. This can be done using re.split with "|".join(special_tokens) as the delimiter (with careful use of re.escape since | may occur in the special tokens). The test test_train_bpe_special_tokens will test for this.
在使用正则模式（通过 re.finditer）进行预分词之前，你应当先从语料中移除所有特殊 token（如果采用并行实现，则是先从每个 chunk 中移除）。务必按特殊 token 进行切分，从而保证不会跨它们所界定的文本边界发生合并。例如，如果你的语料（或某个 chunk）形如 [Doc 1]<|endoftext|>[Doc 2]，那么你应当先按特殊 token <|endoftext|> 切开，再分别对 [Doc 1] 和 [Doc 2] 做预分词，这样就不会跨文档边界发生合并。换句话说，特殊 token 在训练过程中定义了硬性分段边界，但它们本身不应参与 merge 统计。可以使用 re.split，并以 "|".join(special_tokens) 作为分隔符（注意由于特殊 token 中可能包含 |，需要小心使用 re.escape）。测试 test_train_bpe_special_tokens 会检查这一点。

# Optimizing the merging step
# 优化合并步骤

The naïve implementation of BPE training in the stylized example above is slow because for every merge, it iterates over all byte pairs to identify the most frequent pair. However, the only pair counts that change after each merge are those that overlap with the merged pair. Thus, BPE training speed can be improved by indexing the counts of all pairs and incrementally updating these counts, rather than explicitly iterating over each pair of bytes to count pair frequencies. You can get significant speedups with this caching procedure, though we note that the merging part of BPE training is not parallelizable in Python.
上面风格化示例中的朴素 BPE 训练实现之所以很慢，是因为每一次 merge 都要遍历所有字节对以找出频率最高者。然而，每次 merge 之后，真正会变化的只有那些与被合并字节对发生重叠的 pair 计数。因此，你可以为所有 pair 的计数建立索引并做增量更新，而不是每次都显式遍历所有字节对重新统计，这样可以显著提高 BPE 训练速度。通过这种缓存式做法可以获得明显加速，不过需要注意，BPE 训练中的 merge 部分在 Python 中并不容易并行化。

# Low-Resource Tip: Profiling
# 低资源提示：性能分析

You should use profiling tools like cProfile or py-spy to identify the bottlenecks in your implementation, and focus on optimizing those.
你应当使用 cProfile 或 py-spy 之类的性能分析工具来定位实现中的瓶颈，并重点优化这些部分。

# Low-Resource Tip: "Downscaling"
# 低资源提示：“缩小规模”

Instead of jumping to training your tokenizer on the full TinyStories dataset, we recommend you first train on a small subset of the data: a “debug dataset”. For example, you could train your tokenizer on the TinyStories validation set instead, which is 22K documents instead of 2.12M. This illustrates a general strategy of downscaling whenever possible to speed up development: for example, using smaller datasets, smaller model sizes, etc. Choosing the size of the debug dataset or hyperparameter config requires careful consideration: you want your debug set to be large enough to have the same bottlenecks as the full configuration (so that the optimizations you make will generalize), but not so big that it takes forever to run.
我们建议你不要一开始就直接在完整的 TinyStories 数据集上训练分词器，而是先在一个较小的数据子集上训练，也就是一个“debug dataset”。例如，你可以先在 TinyStories 的验证集上训练分词器，该验证集包含 2.2 万个文档，而不是 212 万个。这个建议体现了一种普适的开发策略：在可能的情况下尽量缩小规模以加快开发速度，例如使用更小的数据集、更小的模型等。调试集大小或超参数配置的选择需要仔细权衡：它既要足够大，以便呈现与完整配置相同的瓶颈（这样你的优化才有迁移价值），又不能大到运行时间过长。

# Problem (train_bpe): BPE Tokenizer Training (15 points)
# 题目（train_bpe）：BPE 分词器训练（15 分）

Deliverable: Write a function that, given a path to an input text file, trains a (byte-level) BPE tokenizer. Your BPE training function should handle (at least) the following input parameters:
提交内容：编写一个函数，给定输入文本文件路径后，训练一个（字节级）BPE 分词器。你的 BPE 训练函数至少应支持以下输入参数：

# Input
# 输入

input_path: str  Path to a text file with BPE tokenizer training data.
input_path: str  包含 BPE 分词器训练数据的文本文件路径。

vocab_size: int  A positive integer that defines the maximum final vocabulary size (including the initial byte vocabulary, vocabulary items produced from merging, and any special tokens).
vocab_size: int  一个正整数，用于定义最终词表的最大大小（包括初始字节词表、合并生成的词表项以及所有特殊 token）。

special_tokens: list[str]  A list of strings to add to the vocabulary. During training, treat them as hard boundaries that prevent merges across their spans, but do not include them when computing merge statistics.
special_tokens: list[str]  一个要加入词表的字符串列表。训练时，应将它们视为阻止跨越其跨度发生合并的硬边界，但在计算 merge 统计时不应将它们自身计入。

Your BPE training function should return the resulting vocabulary and merges:
你的 BPE 训练函数应返回最终词表和 merge 结果：

# Output
# 输出

vocab: dict[int, bytes]  The tokenizer vocabulary, a mapping from int (token ID in the vocabulary) to bytes (token bytes).
vocab: dict[int, bytes]  分词器词表，即从 int（词表中的 token ID）到 bytes（token 对应字节串）的映射。

merges: list[tuple[bytes, bytes]]  A list of BPE merges produced from training. Each list item is a tuple of bytes (<token1>, <token2>), representing that <token1> was merged with <token2>. The merges should be ordered by order of creation.
merges: list[tuple[bytes, bytes]]  训练得到的 BPE merge 列表。每个列表项是一个 bytes 二元组（<token1>, <token2>），表示将 <token1> 与 <token2> 合并。merge 列表应按照创建顺序排列。

To test your BPE training function against our provided tests, you will first need to implement the test adapter at [adapters.run_train_bpe]. Then, run uv run pytest tests/test_train_bpe.py. Your implementation should be able to pass all tests. Optionally (this could be a large time-investment), you can implement the key parts of your training method using some systems language, for instance C++ (consider cppyy or nanobind) or Rust (using PyO3). If you do this, be aware of which operations require copying vs reading directly from Python memory, and make sure to leave build instructions, or make sure it builds using only pyproject.toml. Also note that the GPT-2 regex is not well-supported in most regex engines and will be too slow in most that do. We have verified that Oniguruma is reasonably fast and supports negative lookahead, but the regex package in Python is, if anything, even faster.
若要用我们提供的测试来验证你的 BPE 训练函数，你首先需要实现测试适配器 [adapters.run_train_bpe]。然后运行 uv run pytest tests/test_train_bpe.py。你的实现应当能够通过全部测试。可选地（但这可能投入较大时间），你也可以用某种系统语言实现训练方法的关键部分，例如 C++（可考虑 cppyy 或 nanobind）或 Rust（通过 PyO3）。如果这样做，请注意哪些操作需要复制数据，哪些可以直接读取 Python 内存，并确保提供构建说明，或者确保仅凭 pyproject.toml 即可完成构建。另外要注意，GPT-2 使用的正则在多数正则引擎中支持不佳，即便支持也通常较慢。我们验证过 Oniguruma 速度尚可并支持 negative lookahead，但 Python 的 regex 包甚至可能更快。

# Problem (train_bpe_tinystories): BPE Training on TinyStories (2 points)
# 题目（train_bpe_tinystories）：在 TinyStories 上训练 BPE（2 分）

(a) Train a byte-level BPE tokenizer on the TinyStories dataset, using a maximum vocabulary size of 10,000. Make sure to add the TinyStories <|endoftext|> special token to the vocabulary. Serialize the resulting vocabulary and merges to disk for further inspection. How much time and memory did training take? What is the longest token in the vocabulary? Does it make sense?
（a）在 TinyStories 数据集上训练一个字节级 BPE 分词器，最大词表大小设为 10,000。请确保将 TinyStories 中的 <|endoftext|> 特殊 token 加入词表。将得到的词表和 merges 序列化到磁盘以便进一步检查。训练耗时和内存占用是多少？词表中最长的 token 是什么？它合理吗？

Resource requirements: ≤ 30 minutes (no GPUs), ≤ 30 GB RAM
资源要求：≤ 30 分钟（不使用 GPU），≤ 30 GB 内存

Hint  You should be able to get under 2 minutes for BPE training using multiprocessing during pre-tokenization and the following two facts:
提示：如果你在预分词阶段使用 multiprocessing，并利用以下两个事实，BPE 训练时间应能压到 2 分钟以内：

(a) The <|endoftext|> token delimits documents in the data files.
（a）<|endoftext|> token 在数据文件中用于划分文档边界。

(b) The <|endoftext|> token is handled as a special case before the BPE merges are applied.
（b）在应用 BPE merges 之前，<|endoftext|> token 已被作为特殊情况处理。

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

(b) Profile your code. What part of the tokenizer training process takes the most time?
（b）对你的代码进行性能分析。分词器训练过程中哪一部分最耗时？

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

Next, we’ll try training a byte-level BPE tokenizer on the OpenWebText dataset. As before, we recommend taking a look at the dataset to better understand its contents.
接下来，我们会尝试在 OpenWebText 数据集上训练一个字节级 BPE 分词器。和前面一样，我们建议你先浏览一下数据集，以更好地理解其内容。

# Problem (train_bpe_expts_owt): BPE Training on OpenWebText (2 points)
# 题目（train_bpe_expts_owt）：在 OpenWebText 上训练 BPE（2 分）

(a) Train a byte-level BPE tokenizer on the OpenWebText dataset, using a maximum vocabulary size of 32,000. Serialize the resulting vocabulary and merges to disk for further inspection. What is the longest token in the vocabulary? Does it make sense?
（a）在 OpenWebText 数据集上训练一个字节级 BPE 分词器，最大词表大小设为 32,000。将得到的词表和 merges 序列化到磁盘以便进一步检查。词表中最长的 token 是什么？它合理吗？

Resource requirements: ≤ 12 hours (no GPUs), ≤ 100 GB RAM
资源要求：≤ 12 小时（不使用 GPU），≤ 100 GB 内存

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

(b) Compare and contrast the tokenizer that you get training on TinyStories versus OpenWebText.
（b）比较并对比你在 TinyStories 上训练得到的分词器和在 OpenWebText 上训练得到的分词器。

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

# 2.6 BPE Tokenizer: Encoding and Decoding
# 2.6 BPE 分词器：编码与解码

In the previous part of the assignment, we implemented a function to train a BPE tokenizer on input text to obtain a tokenizer vocabulary and a list of BPE merges. Now, we will implement a BPE tokenizer that loads a provided vocabulary and list of merges and uses them to encode and decode text to/from token IDs.
在作业的上一部分中，我们实现了一个在输入文本上训练 BPE 分词器的函数，从而得到分词器词表和 BPE merge 列表。现在，我们将实现一个 BPE 分词器，它会加载给定的词表和 merge 列表，并使用它们在文本和 token ID 之间进行编码与解码。

# 2.6.1 Encoding text
# 2.6.1 文本编码

The process of encoding text by BPE mirrors how we train the BPE vocabulary. There are a few major steps.
使用 BPE 对文本进行编码的过程，与训练 BPE 词表时的流程是对应的。主要包含以下几个步骤。

Step 1: Pre-tokenize. We first pre-tokenize the sequence and represent each pre-token as a sequence of UTF-8 bytes, just as we did in BPE training. We will be merging these bytes within each pre-token into vocabulary elements, handling each pre-token independently (no merges across pre-token boundaries).
步骤 1：预分词。我们首先对序列进行预分词，并像训练 BPE 时那样，把每个 pre-token 表示为 UTF-8 字节序列。随后，我们会在每个 pre-token 内部将这些字节合并成词表项，并且每个 pre-token 独立处理（不会跨 pre-token 边界进行合并）。

Step 2: Apply the merges. We then take the sequence of vocabulary element merges created during BPE training, and apply it to our pre-tokens in the same order of creation.
步骤 2：应用 merges。然后，我们取出在 BPE 训练过程中得到的词表项合并序列，并按照生成时的顺序将其应用到我们的 pre-token 上。

# Example (bpe_encoding): BPE encoding example
# 例子（bpe_encoding）：BPE 编码示例

For example, suppose our input string is 'the cat ate', our vocabulary is {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}, and our learned merges are [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]. First, our pre-tokenizer would split this string into ['the', ' cat', ' ate']. Then, we’ll look at each pre-token and apply the BPE merges.
例如，假设输入字符串是 'the cat ate'，词表为 {0: b' ', 1: b'a', 2: b'c', 3: b'e', 4: b'h', 5: b't', 6: b'th', 7: b' c', 8: b' a', 9: b'the', 10: b' at'}，而学到的 merges 为 [(b't', b'h'), (b' ', b'c'), (b' ', b'a'), (b'th', b'e'), (b' a', b't')]。首先，预分词器会把该字符串切分为 ['the', ' cat', ' ate']。然后，我们会逐个查看每个 pre-token 并应用 BPE merges。

The first pre-token 'the' is initially represented as [b't', b'h', b'e']. Looking at our list of merges, we identify the first applicable merge to be (b't', b'h'), and use that to transform the pre-token into [b'th', b'e']. Then, we go back to the list of merges and identify the next applicable merge to be (b'th', b'e'), which transforms the pre-token into [b'the']. Finally, looking back at the list of merges, we see that there are no more that apply to the string (since the entire pre-token has been merged into a single token), so we are done applying the BPE merges. The corresponding integer sequence is [9].
第一个 pre-token 'the' 初始表示为 [b't', b'h', b'e']。查看 merge 列表后，我们发现第一个可应用的 merge 是 (b't', b'h')，于是把该 pre-token 变成 [b'th', b'e']。接着重新从 merge 列表中查找，下一个可应用的 merge 是 (b'th', b'e')，于是 pre-token 变成 [b'the']。最后再次查看 merge 列表，会发现没有更多适用于该字符串的 merge 了（因为整个 pre-token 已经合并成单个 token），因此合并结束。对应的整数序列是 [9]。

Repeating this process for the remaining pre-tokens, we see that the pre-token ' cat' is represented as [b' c', b'a', b't'] after applying the BPE merges, which becomes the integer sequence [7, 1, 5]. The final pre-token ' ate' is [b' at', b'e'] after applying the BPE merges, which becomes the integer sequence [10, 3]. Thus, the final result of encoding our input string is [9, 7, 1, 5, 10, 3].
对剩余的 pre-token 重复这一过程，可以看到 pre-token ' cat' 在应用 BPE merges 后表示为 [b' c', b'a', b't']，对应整数序列 [7, 1, 5]。最后一个 pre-token ' ate' 在应用 BPE merges 后表示为 [b' at', b'e']，对应整数序列 [10, 3]。因此，输入字符串最终编码结果是 [9, 7, 1, 5, 10, 3]。

# Special tokens
# 特殊 token

Your tokenizer should be able to properly handle user-defined special tokens when encoding text (provided when constructing the tokenizer).
你的分词器在编码文本时应能正确处理用户定义的特殊 token（在构造分词器时提供）。

# Memory considerations
# 内存考虑

Suppose we want to tokenize a large text file that we cannot fit in memory. To efficiently tokenize this large file (or any other stream of data), we need to break it up into manageable chunks and process each chunk in turn, so that the memory complexity is constant as opposed to linear in the size of the text. In doing so, we need to make sure that a token doesn’t cross chunk boundaries, else we’ll get a different tokenization than the naïve method of tokenizing the entire sequence in-memory.
假设我们想对一个无法完整放入内存的大型文本文件进行分词。为了高效地对这个大文件（或任何其他数据流）进行分词，我们需要把它拆成可管理的小块，逐块处理，从而使内存复杂度保持为常数级，而不是随着文本大小线性增长。在这样做时，我们必须确保 token 不会跨越 chunk 边界，否则分词结果就会与将整个序列一次性读入内存后进行分词的朴素方法不同。

# 2.6.2 Decoding text
# 2.6.2 文本解码

To decode a sequence of integer token IDs back to raw text, we can simply look up each ID’s corresponding entries in the vocabulary (a byte sequence), concatenate them together, and then decode the bytes to a Unicode string. Note that input IDs are not guaranteed to map to valid Unicode strings (since a user could input any sequence of integer IDs). In the case that the input token IDs do not produce a valid Unicode string, you should replace the malformed bytes with the official Unicode replacement character U+FFFD.3 The errors argument of bytes.decode controls how Unicode decoding errors are handled, and using errors='replace' will automatically replace malformed data with the replacement marker.
要把整数 token ID 序列解码回原始文本，我们只需要查出每个 ID 在词表中对应的条目（一个字节序列），将它们拼接起来，然后把字节解码为 Unicode 字符串即可。注意，输入的 ID 并不保证一定能映射成合法的 Unicode 字符串（因为用户可能输入任意整数 ID 序列）。如果输入 token ID 不能组成有效的 Unicode 字符串，你应当用官方 Unicode 替换字符 U+FFFD 来替换这些畸形字节。3 bytes.decode 的 errors 参数控制 Unicode 解码错误的处理方式，使用 errors='replace' 会自动用替换字符替代损坏数据。

# Problem (tokenizer): Implementing the tokenizer (15 points)
# 题目（tokenizer）：实现分词器（15 分）

Deliverable: Implement a Tokenizer class that, given a vocabulary and a list of merges, encodes text into integer IDs and decodes integer IDs into text. Your tokenizer should also support user-provided special tokens (appending them to the vocabulary if they aren’t already there). We recommend the following interface:
提交内容：实现一个 Tokenizer 类，给定词表和 merge 列表后，能够把文本编码为整数 ID，并把整数 ID 解码回文本。你的分词器还应支持用户提供的特殊 token（如果它们尚不在词表中，则追加到词表里）。我们推荐如下接口：

def __init__(self, vocab, merges, special_tokens=None) Construct a tokenizer from a given vocabulary, list of merges, and (optionally) a list of special tokens. This function should accept the following parameters:
def __init__(self, vocab, merges, special_tokens=None) 从给定的词表、merge 列表以及（可选的）特殊 token 列表构造一个分词器。该函数应接收以下参数：

```python
vocab: dict[int, bytes]
merges: list[tuple(bytes, bytes)]
special_tokens: list[str] | None = None
```

def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) Class method that constructs and returns a Tokenizer from a serialized vocabulary and list of merges (in the same format that your BPE training code output) and (optionally) a list of special tokens. This method should accept the following additional parameters:
def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None) 类方法，从序列化的词表与 merge 列表（格式应与 BPE 训练代码输出一致）以及（可选的）特殊 token 列表中构造并返回一个 Tokenizer。该方法还应接收以下参数：

```python
vocab_filepath: str
merges_filepath: str
special_tokens: list[str] | None = None
```

def encode(self, text: str) -> list[int] Encode an input text into a sequence of token IDs.
def encode(self, text: str) -> list[int] 将输入文本编码为 token ID 序列。

def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] Given an iterable of strings (e.g., a Python file handle), return a generator that lazily yields token IDs. This is required for memory-efficient tokenization of large files that we cannot directly load into memory.
def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int] 给定一个字符串可迭代对象（例如 Python 文件句柄），返回一个惰性生成 token ID 的生成器。这对于无法直接加载到内存中的大文件实现高内存效率分词是必须的。

def decode(self, ids: list[int]) -> str Decode a sequence of token IDs into text.
def decode(self, ids: list[int]) -> str 将 token ID 序列解码为文本。

To test your Tokenizer against our provided tests, you will first need to implement the test adapter at [adapters.get_tokenizer]. Then, run uv run pytest tests/test_tokenizer.py. Your implementation should be able to pass all tests.
若要使用我们提供的测试来验证你的 Tokenizer，你首先需要实现测试适配器 [adapters.get_tokenizer]。然后运行 uv run pytest tests/test_tokenizer.py。你的实现应当能够通过全部测试。

# 2.7 Experiments
# 2.7 实验

# Problem (tokenizer_experiments): Experiments with tokenizers (4 points)
# 题目（tokenizer_experiments）：分词器实验（4 分）

(a) Sample 10 documents from TinyStories and OpenWebText. Using your previously-trained TinyStories and OpenWebText tokenizers (10K and 32K vocabulary size, respectively), encode these sampled documents into integer IDs. What is each tokenizer’s compression ratio (bytes/token)?
（a）从 TinyStories 和 OpenWebText 中各抽样 10 个文档。使用你之前训练好的 TinyStories 和 OpenWebText 分词器（词表大小分别为 10K 和 32K）将这些样本文档编码为整数 ID。每个分词器的压缩率（bytes/token）是多少？

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

(b) What happens if you tokenize your OpenWebText sample with the TinyStories tokenizer? Compare the compression ratio and/or qualitatively describe what happens.
（b）如果你用 TinyStories 分词器去分词 OpenWebText 的样本，会发生什么？请比较压缩率，并/或定性描述结果。

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

(c) Estimate the throughput of your tokenizer (e.g., in bytes/second). How long would it take to tokenize the Pile dataset (825GB of text)?
（c）估计你的分词器吞吐量（例如 bytes/second）。对 Pile 数据集（825GB 文本）进行分词大约需要多久？

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

(d) Using your TinyStories and OpenWebText tokenizers, encode the respective training and development datasets into a sequence of integer token IDs. We’ll use this later to train our language model. We recommend serializing the token IDs as a NumPy array of datatype uint16. Why is uint16 an appropriate choice?
（d）使用你的 TinyStories 和 OpenWebText 分词器，将各自的训练集和开发集编码为整数 token ID 序列。我们之后会用这些数据来训练语言模型。我们建议你将 token ID 序列序列化为 dtype 为 uint16 的 NumPy 数组。为什么 uint16 是一个合适的选择？

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

# 3 Transformer Language Model Architecture
# 3 Transformer 语言模型结构

A language model takes as input a batched sequence of integer token IDs (i.e., torch.Tensor of shape (batch_size, sequence_length)), and returns a (batched) normalized probability distribution over the vocabulary (i.e., a PyTorch Tensor of shape (batch_size, sequence_length, vocab_size)), where the predicted distribution is over the next word for each input token. When training the language model, we use these next-word predictions to calculate the cross-entropy loss between the actual next word and the predicted next word. When generating text from the language model during inference, we take the predicted next-word distribution from the final time step (i.e., the last item in the sequence) to generate the next token in the sequence (e.g., by taking the token with the highest probability, sampling from the distribution, etc.), add the generated token to the input sequence, and repeat.
语言模型的输入是一个按 batch 组织的整数 token ID 序列（即形状为 (batch_size, sequence_length) 的 torch.Tensor），输出则是词表上的（按 batch 组织的）归一化概率分布（即形状为 (batch_size, sequence_length, vocab_size) 的 PyTorch Tensor），其中每个输入 token 对应一个关于“下一个词”的预测分布。训练语言模型时，我们使用这些下一个词的预测来计算真实下一个词与预测下一个词之间的交叉熵损失。在推理阶段使用语言模型生成文本时，我们从最后一个时间步（也就是序列中的最后一项）对应的下一个词预测分布中生成下一个 token（例如取概率最大的 token、从分布中采样等），然后把生成出来的 token 追加到输入序列中并重复这个过程。

In this part of the assignment, you will build this Transformer language model from scratch. We will begin with a high-level description of the model before progressively detailing the individual components.
在本部分作业中，你将从零开始构建这个 Transformer 语言模型。我们会先给出模型的高层描述，再逐步细化到各个组件。

# 3.1 Transformer LM
# 3.1 Transformer 语言模型

Given a sequence of token IDs, the Transformer language model uses an input embedding to convert token IDs to dense vectors, passes the embedded tokens through num_layers Transformer blocks, and then applies a learned linear projection (the “output embedding” or “LM head”) to produce the predicted next-token logits. See Figure 1 for a schematic representation.
给定一个 token ID 序列，Transformer 语言模型首先使用输入 embedding 将 token ID 转换为稠密向量，然后将这些嵌入向量送入 num_layers 个 Transformer block，最后通过一个可学习的线性投影（即“输出 embedding”或“LM head”）生成预测下一个 token 的 logits。其示意图见图 1。

![](images/17a61358860b31b6f526fd0227e3aec238032da0838ba87a63c2baeffdffebf9.jpg)

Figure 1: An overview of our Transformer language model.
图 1：我们的 Transformer 语言模型概览。

![](images/2ccb1ab0941dff0bdc0b16c6f0707103784c5365cad86286f1316373415bcdbc.jpg)

Figure 2: A pre-norm Transformer block.
图 2：一个 pre-norm Transformer block。

# Token Embeddings
# Token 嵌入

In the very first step, the Transformer embeds the (batched) sequence of token IDs into a sequence of vectors containing information on the token identity (red blocks in Figure 1).
在第一步中，Transformer 会将（按 batch 组织的）token ID 序列嵌入为一串向量，这些向量包含 token 身份信息（见图 1 中的红色模块）。

More specifically, given a sequence of token IDs, the Transformer language model uses a token embedding layer to produce a sequence of vectors. Each embedding layer takes in a tensor of integers of shape (batch_size, sequence_length) and produces a sequence of vectors of shape (batch_size, sequence_length, d_model).
更具体地说，给定一个 token ID 序列，Transformer 语言模型会使用 token embedding 层生成一个向量序列。每个 embedding 层接收一个形状为 (batch_size, sequence_length) 的整型张量，并输出形状为 (batch_size, sequence_length, d_model) 的向量序列。

# Pre-norm Transformer Block
# Pre-norm Transformer Block

After embedding, the activations are processed by several identically structured neural net layers. A standard decoder-only Transformer language model consists of num_layers identical layers (commonly called Transformer “blocks”). Each Transformer block takes in an input of shape (batch_size, sequence_length, d_model) and returns an output of shape (batch_size, sequence_length, d_model). Each block aggregates information across the sequence (via self-attention) and non-linearly transforms it (via the feed-forward layers).
在 embedding 之后，激活值会经过若干个结构相同的神经网络层。一个标准的 decoder-only Transformer 语言模型由 num_layers 个相同的层组成（通常称为 Transformer “blocks”）。每个 Transformer block 接受形状为 (batch_size, sequence_length, d_model) 的输入，并返回相同形状的输出。每个 block 一方面通过 self-attention 在序列范围内聚合信息，另一方面通过前馈层进行非线性变换。

After num_layers Transformer blocks, we will take the final activations and turn them into a distribution over the vocabulary.
经过 num_layers 个 Transformer block 后，我们会将最终激活值转换成词表上的一个分布。

We will implement the “pre-norm” Transformer block (detailed in Section 3.4), which additionally requires the use of layer normalization (detailed below) after the final Transformer block to ensure its outputs are properly scaled.
我们将实现“pre-norm” Transformer block（详见第 3.4 节），它还要求在最后一个 Transformer block 之后再使用一次 layer normalization（后文会详细说明），以确保输出尺度合适。

After this normalization, we will use a standard learned linear transformation to convert the output of the Transformer blocks into predicted next-token logits (see, e.g., A. Radford et al. [7] equation 2).
在该归一化之后，我们会使用一个标准的可学习线性变换，把 Transformer blocks 的输出转换成预测下一个 token 的 logits（例如可参考 A. Radford et al. [7] 的公式 2）。

# 3.2 Remark: Batching, Einsum and Efficient Computation
# 3.2 说明：Batching、Einsum 与高效计算

Throughout the Transformer, we will be performing the same computation applied to many batch-like inputs. Here are a few examples:
在整个 Transformer 中，我们会对许多具有 batch 特征的输入重复执行同样的计算。下面是几个例子：

• Elements of a batch: we apply the same Transformer forward operation on each batch element.
• Batch 中的各元素：我们会对每个 batch 元素应用相同的 Transformer 前向计算。

• Sequence length: the “position-wise” operations like RMSNorm and feed-forward operate identically on each position of a sequence.
• 序列长度维：像 RMSNorm 和前馈网络这类“逐位置”操作，会在序列的每个位置上以相同方式执行。

• Attention heads: the attention operation is batched across attention heads in a “multi-headed” attention operation.
• Attention 头：在多头注意力中，attention 操作会沿着注意力头这一维进行批处理。

It is useful to have an ergonomic way of performing such operations in a way that fully utilizes the GPU, and is easy to read and understand. Many PyTorch operations can take in excess “batch-like” dimensions at the start of a tensor and repeat/broadcast the operation across these dimensions efficiently.
我们需要一种既易于书写和理解、又能充分利用 GPU 的方式来实现这类操作。许多 PyTorch 操作都可以在张量开头接受额外的“batch-like”维度，并高效地在这些维度上重复或广播计算。

For instance, say we are doing a position-wise, batched operation. We have a “data tensor” D of shape (batch_size, sequence_length, d_model), and we would like to do a batched vector-matrix multiply against a matrix A of shape (d_model, d_model). In this case, D @ A will do a batched matrix multiply, which is an efficient primitive in PyTorch, where the (batch_size, sequence_length) dimensions are batched over.
例如，假设我们在做一个逐位置、按 batch 处理的操作。我们有一个形状为 (batch_size, sequence_length, d_model) 的“数据张量” D，并希望把它与一个形状为 (d_model, d_model) 的矩阵 A 做批量向量-矩阵乘法。在这种情况下，D @ A 就会执行 batched matrix multiply，这是 PyTorch 中一个高效的基础操作，其中 (batch_size, sequence_length) 两个维度会被当作 batch 维度。

Because of this, it is helpful to assume that your functions may be given additional batch-like dimensions and to keep those dimensions at the start of the PyTorch shape. To organize tensors so they can be batched in this manner, they might need to be shaped using many steps of view, reshape and transpose. This can be a bit of a pain, and it often gets hard to read what the code is doing and what the shapes of your tensors are.
正因如此，最好假设你的函数可能会收到额外的 batch-like 维度，并把这些维度始终放在 PyTorch 张量形状的前面。为了让张量能够以这种方式被批处理，有时需要通过多次 view、reshape 和 transpose 来整理张量形状。这会比较麻烦，也常常导致代码难读，让人不容易看清它到底在做什么、张量形状又是什么。

A more ergonomic option is to use einsum notation within torch.einsum, or rather use framework-agnostic libraries like einops or einx. The two key ops are einsum, which can do tensor contractions with arbitrary dimensions of input tensors, and rearrange, which can reorder, concatenate, and split arbitrary dimensions. It turns out almost all operations in machine learning are some combination of dimension juggling and tensor contraction with the occasional (usually pointwise) nonlinear function. This means that a lot of your code can be more readable and flexible when using einsum notation.
更方便的做法是使用 torch.einsum 中的 einsum 记法，或者更推荐使用与框架无关的库，如 einops 或 einx。两个关键操作分别是 einsum，它可以对任意维度的输入张量执行张量收缩；以及 rearrange，它可以重排、拼接和拆分任意维度。事实上，机器学习中的几乎所有操作都可以看作“维度整理 + 张量收缩 + 偶尔出现的（通常是逐点）非线性函数”的组合。因此，使用 einsum 记法往往能让代码更易读、更灵活。

We strongly recommend learning and using einsum notation for the class. Students who have not been exposed to einsum notation before should use einops (docs here), and students who are already comfortable with einops should learn the more general einx (here).4 Both packages are already installed in the environment we’ve supplied.
我们强烈建议你在这门课中学习并使用 einsum 记法。之前没有接触过 einsum 的同学应使用 einops（文档见此），而已经熟悉 einops 的同学则可以学习更通用的 einx（见此）。4 这两个包都已经安装在我们提供的环境中。

Here we give some examples of how einsum notation can be used. These are a supplement to the documentation for einops, which you should read first.
下面我们给出一些使用 einsum 记法的示例。这些内容是对 einops 文档的补充，而你应先阅读官方文档。

Example (einstein_example1): Batched matrix multiplication with einops.einsum
例子（einstein_example1）：使用 einops.einsum 的批量矩阵乘法

```python
import torch
from einops import rearrange, einsum

## Basic implementation
Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?

## Einsum is self-documenting and robust
#
D A -> Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

Example (einstein_example2): Broadcasted operations with einops.rearrange
例子（einstein_example2）：使用 einops.rearrange 的广播操作

```python
We have a batch of images, and for each image we want to generate 10 dimmed versions based on some scaling factor:

images = torch.randn(64, 128, 128, 3)  # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## Or in one go:
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

Example (einstein_example3): Pixel mixing with einops.rearrange
例子（einstein_example3）：使用 einops.rearrange 的像素混合

```txt
Suppose we have a batch of images represented as a tensor of shape (batch, height, width, channel), and we want to perform a linear transformation across all pixels of the image, but this transformation should happen independently for each channel. Our linear transformation is represented as a matrix B of shape (height * width, height * width).

channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)

Instead, using einops:

height = width = 32

## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height = height, width = width
)

Or, if you're feeling crazy: all in one go using einx.dot (einx equivalent of einops.einsum)

height = width = 32

channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in = width, col_out = width
)
```

The first implementation here could be improved by placing comments before and after to indicate what the input and output shapes are, but this is clunky and susceptible to bugs. With einsum notation, documentation is implementation!
这里第一种实现方式可以通过在前后加注释来说明输入输出形状，但那样写起来既笨拙又容易出错。使用 einsum 记法时，文档本身就是实现！

Einsum notation can handle arbitrary input batching dimensions, but also has the key benefit of being self-documenting. It’s much clearer what the relevant shapes of your input and output tensors are in code that uses einsum notation. For the remaining tensors, you can consider using Tensor type hints, for instance using the jaxtyping library (not specific to JAX).
Einsum 记法可以处理任意的输入 batch 维度，同时还具备一个关键优势：它具有自说明性。使用 einsum 记法的代码能更清晰地展示输入输出张量的关键形状。对于其他张量，你还可以考虑使用 Tensor 类型标注，例如通过 jaxtyping 库（它并不局限于 JAX）。

We will talk more about the performance implications of using einsum notation in assignment 2, but for now know that they’re almost always better than the alternative!
我们会在作业 2 中进一步讨论使用 einsum 记法的性能影响，但目前你只需要知道，它几乎总是优于其他替代写法。

# 3.2.1 Mathematical Notation and Memory Ordering
# 3.2.1 数学记号与内存布局顺序

Many machine learning papers use row vectors in their notation, which result in representations that mesh well with the row-major memory ordering used by default in NumPy and PyTorch. With row vectors, a linear transformation looks like
很多机器学习论文在记号中使用行向量，这与 NumPy 和 PyTorch 默认采用的 row-major 内存布局非常契合。使用行向量时，线性变换写作：

$$
y = x W ^ {\top}, \tag {1}
$$

for row-major $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ and row-vector $x \in \mathbb { R } ^ { 1 \times d _ { \mathrm { i n } } }$ . Notice that this lets us batch inputs by increasing the outermost dimension of x, meaning we can substitute vector input x for matrix input $X \in \mathbb { R } ^ { \mathrm { b a t c h } \times d _ { \mathrm { i n } } }$ .
其中，$W \in \mathbb { R } ^ { d _ { \mathrm { out } } \times d _ { \mathrm { in } } }$ 按 row-major 存储，$x \in \mathbb { R } ^ { 1 \times d _ { \mathrm { in } } }$ 是行向量。注意，这样可以通过增加 x 的最外层维度来对输入做 batch 化，也就是把向量输入 x 替换为矩阵输入 $X \in \mathbb { R } ^ { \mathrm { batch } \times d _ { \mathrm { in } } }$。

In linear algebra it’s generally more common to use column vectors, where linear transformations look like
而在线性代数中，更常见的是使用列向量，此时线性变换写作：

$$
y = W x, \tag {2}
$$

given a row-major $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d _ { \mathrm { i n } } }$ and column-vector $x \in \mathbb { R } ^ { d _ { \mathrm { i n } } }$ . To batch the input in this setting, the batch dimension to x would have to come last, so x would need to be replaced with a matrix $\tilde { X } \in \mathbb { R } ^ { d _ { \mathrm { i n } } \times \mathrm { batch } }$.
其中，$W \in \mathbb { R } ^ { d _ { \mathrm { out } } \times d _ { \mathrm { in } } }$ 仍按 row-major 存储，而 $x \in \mathbb { R } ^ { d _ { \mathrm { in } } }$ 是列向量。在这种设定下，如果要对输入做 batch 化，那么 batch 维必须放在 x 的最后，所以 x 需要被替换为一个矩阵 $\tilde { X } \in \mathbb { R } ^ { d _ { \mathrm { in } } \times \mathrm { batch } }$。

We will use mostly column vectors for mathematical notation in this assignment, since math generally follows this notation. You should keep in mind that if you want to use plain matrix multiplication notation, you will have to apply matrices with a transpose as seen in the row vector convention in Equation 1, since PyTorch uses row-major memory ordering. If you use einsum for your linear algebra operations, this should be a non-issue as long as you label your axes correctly. As an aside, it’s worth noting that other languages/linear algebra packages like Matlab, Julia and Fortran all use column-major memory ordering, meaning batching dimensions come last, but Python and related packages have adopted the C-standard of row-major ordering.
在本作业中，我们在数学记号上会主要使用列向量，因为数学中通常遵循这种记法。你需要记住，如果你想直接使用普通矩阵乘法记号，那么由于 PyTorch 使用 row-major 内存布局，你在代码里就往往需要像公式 1 中的行向量约定那样对矩阵取转置。如果你使用 einsum 来实现线性代数操作，那么只要轴标签标对了，这基本不会成为问题。顺便一提，Matlab、Julia 和 Fortran 等其他语言或线性代数包都使用 column-major 内存布局，也就是 batch 维通常在最后；而 Python 及相关包则沿用了 C 标准中的 row-major 布局。

# 3.3 Basic Building Blocks: Linear and Embedding Modules
# 3.3 基础构件：Linear 与 Embedding 模块

# 3.3.1 Parameter Initialization
# 3.3.1 参数初始化

Training neural networks effectively often requires careful initialization of the model parameters—bad initializations can lead to undesirable behavior such as vanishing or exploding gradients. Pre-norm transformers are unusually robust to initializations, but they can still have a significant impact on training speed and convergence. Since this assignment is already long, we will save the details for assignment 3, and instead give you some approximate initializations that should work well for most cases. For now, use:
要高效训练神经网络，通常需要仔细初始化模型参数；糟糕的初始化可能导致梯度消失或爆炸等不良现象。Pre-norm Transformer 对初始化异常稳健，但初始化仍会对训练速度和收敛性产生显著影响。由于本作业已经很长，我们会把更详细的内容留到作业 3 中，这里只给出一些在大多数情况下都应有效的近似初始化方案。目前，请使用：

• Linear weights: $\mathcal { N } \Big ( \mu = 0 , \sigma ^ { 2 } = \frac { 2 } { d _ { \mathrm { i n } } + d _ { \mathrm { o u t } } } \Big )$ truncated at $[ - 3 \sigma , 3 \sigma ]$ .
• Linear 权重：服从 $\mathcal { N } \Big ( \mu = 0 , \sigma ^ { 2 } = \frac { 2 } { d _ { \mathrm { in } } + d _ { \mathrm { out } } } \Big )$，并在区间 $[ - 3 \sigma , 3 \sigma ]$ 上截断。

• Embedding: $\mathcal { N } ( \mu = 0 , \sigma ^ { 2 } = 1 )$ truncated at $[ - 3 , 3 ]$
• Embedding：服从 $\mathcal { N } ( \mu = 0 , \sigma ^ { 2 } = 1 )$，并在区间 $[ - 3 , 3 ]$ 上截断。

• RMSNorm: 1
• RMSNorm：1

You should use torch.nn.init.trunc_normal_ to initialize the truncated normal weights.
你应使用 torch.nn.init.trunc_normal_ 来初始化截断正态分布权重。

# 3.3.2 Linear Module
# 3.3.2 Linear 模块

Linear layers are a fundamental building block of Transformers and neural nets in general. First, you will implement your own Linear class that inherits from torch.nn.Module and performs a linear transformation:
Linear 层是 Transformer 以及一般神经网络中的基础构件。首先，你将实现自己的 Linear 类，它继承自 torch.nn.Module，并执行如下线性变换：

$$
y = W x. \tag {3}
$$

Note that we do not include a bias term, following most modern LLMs.
注意，和大多数现代大语言模型一样，我们这里不包含 bias 项。

Problem (linear): Implementing the linear module (1 point)
题目（linear）：实现线性模块（1 分）

Deliverable: Implement a Linear class that inherits from torch.nn.Module and performs a linear transformation. Your implementation should follow the interface of PyTorch's built-in nn.Linear module, except for not having a bias argument or parameter. We recommend the following interface:
提交内容：实现一个继承自 torch.nn.Module 的 Linear 类，用于执行线性变换。你的实现应遵循 PyTorch 内置 nn.Linear 模块的接口，但不包含 bias 参数或 bias 张量。我们推荐如下接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None)
```

Construct a linear transformation module. This function should accept the following parameters:
构造一个线性变换模块。该函数应接收以下参数：

in_features: int final dimension of the input
in_features: int 输入的最后一维大小

out_features: int final dimension of the output
out_features: int 输出的最后一维大小

device: torch.device | None = None Device to store the parameters on
device: torch.device | None = None 参数存储所在设备

dtype: torch.dtype | None = None Data type of the parameters
dtype: torch.dtype | None = None 参数的数据类型

```python
def forward(self, x: torch.Tensor) -> torch.Tensor
```

Apply the linear transformation to the input.
对输入应用线性变换。

Make sure to:
请确保：

• subclass nn.Module
• 继承 nn.Module

• call the superclass constructor
• 调用父类构造函数

• construct and store your parameter as W (not W^T), putting it in an nn.Parameter
• 将参数按 W（而不是 W^T）的形式构造并存储为 nn.Parameter

• of course, don’t use nn.Linear or nn.functional.linear
• 当然，不要使用 nn.Linear 或 nn.functional.linear

For initializations, use the settings from above along with torch.nn.init.trunc_normal_ to initialize the weights.
初始化时，请使用上面给出的设置，并结合 torch.nn.init.trunc_normal_ 来初始化权重。

To test your Linear module, implement the test adapter at [adapters.run_linear]. The adapter should load the given weights into your Linear module. You can use Module.load_state_dict for this purpose. Then, run uv run pytest -k test_linear.
若要测试你的 Linear 模块，请实现测试适配器 [adapters.run_linear]。该适配器应把给定权重加载到你的 Linear 模块中，你可以使用 Module.load_state_dict 来完成。然后运行 uv run pytest -k test_linear。

# 3.3.3 Embedding Module
# 3.3.3 Embedding 模块

As discussed above, the first layer of the Transformer is an embedding layer that maps integer token IDs into a vector space of dimension d_model. We will implement a custom Embedding class that inherits from torch.nn.Module (so you should not use nn.Embedding). The forward method should select the embedding vector for each token ID by indexing into an embedding matrix of shape (vocab_size, d_model) using a torch.LongTensor of token IDs with shape (batch_size, sequence_length).
如前所述，Transformer 的第一层是 embedding 层，它将整数 token ID 映射到一个 d_model 维的向量空间中。我们将实现一个自定义的 Embedding 类，它继承自 torch.nn.Module（因此你不应使用 nn.Embedding）。其 forward 方法应当通过一个形状为 (batch_size, sequence_length) 的 torch.LongTensor token ID 张量，对形状为 (vocab_size, d_model) 的 embedding 矩阵进行索引，从而为每个 token ID 选取对应的 embedding 向量。

Problem (embedding): Implement the embedding module (1 point)
题目（embedding）：实现 embedding 模块（1 分）

Deliverable: Implement the Embedding class that inherits from torch.nn.Module and performs an embedding lookup. Your implementation should follow the interface of PyTorch’s built-in nn.Embedding module. We recommend the following interface:
提交内容：实现一个继承自 torch.nn.Module 的 Embedding 类，用于执行 embedding 查找。你的实现应遵循 PyTorch 内置 nn.Embedding 模块的接口。我们推荐如下接口：

def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) Construct an embedding module. This function should accept the following parameters:
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None) 构造一个 embedding 模块。该函数应接收以下参数：

num_embeddings: int Size of the vocabulary
num_embeddings: int 词表大小

embedding_dim: int Dimension of the embedding vectors, i.e., d_model
embedding_dim: int embedding 向量维度，即 d_model

device: torch.device | None = None Device to store the parameters on
device: torch.device | None = None 参数存储所在设备

dtype: torch.dtype | None = None Data type of the parameters
dtype: torch.dtype | None = None 参数的数据类型

def forward(self, token_ids: torch.Tensor) -> torch.Tensor Lookup the embedding vectors for the given token IDs.
def forward(self, token_ids: torch.Tensor) -> torch.Tensor 为给定 token ID 查找对应的 embedding 向量。

Make sure to:
请确保：

• subclass nn.Module
• 继承 nn.Module

• call the superclass constructor
• 调用父类构造函数

• initialize your embedding matrix as an nn.Parameter
• 将 embedding 矩阵初始化为 nn.Parameter

• store the embedding matrix with the d_model being the final dimension
• 以 d_model 作为最后一维存储 embedding 矩阵

• of course, don’t use nn.Embedding or nn.functional.embedding
• 当然，不要使用 nn.Embedding 或 nn.functional.embedding

Again, use the settings from above for initialization, and use torch.nn.init.trunc_normal_ to initialize the weights.
同样，请使用上面给出的初始化设置，并使用 torch.nn.init.trunc_normal_ 初始化权重。

To test your implementation, implement the test adapter at [adapters.run_embedding]. Then, run uv run pytest -k test_embedding.
若要测试你的实现，请完成测试适配器 [adapters.run_embedding]。然后运行 uv run pytest -k test_embedding。

# 3.4 Pre-Norm Transformer Block
# 3.4 Pre-Norm Transformer Block

Each Transformer block has two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network ([A. Vaswani et al., 2017], section 3.1).
每个 Transformer block 都有两个子层：一个多头自注意力机制，以及一个逐位置前馈网络（[A. Vaswani et al., 2017] 第 3.1 节）。

In the original Transformer paper, the model uses a residual connection around each of the two sublayers, followed by layer normalization. This architecture is commonly known as the “post-norm” Transformer, since layer normalization is applied to the sub-layer output. However, a variety of work has found that moving layer normalization from the output of each sub-layer to the input of each sub-layer (with an additional layer normalization after the final Transformer block) improves Transformer training stability [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]; see Figure 2 for a visual representation of this “pre-norm” Transformer block. The output of each Transformer block sub-layer is then added to the sub-layer input via the residual connection (A. Vaswani et al. [8], section 5.4). An intuition for pre-norm is that there is a clean “residual stream” without any normalization going from the input embeddings to the final output of the Transformer, which is purported to improve gradient flow. This pre-norm Transformer is now the standard used in language models today (e.g., GPT-3, LLaMA, PaLM, etc.), so we will implement this variant. We will walk through each of the components of a pre-norm Transformer block, implementing them in sequence.
在最初的 Transformer 论文中，模型在两个子层外都包裹残差连接，并在其后进行 layer normalization。这种结构通常被称为 “post-norm” Transformer，因为 layer normalization 作用于子层输出。然而，很多工作发现，把 layer normalization 从每个子层的输出端移动到输入端（并在最后一个 Transformer block 后额外加一层归一化）能够提高 Transformer 训练的稳定性 [T. Q. Nguyen et al., 2019; R. Xiong et al., 2020]；图 2 展示了这种 “pre-norm” Transformer block 的可视化结构。每个 Transformer block 子层的输出随后会通过残差连接加回到子层输入上（A. Vaswani et al. [8]，第 5.4 节）。对 pre-norm 的一种直观理解是：从输入 embedding 到 Transformer 最终输出之间存在一条不经过归一化的“干净残差流”，这被认为有助于梯度传播。如今，pre-norm Transformer 已成为语言模型中的标准配置（例如 GPT-3、LLaMA、PaLM 等），因此我们将实现这一变体。接下来，我们会逐步介绍 pre-norm Transformer block 的各个组件，并依次实现它们。

# 3.4.1 Root Mean Square Layer Normalization
# 3.4.1 均方根层归一化

The original Transformer implementation of A. Vaswani et al. [8] uses layer normalization [J. L. Ba et al., 2016] to normalize activations. Following H. Touvron et al. [12], we will use root mean square layer normalization (RMSNorm; B. Zhang et al. [13], equation 4) for layer normalization. Given a vector $a \in \mathbb { R } ^ { d _ { \mathrm { m o d e l } } }$ of activations, RMSNorm will rescale each activation $a _ { i }$ as follows:
最初的 Transformer 实现（A. Vaswani et al. [8]）使用 layer normalization [J. L. Ba et al., 2016] 对激活值进行归一化。按照 H. Touvron et al. [12] 的做法，我们将使用 root mean square layer normalization（RMSNorm；B. Zhang et al. [13] 公式 4）来做层归一化。给定一个激活向量 $a \in \mathbb { R } ^ { d _ { \mathrm { model } } }$，RMSNorm 会将每个激活 $a _ { i }$ 按如下方式缩放：

$$
\operatorname{RMSNorm} \left(a _ {i}\right) = \frac {a _ {i}}{\operatorname{RMS} (a)} g _ {i}, \tag {4}
$$

where $\operatorname { RMS } ( a ) = \sqrt { \frac { 1 } { d _ { \mathrm { model } } } \sum _ { i = 1 } ^ { d _ { \mathrm { model } } } a _ { i } ^ { 2 } + \varepsilon }$. Here, $g_i$ is a learnable “gain” parameter (there are d_model such parameters total), and $\varepsilon$ is a hyperparameter that is often fixed at 1e-5.
其中，$\operatorname { RMS } ( a ) = \sqrt { \frac { 1 } { d _ { \mathrm { model } } } \sum _ { i = 1 } ^ { d _ { \mathrm { model } } } a _ { i } ^ { 2 } + \varepsilon }$。这里，$g_i$ 是可学习的“增益”参数（总共有 d_model 个），而 $\varepsilon$ 是一个超参数，通常固定为 1e-5。

You should upcast your input to torch.float32 to prevent overflow when you square the input. Overall, your forward method should look like:
你应当先将输入上转型为 torch.float32，以防在对输入平方时发生溢出。总体上，你的 forward 方法应大致如下：

```python
in_dtype = x.dtype
x = x.to(torch.float32)

# Your code here performing RMSNorm
...
result = ...

# Return the result in the original dtype
return result.to(in_dtype)
```

Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)
题目（rmsnorm）：均方根层归一化（1 分）

Deliverable: Implement RMSNorm as a torch.nn.Module. We recommend the following interface:
提交内容：将 RMSNorm 实现为一个 torch.nn.Module。推荐接口如下：

def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) Construct the RMSNorm module. This function should accept the following parameters:
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None) 构造 RMSNorm 模块。该函数应接收以下参数：

```python
d_model: int Hidden dimension of the model
eps: float = 1e-5 Epsilon value for numerical stability
device: torch.device | None = None Device to store the parameters on
dtype: torch.dtype | None = None Data type of the parameters
```

def forward(self, x: torch.Tensor) -> torch.Tensor Process an input tensor of shape (batch_size, sequence_length, d_model) and return a tensor of the same shape.
def forward(self, x: torch.Tensor) -> torch.Tensor 处理一个形状为 (batch_size, sequence_length, d_model) 的输入张量，并返回相同形状的张量。

Note: Remember to upcast your input to torch.float32 before performing the normalization (and later downcast to the original dtype), as described above.
注意：如上所述，归一化前请记得先把输入上转型到 torch.float32（之后再降回原始 dtype）。

To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run uv run pytest -k test_rmsnorm.
若要测试你的实现，请完成测试适配器 [adapters.run_rmsnorm]。然后运行 uv run pytest -k test_rmsnorm。

# 3.4.2 Position-Wise Feed-Forward Network
# 3.4.2 逐位置前馈网络

In the original Transformer paper (section 3.3 of A. Vaswani et al. [8]), the Transformer feed-forward network consists of two linear transformations with a ReLU activation between them. In that original architecture, the dimensionality of the inner feed-forward layer is typically 4x the input dimensionality.
在最初的 Transformer 论文中（A. Vaswani et al. [8] 第 3.3 节），Transformer 的前馈网络由两个线性变换和中间的 ReLU 激活组成。在原始架构中，前馈网络内部层的维度通常是输入维度的 4 倍。

However, modern language models tend to incorporate two main changes compared to this original design: they use another activation function and employ a gating mechanism. Specifically, we will implement the “SwiGLU” activation function adopted in LLMs like Llama 3 [A. Grattafiori et al., 2024] and Qwen 2.5 [A. Yang et al., 2024], which combines the SiLU (often called Swish) activation with a gating mechanism called a Gated Linear Unit (GLU). We will also omit the bias terms sometimes used in linear layers, following most modern LLMs since PaLM [A. Chowdhery et al., 2022] and LLaMA [H. Touvron et al., 2023].
不过，相比这一原始设计，现代语言模型通常会引入两个主要变化：使用另一种激活函数，并引入门控机制。具体来说，我们将实现 Llama 3 [A. Grattafiori et al., 2024] 和 Qwen 2.5 [A. Yang et al., 2024] 等大语言模型采用的 “SwiGLU” 激活函数，它将 SiLU（通常也称 Swish）激活与一种称为 Gated Linear Unit（GLU）的门控机制结合起来。与此同时，我们也会像 PaLM [A. Chowdhery et al., 2022] 和 LLaMA [H. Touvron et al., 2023] 之后的大多数现代大语言模型一样，省略线性层中有时会使用的 bias 项。

The SiLU or Swish activation function [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] is defined as follows:
SiLU 或 Swish 激活函数 [D. Hendrycks et al., 2016; S. Elfwing et al., 2017] 定义如下：

$$
\operatorname{SiLU} (x) = x \cdot \sigma (x) = \frac {x}{1 + e ^ {- x}} \tag {5}
$$

As can be seen in Figure 3, the SiLU activation function is similar to the ReLU activation function, but is smooth at zero.
如图 3 所示，SiLU 激活函数与 ReLU 激活函数相似，但在 0 点处是平滑的。

Gated Linear Units (GLUs) were originally defined by Y. N. Dauphin et al. [19] as the element-wise product of a linear transformation passed through a sigmoid function and another linear transformation:
门控线性单元（GLU）最初由 Y. N. Dauphin 等人 [19] 定义为：一个线性变换经过 sigmoid 函数后的结果，与另一个线性变换结果做逐元素乘积：

$$
\mathrm{GLU} (x, W _ {1}, W _ {2}) = \sigma (W _ {1} x) \odot W _ {2} x, \tag {6}
$$

where $\odot$ represents element-wise multiplication. Gated Linear Units are suggested to “reduce the vanishing gradient problem for deep architectures by providing a linear path for the gradients while retaining non-linear capabilities.”
其中，$\odot$ 表示逐元素乘法。有人认为，GLU 通过为梯度提供一条线性传播路径、同时保留非线性能力，能够“减轻深层结构中的梯度消失问题”。

Putting the SiLU/Swish and GLU together, we get the SwiGLU, which we will use for our feed-forward networks:
将 SiLU/Swish 与 GLU 结合起来，就得到我们将在前馈网络中使用的 SwiGLU：

$$
\operatorname{FFN} (x) = \operatorname{SwiGLU} \left(x, W _ {1}, W _ {2}, W _ {3}\right) = W _ {2} \left(\operatorname{SiLU} \left(W _ {1} x\right) \odot W _ {3} x\right), \tag {7}
$$

where $x \in \mathbb { R } ^ { d _ { \mathrm { model } } } , W _ { 1 } , W _ { 3 } \in \mathbb { R } ^ { d _ { \mathrm { ff } } \times d _ { \mathrm { model } } } , W _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { model } } \times d _ { \mathrm { ff } } }$, and canonically, $d _ { \mathrm { ff } } = \frac { 8 } { 3 } d _ { \mathrm { model } }$. For concrete implementations, it is fine to round this to a nearby multiple of 64 for hardware efficiency.
其中，$x \in \mathbb { R } ^ { d _ { \mathrm { model } } }$，$W _ { 1 } , W _ { 3 } \in \mathbb { R } ^ { d _ { \mathrm { ff } } \times d _ { \mathrm { model } } }$，$W _ { 2 } \in \mathbb { R } ^ { d _ { \mathrm { model } } \times d _ { \mathrm { ff } } }$，并且通常取 $d _ { \mathrm { ff } } = \frac { 8 } { 3 } d _ { \mathrm { model } }$。在具体实现中，可以将其四舍五入到邻近的 64 的倍数，以提高硬件效率。

N. Shazeer [20] first proposed combining the SiLU/Swish activation with GLUs and conducted experiments showing that SwiGLU outperforms baselines like ReLU and SiLU (without gating) on language modeling tasks. Later in the assignment, you will compare SwiGLU and SiLU. Though we’ve mentioned some heuristic arguments for these components (and the papers provide more supporting evidence), it’s good to keep an empirical perspective: a now famous quote from Shazeer’s paper is
N. Shazeer [20] 首先提出将 SiLU/Swish 激活与 GLU 结合，并通过实验表明，在语言建模任务上，SwiGLU 的表现优于 ReLU 和不带门控的 SiLU 等基线。后面你还会亲自比较 SwiGLU 与 SiLU。虽然我们已经给出了一些关于这些组件的启发式解释（论文中也提供了更多证据），但仍然要保持经验主义视角：Shazeer 的论文中有一句如今很出名的话：

“We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.”
“我们无法解释这些结构为何看起来有效；和其他一切一样，我们把它们的成功归因于神的慈悲。”

# Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)
# 题目（positionwise_feedforward）：实现逐位置前馈网络（2 分）

Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.
提交内容：实现 SwiGLU 前馈网络，它由 SiLU 激活函数和一个 GLU 组成。

Note: in this particular case, you should feel free to use torch.sigmoid in your implementation for numerical stability.
注意：在这个实现中，你可以直接使用 torch.sigmoid，以获得更好的数值稳定性。

You should set $d _ { \mathrm { ff } }$ to approximately $\frac { 8 } { 3 } \times d _ { \mathrm { model } }$ in your implementation, while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware. To test your implementation against our provided tests, you will need to implement the test adapter at [adapters.run_swiglu]. Then, run uv run pytest -k test_swiglu to test your implementation.
在实现中，你应将 $d _ { \mathrm { ff } }$ 设为大约 $\frac { 8 } { 3 } \times d _ { \mathrm { model } }$，同时确保前馈网络内部层的维度是 64 的倍数，以充分利用硬件。若要使用我们提供的测试来验证你的实现，你需要完成测试适配器 [adapters.run_swiglu]。然后运行 uv run pytest -k test_swiglu。

# 3.4.3 Relative Positional Embeddings
# 3.4.3 相对位置嵌入

To inject positional information into the model, we will implement Rotary Position Embeddings [J. Su et al., 2021], often called RoPE. For a given query token $q ^ { ( i ) } = W _ { q } x ^ { ( i ) } \in \mathbb { R } ^ { d }$ at token position $i$, we will apply a pairwise rotation matrix $R ^ { i }$, giving us $q ^ { \prime ( i ) } = R ^ { i } q ^ { ( i ) } = R ^ { i } W _ { q } x ^ { ( i ) }$. Here, $R ^ i$ rotates pairs of embedding elements $q _ { 2 k - 1 : 2 k } ^ { ( i ) }$ as 2D vectors by the angle $\theta _ { i , k } = \frac { i } { \Theta ^ { ( 2 k - 2 ) / d } }$ for $k \in \{ 1 , . . . , d / 2 \}$ and some constant $\Theta$. Thus, we can consider $R ^ i$ to be a block-diagonal matrix of size $d \times d$, with blocks $R _ k ^ i$ for $k \in \{ 1 , . . . , d / 2 \}$, with
为了向模型注入位置信息，我们将实现 Rotary Position Embeddings [J. Su et al., 2021]，通常称为 RoPE。对于位于位置 $i$ 的某个 query token，记 $q ^ { ( i ) } = W _ { q } x ^ { ( i ) } \in \mathbb { R } ^ { d }$，我们会对其施加一个成对旋转矩阵 $R ^ { i }$，从而得到 $q ^ { \prime ( i ) } = R ^ { i } q ^ { ( i ) } = R ^ { i } W _ { q } x ^ { ( i ) }$。这里，$R ^ i$ 会把 embedding 中成对的元素 $q _ { 2 k - 1 : 2 k } ^ { ( i ) }$ 当作二维向量旋转，旋转角度为 $\theta _ { i , k } = \frac { i } { \Theta ^ { ( 2 k - 2 ) / d } }$，其中 $k \in \{ 1 , . . . , d / 2 \}$，$\Theta$ 为某个常数。因此，我们可以把 $R ^ i$ 看成一个大小为 $d \times d$ 的块对角矩阵，其对角块为 $R _ k ^ i$，其中 $k \in \{ 1 , . . . , d / 2 \}$，并且

$$
R _ {k} ^ {i} = \left( \begin{array}{c c} \cos (\theta_ {i, k}) & - \sin (\theta_ {i, k}) \\ \sin (\theta_ {i, k}) & \cos (\theta_ {i, k}) \end{array} \right) \tag {8}
$$

Thus we get the full rotation matrix
于是我们得到完整的旋转矩阵

$$
R ^ {i} = \left( \begin{array}{c c c c c} R _ {1} ^ {i} & 0 & 0 & \dots & 0 \\ 0 & R _ {2} ^ {i} & 0 & \dots & 0 \\ 0 & 0 & R _ {3} ^ {i} & \dots & 0 \\ \vdots & \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & 0 & \dots & R _ {d / 2} ^ {i} \end{array} \right), \tag {9}
$$

where 0s represent $2 \times 2$ zero matrices. While one could construct the full $d \times d$ matrix, a good solution should use the properties of this matrix to implement the transformation more efficiently. Since we only care about the relative rotation of tokens within a given sequence, we can reuse the values we compute for $\cos \left( \theta _ { i , k } \right)$ and $\sin \left( \theta _ { i , k } \right)$ across layers, and different batches. If you would like to optimize it, you may use a single RoPE module referenced by all layers, and it can have a 2D pre-computed buffer of sin and cos values created during init with self.register_buffer(persistent=False), instead of an nn.Parameter (because we do not want to learn these fixed cosine and sine values). The exact same rotation process we did for our $q ^ { ( i ) }$ is then done for $k ^ { ( j ) }$, rotating by the corresponding angle. Notice that this layer has no learnable parameters.
其中的 0 表示 $2 \times 2$ 的零矩阵。虽然你当然可以直接构造完整的 $d \times d$ 矩阵，但更好的实现应利用该矩阵的结构性质，以更高效地完成变换。由于我们只关心给定序列内部 token 之间的相对旋转，因此对 $\cos \left( \theta _ { i , k } \right)$ 和 $\sin \left( \theta _ { i , k } \right)$ 的计算结果可以跨层、跨 batch 复用。如果你想进一步优化，可以让所有层共享同一个 RoPE 模块，并在初始化时通过 self.register_buffer(persistent=False) 预先建立一个二维 sin/cos 值缓冲区，而不是使用 nn.Parameter（因为这些固定的正弦余弦值并不是要学习的参数）。对 $q ^ { ( i ) }$ 所做的同样旋转过程，也会对应地应用到 $k ^ { ( j ) }$ 上。注意，这一层没有任何可学习参数。

Problem (rope): Implement RoPE (2 points)
题目（rope）：实现 RoPE（2 分）

Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input tensor.
提交内容：实现一个 RotaryPositionalEmbedding 类，将 RoPE 应用于输入张量。

The following interface is recommended:
推荐使用如下接口：

def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) Construct the RoPE module and create buffers if needed.
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None) 构造 RoPE 模块，并在需要时创建缓冲区。

theta: float $\Theta$ value for the RoPE
theta: float RoPE 中的 $\Theta$ 参数

d_k: int dimension of query and key vectors
d_k: int query 和 key 向量维度

max_seq_len: int Maximum sequence length that will be input
max_seq_len: int 可能输入的最大序列长度

device: torch.device | None = None Device to store the buffer on
device: torch.device | None = None 缓冲区所在设备

def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape. Note that you should tolerate x with an arbitrary number of batch dimensions. You should assume that the token positions are a tensor of shape (..., seq_len) specifying the token positions of x along the sequence dimension.
def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor 处理一个形状为 (..., seq_len, d_k) 的输入张量，并返回相同形状的张量。注意，你应当支持 x 含有任意数量的 batch 维度。同时假设 token_positions 是一个形状为 (..., seq_len) 的张量，用于指定 x 在序列维上的 token 位置。

You should use the token positions to slice your (possibly precomputed) cos and sin tensors along the sequence dimension.
你应当利用 token positions 在序列维上切片你预先计算好的（或动态计算的）cos 与 sin 张量。

To test your implementation, complete [adapters.run_rope] and make sure it passes uv run pytest -k test_rope.
若要测试你的实现，请完成 [adapters.run_rope]，并确保它能通过 uv run pytest -k test_rope。

# 3.4.4 Scaled Dot-Product Attention
# 3.4.4 缩放点积注意力

We will now implement scaled dot-product attention as described in A. Vaswani et al. [8] (section 3.2.1). As a preliminary step, the definition of the Attention operation will make use of softmax, an operation that takes an unnormalized vector of scores and turns it into a normalized distribution:
现在我们将按照 A. Vaswani et al. [8]（第 3.2.1 节）的描述来实现 scaled dot-product attention。作为准备步骤，Attention 的定义会用到 softmax。softmax 是一种把未归一化分数向量变成归一化分布的操作：

$$
\operatorname{softmax} (v) _ {i} = \frac {\exp (v _ {i})}{\sum_ {j = 1} ^ {n} \exp (v _ {j})}. \tag {10}
$$

Note that $\exp ( v _ { i } )$ can become inf for large values (then, inf/inf = NaN). We can avoid this by noticing that the softmax operation is invariant to adding any constant $c$ to all inputs. We can leverage this property for numerical stability—typically, we will subtract the largest entry of $v$ from all elements of $v$, making the new largest entry 0. You will now implement softmax, using this trick for numerical stability.
注意，当输入值很大时，$\exp ( v _ { i } )$ 可能会变成 inf（此时 inf/inf 就会得到 NaN）。我们可以利用 softmax 对“给所有输入同时加上同一个常数 $c$”这一操作不变的性质来避免这个问题。通常的做法是从 $v$ 的所有元素中减去它的最大值，使新的最大值变成 0，从而提升数值稳定性。现在你需要基于这一技巧实现 softmax。

Problem (softmax): Implement softmax (1 point)
题目（softmax）：实现 softmax（1 分）

Deliverable: Write a function to apply the softmax operation on a tensor. Your function should take two parameters: a tensor and a dimension $dim$, and apply softmax to the $dim$-th dimension of the input tensor. The output tensor should have the same shape as the input tensor, but its $dim$-th dimension will now have a normalized probability distribution. Use the trick of subtracting the maximum value in the $dim$-th dimension from all elements of the $dim$-th dimension to avoid numerical stability issues.
提交内容：编写一个函数，对张量执行 softmax 操作。函数应接收两个参数：一个张量和一个维度 $dim$，并对输入张量的第 $dim$ 维应用 softmax。输出张量应与输入张量具有相同形状，但其第 $dim$ 维将表示一个归一化的概率分布。请使用“从该维度所有元素中减去最大值”的技巧，以避免数值稳定性问题。

To test your implementation, complete [adapters.run_softmax] and make sure it passes uv run pytest -k test_softmax_matches_pytorch.
若要测试你的实现，请完成 [adapters.run_softmax]，并确保它通过 uv run pytest -k test_softmax_matches_pytorch。

We can now define the Attention operation mathematically as follows:
现在我们可以将 Attention 操作数学化地定义如下：

$$
\operatorname{Attention} (Q, K, V) = \operatorname{softmax} \left(\frac {Q K ^ {T}}{\sqrt {d _ {k}}}\right) V \tag {11}
$$

where $Q \in \mathbb { R } ^ { n \times d _ { k } } , K \in \mathbb { R } ^ { m \times d _ { k } }$ , and $V \in \mathbb { R } ^ { m \times d _ { v } }$. Here, $Q$, $K$ and $V$ are all inputs to this operation—note that these are not the learnable parameters.
其中，$Q \in \mathbb { R } ^ { n \times d _ { k } }$，$K \in \mathbb { R } ^ { m \times d _ { k } }$，而 $V \in \mathbb { R } ^ { m \times d _ { v } }$。这里的 $Q$、$K$ 和 $V$ 都是这个操作的输入，而不是可学习参数。

Masking: It is sometimes convenient to mask the output of an attention operation. A mask should have the shape $M \in \{ \mathrm { True } , \mathrm { False } \} ^ { n \times m }$, and each row $i$ of this boolean matrix indicates which keys the query $i$ should attend to. Canonically (and slightly confusingly), a value of True at position $( i , j )$ indicates that the query $i$ does attend to the key $j$, and a value of False indicates that the query does not attend to the key. In other words, “information flows” at $( i , j )$ pairs with value True. For example, consider a $1 \times 3$ mask matrix with entries [[True, True, False]]. The single query vector attends only to the first two keys.
Masking：有时我们希望对 attention 的输出加掩码。mask 的形状应为 $M \in \{ \mathrm { True } , \mathrm { False } \} ^ { n \times m }$，其中这个布尔矩阵的每一行 $i$ 表示 query $i$ 应该关注哪些 key。按惯例（虽然略显绕），位置 $( i , j )$ 上为 True 表示 query $i$ 可以关注 key $j$，而 False 表示 query 不应关注该 key。也就是说，值为 True 的 $( i , j )$ 位置允许“信息流动”。例如，考虑一个形状为 $1 \times 3$ 的 mask 矩阵 [[True, True, False]]，那么这个单独的 query 向量只关注前两个 key。

Computationally, it will be much more efficient to use masking than to compute attention on subsequences, and we can do this by taking the pre-softmax values $\left( Q K ^ { \top } / \sqrt { d _ { k } } \right)$ and adding a $-\infty$ to any entry of the mask matrix that is False.
从计算角度看，使用 mask 会比对各个子序列分别计算 attention 高效得多。做法是在 softmax 前的分数矩阵 $\left( Q K ^ { \top } / \sqrt { d _ { k } } \right)$ 中，对 mask 为 False 的位置加上 $-\infty$。

# Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)
# 题目（scaled_dot_product_attention）：实现缩放点积注意力（5 分）

Deliverable: Implement the scaled dot-product attention function. Your implementation should handle keys and queries of shape (batch_size, ..., seq_len, d_k) and values of shape (batch_size, ..., seq_len, d_v), where ... represents any number of other batch-like dimensions (if provided). The implementation should return an output with the shape (batch_size, ..., seq_len, d_v). See Section 3.2 for a discussion on batch-like dimensions.
提交内容：实现 scaled dot-product attention 函数。你的实现应支持形状为 (batch_size, ..., seq_len, d_k) 的 key 和 query，以及形状为 (batch_size, ..., seq_len, d_v) 的 value，其中 ... 表示任意数量的其他 batch-like 维度（如果存在）。输出应具有形状 (batch_size, ..., seq_len, d_v)。关于 batch-like 维度的讨论可参见第 3.2 节。

Your implementation should also support an optional user-provided boolean mask of shape (seq_len, seq_len). The attention probabilities of positions with a mask value of True should collectively sum to 1, and the attention probabilities of positions with a mask value of False should be zero.
你的实现还应支持一个可选的、由用户提供的布尔 mask，其形状为 (seq_len, seq_len)。mask 为 True 的位置对应的注意力概率应归一化后总和为 1，而 mask 为 False 的位置的注意力概率应为 0。

To test your implementation against our provided tests, you will need to implement the test adapter at [adapters.run_scaled_dot_product_attention]. uv run pytest -k test_scaled_dot_product_attention tests your implementation on third-order input tensors, while uv run pytest -k test_4d_scaled_dot_product_attention tests your implementation on fourth-order input tensors.
若要使用我们提供的测试来验证你的实现，你需要完成测试适配器 [adapters.run_scaled_dot_product_attention]。其中，uv run pytest -k test_scaled_dot_product_attention 会测试三阶输入张量，uv run pytest -k test_4d_scaled_dot_product_attention 会测试四阶输入张量。

# 3.4.5 Causal Multi-Head Self-Attention
# 3.4.5 因果多头自注意力

We will implement multi-head self-attention as described in section 3.2.2 of A. Vaswani et al. [8]. Recall that, mathematically, the operation of applying multi-head attention is defined as follows:
我们将按照 A. Vaswani et al. [8] 第 3.2.2 节的描述来实现多头自注意力。回顾一下，从数学上，多头注意力定义如下：

$$
\operatorname{MultiHead} (Q, K, V) = \operatorname{Concat} \left(\text { head } _ {1}, \dots , \text { head } _ {h}\right) \tag {12}
$$

$$
\text { for   head } _ {i} = \text { Attention } (Q _ {i}, K _ {i}, V _ {i}) \tag {13}
$$

with $Q _ { i } , K _ { i } , V _ { i }$ being slice number $i \in \{ 1 , . . . , h \}$ of size $d _ { k }$ or $d _ { v }$ of the embedding dimension for $Q , K$ , and $V$ respectively. With Attention being the scaled dot-product attention operation defined in Section 3.4.4. From this we can form the multi-head self-attention operation:
其中，$Q _ { i } , K _ { i } , V _ { i }$ 分别是 $Q$、$K$、$V$ 在 embedding 维上切出的第 $i \in \{ 1 , . . . , h \}$ 个分块，其大小分别为 $d _ { k }$ 或 $d _ { v }$。这里的 Attention 就是第 3.4.4 节定义的 scaled dot-product attention。由此，我们可以构造多头自注意力操作：

$$
\text { MultiHeadSelfAttention } (x) = W _ {O} \text { MultiHead } \left(W _ {Q} x, W _ {K} x, W _ {V} x\right) \tag {14}
$$

Here, the learnable parameters are $W _ { Q } \in \mathbb { R } ^ { h d _ { k } \times d _ { \mathrm { model } } }$ , $W _ { K } \in \mathbb { R } ^ { h d _ { k } \times d _ { \mathrm { model } } }$ , $W _ { V } \in \mathbb R ^ { h d _ { v } \times d _ { \mathrm { model } } }$ , and $W _ { O } \in \mathbb { R } ^ { d _ { \mathrm { model } } \times h d _ { v } }$. Since the Qs, Ks, and Vs are sliced in the multi-head attention operation, we can think of $W _ { Q } , W _ { K }$ and $W _ { V }$ as being separated for each head along the output dimension. When you have this working, you should be computing the key, value, and query projections in a total of three matrix multiplies.5
这里的可学习参数分别是 $W _ { Q } \in \mathbb { R } ^ { h d _ { k } \times d _ { \mathrm { model } } }$、$W _ { K } \in \mathbb { R } ^ { h d _ { k } \times d _ { \mathrm { model } } }$、$W _ { V } \in \mathbb { R } ^ { h d _ { v } \times d _ { \mathrm { model } } }$，以及 $W _ { O } \in \mathbb { R } ^ { d _ { \mathrm { model } } \times h d _ { v } }$。由于在多头注意力中，Q、K、V 都会按头进行切分，因此你可以把 $W _ { Q }$、$W _ { K }$ 和 $W _ { V }$ 理解为沿输出维为每个头分别提供参数。实现正确后，你应当总共只用三次矩阵乘法就完成 key、value 和 query 的投影。5

# Causal masking
# 因果掩码

Your implementation should prevent the model from attending to future tokens in the sequence. In other words, if the model is given a token sequence $t _ { 1 } , . . . , t _ { n }$, and we want to calculate the next-word predictions for the prefix $t _ { 1 } , . . . , t _ { i }$ (where $i < n$), the model should not be able to access (attend to) the token representations at positions $t _ { i + 1 } , . . . , t _ { n }$ since it will not have access to these tokens when generating text during inference (and these future tokens leak information about the identity of the true next word, trivializing the language modeling pre-training objective). For an input token sequence $t _ { 1 } , . . . , t _ { n }$ we can naively prevent access to future tokens by running multi-head self-attention $n$ times (for the $n$ unique prefixes in the sequence). Instead, we’ll use causal attention masking, which allows token $i$ to attend to all positions $j \le i$ in the sequence. You can use torch.triu or a broadcasted index comparison to construct this mask, and you should take advantage of the fact that your scaled dot-product attention implementation from Section 3.4.4 already supports attention masking.
你的实现应阻止模型关注序列中的未来 token。也就是说，如果模型输入的 token 序列是 $t _ { 1 } , . . . , t _ { n }$，而我们想计算前缀 $t _ { 1 } , . . . , t _ { i }$（其中 $i < n$）的下一个词预测，那么模型不应访问（注意到）位置 $t _ { i + 1 } , . . . , t _ { n }$ 对应的 token 表示，因为在推理生成时它本来就拿不到这些未来 token（而这些未来 token 会泄漏真实下一个词的信息，使语言模型预训练目标变得过于简单）。对于输入序列 $t _ { 1 } , . . . , t _ { n }$，一种朴素做法是对这 $n$ 个不同前缀分别运行 $n$ 次多头自注意力。相反，我们会使用因果注意力掩码，它允许 token $i$ 关注序列中所有满足 $j \le i$ 的位置。你可以使用 torch.triu 或基于广播的索引比较来构造这个掩码，并利用你在第 3.4.4 节中实现的 attention masking 支持。

# Applying RoPE
# 应用 RoPE

RoPE should be applied to the query and key vectors, but not the value vectors. Also, the head dimension should be handled as a batch dimension, because in multi-head attention, attention is being applied independently for each head. This means that precisely the same RoPE rotation should be applied to the query and key vectors for each head.
RoPE 应应用在 query 和 key 向量上，而不是 value 向量上。此外，head 维应视为 batch 维，因为在多头注意力中，每个头的 attention 都是独立计算的。这意味着对每个头的 query 和 key，都应施加完全相同的 RoPE 旋转。

# Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)
# 题目（multihead_self_attention）：实现因果多头自注意力（5 分）

Deliverable: Implement causal multi-head self-attention as a torch.nn.Module. Your implementation should accept (at least) the following parameters:
提交内容：将因果多头自注意力实现为一个 torch.nn.Module。你的实现至少应接收以下参数：

d_model: int Dimensionality of the Transformer block inputs.
d_model: int Transformer block 输入的维度。

num_heads: int Number of heads to use in multi-head self-attention.
num_heads: int 多头自注意力中使用的头数。

Following A. Vaswani et al. [8], set $d _ { k } = d _ { v } = \frac { d _ { \mathrm { model } } } { h }$. To test your implementation against our provided tests, implement the test adapter at [adapters.run_multihead_self_attention]. Then, run uv run pytest -k test_multihead_self_attention to test your implementation.
按照 A. Vaswani et al. [8]，设置 $d _ { k } = d _ { v } = \frac { d _ { \mathrm { model } } } { h }$。若要使用我们提供的测试来验证实现，请完成测试适配器 [adapters.run_multihead_self_attention]。然后运行 uv run pytest -k test_multihead_self_attention。

# 3.5 The Full Transformer LM
# 3.5 完整的 Transformer 语言模型

Let’s begin by assembling the Transformer block (it will be helpful to refer back to Figure 2). A Transformer block contains two ‘sub-layers’, one for the multi-head self attention, and another for the SwiGLU feed-forward network. In each sub-layer, we first perform RMSNorm, then the main operation (MHA/FF), finally adding in the residual connection.
现在开始把 Transformer block 组装起来（此时回看图 2 会很有帮助）。一个 Transformer block 包含两个“子层”，一个是多头自注意力，另一个是 SwiGLU 前馈网络。在每个子层中，我们先做 RMSNorm，然后执行主操作（MHA 或 FF），最后再加上残差连接。

To be concrete, the first half (the first ‘sub-layer’) of the Transformer block should be implementing the following set of updates to produce an output $y$ from an input $x$,
具体来说，Transformer block 的前半部分（也就是第一个“子层”）应实现如下更新：由输入 $x$ 生成输出 $y$，

$$
y = x + \text { MultiHeadSelfAttention } (\text { RMSNorm } (x)). \tag {15}
$$

# Problem (transformer_block): Implement the Transformer block (3 points)
# 题目（transformer_block）：实现 Transformer block（3 分）

Implement the pre-norm Transformer block as described in Section 3.4 and illustrated in Figure 2. Your Transformer block should accept (at least) the following parameters.
请按照第 3.4 节的描述和图 2 所示，实现 pre-norm Transformer block。你的 Transformer block 至少应接收以下参数。

d_model: int Dimensionality of the Transformer block inputs.
d_model: int Transformer block 输入维度。

num_heads: int Number of heads to use in multi-head self-attention.
num_heads: int 多头自注意力中使用的头数。

d_ff: int Dimensionality of the position-wise feed-forward inner layer.
d_ff: int 逐位置前馈网络内部层的维度。

To test your implementation, implement the adapter [adapters.run_transformer_block]. Then run uv run pytest -k test_transformer_block to test your implementation.
若要测试你的实现，请完成适配器 [adapters.run_transformer_block]。然后运行 uv run pytest -k test_transformer_block。

Deliverable: Transformer block code that passes the provided tests.
提交内容：能够通过所提供测试的 Transformer block 代码。

Now we put the blocks together, following the high-level diagram in Figure 1. Follow our description of the embedding in Section 3.1, feed this into num_layers Transformer blocks, and then pass that into the final layer norm and LM head to obtain an unnormalized distribution over the vocabulary (the logits).
现在我们根据图 1 的高层结构，把这些 block 组合起来。按照第 3.1 节对 embedding 的描述，将其输出送入 num_layers 个 Transformer blocks，之后再通过最终的 layer norm 和 LM head，得到词表上的未归一化分布（即 logits）。

# Problem (transformer_lm): Implementing the Transformer LM (3 points)
# 题目（transformer_lm）：实现 Transformer 语言模型（3 分）

Time to put it all together! Implement the Transformer language model as described in Section 3.1 and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned construction parameters for the Transformer block, as well as these additional parameters:
现在把一切整合起来！请按照第 3.1 节的描述和图 1 的示意，实现 Transformer 语言模型。至少来说，你的实现应接收前面提到的所有 Transformer block 构造参数，以及下面这些额外参数：

vocab_size: int The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
vocab_size: int 词表大小，用于确定 token embedding 矩阵的维度。

context_length: int The maximum context length, necessary for determining the dimensionality of the RoPE sin and cos buffer.
context_length: int 最大上下文长度，用于确定 RoPE 的 sin/cos 缓冲区维度。

num_layers: int The number of Transformer blocks to use.
num_layers: int Transformer block 的层数。

To test your implementation against our provided tests, you will first need to implement the test adapter at [adapters.run_transformer_lm]. Then, run uv run pytest -k test_transformer_lm to test your implementation.
若要使用我们提供的测试验证你的实现，你首先需要完成测试适配器 [adapters.run_transformer_lm]。然后运行 uv run pytest -k test_transformer_lm。

Deliverable: A Transformer LM module that passes the above tests.
提交内容：一个能够通过上述测试的 Transformer 语言模型模块。

# Resource accounting
# 资源核算

It is useful to be able to understand how the various parts of the Transformer consume compute and memory. We will go through the steps to do some basic “FLOPs accounting.” The vast majority of FLOPS in a Transformer are matrix multiplies, so our core approach is simple:
理解 Transformer 各部分如何消耗算力和内存是很有价值的。下面我们会介绍如何做一些基础的 “FLOPs 核算”。Transformer 中绝大多数 FLOPs 都来自矩阵乘法，因此核心思路很简单：

1. Write down all the matrix multiplies in a Transformer forward pass.
1. 列出 Transformer 前向传播中的所有矩阵乘法。

2. Convert each matrix multiply into FLOPs required.
2. 把每一个矩阵乘法换算成所需 FLOPs。

For this second step, the following fact will be useful:
在第二步中，下面这个事实会很有用：

Rule: Given $A \in \mathbb { R } ^ { m \times n }$ and $B \in \mathbb { R } ^ { n \times p }$, the matrix-matrix product $AB$ requires $2mnp$ FLOPs.
规则：给定 $A \in \mathbb { R } ^ { m \times n }$ 和 $B \in \mathbb { R } ^ { n \times p }$，矩阵乘法 $AB$ 需要 $2mnp$ 次 FLOPs。

To see this, note that $( A B ) [ i , j ] = A [ i , : ] \cdot B [ : , j ]$, and that this dot product requires $n$ additions and $n$ multiplications (共 $2n$ FLOPs). Then, since the matrix product $AB$ has $m \times p$ entries, the total number of FLOPs is $( 2 n ) ( m p ) = 2 m n p$.
原因是：$( A B ) [ i , j ] = A [ i , : ] \cdot B [ : , j ]$，而这个点积需要 $n$ 次加法和 $n$ 次乘法（共 $2n$ FLOPs）。又由于矩阵乘积 $AB$ 共有 $m \times p$ 个元素，总 FLOPs 即为 $( 2 n ) ( m p ) = 2 m n p$。

Now, before you do the next problem, it can be helpful to go through each component of your Transformer block and Transformer LM, and list out all the matrix multiplies and their associated FLOPs costs.
因此，在做下一题之前，先把你的 Transformer block 和 Transformer 语言模型的各个组件都过一遍，列出其中所有矩阵乘法及其对应的 FLOPs 代价，会很有帮助。

# Problem (transformer_accounting): Transformer LM resource accounting (5 points)
# 题目（transformer_accounting）：Transformer 语言模型资源核算（5 分）

(a) Consider a GPT-2 XL-sized model using our assignment architecture, which has the following configuration:
（a）考虑一个使用本作业架构、大小相当于 GPT-2 XL 的模型，其配置如下：

vocab_size: 50,257
vocab_size: 50,257

context_length: 1,024
context_length: 1,024

num_layers: 48
num_layers: 48

d_model: 1,600
d_model: 1,600

num_heads: 25
num_heads: 25

d_ff: 4,288 (the nearest multiple of 64 to $\frac{8}{3} \times 1,600$)
d_ff: 4,288（即最接近 $\frac{8}{3} \times 1,600$ 的 64 倍数）

Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?
假设我们用该配置构建模型。那么模型一共有多少可训练参数？如果每个参数都用单精度浮点数表示，仅加载这个模型需要多少内存？

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has context_length tokens.
（b）请找出完成我们这个 GPT-2 XL 形状模型一次前向传播所需的所有矩阵乘法。这些矩阵乘法总共需要多少 FLOPs？假设输入序列长度为 context_length。

Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs required.
提交内容：列出所有矩阵乘法（附说明），以及所需 FLOPs 总数。

(c) Based on your analysis above, which parts of the model require the most FLOPs?
（c）基于上述分析，模型的哪些部分需要最多 FLOPs？

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?
（d）请对 GPT-2 small（12 层，768 维 d_model，12 个头）、GPT-2 medium（24 层，1024 维 d_model，16 个头）和 GPT-2 large（36 层，1280 维 d_model，20 个头）重复上述分析。随着模型规模增大，Transformer 语言模型中哪些部分所占 FLOPs 比例会上升或下降？

Deliverable: For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.
提交内容：对每个模型，给出各组件及其对应 FLOPs 的分解（占一次前向传播总 FLOPs 的比例）。此外，再用 1 到 2 句话说明随着模型规模变化，各组件 FLOPs 占比如何变化。

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How does the relative contribution of FLOPs of the model components change?
（e）将 GPT-2 XL 的上下文长度增加到 16,384。一次前向传播的总 FLOPs 会如何变化？模型各组件 FLOPs 的相对占比又会如何变化？

Deliverable: A one-to-two sentence response.
提交内容：1 到 2 句话回答。

# 4 Training a Transformer LM
# 4 训练 Transformer 语言模型

We now have the steps to preprocess the data (via tokenizer) and the model (Transformer). What remains is to build all of the code to support training. This consists of the following:
现在，我们已经具备了数据预处理（通过 tokenizer）和模型（Transformer）的基本组件。剩下的就是构建支撑训练所需的所有代码，主要包括以下部分：

• Loss: we need to define the loss function (cross-entropy).
• Loss：我们需要定义损失函数（交叉熵）。

• Optimizer: we need to define the optimizer to minimize this loss (AdamW).
• Optimizer：我们需要定义用来最小化该损失的优化器（AdamW）。

• Training loop: we need all the supporting infrastructure that loads data, saves checkpoints, and manages training.
• Training loop：我们需要加载数据、保存 checkpoint、管理训练过程等支撑性基础设施。

# 4.1 Cross-entropy loss
# 4.1 交叉熵损失

Recall that the Transformer language model defines a distribution $p _ { \theta } ( x _ { i + 1 } \mid x _ { 1 : i } )$ for each sequence $x$ of length $m + 1$ and $i = 1 , . . . , m$. Given a training set $D$ consisting of sequences of length $m + 1$, we define the standard cross-entropy (negative log-likelihood) loss function:
回顾一下，对于每个长度为 $m + 1$ 的序列 $x$ 以及每个 $i = 1 , . . . , m$，Transformer 语言模型定义了一个分布 $p _ { \theta } ( x _ { i + 1 } \mid x _ { 1 : i } )$。给定一个由长度为 $m + 1$ 的序列构成的训练集 $D$，我们定义标准交叉熵（负对数似然）损失函数如下：

$$
\ell (\theta ; D) = \frac {1}{| D | m} \sum_ {x \in D} \sum_ {i = 1} ^ {m} - \log p _ {\theta} (x _ {i + 1} \mid x _ {1: i}). \tag {16}
$$

(Note that a single forward pass in the Transformer yields $p _ { \theta } ( x _ { i + 1 } \mid x _ { 1 : i } )$ for all $i = 1 , . . . , m$.)
（注意，Transformer 只需一次前向传播，就可以同时得到所有 $i = 1 , . . . , m$ 的 $p _ { \theta } ( x _ { i + 1 } \mid x _ { 1 : i } )$。）

In particular, the Transformer computes logits $o _ { i } \in \mathbb { R } ^ { \mathrm { vocab\_size } }$ for each position $i$, which results in:
具体来说，Transformer 会为每个位置 $i$ 计算 logits $o _ { i } \in \mathbb { R } ^ { \mathrm { vocab\_size } }$，从而得到：

$$
p (x _ {i + 1} \mid x _ {1: i}) = \text { softmax } (o _ {i}) [ x _ {i + 1} ] = \frac {\exp (o _ {i} [ x _ {i + 1} ])}{\sum_ {a = 1} ^ {\text { vocab\_size }} \exp (o _ {i} [ a ])}. \tag {17}
$$

The cross-entropy loss is generally defined with respect to the vector of logits $o _ { i } \in \mathbb { R } ^ { \mathrm { vocab\_size } }$ and target $x _ { i + 1 }$.
交叉熵损失通常是相对于 logits 向量 $o _ { i } \in \mathbb { R } ^ { \mathrm { vocab\_size } }$ 以及目标值 $x _ { i + 1 }$ 来定义的。

Implementing the cross-entropy loss requires some care with numerical issues, just like in the case of softmax.
实现交叉熵损失时同样需要注意数值稳定性问题，就像实现 softmax 时那样。

# Problem (cross_entropy): Implement cross-entropy (1 point)
# 题目（cross_entropy）：实现交叉熵（1 分）

Deliverable: Write a function to compute the cross-entropy loss, which takes in predicted logits $( o _ { i } )$ and targets $( x _ { i + 1 } )$ and computes the cross-entropy $\ell _ { i } = - \log \operatorname { softmax } ( o _ { i } ) [ x _ { i + 1 } ]$. Your function should handle the following:
提交内容：编写一个函数来计算交叉熵损失。该函数接收预测 logits $( o _ { i } )$ 和目标值 $( x _ { i + 1 } )$，并计算交叉熵 $\ell _ { i } = - \log \operatorname { softmax } ( o _ { i } ) [ x _ { i + 1 } ]$。你的函数应处理以下要求：

• Subtract the largest element for numerical stability.
• 为数值稳定性减去最大元素。

• Cancel out log and exp whenever possible.
• 在可能时尽量对 log 与 exp 做代数消去。

• Handle any additional batch dimensions and return the average across the batch. As with Section 3.2, we assume batch-like dimensions always come first, before the vocabulary size dimension.
• 处理任意额外的 batch 维度，并返回 batch 上的平均值。与第 3.2 节相同，我们假设所有 batch-like 维度都位于最前面，在词表维之前。

Implement [adapters.run_cross_entropy], then run uv run pytest -k test_cross_entropy to test your implementation.
完成 [adapters.run_cross_entropy]，然后运行 uv run pytest -k test_cross_entropy 来测试你的实现。

# Perplexity
# 困惑度

Cross-entropy suffices for training, but when we evaluate the model, we also want to report perplexity. For a sequence of length $m$ where we suffer cross-entropy losses $\ell _ { 1 } , . . . , \ell _ { m }$:
交叉熵足以用于训练，但在评估模型时，我们通常还希望报告 perplexity（困惑度）。对于一个长度为 $m$ 的序列，若其交叉熵损失为 $\ell _ { 1 } , . . . , \ell _ { m }$，则：

$$
\text { perplexity } = \exp \left(\frac {1}{m} \sum_ {i = 1} ^ {m} \ell_ {i}\right). \tag {18}
$$
