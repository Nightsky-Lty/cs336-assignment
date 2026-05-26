from cs336_basics import tokenizer
import pickle, argparse
import numpy as np
import time

def train_tokenizer(
    vocab_path: str,
    merges_path: str,
    vocab_size: int,
    special_tokens: list[str],
    train_txt_path: str
):
    print("strat training...")
    vocab, merges = tokenizer.train_bpe(train_txt_path, vocab_size, special_tokens)

    with open(vocab_path, "wb") as v:
        pickle.dump(vocab, v)
    with open(merges_path, "wb") as m:
        pickle.dump(merges, m)

def encode_txt(
    txt_path: str,
    vocab_path: str,
    merges_path: str,
    special_tokens: list[str],
    out_path: str
):
    tk = tokenizer.Tokenizer.from_files(vocab_path, merges_path, special_tokens)
    with open(txt_path, "r", encoding="utf-8") as f:
        tokens = tk.encode_iterable(f)
        tokens_arr = np.fromiter(tokens, dtype=np.uint16)
    tokens_arr.tofile(out_path)

def main():
    strat_time = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--txt_path", type=str, default=None)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--vocab_path", type=str, default="data/tinystories_vocab.pkl")
    parser.add_argument("--merges_path", type=str, default="data/tinystories_merges.pkl")
    parser.add_argument("--out_path", type=str, default="data/encode_result.bin")
    parser.add_argument("--train_txt_path", type=str, default="data/TinyStoriesV2-GPT4-train.txt")

    args = parser.parse_args()

    special_tokens = ["<|endoftext|>"]

    if args.mode == "train_tokenizer":
        train_tokenizer(
            special_tokens=special_tokens,
            vocab_size=args.vocab_size,
            vocab_path=args.vocab_path,
            merges_path=args.merges_path,
            train_txt_path=args.train_txt_path
        )
    elif args.mode == "encode_txt":
        if args.txt_path is None:
            raise ValueError("Invalid txt_path")
        encode_txt(
            txt_path=args.txt_path,
            vocab_path=args.vocab_path,
            merges_path=args.merges_path,
            special_tokens=special_tokens,
            out_path=args.out_path
        )
    else:
        raise ValueError("Invalid mode")

    print(f"use time:{time.time() - strat_time:.2f}")

if __name__ == "__main__":
    main()