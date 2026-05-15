import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path: str, vocab_siz: int, special_tokens: list[str]):
    vocab: dict[int,bytes] = {i : bytes([i]) for i in range(256)}
    current_idx = len(vocab)
    for special_token in special_tokens:
        vocab[current_idx] = special_token.encode("utf-8")
        current_idx += 1
    
    merge: list[tuple[bytes,bytes]] = []

    ordered_special_tokens = sorted(special_tokens, key = len, reverse = True)
    with open(input_path, "r", encoding = "utf-8") as f:
        text = f.read()
    text_segments: list[str] = []
    i = 0
    current_str = ""
    while i < len(text):
        is_special_token = False

        for special_token in ordered_special_tokens:
            if text.startswith(special_token, i):
                if not (current_str == ""):
                    text_segments.append(current_str)
                current_str = ""
                i += len(special_token)
                is_special_token = True
                break

        if not is_special_token:
            current_str += text[i]
            i += 1
    if not(current_str == ""):
        text_segments.append(current_str)

    pretoken_counts: dict[bytes, int] = {}
    for text_segment in text_segments:
        for match in re.finditer(PAT, text_segment):
            m = match.group(0).encode("utf-8")
            if m in pretoken_counts:
                pretoken_counts[m] += 1
            else:
                pretoken_counts[m] = 1
    
    pretoken_token_sequences: dict[bytes,list[bytes]] = {}
    for pretoken_bytes in pretoken_counts:
        pretoken_token_sequences[pretoken_bytes] = [bytes([b]) for b in pretoken_bytes]

    return vocab, merge