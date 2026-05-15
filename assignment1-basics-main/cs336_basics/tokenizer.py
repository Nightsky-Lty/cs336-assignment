import regex as re
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
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

    while current_idx < vocab_size:
        pair_counts: dict[tuple[bytes, bytes], int] = {}
        for pretoken_bytes in pretoken_token_sequences:
            i = 0
            token_seq = pretoken_token_sequences[pretoken_bytes]
            while i + 1 < len(token_seq):
                pair = (token_seq[i], token_seq[i + 1])
                if pair in pair_counts:
                    pair_counts[pair] += pretoken_counts[pretoken_bytes]
                else:
                    pair_counts[pair] = pretoken_counts[pretoken_bytes]
                i += 1
        max_counts = None
        max_pair = None
        for pair in pair_counts:
            counts = pair_counts[pair]
            if max_pair is None or counts > max_counts or (max_counts == counts and max_pair < pair):
                max_pair = pair
                max_counts = counts
        if max_pair == None:
            break
        merge.append(max_pair)
        merged_bytes = max_pair[0] + max_pair[1]
        vocab[current_idx] = merged_bytes
        current_idx += 1
        for pretoken_bytes in pretoken_token_sequences:
            i = 0
            new_seq = []
            token_seq = pretoken_token_sequences[pretoken_bytes]
            while i < len(token_seq):
                if i + 1 == len(token_seq):
                    new_seq.append(token_seq[i])
                    break
                pair = (token_seq[i], token_seq[i + 1])
                if pair == max_pair:
                    new_seq.append(merged_bytes)
                    i += 2
                else:
                    new_seq.append(token_seq[i])
                    i += 1
            pretoken_token_sequences[pretoken_bytes] = new_seq

    return vocab, merge