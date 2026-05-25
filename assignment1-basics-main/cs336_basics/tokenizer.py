import regex as re
import pickle
import multiprocessing as mp
from collections import Counter
from collections.abc import Iterable
import math

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)

def count_chunks(chunks: list[str]) -> Counter[bytes]:
    counts = Counter()
    for chunk in chunks:
        for match in PAT_RE.finditer(chunk):
            m = match.group(0).encode("utf-8")
            counts[m] += 1
    return counts

def train_bpe(
    input_path: str, 
    vocab_size: int, 
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    vocab: dict[int, bytes] = {i : bytes([i]) for i in range(256)}
    current_idx = len(vocab)
    for special_token in special_tokens:
        vocab[current_idx] = special_token.encode("utf-8")
        current_idx += 1
    
    merges: list[tuple[bytes, bytes]] = []

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

    pretoken_counts: Counter[bytes] = Counter()
    
    # for text_segment in text_segments:
    #     for match in PAT_RE.finditer(text_segment):
    #         m = match.group(0).encode("utf-8")
    #         pretoken_counts[m] += 1
    
    num_workers = 4
    batch_size = math.ceil(len(text_segments) / num_workers)
    batches = [text_segments[i: i + batch_size] for i in range(0, len(text_segments), batch_size)]
    with mp.get_context("spawn").Pool(processes=num_workers) as pool:
        partial_counts = pool.map(count_chunks, batches)
    for c in partial_counts:
        pretoken_counts.update(c)

    pretoken_token_sequences: dict[bytes,list[bytes]] = {}
    for pretoken_bytes in pretoken_counts:
        pretoken_token_sequences[pretoken_bytes] = [bytes([b]) for b in pretoken_bytes]

    while current_idx < vocab_size:
        pair_counts: Counter[tuple[bytes, bytes]] = Counter()
        for pretoken_bytes in pretoken_token_sequences:
            i = 0
            token_seq = pretoken_token_sequences[pretoken_bytes]
            while i + 1 < len(token_seq):
                pair = (token_seq[i], token_seq[i + 1])
                pair_counts[pair] += pretoken_counts[pretoken_bytes]
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
        merges.append(max_pair)
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

    return vocab, merges

class Tokenizer:

    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
        ):
        self.vocab = dict(vocab)
        self.merges = list(merges)
        self.merge_rank: dict[tuple[bytes, bytes], int] = {b : rk for rk, b in enumerate(merges)}
        self.bytes_to_id: dict[bytes,int] = {
            v: k for k,v in self.vocab.items()
        }
        self.special_tokens = special_tokens
        if self.special_tokens is None:
            self.special_tokens = []
        self.ordered_special_tokens = sorted(self.special_tokens, key = len,reverse=True)
        
        if special_tokens:
            current_idx = len(self.vocab)
            for token in special_tokens:
                token_bytes = token.encode("utf-8")
                if token_bytes not in self.bytes_to_id:
                    self.bytes_to_id[token_bytes] = current_idx
                    self.vocab[current_idx] = token_bytes
                    current_idx += 1
    
    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath,"rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath,"rb") as f:
            merges = pickle.load(f)
        tokenizer = cls(vocab, merges, special_tokens)
        return tokenizer
    
    def decode(self, ids: list[int]) -> str:
        final_bytes = b"".join(self.vocab[idx] for idx in ids)
        string = bytes.decode(final_bytes, "utf-8", "replace")
        return string

    def encode_pretoken(self, pretoken: list[bytes]) -> list[int]:
        while True:
            target = None
            priority = None
            i = 0
            while i + 1 < len(pretoken):
                merged_bytes = (pretoken[i],pretoken[i + 1])
                if merged_bytes in self.merge_rank:
                    if priority is None:
                        priority = self.merge_rank[merged_bytes]
                        target = merged_bytes
                    elif self.merge_rank[merged_bytes] < priority:
                        priority = self.merge_rank[merged_bytes]
                        target = merged_bytes
                i += 1
            if target is None:
                break
            i = 0
            new_seq: list[bytes] = []
            while i < len(pretoken):
                if i + 1 < len(pretoken) and target == (pretoken[i], pretoken[i + 1]):
                    new_seq.append(pretoken[i] + pretoken[i + 1])
                    i += 2
                else:
                    new_seq.append(pretoken[i])
                    i += 1
            pretoken = new_seq
        
        tokens: list[int] = []
        for b in pretoken:
            tokens.append(self.bytes_to_id[b])
        return tokens
    
    def encode(self, text: str) -> list[int]:
        text_segments: list[str] = []
        ordered_special_tokens = self.ordered_special_tokens
        i = 0
        current_str = ""
        while i < len(text):
            is_special_token = False
            for special_token in ordered_special_tokens:
                if text.startswith(special_token, i):
                    is_special_token = True
                    if not (current_str == ""):
                        text_segments.append(current_str)
                    current_str = ""
                    text_segments.append(special_token)
                    i += len(special_token)
                    break
            if not is_special_token:
                current_str += text[i]
                i += 1
        if not (current_str == ""):
            text_segments.append(current_str)
        
        pretokens: list[list[bytes]] = []
        for text_segment in text_segments:
            is_special_token = False
            for special_token in ordered_special_tokens:
                if text_segment == special_token:
                    is_special_token = True
                    break
            if is_special_token:
                pretokens.append([text_segment.encode("utf-8")])
            else:
                for match in PAT_RE.finditer(text_segment):
                    m = match.group(0).encode("utf-8")
                    pretokens.append([bytes([b]) for b in m])

        tokens: list[int] = []
        for pretoken in pretokens:
            tokens.extend(self.encode_pretoken(pretoken))

        return tokens
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterable[int]:
        buffer: str = ""
        if not len(self.ordered_special_tokens) == 0:
            max_len = len(self.ordered_special_tokens[0])
        else:
            max_len = 0
        for chunk in iterable:
            buffer += chunk
            buffer_suffixes: list[str] = []
            protected_suffix = ""
            for i in range(min(max_len,len(buffer)), 0, -1):
                buffer_suffixes.append(buffer[-i:])
            for suffix in buffer_suffixes:
                if any(special_token.startswith(suffix) for special_token in self.ordered_special_tokens):
                    protected_suffix = buffer[-len(suffix):]
                    buffer = buffer[:-len(suffix)]
                    break
            
            if len(self.ordered_special_tokens) > 0:
                i = 0
                st = 0
                while i < len(buffer):
                    for special_token in self.ordered_special_tokens:
                        if buffer.startswith(special_token, i):
                            yield from self.encode(buffer[st: i])
                            yield self.bytes_to_id[special_token.encode("utf-8")]
                            st = i + len(special_token)
                            i = i + len(special_token) - 1
                            break
                    i += 1
                buffer = buffer[st:]

            total_len = 0
            matchs = PAT_RE.findall(buffer)
            matchs = matchs[:-1]
            for match in matchs:
                yield from self.encode(match)
                total_len += len(match)
            buffer = buffer[total_len:]
            buffer = buffer + protected_suffix

        yield from self.encode(buffer)