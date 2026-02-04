from __future__ import annotations

import re
from dataclasses import dataclass
from collections import Counter
from typing import Iterable, List, Dict, Tuple

SPECIAL_TOKENS = {
    "pad": "[PAD]",
    "unk": "[UNK]",
    "cls": "[CLS]",
}

_token_re = re.compile(r"""[A-Za-z0-9]+|[^\sA-Za-z0-9]""", re.UNICODE)

def basic_tokenize(text: str) -> List[str]:
    # Lowercase + simple punctuation-aware tokenization.
    text = text.lower().strip()
    return _token_re.findall(text)

@dataclass
class Vocab:
    stoi: Dict[str, int]
    itos: List[str]

    @property
    def pad_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["pad"]]

    @property
    def unk_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["unk"]]

    @property
    def cls_id(self) -> int:
        return self.stoi[SPECIAL_TOKENS["cls"]]

def build_vocab(texts: Iterable[str], max_vocab: int, min_freq: int = 2) -> Vocab:
    counter = Counter()
    for t in texts:
        counter.update(basic_tokenize(t))

    # Reserve 3 special tokens at the front
    itos = [SPECIAL_TOKENS["pad"], SPECIAL_TOKENS["unk"], SPECIAL_TOKENS["cls"]]
    stoi = {tok: i for i, tok in enumerate(itos)}

    # Most common tokens
    for tok, freq in counter.most_common(max(0, max_vocab - len(itos))):
        if freq < min_freq:
            break
        if tok in stoi:
            continue
        stoi[tok] = len(itos)
        itos.append(tok)

    return Vocab(stoi=stoi, itos=itos)

def encode(text: str, vocab: Vocab, max_len: int) -> List[int]:
    toks = basic_tokenize(text)
    ids = [vocab.cls_id]  # prepend CLS
    for tok in toks[: max_len - 1]:
        ids.append(vocab.stoi.get(tok, vocab.unk_id))
    return ids
