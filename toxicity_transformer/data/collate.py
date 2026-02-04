from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import torch

@dataclass
class Batch:
    input_ids: torch.Tensor      # [B, T]
    attention_mask: torch.Tensor # [B, T] 1 for tokens, 0 for pad
    labels: torch.Tensor         # [B]

def collate_fn(features: List[Dict[str, Any]], pad_id: int) -> Batch:
    # features contain: input_ids (list[int]), label (int)
    max_len = max(len(f["input_ids"]) for f in features)
    bsz = len(features)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
    labels = torch.zeros((bsz,), dtype=torch.float32)

    for i, f in enumerate(features):
        ids = torch.tensor(f["input_ids"], dtype=torch.long)
        L = ids.numel()
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
        labels[i] = float(f["label"])

    return Batch(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
