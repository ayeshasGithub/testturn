import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import PreTrainedTokenizerBase


@dataclass
class DataConfig:
    train_csv: str
    test_csv: str
    sample_sub_csv: str
    text_col: str = "comment_text"
    target_col: str = "target"
    max_len: int = 128


class OnTheFlyTokenizedDataset(Dataset):
    """Tokenize on-the-fly."""

    def __init__(
        self,
        df: pd.DataFrame,
        tokenizer: PreTrainedTokenizerBase,
        max_len: int,
        text_col: str,
        target_col: str,
        is_test: bool,
    ):
        self.texts = df[text_col].astype(str).values
        self.is_test = is_test
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.text_col = text_col
        self.target_col = target_col
        if not is_test:
            self.targets = df[target_col].astype(np.float32).values

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        tok = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt",
        )
        item = {
            "input_ids": tok["input_ids"].squeeze(0),
            "attention_mask": tok["attention_mask"].squeeze(0),
        }
        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


def write_memmap_meta(ids_path: str, num_rows: int) -> None:
    meta_path = ids_path.replace("_input_ids.memmap", "_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({"num_rows": int(num_rows)}, f)


def preprocess_to_memmap(
    df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    out_dir: str,
    split_name: str,
    text_col: str,
    target_col: str,
    is_test: bool,
) -> Tuple[str, str, Optional[str]]:
    """Pre-tokenize and store as memmap arrays."""
    os.makedirs(out_dir, exist_ok=True)
    ids_path = os.path.join(out_dir, f"{split_name}_input_ids.memmap")
    mask_path = os.path.join(out_dir, f"{split_name}_attention_mask.memmap")
    tgt_path = os.path.join(out_dir, f"{split_name}_targets.memmap") if not is_test else None

    N = len(df)
    input_ids = np.memmap(ids_path, mode="w+", dtype=np.int32, shape=(N, max_len))
    attn_mask = np.memmap(mask_path, mode="w+", dtype=np.int8, shape=(N, max_len))
    targets = None
    if not is_test:
        targets = np.memmap(tgt_path, mode="w+", dtype=np.float32, shape=(N,))

    texts = df[text_col].astype(str).values
    for i, t in enumerate(texts):
        tok = tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors=None,
        )
        input_ids[i] = np.asarray(tok["input_ids"], dtype=np.int32)
        attn_mask[i] = np.asarray(tok["attention_mask"], dtype=np.int8)
        if not is_test:
            targets[i] = np.float32(df[target_col].iloc[i])

    input_ids.flush()
    attn_mask.flush()
    if targets is not None:
        targets.flush()

    write_memmap_meta(ids_path, N)
    return ids_path, mask_path, tgt_path


class MemmapTokenizedDataset(Dataset):
    """Load tokenized arrays from memmap for fast training."""

    def __init__(
        self,
        ids_path: str,
        mask_path: str,
        tgt_path: Optional[str],
        max_len: int,
        is_test: bool,
    ):
        self.is_test = is_test
        meta_path = ids_path.replace("_input_ids.memmap", "_meta.json")
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        N = int(meta["num_rows"])

        self.input_ids = np.memmap(ids_path, mode="r", dtype=np.int32, shape=(N, max_len))
        self.attn_mask = np.memmap(mask_path, mode="r", dtype=np.int8, shape=(N, max_len))
        self.targets = None
        if not is_test and tgt_path is not None:
            self.targets = np.memmap(tgt_path, mode="r", dtype=np.float32, shape=(N,))

    def __len__(self) -> int:
        return int(self.input_ids.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = {
            "input_ids": torch.tensor(self.input_ids[idx], dtype=torch.long),
            "attention_mask": torch.tensor(self.attn_mask[idx], dtype=torch.long),
        }
        if not self.is_test:
            item["target"] = torch.tensor(self.targets[idx], dtype=torch.float32)
        return item


def build_loaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    tokenizer: PreTrainedTokenizerBase,
    max_len: int,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    use_memmap: bool,
    memmap_dir: str,
    text_col: str = "comment_text",
    target_col: str = "target",
    world_size: int = 1,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create DataLoaders with optional memmap caching."""

    if use_memmap:
        # Tokenization should happen once; recommended to run from rank0, but callers must barrier themselves.
        os.makedirs(memmap_dir, exist_ok=True)

        train_ds = MemmapTokenizedDataset(
            ids_path=os.path.join(memmap_dir, "train_input_ids.memmap"),
            mask_path=os.path.join(memmap_dir, "train_attention_mask.memmap"),
            tgt_path=os.path.join(memmap_dir, "train_targets.memmap"),
            max_len=max_len,
            is_test=False,
        )
        val_ds = MemmapTokenizedDataset(
            ids_path=os.path.join(memmap_dir, "val_input_ids.memmap"),
            mask_path=os.path.join(memmap_dir, "val_attention_mask.memmap"),
            tgt_path=os.path.join(memmap_dir, "val_targets.memmap"),
            max_len=max_len,
            is_test=False,
        )
        test_ds = MemmapTokenizedDataset(
            ids_path=os.path.join(memmap_dir, "test_input_ids.memmap"),
            mask_path=os.path.join(memmap_dir, "test_attention_mask.memmap"),
            tgt_path=None,
            max_len=max_len,
            is_test=True,
        )
    else:
        train_ds = OnTheFlyTokenizedDataset(train_df, tokenizer, max_len, text_col, target_col, is_test=False)
        val_ds = OnTheFlyTokenizedDataset(val_df, tokenizer, max_len, text_col, target_col, is_test=False)
        test_ds = OnTheFlyTokenizedDataset(test_df, tokenizer, max_len, text_col, target_col, is_test=True)

    train_sampler = DistributedSampler(train_ds, shuffle=True) if world_size > 1 else None
    val_sampler = DistributedSampler(val_ds, shuffle=False) if world_size > 1 else None
    test_sampler = DistributedSampler(test_ds, shuffle=False) if world_size > 1 else None

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=train_sampler,
        shuffle=(train_sampler is None),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        sampler=test_sampler,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
