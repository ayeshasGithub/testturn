from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from datasets import load_dataset, Dataset, DatasetDict

from .tokenizer import Vocab, build_vocab, encode

@dataclass
class DataConfig:
    dataset_name: str = "civil_comments"
    text_col: str = "text"
    target_col: str = "target"
    toxic_threshold: float = 0.5
    max_vocab: int = 30000
    min_freq: int = 2
    max_len: int = 256
    val_size: float = 0.1
    test_size: float = 0.1
    seed: int = 42
    dataset_path: Optional[str] = None  # optional local CSV/Parquet path

def _toxic_label(target: float, thr: float) -> int:
    return int(float(target) >= thr)

def load_civil_comments(cfg: DataConfig, max_examples: Optional[int] = None) -> Dataset:
    if cfg.dataset_path:
        # expects a local file with columns: cfg.text_col, cfg.target_col
        ext = os.path.splitext(cfg.dataset_path)[1].lower()
        if ext in [".csv"]:
            ds = load_dataset("csv", data_files=cfg.dataset_path)["train"]
        elif ext in [".parquet"]:
            ds = load_dataset("parquet", data_files=cfg.dataset_path)["train"]
        else:
            raise ValueError(f"Unsupported dataset_path extension: {ext}")
    else:
        ds = load_dataset(cfg.dataset_name, split="train")

    if max_examples:
        ds = ds.select(range(min(max_examples, len(ds))))
    return ds

def add_binary_label(ds: Dataset, cfg: DataConfig) -> Dataset:
    def _map(ex):
        return {"label": _toxic_label(ex[cfg.target_col], cfg.toxic_threshold)}
    return ds.map(_map, remove_columns=[], desc="Binarizing labels")

def stratified_split(ds: Dataset, cfg: DataConfig) -> DatasetDict:
    # HuggingFace datasets doesn't directly offer stratified splitting, so we do it via indices.
    labels = np.array(ds["label"], dtype=np.int64)
    n = len(labels)
    rng = np.random.RandomState(cfg.seed)

    idx = np.arange(n)
    # stratified split by shuffling indices within each class then slicing proportions
    train_idx, val_idx, test_idx = [], [], []
    for y in [0, 1]:
        cls_idx = idx[labels == y]
        rng.shuffle(cls_idx)
        n_test = int(len(cls_idx) * cfg.test_size)
        n_val = int(len(cls_idx) * cfg.val_size)
        test_part = cls_idx[:n_test]
        val_part = cls_idx[n_test:n_test + n_val]
        train_part = cls_idx[n_test + n_val:]
        train_idx.append(train_part)
        val_idx.append(val_part)
        test_idx.append(test_part)

    train_idx = np.concatenate(train_idx)
    val_idx = np.concatenate(val_idx)
    test_idx = np.concatenate(test_idx)

    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(test_idx)

    return DatasetDict({
        "train": ds.select(train_idx.tolist()),
        "val": ds.select(val_idx.tolist()),
        "test": ds.select(test_idx.tolist()),
    })

def build_vocab_from_split(dsd: DatasetDict, cfg: DataConfig, max_texts: Optional[int] = None) -> Vocab:
    texts = dsd["train"][cfg.text_col]
    if max_texts:
        texts = texts[:max_texts]
    return build_vocab(texts, max_vocab=cfg.max_vocab, min_freq=cfg.min_freq)

def tokenize_dataset(
    ds: Dataset,
    cfg: DataConfig,
    vocab: Vocab,
    pretokenize_dir: Optional[str] = None,
) -> Dataset:
    # If pretokenize_dir is provided, store token IDs on disk as an arrow dataset.
    def _map(ex):
        ids = encode(ex[cfg.text_col], vocab=vocab, max_len=cfg.max_len)
        return {"input_ids": ids, "length": len(ids)}

    tokenized = ds.map(_map, remove_columns=[], desc="Tokenizing")
    if pretokenize_dir:
        os.makedirs(pretokenize_dir, exist_ok=True)
        tokenized.save_to_disk(pretokenize_dir)
    return tokenized

def load_tokenized_from_disk(path: str) -> Dataset:
    from datasets import load_from_disk
    return load_from_disk(path)

def compute_pos_weight(train_ds: Dataset) -> float:
    labels = np.array(train_ds["label"], dtype=np.int64)
    pos = labels.sum()
    neg = len(labels) - pos
    if pos == 0:
        return 1.0
    return float(neg / max(pos, 1))
