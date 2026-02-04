from __future__ import annotations

import os
import argparse
import random
import numpy as np
import torch

from toxicity_transformer.data.dataset import (
    DataConfig, load_civil_comments, add_binary_label, stratified_split,
    build_vocab_from_split, tokenize_dataset, load_tokenized_from_disk,
    compute_pos_weight
)
from toxicity_transformer.data.collate import collate_fn
from toxicity_transformer.models.transformer import ModelConfig, ToxicityTransformer
from toxicity_transformer.training.distributed import init_distributed, cleanup_distributed, is_main_process
from toxicity_transformer.training.trainer import TrainConfig, make_loader, train

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def parse_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--dataset-path", type=str, default=None, help="Optional local CSV/Parquet with columns text,target")
    p.add_argument("--max-examples", type=int, default=None, help="Limit total examples for faster runs.")
    p.add_argument("--max-train-examples", type=int, default=None)
    p.add_argument("--max-val-examples", type=int, default=None)
    p.add_argument("--max-test-examples", type=int, default=None)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--test-size", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-vocab", type=int, default=30000)
    p.add_argument("--min-freq", type=int, default=2)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--pretokenize", action="store_true", help="Tokenize once and save to disk under out_dir/tokenized/*")
    # model
    p.add_argument("--d-model", type=int, default=192)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=4)
    p.add_argument("--mlp-ratio", type=float, default=4.0)
    p.add_argument("--dropout", type=float, default=0.1)
    # train
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--no-amp", action="store_true")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", type=str, default="runs/latest")
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)

    init_distributed()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dcfg = DataConfig(
        dataset_path=args.dataset_path,
        max_vocab=args.max_vocab,
        min_freq=args.min_freq,
        max_len=args.max_len,
        val_size=args.val_size,
        test_size=args.test_size,
        seed=args.seed,
    )

    # Load and split
    raw = load_civil_comments(dcfg, max_examples=args.max_examples)
    raw = add_binary_label(raw, dcfg)
    dsd = stratified_split(raw, dcfg)

    # Optionally truncate splits
    if args.max_train_examples:
        dsd["train"] = dsd["train"].select(range(min(args.max_train_examples, len(dsd["train"]))))
    if args.max_val_examples:
        dsd["val"] = dsd["val"].select(range(min(args.max_val_examples, len(dsd["val"]))))
    if args.max_test_examples:
        dsd["test"] = dsd["test"].select(range(min(args.max_test_examples, len(dsd["test"]))))

    # Build vocab on train split only
    vocab = build_vocab_from_split(dsd, dcfg)

    # Tokenize (on the fly mapping or pretokenize)
    tok_root = os.path.join(args.out_dir, "tokenized")
    if args.pretokenize:
        train_ds = tokenize_dataset(dsd["train"], dcfg, vocab, pretokenize_dir=os.path.join(tok_root, "train"))
        val_ds = tokenize_dataset(dsd["val"], dcfg, vocab, pretokenize_dir=os.path.join(tok_root, "val"))
        test_ds = tokenize_dataset(dsd["test"], dcfg, vocab, pretokenize_dir=os.path.join(tok_root, "test"))
    else:
        train_ds = tokenize_dataset(dsd["train"], dcfg, vocab, pretokenize_dir=None)
        val_ds = tokenize_dataset(dsd["val"], dcfg, vocab, pretokenize_dir=None)
        test_ds = tokenize_dataset(dsd["test"], dcfg, vocab, pretokenize_dir=None)

    pos_weight = compute_pos_weight(train_ds)

    collate = lambda feats: collate_fn(feats, pad_id=vocab.pad_id)
    train_loader, _ = make_loader(train_ds, collate, batch_size=args.batch_size, shuffle=True)
    val_loader, _ = make_loader(val_ds, collate, batch_size=args.batch_size, shuffle=False)
    test_loader, _ = make_loader(test_ds, collate, batch_size=args.batch_size, shuffle=False)

    mcfg = ModelConfig(
        vocab_size=len(vocab.itos),
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )

    model = ToxicityTransformer(mcfg, pad_id=vocab.pad_id, cls_id=vocab.cls_id).to(device)

    # DDP wrap if needed
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        # move model to correct device already
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[torch.cuda.current_device()] if device.type == "cuda" else None
        )

    tcfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        max_steps=args.max_steps,
        grad_clip=args.grad_clip,
        amp=not args.no_amp,
        log_every=args.log_every,
        out_dir=args.out_dir,
    )

    if is_main_process():
        os.makedirs(args.out_dir, exist_ok=True)
        # save vocab for eval
        import json
        with open(os.path.join(args.out_dir, "vocab.json"), "w", encoding="utf-8") as f:
            json.dump({"itos": vocab.itos}, f)

    train(model, train_loader, val_loader, device, pos_weight=pos_weight, cfg=tcfg)

    cleanup_distributed()

if __name__ == "__main__":
    main()
