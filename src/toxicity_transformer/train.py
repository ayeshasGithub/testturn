import argparse
import os
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.data import DistributedSampler
from transformers import AutoTokenizer

from toxicity_transformer.data import build_loaders, preprocess_to_memmap
from toxicity_transformer.model import SmallTransformerEncoder
from toxicity_transformer.utils import (
    seed_everything,
    init_distributed,
    get_rank,
    get_world_size,
    is_main_process,
    ddp_barrier,
    cleanup_memory,
    sigmoid_np,
)


def train_one_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float,
    use_amp: bool,
    bce_pos_weight: torch.Tensor,
) -> float:
    model.train()
    losses = []
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(
                logits, targets, pos_weight=bce_pos_weight
            )

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_metrics(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()
    losses = []
    all_logits = []
    all_targets = []

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["target"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
            loss = F.binary_cross_entropy_with_logits(logits, targets)

        losses.append(loss.item())
        all_logits.append(logits.detach().cpu().numpy())
        all_targets.append(targets.detach().cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    targets_np = np.concatenate(all_targets, axis=0)

    y_true = (targets_np >= threshold).astype(np.int32)
    y_prob = sigmoid_np(logits_np)
    y_pred = (y_prob >= threshold).astype(np.int32)

    val_loss = float(np.mean(losses)) if losses else float("nan")

    if y_true.min() == y_true.max():
        val_auc = float("nan")
    else:
        val_auc = float(roc_auc_score(y_true, y_prob))

    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    return {
        "val_loss": val_loss,
        "val_auc": val_auc,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


@torch.no_grad()
def predict_test_probs(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    use_amp: bool,
) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(input_ids, attention_mask)
        probs.append(sigmoid_np(logits.detach().cpu().numpy()))
    return np.concatenate(probs, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()

    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, required=True)
    p.add_argument("--sample_sub_csv", type=str, required=True)

    p.add_argument("--text_col", type=str, default="comment_text")
    p.add_argument("--target_col", type=str, default="target")

    p.add_argument("--tokenizer_name", type=str, default="distilbert-base-uncased")
    p.add_argument("--max_len", type=int, default=128)

    # model
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--n_layers", type=int, default=4)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.1)

    # train
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=2)
    p.add_argument("--use_amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    # data loading
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--pin_memory", action="store_true")
    p.add_argument("--use_memmap", action="store_true")
    p.add_argument("--memmap_dir", type=str, default="./tokenized_memmap")

    p.add_argument("--out_dir", type=str, default="./outputs")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    init_distributed()
    seed_everything(args.seed)

    world_size = get_world_size()
    rank = get_rank()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if is_main_process():
        os.makedirs(args.out_dir, exist_ok=True)

    # Load CSVs
    train_df = pd.read_csv(args.train_csv)
    test_df = pd.read_csv(args.test_csv)
    sub_df = pd.read_csv(args.sample_sub_csv)

    # Clean + enforce types
    train_df = train_df.dropna(subset=[args.text_col]).copy()
    train_df[args.target_col] = train_df[args.target_col].astype(float)

    # Stratified split by toxic>=0.5
    strat = (train_df[args.target_col].values >= 0.5).astype(int)
    tr_df, va_df = train_test_split(
        train_df, test_size=0.2, random_state=args.seed, stratify=strat
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Optional: build memmaps once (rank0), then all ranks read them.
    if args.use_memmap:
        if is_main_process():
            os.makedirs(args.memmap_dir, exist_ok=True)
            preprocess_to_memmap(
                tr_df,
                tokenizer,
                args.max_len,
                args.memmap_dir,
                "train",
                text_col=args.text_col,
                target_col=args.target_col,
                is_test=False,
            )
            preprocess_to_memmap(
                va_df,
                tokenizer,
                args.max_len,
                args.memmap_dir,
                "val",
                text_col=args.text_col,
                target_col=args.target_col,
                is_test=False,
            )
            preprocess_to_memmap(
                test_df,
                tokenizer,
                args.max_len,
                args.memmap_dir,
                "test",
                text_col=args.text_col,
                target_col=args.target_col,
                is_test=True,
            )
        ddp_barrier()

    train_loader, val_loader, test_loader = build_loaders(
        tr_df,
        va_df,
        test_df,
        tokenizer=tokenizer,
        max_len=args.max_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory and torch.cuda.is_available(),
        use_memmap=args.use_memmap,
        memmap_dir=args.memmap_dir,
        text_col=args.text_col,
        target_col=args.target_col,
        world_size=world_size,
    )

    # Model
    model = SmallTransformerEncoder(
        vocab_size=tokenizer.vocab_size,
        max_len=args.max_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        pad_id=pad_id,
    ).to(device)

    if world_size > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp and torch.cuda.is_available())

    # Class imbalance: pos_weight = neg/pos (computed on train split)
    y = (tr_df[args.target_col].values >= 0.5).astype(np.int32)
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    pos_weight_val = (n_neg / max(n_pos, 1))  # avoid divide by zero
    bce_pos_weight = torch.tensor(pos_weight_val, device=device, dtype=torch.float32)

    if is_main_process():
        print(f"[info] world_size={world_size} device={device}")
        print(f"[info] train toxic rate={n_pos/len(y):.4f} pos_weight={pos_weight_val:.3f}")

    best_val_loss = float("inf")
    patience_left = args.patience
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        if isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        t0 = time.time()

        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            max_grad_norm=args.max_grad_norm,
            use_amp=args.use_amp and torch.cuda.is_available(),
            bce_pos_weight=bce_pos_weight,
        )

        # Simple evaluation: run on rank0 only for speed/clarity.
        ddp_barrier()
        if is_main_process():
            metrics = evaluate_metrics(
                model=model.module if hasattr(model, "module") else model,
                loader=val_loader,
                device=device,
                use_amp=args.use_amp and torch.cuda.is_available(),
                threshold=0.5,
            )
            dt = time.time() - t0
            print(
                f"[Epoch {epoch:02d}] "
                f"train_loss={train_loss:.4f} "
                f"val_loss={metrics['val_loss']:.4f} "
                f"val_auc={metrics['val_auc']:.4f} "
                f"acc={metrics['accuracy']:.4f} "
                f"prec={metrics['precision']:.4f} "
                f"rec={metrics['recall']:.4f} "
                f"f1={metrics['f1']:.4f} "
                f"time={dt:.1f}s"
            )

            if metrics["val_loss"] < best_val_loss:
                best_val_loss = metrics["val_loss"]
                patience_left = args.patience
                state_dict = (model.module.state_dict() if hasattr(model, "module") else model.state_dict())
                torch.save({"model": state_dict, "args": vars(args)}, best_path)
                print(f"  ✓ saved best checkpoint: {best_path}")
            else:
                patience_left -= 1
                print(f"  patience left: {patience_left}")
                if patience_left <= 0:
                    print("  early stopping.")
                    break

        ddp_barrier()
        cleanup_memory()

    ddp_barrier()
    if is_main_process():
        ckpt = torch.load(best_path, map_location=device)
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])

        probs = predict_test_probs(
            model=model.module if hasattr(model, "module") else model,
            loader=test_loader,
            device=device,
            use_amp=args.use_amp and torch.cuda.is_available(),
        )

        out_sub = sub_df.copy()
        out_sub["prediction"] = probs
        out_path = os.path.join(args.out_dir, "submission.csv")
        out_sub.to_csv(out_path, index=False)
        print(f"Saved submission -> {out_path}")


if __name__ == "__main__":
    main()
