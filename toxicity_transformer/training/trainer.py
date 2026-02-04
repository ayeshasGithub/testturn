from __future__ import annotations

import os
import time
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from .distributed import is_main_process, get_world_size, is_dist_avail_and_initialized
from .metrics import compute_metrics, as_dict

@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 128
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 200
    max_steps: Optional[int] = None
    grad_clip: float = 1.0
    amp: bool = True
    log_every: int = 50
    out_dir: str = "runs/latest"

def cosine_schedule(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + np.cos(np.pi * progress))

def make_loader(ds, collate, batch_size: int, shuffle: bool, num_workers: int = 4):
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(ds, shuffle=shuffle)
        shuffle = False
    else:
        sampler = None
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, sampler=sampler, collate_fn=collate,
                      num_workers=num_workers, pin_memory=torch.cuda.is_available(), drop_last=False), sampler

def save_checkpoint(path: str, model: nn.Module, optimizer: torch.optim.Optimizer, step: int, best_metric: float, extra: Dict[str, Any]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    state = {
        "model": (model.module.state_dict() if hasattr(model, "module") else model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "step": step,
        "best_metric": best_metric,
        "extra": extra,
    }
    torch.save(state, path)

@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Dict[str, float]:
    model.eval()
    all_y = []
    all_p = []
    for batch in loader:
        input_ids = batch.input_ids.to(device)
        attention_mask = batch.attention_mask.to(device)
        labels = batch.labels.to(device)
        logits = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(logits)
        all_y.append(labels.detach().cpu().numpy())
        all_p.append(probs.detach().cpu().numpy())
    y = np.concatenate(all_y, axis=0)
    p = np.concatenate(all_p, axis=0)
    m = compute_metrics(y_true=y, y_prob=p)
    return as_dict(m)

def train(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device,
          pos_weight: float, cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    if is_main_process():
        with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg.__dict__, f, indent=2)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))

    model.train()
    global_step = 0
    best_f1 = -1.0
    start = time.time()

    total_steps = cfg.max_steps
    if total_steps is None:
        total_steps = cfg.epochs * len(train_loader)

    pbar = tqdm(total=total_steps, disable=not is_main_process(), desc="train")
    for epoch in range(cfg.epochs):
        if hasattr(train_loader, "sampler") and isinstance(train_loader.sampler, DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            model.train()
            input_ids = batch.input_ids.to(device, non_blocking=True)
            attention_mask = batch.attention_mask.to(device, non_blocking=True)
            labels = batch.labels.to(device, non_blocking=True)

            lr_scale = cosine_schedule(global_step, total_steps, cfg.warmup_steps)
            for pg in optimizer.param_groups:
                pg["lr"] = cfg.lr * lr_scale

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                logits = model(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            if cfg.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            if is_main_process() and global_step % cfg.log_every == 0:
                elapsed = time.time() - start
                pbar.write(f"step={global_step} loss={loss.item():.4f} lr={optimizer.param_groups[0]['lr']:.2e} t={elapsed:.1f}s")

            global_step += 1
            pbar.update(1)

            if global_step >= total_steps:
                break

        # Validation each epoch (main process only for logging/checkpointing)
        if is_main_process():
            metrics = evaluate(model, val_loader, device)
            pbar.write(f"[val] epoch={epoch} " + " ".join([f"{k}={v:.4f}" for k,v in metrics.items()]))
            # checkpoint best by F1
            if metrics["f1"] > best_f1:
                best_f1 = metrics["f1"]
                save_checkpoint(os.path.join(cfg.out_dir, "best.pt"), model, optimizer, global_step, best_f1, {"val": metrics})
            save_checkpoint(os.path.join(cfg.out_dir, "last.pt"), model, optimizer, global_step, best_f1, {"val": metrics})

        if global_step >= total_steps:
            break

    pbar.close()
    return {"best_f1": best_f1, "steps": global_step}
