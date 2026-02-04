# Robust Toxicity Classification — Custom Lightweight Transformer (PyTorch)

This repository implements a **small Transformer encoder** for toxicity classification on the **Civil Comments** dataset.
It uses a **custom architecture** (no off-the-shelf pretrained weights) built from standard PyTorch layers.

**Labeling rule**: `target >= 0.5` → **Toxic (1)** else **Non-toxic (0)**.

## Highlights
- Custom **pre-layernorm Transformer** implemented from scratch using `nn.MultiheadAttention`, `nn.Linear`, `nn.LayerNorm`.
- Efficient data pipeline with an explicit decision between **on-the-fly tokenization** and **pre-tokenization**.
- Single-GPU and **multi-GPU (DDP)** training supported (via `torchrun`).
- Clean package structure + reproducible configs.
- No dataset files or trained model weights are included (downloads happen at runtime).

---

## Quickstart

### 1) Create environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Run a small smoke-train (CPU or single GPU)
```bash
python scripts/train.py --max-train-examples 20000 --max-val-examples 5000 --epochs 2
```

### 3) Multi-GPU (DDP) on a single node
```bash
torchrun --standalone --nproc_per_node=2 scripts/train.py --epochs 2
```

### 4) Evaluate
```bash
python scripts/eval.py --checkpoint runs/latest/best.pt
```

---

## Dataset
This code uses the **HuggingFace `datasets`** loader to fetch the Civil Comments dataset. If your environment cannot
download from the internet, you can:
1. Download/export the dataset externally, and
2. Point `--dataset-path` to a local file (CSV/Parquet) with `text` and `target` columns.

---

## Reproducibility
All key hyperparameters are CLI arguments (see `--help` on scripts). Seeds are set for Python / NumPy / PyTorch.

---

## Repo Layout
- `toxicity_transformer/`
  - `data/` dataset + tokenization + collate
  - `models/` custom Transformer encoder classifier
  - `training/` trainer, metrics, distributed utils
- `scripts/` entrypoints (`train.py`, `eval.py`)
- `report.md` decision log and post-mortem

---

## Notes for reviewers
- The model is intentionally small and fast for iteration and future optimization.
- The goal is correctness, clarity, and solid engineering (not SOTA performance).
