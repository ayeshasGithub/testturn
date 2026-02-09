# Toxicity Transformer (Civil Comments) — uv workflow

This repo trains a **small Transformer encoder implemented from scratch in PyTorch** (pre-layer norm blocks) for toxicity classification on the **Civil Comments dataset** (Kaggle: *Jigsaw Unintended Bias in Toxicity Classification*).

**Label rule:** `toxic = (target >= 0.5)`.

No data or trained models are included.

---

## 1) Dataset

Download from Kaggle competition:
- *Jigsaw Unintended Bias in Toxicity Classification* (Civil Comments)

Place these files locally (recommended):

data/
train.csv
test.csv
sample_submission.csv

The code expects:
- `train.csv` with columns `comment_text`, `target`
- `test.csv` with column `comment_text`
- `sample_submission.csv` with `id` and `prediction` columns

---

## 2) Install uv

Install `uv` (Astral). Any of these work:

### macOS / Linux
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh

Windows (PowerShell)
irm https://astral.sh/uv/install.ps1 | iex

Verify:
uv --version


3) Create env + install deps (uv)
You might use cod . to run it from vscode
From the repo root(turnit_transformer_repo/toxicity_transformer_repo):

uv venv
uv sync

This creates a local virtual environment and installs dependencies from pyproject.toml.

4) Run training (single GPU or CPU)
Option A: run the console script
uv run toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs \
  --epochs 5 \
  --batch_size 32 \
  --max_len 128 \
  --tokenizer_name distilbert-base-uncased

Option B: run as a module
uv run python -m toxicity_transformer.train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs

Outputs:
outputs/best.pt (best checkpoint by validation loss)
outputs/submission.csv (Kaggle submission file)

5) Multi-GPU training (DDP)
set USE_LIBUV=0
uv run python -c "import os; print(os.getenv('USE_LIBUV'))"

Example on 2 GPUs (single machine):
uv run torchrun --nproc_per_node=2 -m toxicity_transformer.train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs \
  --epochs 5 \
  --batch_size 32 \
  --use_amp

Notes:
DDP uses torch.distributed and DistributedSampler.
Only rank 0 writes checkpoints and submission artifacts.

6) Tokenization strategy (on-the-fly vs memmap)
Default: on-the-fly tokenization (flexible; simplest).
For faster multi-epoch training, use pre-tokenized memmaps:
uv run toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --use_memmap \
  --memmap_dir tokenized_memmap \
  --out_dir outputs

This creates:
tokenized_memmap/*_input_ids.memmap
tokenized_memmap/*_attention_mask.memmap
tokenized_memmap/*_targets.memmap (train/val only)
tokenized_memmap/*_meta.json (shape metadata)

7) Project layout
src/toxicity_transformer/
  data.py     # datasets + loaders + memmap option
  model.py    # pre-LN transformer encoder
  train.py    # training / evaluation / DDP entrypoint
  utils.py    # seeding, DDP helpers
report.md     # short design report


Troubleshooting
If you run on CPU, omit --use_amp.
If transformers can’t download the tokenizer, ensure you have internet access or set/cache Hugging Face assets:
You can set HF_HOME to a writable cache directory.













