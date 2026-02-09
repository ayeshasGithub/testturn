# Toxicity Transformer (Civil Comments) — `uv` Workflow

This repository trains a **small Transformer encoder implemented from scratch in PyTorch** (pre-layer norm blocks) for toxicity classification on the **Civil Comments dataset** from Kaggle (*Jigsaw Unintended Bias in Toxicity Classification*).

**Label rule**

```text
toxic = (target >= 0.5)
```

> ⚠️ No dataset files or trained models are included in this repository.

---

## 1. Dataset

Download the dataset from the Kaggle competition:

* **Jigsaw Unintended Bias in Toxicity Classification** (Civil Comments)

Place the files locally (recommended structure):

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

The code expects:

* `train.csv` with columns:

  * `comment_text`
  * `target`
* `test.csv` with column:

  * `comment_text`
* `sample_submission.csv` with columns:

  * `id`
  * `prediction`

---

## 2. Install `uv`

Install **uv** (Astral). Any of the following methods work.

### macOS / Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows (PowerShell)

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

### Verify installation

```bash
uv --version
```
### 2. Option B: Docker (optional)
```bash
docker build -t toxicity-transformer .
docker run --rm -it -v "$PWD:/work" -w /work toxicity-transformer toxicity-train --help
```

---

## 3. Create Environment & Install Dependencies

From the repository root (e.g. `toxicity_transformer_repo/`):

```bash
uv venv
uv sync
```

This creates a local virtual environment and installs dependencies from `pyproject.toml`.

> Tip: You can open the repo in VS Code using `code .`

---

## 4. Run Training (Single GPU or CPU)

### Option A: Run the console script

```bash
uv run toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs \
  --epochs 5 \
  --batch_size 32 \
  --max_len 128 \
  --tokenizer_name distilbert-base-uncased
```

### Option B: Run as a Python module

```bash
uv run python -m toxicity_transformer.train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs
```

### Outputs

```text
outputs/
├── best.pt            # Best checkpoint (by validation loss)
└── submission.csv     # Kaggle submission file
```

---

## 5. Multi-GPU Training (DDP)

Disable libuv if needed:

```bash
export USE_LIBUV=0
uv run python -c "import os; print(os.getenv('USE_LIBUV'))"
```

### Example: 2 GPUs on a single machine

```bash
uv run torchrun --nproc_per_node=2 -m toxicity_transformer.train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs \
  --epochs 5 \
  --batch_size 32 \
  --use_amp
```

**Notes**

* Uses `torch.distributed` with `DistributedSampler`
* Only rank 0 writes checkpoints and submission artifacts

---

## 6. Tokenization Strategy

### Default: On-the-fly tokenization

* Flexible
* Simplest setup
* Recommended for experimentation

### Optional: Pre-tokenized memmaps (faster for multi-epoch training)

```bash
uv run toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --use_memmap \
  --memmap_dir tokenized_memmap \
  --out_dir outputs
```

This generates:

```text
tokenized_memmap/
├── *_input_ids.memmap
├── *_attention_mask.memmap
├── *_targets.memmap      # train/val only
└── *_meta.json           # shape metadata
```

---

## 7. Project Layout

```text
src/toxicity_transformer/
├── data.py      # datasets, loaders, memmap support
├── model.py     # pre-LN Transformer encoder
├── train.py     # training, evaluation, DDP entrypoint
├── utils.py     # seeding, DDP helpers
report.md        # short design report
```

---

## Troubleshooting

* **CPU-only training**: omit `--use_amp`
* **Tokenizer download issues**:

  * Ensure internet access, or
  * Set a writable Hugging Face cache directory:

    ```bash
    export HF_HOME=/path/to/cache
    ```

