# Toxicity Transformer (Civil Comments) — `uv` + Docker Workflow

This repository trains a **small Transformer encoder implemented from scratch in PyTorch** (pre-layer norm blocks) for toxicity classification on the **Civil Comments** dataset from Kaggle (*Jigsaw Unintended Bias in Toxicity Classification*).

**Labeling rule**

```text
toxic = (target >= 0.5)
```

> ⚠️ This repo does **not** include the Kaggle dataset files or any trained model checkpoints.

---

## 1. Dataset

Download the Kaggle competition data:

* **Jigsaw Unintended Bias in Toxicity Classification** (Civil Comments)

Place files locally using this structure:

```text
data/
├── train.csv
├── test.csv
└── sample_submission.csv
```

Expected columns:

* `train.csv`: `comment_text`, `target`
* `test.csv`: `comment_text`
* `sample_submission.csv`: `id`, `prediction`

---

## 2. Setup

Choose one of the following options.

### Option A: Install `uv` (recommended)

Install **uv** (Astral):

**macOS / Linux**

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows (PowerShell)**

```powershell
irm https://astral.sh/uv/install.ps1 | iex
```

Verify:

```bash
uv --version
```

### Option B: Docker (optional)

Build and run the image:

```bash
docker build -t toxicity-transformer .
docker run --rm -it toxicity-transformer toxicity-train --help
```

If you want the container to access your local `data/` and write `outputs/` back to your machine:

```bash
docker run --rm -it \
  -v "$PWD:/work" -w /work \
  toxicity-transformer \
  toxicity-train --help
```

---

## 3. Create Environment & Install Dependencies (`uv`)

From the repository root:

```bash
uv venv
uv sync
```

This creates a local virtual environment and installs dependencies from `pyproject.toml`.

> Tip: open the repo in VS Code with `code .`

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

### Option B: Run as a module

```bash
uv run python -m toxicity_transformer.train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs
```

Outputs:

```text
outputs/
├── best.pt            # best checkpoint (validation loss)
└── submission.csv     # Kaggle submission file
```

---

## 5. Multi-GPU Training (DDP)

If your environment has issues with libuv, disable it:

```bash
export USE_LIBUV=0
uv run python -c "import os; print(os.getenv('USE_LIBUV'))"
```

Example: 2 GPUs on one machine

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

Notes:

* Uses `torch.distributed` + `DistributedSampler`
* Only rank 0 writes checkpoints and `submission.csv`

---

## 6. Tokenization Strategy

### Default: on-the-fly tokenization

* Flexible
* Simplest setup
* Recommended for iteration

### Optional: pre-tokenized memmaps (faster for multi-epoch training)

```bash
uv run toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --use_memmap \
  --memmap_dir tokenized_memmap \
  --out_dir outputs
```

Generated files:

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
├── train.py     # training, eval, DDP entrypoint
├── utils.py     # seeding, DDP helpers
report.md        # short design report
```

---

## Troubleshooting

* **CPU-only training**: omit `--use_amp`
* **Tokenizer download issues**:

  * Ensure you have internet access, or
  * Set a writable Hugging Face cache directory:

    ```bash
    export HF_HOME=/path/to/cache
    ```
