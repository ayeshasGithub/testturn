# Toxicity Transformer (Civil Comments)

This repo trains a **small Transformer encoder implemented from scratch in PyTorch** (pre-layer norm blocks) for toxicity classification on the **Civil Comments dataset** (Kaggle: *Jigsaw Unintended Bias in Toxicity Classification*).

**Label rule:** `toxic = (target >= 0.5)`.

No data or trained models are included.

---

## 1) Dataset

Download from Kaggle competition:
- *Jigsaw Unintended Bias in Toxicity Classification* (Civil Comments)

Place these files locally (recommended):
```
data/
  train.csv
  test.csv
  sample_submission.csv
```

The code expects:
- `train.csv` with columns `comment_text`, `target`
- `test.csv` with column `comment_text`
- `sample_submission.csv` with `id` and `prediction` columns

---

## 2) Install

### Option A: pip (editable)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Option B: Docker (optional)
```bash
docker build -t toxicity-transformer .
docker run --rm -it -v "$PWD:/work" -w /work toxicity-transformer toxicity-train --help
```

---

## 3) Train (single GPU or CPU)

```bash
toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs \
  --epochs 5 \
  --batch_size 32 \
  --max_len 128 \
  --tokenizer_name distilbert-base-uncased
```

Outputs:
- `outputs/best.pt` (best checkpoint by validation loss)
- `outputs/submission.csv` (Kaggle submission file)

---

## 4) Train with multiple GPUs (DDP)

Example on 2 GPUs (single machine):
```bash
torchrun --nproc_per_node=2 -m toxicity_transformer.train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs \
  --epochs 5 \
  --batch_size 32 \
  --use_amp
```

Notes:
- DDP uses `torch.distributed` and `DistributedSampler`.
- Only rank 0 writes checkpoints and submission artifacts.

---

## 5) Tokenization strategy (on-the-fly vs memmap)

Default: **on-the-fly tokenization** (more flexible; simpler).
For faster multi-epoch training, use **pre-tokenized memmaps**:

```bash
toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --use_memmap \
  --memmap_dir tokenized_memmap \
  --out_dir outputs
```

This creates:
- `tokenized_memmap/*_input_ids.memmap`
- `tokenized_memmap/*_attention_mask.memmap`
- `tokenized_memmap/*_targets.memmap` (train/val only)
- `tokenized_memmap/*_meta.json` (shape metadata)

---

## 6) Project layout

```
src/toxicity_transformer/
  data.py     # datasets + loaders + memmap option
  model.py    # pre-LN transformer encoder
  train.py    # training / evaluation / DDP entrypoint
  utils.py    # seeding, DDP helpers
report.md     # short design report
```

---

## Troubleshooting

- If you run on CPU, omit `--use_amp`.
- If `transformers` canâ€™t download the tokenizer, pre-download models or provide `HF_HOME` cache volume in Docker.














# Toxicity Transformer (Civil Comments)

This repo trains a **small Transformer encoder implemented from scratch in PyTorch** (pre-layer norm blocks) for toxicity classification on the **Civil Comments dataset** (Kaggle: *Jigsaw Unintended Bias in Toxicity Classification*).

**Label rule:** `toxic = (target >= 0.5)`.

No data or trained models are included.

---

## 1) Dataset

Download from Kaggle competition:
- *Jigsaw Unintended Bias in Toxicity Classification* (Civil Comments)

Place these files locally (recommended):
```
data/
  train.csv
  test.csv
  sample_submission.csv
```

The code expects:
- `train.csv` with columns `comment_text`, `target`
- `test.csv` with column `comment_text`
- `sample_submission.csv` with `id` and `prediction` columns

---

## 2) Install

### Option A: pip (editable)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
```

### Option B: Docker (optional)
```bash
docker build -t toxicity-transformer .
docker run --rm -it -v "$PWD:/work" -w /work toxicity-transformer toxicity-train --help
```

---

## 3) Train (single GPU or CPU)

```bash
toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs \
  --epochs 5 \
  --batch_size 32 \
  --max_len 128 \
  --tokenizer_name distilbert-base-uncased
```

Outputs:
- `outputs/best.pt` (best checkpoint by validation loss)
- `outputs/submission.csv` (Kaggle submission file)

---

## 4) Train with multiple GPUs (DDP)

Example on 2 GPUs (single machine):
```bash
torchrun --nproc_per_node=2 -m toxicity_transformer.train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --out_dir outputs \
  --epochs 5 \
  --batch_size 32 \
  --use_amp
```

Notes:
- DDP uses `torch.distributed` and `DistributedSampler`.
- Only rank 0 writes checkpoints and submission artifacts.

---

## 5) Tokenization strategy (on-the-fly vs memmap)

Default: **on-the-fly tokenization** (more flexible; simpler).
For faster multi-epoch training, use **pre-tokenized memmaps**:

```bash
toxicity-train \
  --train_csv data/train.csv \
  --test_csv data/test.csv \
  --sample_sub_csv data/sample_submission.csv \
  --use_memmap \
  --memmap_dir tokenized_memmap \
  --out_dir outputs
```

This creates:
- `tokenized_memmap/*_input_ids.memmap`
- `tokenized_memmap/*_attention_mask.memmap`
- `tokenized_memmap/*_targets.memmap` (train/val only)
- `tokenized_memmap/*_meta.json` (shape metadata)

---

## 6) Project layout

```
src/toxicity_transformer/
  data.py     # datasets + loaders + memmap option
  model.py    # pre-LN transformer encoder
  train.py    # training / evaluation / DDP entrypoint
  utils.py    # seeding, DDP helpers
report.md     # short design report
```

---

## Troubleshooting

- If you run on CPU, omit `--use_amp`.
- If `transformers` can’t download the tokenizer, pre-download models or provide `HF_HOME` cache volume in Docker.
