# Short Report: Small Pre-LN Transformer for Civil Comments Toxicity

## Problem framing & data
We use the Civil Comments dataset from Kaggle (Jigsaw Unintended Bias in Toxicity Classification).
The provided `target` is continuous in `[0,1]`. We define a binary label:

- **Toxic** if `target >= 0.5`
- **Non-toxic** otherwise

This aligns with common baselines used on the competition.

## Architecture choice
We implement a lightweight **Transformer encoder** from scratch using standard PyTorch layers:

- Token embedding + learned positional embedding
- `N` × Transformer blocks with **pre-layer normalisation**
- CLS pooling (use the first token representation)
- Linear head → 1 logit

### Pre-LayerNorm block
For each block:
1. `h = LN(x)`
2. `x = x + MHA(h)` (residual)
3. `h = LN(x)`
4. `x = x + MLP(h)` (residual)

**Why pre-LN?**
Pre-LN generally stabilizes optimization for deep residual Transformers because gradients can flow through the residual path without being affected by normalization after the residual addition.

### Default hyperparameters (configurable via CLI)
- Tokenizer: `distilbert-base-uncased` (BERT WordPiece vocabulary)
- `vocab_size`: tokenizer vocab (≈30k for DistilBERT)
- `max_len`: 128
- Embedding / model dim `d_model`: 256
- Heads `n_heads`: 4
- Layers `n_layers`: 4
- FFN size `d_ff`: 1024
- Dropout: 0.1

This is intentionally smaller than BERT; it is meant to demonstrate architecture implementation and engineering.

## Loss function
We use `BCEWithLogits`:
- Numerically stable sigmoid + binary cross-entropy in one function
- The model outputs a **single logit** per input

## Handling class imbalance
Civil Comments is imbalanced (toxic is the minority). Two common strategies:

1. **Weighted loss** (implemented): use `pos_weight` in `BCEWithLogitsLoss` computed from training split:
   - `pos_weight = (#negative / #positive)`
   - This upweights toxic examples and improves recall / AUC in many settings.

2. **Sampling** (not enabled by default): could use `WeightedRandomSampler` or over/under-sampling.
   - We prefer loss weighting since it is simpler and works well with DDP samplers.

The code logs AUC and classification metrics at threshold 0.5.

## Data pipeline & split strategy
- We perform a **strict stratified** train/val split based on `target>=0.5` to preserve class ratio.
- Two tokenization modes are provided:
  - **On-the-fly** tokenization (default): flexible and simple.
  - **Memmap pre-tokenization** (`--use_memmap`): faster for multi-epoch runs and large data. It stores `input_ids` and `attention_mask` arrays on disk and memory-maps them.

## Engineering & MLOps notes
- Supports single-process training or multi-GPU distributed training via **torchrun** (DDP).
- Uses `DistributedSampler` for correct sharding across processes.
- Only rank 0 saves checkpoints and writes submission files.
- Mixed precision is supported via `--use_amp` (when CUDA is available).

## Post-mortem: what to change for 10k req/s
If deploying at 10k requests/second, I would:
1. **Export / optimize inference**: TorchScript or `torch.compile`, or ONNX/TensorRT on GPU.
2. **Batching**: dynamic micro-batching at the service layer; avoid per-request launches.
3. **Tokenizer optimization**: move tokenization off the critical path (pre-tokenize in client, or use a fast Rust tokenizer service).
4. **Model improvements**: add quantization (INT8) and potentially distill further.
5. **Serving architecture**: horizontally scale stateless inference pods behind a load balancer, keep a warm tokenizer/model cache.
6. **Monitoring**: latency SLOs, drift detection, and tracing.
