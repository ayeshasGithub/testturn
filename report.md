# Short report — Robust Toxicity Classification (Civil Comments)

## 1) Architecture choice
**Model**: a compact Transformer encoder for sequence classification.

- **Token embeddings**: `nn.Embedding(vocab_size, d_model)`
- **Positional embeddings** (learned): `nn.Embedding(max_len, d_model)`
- **N Transformer blocks** (pre-layernorm):
  - `x = x + SelfAttention(LN(x))`
  - `x = x + MLP(LN(x))`
- **Pooling**: prepend a learned **[CLS] token**; use its final hidden state for classification.
- **Classifier head**: `Linear(d_model -> 1)` producing a logit.

### Pre-layernorm (Pre-LN) rationale
Pre-LN places `LayerNorm` **before** attention/MLP sublayers. Compared to Post-LN, Pre-LN often yields:
- more stable optimization (especially for deeper stacks),
- better gradient flow through residual paths,
- fewer training instabilities at higher learning rates.

Implementation uses standard PyTorch modules (`nn.MultiheadAttention`, `nn.Linear`, `nn.LayerNorm`) only.

## 2) Loss, imbalance handling, and metrics
**Loss**: `BCEWithLogitsLoss(pos_weight=...)`

- Binary toxicity label from `target >= 0.5`.
- Civil Comments is typically imbalanced (toxic << non-toxic). To compensate, we compute:
  - `pos_weight = (#negative / #positive)` on the **training split**.
  - This increases the loss contribution from toxic examples without oversampling.

**Metrics**: accuracy, precision, recall, F1, AUROC (computed on validation/test).

## 3) Data pipeline decisions
Two supported approaches:
1. **On-the-fly tokenization** (default):
   - Pros: simple, flexible, avoids storing extra artifacts.
   - Cons: CPU tokenization cost each epoch.
2. **Pre-tokenization** (`--pretokenize`):
   - Tokenize once, store integer IDs to disk.
   - Pros: faster subsequent epochs/runs.
   - Cons: storage + complexity.

For a take-home, on-the-fly is the default with the option to pretokenize when scaling.

### Tokenizer
A lightweight tokenizer is implemented:
- lowercasing + basic punctuation splitting,
- vocabulary built from the training split with `max_vocab`,
- special tokens: `[PAD]`, `[UNK]`, `[CLS]`.

This keeps full control over the full stack and avoids pretrained tokenizers/weights.

## 4) Train/val/test split
- Uses a strict split with **stratification** on the binary toxic label.
- Default: 80/10/10, configurable.
- Stratification preserves the label distribution across splits.

## 5) Hyperparameters (defaults; change via CLI)
- `max_vocab=30000`
- `d_model=192`
- `n_layers=4`
- `n_heads=4`
- `mlp_ratio=4.0` (FFN dim = `4*d_model`)
- `dropout=0.1`
- `max_len=256`
- Optimizer: `AdamW(lr=3e-4, weight_decay=0.01)`
- LR schedule: cosine w/ warmup (configurable)
- Batch size: 128 (per GPU)

Reasoning: keep the model small enough for fast iteration and future pruning/quantization.

## 6) Post-mortem: deploying at 10k requests/sec
If deploying for 10k RPS, I would:
1. **Optimize tokenization**
   - Pre-tokenize common phrases / use a faster compiled tokenizer implementation.
   - Consider a character-level fallback for OOV robustness.
2. **Quantize + fuse**
   - Apply INT8 quantization (PTQ/QAT) for linear layers.
   - Use kernel fusions where possible (e.g., FlashAttention-style attention if allowed).
3. **Batching + concurrency**
   - Micro-batch requests (e.g., 4–16ms window) to increase GPU utilization.
   - Run multiple model replicas with a load balancer.
4. **Model changes**
   - Reduce `d_model` / layers; distill into a smaller student.
   - Replace standard attention with linear attention variants if long context needed.
5. **Serving**
   - Export to TorchScript / ONNX, run in Triton or a custom gRPC service.
   - Add caching for repeated comments, aggressive timeouts, and health checks.
6. **Monitoring**
   - Track drift, calibration, latency percentiles, and false positive rate on curated eval sets.
