from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class ModelConfig:
    vocab_size: int
    max_len: int = 256
    d_model: int = 192
    n_heads: int = 4
    n_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1

class PreLNTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        hidden = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None, key_padding_mask: torch.Tensor | None = None):
        # Pre-LN self-attention
        h = self.ln1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)
        x = x + self.drop1(attn_out)
        # Pre-LN FFN
        x = x + self.mlp(self.ln2(x))
        return x

class ToxicityTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, pad_id: int, cls_id: int):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.cls_id = cls_id

        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        self.blocks = nn.ModuleList([
            PreLNTransformerBlock(cfg.d_model, cfg.n_heads, cfg.mlp_ratio, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, 1)

        # Learned CLS embedding (optional; we already insert CLS id in tokenizer)
        # Keeping it simple: CLS token uses regular token embedding.

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids: [B,T], attention_mask: [B,T] with 1 for tokens, 0 for pad
        B, T = input_ids.shape
        if T > self.cfg.max_len:
            input_ids = input_ids[:, : self.cfg.max_len]
            attention_mask = attention_mask[:, : self.cfg.max_len]
            T = input_ids.shape[1]

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)  # [B,T]
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        # key_padding_mask: True for pad positions
        key_padding_mask = attention_mask == 0

        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.ln_f(x)

        # CLS pooling: CLS is first token
        cls = x[:, 0, :]  # [B, d_model]
        logits = self.head(cls).squeeze(-1)  # [B]
        return logits
