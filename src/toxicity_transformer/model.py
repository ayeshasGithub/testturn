from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PreLNTransformerBlock(nn.Module):
    """Pre-LayerNorm Transformer encoder block:
      x = x + MHA(LN(x))
      x = x + MLP(LN(x))
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=True
        )
        self.drop1 = nn.Dropout(dropout)

        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop_ff = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor]) -> torch.Tensor:
        h = self.ln1(x)
        attn_out, _ = self.mha(
            h, h, h, key_padding_mask=key_padding_mask, need_weights=False
        )
        x = x + self.drop1(attn_out)

        h = self.ln2(x)
        h = self.fc2(self.drop_ff(F.gelu(self.fc1(h))))
        x = x + self.drop2(h)
        return x


class SmallTransformerEncoder(nn.Module):
    """Minimal text encoder:
    - Token + positional embeddings
    - N x pre-LN transformer blocks
    - CLS pooling (first token) + linear head
    """

    def __init__(
        self,
        vocab_size: int,
        max_len: int,
        d_model: int = 256,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 1024,
        dropout: float = 0.1,
        pad_id: int = 0,
    ):
        super().__init__()
        self.max_len = max_len
        self.pad_id = pad_id

        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [
                PreLNTransformerBlock(
                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout
                )
                for _ in range(n_layers)
            ]
        )
        self.ln_final = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

        # init (simple)
        nn.init.normal_(self.tok_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward.

        input_ids: (B, T)
        attention_mask: (B, T) with 1 for real tokens, 0 for pad
        """
        B, T = input_ids.shape
        if T > self.max_len:
            raise ValueError(f"Sequence length {T} exceeds max_len {self.max_len}")

        pos = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)

        key_padding_mask = attention_mask == 0  # True for PAD tokens
        for blk in self.blocks:
            x = blk(x, key_padding_mask=key_padding_mask)

        x = self.ln_final(x)
        cls = x[:, 0, :]
        logits = self.head(cls).squeeze(-1)
        return logits
