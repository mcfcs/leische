"""Stage 1: shared XLM-R encoder + pooling + projection (guide §4.1).

ONE encoder for every text unit (target, conversational items, temporal items,
retrieval exemplars) — a deliberate small-data choice from the thesis; do not
give each channel its own encoder.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class SharedEncoder(nn.Module):
    def __init__(self, encoder_name: str, d_model: int, pooling: str = "mean",
                 freeze_bottom_layers: int = 0):
        super().__init__()
        if pooling not in ("mean", "cls"):
            raise ValueError(f"pooling must be mean|cls, got {pooling!r}")
        self.backbone = AutoModel.from_pretrained(encoder_name)
        self.pooling = pooling
        hidden = self.backbone.config.hidden_size
        self.proj = nn.Linear(hidden, d_model)
        if freeze_bottom_layers > 0:
            self._freeze_bottom(freeze_bottom_layers)

    def _freeze_bottom(self, n_layers: int) -> None:
        """§9.4: freeze embeddings + bottom n encoder layers."""
        for p in self.backbone.embeddings.parameters():
            p.requires_grad = False
        for layer in self.backbone.encoder.layer[:n_layers]:
            for p in layer.parameters():
                p.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """[n_texts, seq] → [n_texts, d_model]."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state  # [n, seq, hidden]
        if self.pooling == "mean":
            mask = attention_mask.unsqueeze(-1).to(h.dtype)
            pooled = (h * mask).sum(1) / mask.sum(1).clamp(min=1e-6)
        else:
            pooled = h[:, 0]
        return self.proj(pooled)

    def forward_chunked(self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
                        chunk: int = 64) -> torch.Tensor:
        """Encode a large flattened batch of context items without OOM (8 GB VRAM)."""
        if input_ids.shape[0] <= chunk:
            return self.forward(input_ids, attention_mask)
        parts = [
            self.forward(input_ids[i:i + chunk], attention_mask[i:i + chunk])
            for i in range(0, input_ids.shape[0], chunk)
        ]
        return torch.cat(parts, dim=0)


class Tokenize:
    """Tokenizer with the guide's per-type truncation budgets."""

    def __init__(self, encoder_name: str, max_len_target: int = 192,
                 max_len_context: int = 96, max_len_selftext: int = 128):
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.budgets = {"target": max_len_target, "context": max_len_context,
                        "selftext": max_len_selftext}

    def __call__(self, texts: list[str], kind: str = "context") -> dict[str, torch.Tensor]:
        if not texts:
            return {"input_ids": torch.zeros(0, 1, dtype=torch.long),
                    "attention_mask": torch.zeros(0, 1, dtype=torch.long)}
        enc = self.tokenizer(texts, truncation=True, max_length=self.budgets[kind],
                             padding=True, return_tensors="pt")
        return {"input_ids": enc["input_ids"], "attention_mask": enc["attention_mask"]}

    def token_count(self, text: str) -> int:
        return len(self.tokenizer(text, truncation=False)["input_ids"])
