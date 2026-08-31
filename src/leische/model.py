"""ContextAwareSarcasmModel — all five stages of guide §4 in one class.

The RQ1 context-agnostic baseline IS this class with use_conv=use_temp=use_ret
=False (classifier reduces exactly to MLP([t])), so baseline and full model can
never drift apart. Disabled channels are REMOVED from the gate softmax
(renormalized over active ones), not zero-filled (§7.3).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import LeischeConfig
from .encoders import SharedEncoder

CHANNELS = ("conv", "temp", "ret")


def scatter_items(flat: torch.Tensor, batch_idx: torch.Tensor, batch_size: int
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, d] flattened items + owner index → padded [B, M, d] and bool mask [B, M]."""
    d = flat.shape[-1]
    counts = torch.bincount(batch_idx, minlength=batch_size)
    max_items = int(counts.max().item()) if counts.numel() and counts.max() > 0 else 1
    out = flat.new_zeros(batch_size, max_items, d)
    mask = torch.zeros(batch_size, max_items, dtype=torch.bool, device=flat.device)
    slot = torch.zeros(batch_size, dtype=torch.long, device=flat.device)
    for n in range(flat.shape[0]):
        b = batch_idx[n]
        out[b, slot[b]] = flat[n]
        mask[b, slot[b]] = True
        slot[b] += 1
    return out, mask


class TargetAttention(nn.Module):
    """Stage-2/3 building block: scaled dot-product attention, target as query."""

    def __init__(self, d_model: int):
        super().__init__()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.scale = math.sqrt(d_model)

    def forward(self, target: torch.Tensor, items: torch.Tensor, mask: torch.Tensor
                ) -> torch.Tensor:
        """target [B,d], items [B,M,d], mask [B,M] → [B,d]; all-empty rows → zeros."""
        q = self.w_q(target).unsqueeze(1)                      # [B,1,d]
        k, v = self.w_k(items), self.w_v(items)                # [B,M,d]
        scores = (q @ k.transpose(1, 2)).squeeze(1) / self.scale  # [B,M]
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        attn = F.softmax(scores, dim=-1)
        empty = ~mask.any(dim=-1)                              # rows with no items
        attn = torch.where(empty.unsqueeze(-1), torch.zeros_like(attn), attn)
        return (attn.unsqueeze(1) @ v).squeeze(1)              # [B,d]


class ContextAwareSarcasmModel(nn.Module):
    def __init__(self, cfg: LeischeConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.encoder = SharedEncoder(cfg.encoder_name, d, cfg.pooling,
                                     cfg.freeze_bottom_layers)
        self.active = [c for c, on in zip(CHANNELS, (cfg.use_conv, cfg.use_temp, cfg.use_ret)) if on]

        # stage 2 — conversational
        if cfg.use_conv:
            self.conv_attn = TargetAttention(d)
            if cfg.conv_role_embeddings:
                self.role_emb = nn.Embedding(3, d)       # submission / ancestor / reply
                self.submitter_emb = nn.Embedding(2, d)  # is_submitter flag

        # stage 2 — temporal (decay exp(−λ·Δt), Δt in days)
        if cfg.use_temp:
            self.temp_attn = TargetAttention(d)
            # inverse-softplus init so softplus(raw) == lambda_init exactly
            raw = math.log(math.expm1(cfg.temporal_lambda_init))
            self.temporal_lambda_raw = nn.Parameter(
                torch.tensor(raw, dtype=torch.float32),
                requires_grad=cfg.temporal_lambda_learnable,
            )

        # stage 3 — retrieval: two banks, separate attention, concat + project
        if cfg.use_ret:
            self.ret_attn_sarc = TargetAttention(d)
            self.ret_attn_nonsarc = TargetAttention(d)
            self.ret_proj = nn.Linear(2 * d, d)

        # learned "missing channel" embeddings (alternative to zeros, guide §4.2)
        if self.active and cfg.missing_channel == "learned":
            self.missing_emb = nn.ParameterDict(
                {c: nn.Parameter(torch.zeros(d)) for c in self.active})

        # stage 4 — gated fusion over ACTIVE channels only
        if len(self.active) >= 2:
            self.gate = nn.Linear(len(self.active) * d, len(self.active))

        # stage 5 — classifier MLP([t ; c_fused]) (or MLP([t]) for the baseline)
        in_dim = d * (2 if self.active else 1)
        self.classifier = nn.Sequential(
            nn.Linear(in_dim, cfg.mlp_hidden), nn.GELU(),
            nn.Dropout(cfg.dropout), nn.Linear(cfg.mlp_hidden, 2),
        )

        # §9.3 auxiliary heads (constructed only when flagged)
        if cfg.aux_cue_heads:
            self.cue_head = nn.Linear(in_dim, 4)
        if cfg.aux_polarity_shift:
            self.shift_head = nn.Linear(in_dim, 2)
        if cfg.aux_language_head:
            self.lang_head = nn.Linear(in_dim, 3)
        # NOTE: the RQ3 stage-2 intended-sentiment head is deliberately NOT part
        # of this model — it is trained post-hoc on frozen [t; c_fused] features
        # (notebook 06) so RQ3 supervision never touches the sarcasm encoder.

    @property
    def temporal_lambda(self) -> torch.Tensor:
        return F.softplus(self.temporal_lambda_raw)

    # ------------------------------------------------------------------ channels

    def _encode_items(self, part: dict, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        if part["input_ids"].shape[0] == 0:
            d = self.cfg.d_model
            dev = next(self.parameters()).device
            return (torch.zeros(batch_size, 1, d, device=dev),
                    torch.zeros(batch_size, 1, dtype=torch.bool, device=dev))
        flat = self.encoder.forward_chunked(part["input_ids"], part["attention_mask"])
        return scatter_items(flat, part["batch_idx"], batch_size)

    def _apply_missing(self, out: torch.Tensor, empty: torch.Tensor, channel: str) -> torch.Tensor:
        """empty: [B] bool — rows whose channel had no items at all."""
        if self.cfg.missing_channel == "learned":
            fill = self.missing_emb[channel].to(out.dtype)
            out = torch.where(empty.unsqueeze(-1), fill.expand_as(out), out)
        return out  # "zeros": attention already yields zeros for empty rows

    def _conv_channel(self, batch: dict, t: torch.Tensor) -> torch.Tensor:
        items, mask = self._encode_items(batch["conv"], t.shape[0])
        if self.cfg.conv_role_embeddings and batch["conv"]["input_ids"].shape[0] > 0:
            role_flat = self.role_emb(batch["conv"]["role"])
            subm_flat = self.submitter_emb(batch["conv"]["is_submitter"])
            extra, _ = scatter_items(role_flat + subm_flat, batch["conv"]["batch_idx"], t.shape[0])
            items = items + extra
        out = self.conv_attn(t, items, mask)
        return self._apply_missing(out, ~mask.any(dim=-1), "conv")

    def _temp_channel(self, batch: dict, t: torch.Tensor) -> torch.Tensor:
        items, mask = self._encode_items(batch["temp"], t.shape[0])
        if batch["temp"]["input_ids"].shape[0] > 0:
            decay_flat = torch.exp(-self.temporal_lambda * batch["temp"]["delta_days"])
            decay, _ = scatter_items(decay_flat.unsqueeze(-1), batch["temp"]["batch_idx"], t.shape[0])
            items = items * decay.to(items.dtype)  # K/V · exp(−λ·Δt) (guide §4.2)
        out = self.temp_attn(t, items, mask)
        return self._apply_missing(out, ~mask.any(dim=-1), "temp")

    def _ret_channel(self, batch: dict, t: torch.Tensor) -> torch.Tensor:
        s_items, s_mask = self._encode_items(batch["ret_sarc"], t.shape[0])
        n_items, n_mask = self._encode_items(batch["ret_nonsarc"], t.shape[0])
        c_s = self.ret_attn_sarc(t, s_items, s_mask)
        c_n = self.ret_attn_nonsarc(t, n_items, n_mask)
        out = self.ret_proj(torch.cat([c_s, c_n], dim=-1))
        # the two banks are padded to different widths (sarcastic bank can hold
        # fewer than k exemplars) — combine emptiness per row, not per mask
        empty = ~(s_mask.any(dim=-1) | n_mask.any(dim=-1))
        return self._apply_missing(out, empty, "ret")

    # ------------------------------------------------------------------ forward

    def forward(self, batch: dict) -> dict:
        t = self.encoder(batch["target"]["input_ids"], batch["target"]["attention_mask"])
        B = t.shape[0]

        outputs: dict = {"target_emb": t}
        chans: list[torch.Tensor] = []
        for name in self.active:
            c = {"conv": self._conv_channel, "temp": self._temp_channel,
                 "ret": self._ret_channel}[name](batch, t)
            chans.append(c)
            outputs[f"c_{name}"] = c

        if len(self.active) == 0:
            feats = t
            outputs["gates"] = None
        elif len(self.active) == 1:
            feats = torch.cat([t, chans[0]], dim=-1)
            outputs["gates"] = torch.ones(B, 1, device=t.device)
        else:
            g = F.softmax(self.gate(torch.cat(chans, dim=-1)), dim=-1)  # [B, n_active]
            c_fused = sum(g[:, i:i + 1] * chans[i] for i in range(len(chans)))
            feats = torch.cat([t, c_fused], dim=-1)
            outputs["gates"] = g

        outputs["features"] = feats
        outputs["logits"] = self.classifier(feats)
        if self.cfg.aux_cue_heads:
            outputs["cue_logits"] = self.cue_head(feats)
        if self.cfg.aux_polarity_shift:
            outputs["shift_logits"] = self.shift_head(feats)
        if self.cfg.aux_language_head:
            outputs["lang_logits"] = self.lang_head(feats)
        return outputs

    def gate_channels(self) -> list[str]:
        return list(self.active)
