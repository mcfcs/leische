"""Architecture checks for the notebook's model cell — no GPU, no HF download.

Extracts the model classes straight out of `leische_pipeline.ipynb` (so the
notebook stays the single source of truth), stubs the XLM-R backbone with a
tiny random encoder, and asserts the properties that the manuscript's
Chapter III actually specifies:

  M1  the Stage-4 gate is conditioned on the target embedding, one sigmoid
      gate per channel, normalised to sum to one over the ACTIVE channels
  M2  the temporal channel decays on Δt in HOURS
  §7.3 disabled channels leave the gate entirely rather than being zero-filled
  §4.5 the all-off configuration reduces exactly to the RQ1 baseline MLP([t])

    uv run python tools/check_model_contract.py
"""

from __future__ import annotations

import json
import math
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
D_HIDDEN = 32


# ------------------------------------------------------------------- stubs
class _StubBackbone(nn.Module):
    """Stands in for xlm-roberta-base: same interface, 32 dims, no download."""

    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=D_HIDDEN)
        self.embeddings = nn.Embedding(64, D_HIDDEN)
        layer = nn.ModuleList([nn.Linear(D_HIDDEN, D_HIDDEN) for _ in range(4)])
        self.encoder = types.SimpleNamespace(layer=layer)
        self._layers = layer  # keep them registered as real parameters

    def forward(self, input_ids=None, attention_mask=None, **_):
        h = self.embeddings(input_ids.clamp(0, 63))
        for lyr in self._layers:
            h = torch.tanh(lyr(h))
        return types.SimpleNamespace(last_hidden_state=h)


@dataclass
class Cfg:
    encoder_name: str = "stub"
    pooling: str = "mean"
    d_model: int = 16
    max_len_target: int = 8
    max_len_context: int = 8
    max_len_selftext: int = 8
    use_conv: bool = False
    use_temp: bool = False
    use_ret: bool = False
    conv_role_embeddings: bool = True
    temporal_k: int = 5
    temporal_window_hours: float | None = 48.0
    temporal_lambda_init: float = 0.0289
    temporal_lambda_learnable: bool = True
    missing_channel: str = "zeros"
    retrieval_encoder: str = "sentence_transformer"
    retrieval_k: int = 3
    mlp_hidden: int = 16
    dropout: float = 0.0
    freeze_bottom_layers: int = 0
    aux_cue_heads: bool = False
    aux_polarity_shift: bool = False
    aux_language_head: bool = False


def load_model_cell() -> dict:
    """Exec the notebook's model cell against stubbed transformers."""
    nb = json.loads((ROOT / "leische_pipeline.ipynb").read_text(encoding="utf-8"))
    cell = next(
        "".join(c["source"]) for c in nb["cells"]
        if c["cell_type"] == "code" and "class ContextAwareSarcasmModel" in "".join(c["source"])
    )
    ns = {
        "torch": torch, "nn": nn, "F": F, "math": math,
        "AutoModel": types.SimpleNamespace(from_pretrained=lambda *_a, **_k: _StubBackbone()),
        "AutoTokenizer": types.SimpleNamespace(from_pretrained=lambda *_a, **_k: None),
        "Config": Cfg, "print": lambda *a, **k: None,
    }
    exec(compile(cell, "model_cell", "exec"), ns)
    return ns


def batch(n=6, n_items=3, d_ids=8):
    def part(count):
        return {"input_ids": torch.randint(1, 60, (count, d_ids)),
                "attention_mask": torch.ones(count, d_ids, dtype=torch.long),
                "batch_idx": torch.arange(count) % n}
    conv = part(n * n_items)
    conv["role"] = torch.randint(0, 3, (n * n_items,))
    conv["is_submitter"] = torch.randint(0, 2, (n * n_items,))
    temp = part(n * n_items)
    # hours, inside the 48 h window
    temp["delta_hours"] = torch.rand(n * n_items) * 48.0
    return {"target": part(n), "conv": conv, "temp": temp,
            "ret_sarc": part(n * 2), "ret_nonsarc": part(n * 2)}


def main() -> int:
    ns = load_model_cell()
    Model = ns["ContextAwareSarcasmModel"]
    torch.manual_seed(13)
    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
        if not cond:
            failures.append(name)

    print("§7.3 · all 8 ablation conditions")
    for c, t_, r_ in [(a, b, d) for a in (0, 1) for b in (0, 1) for d in (0, 1)]:
        cfg = Cfg(use_conv=bool(c), use_temp=bool(t_), use_ret=bool(r_))
        model = Model(cfg).eval()
        with torch.no_grad():
            out = model(batch())
        n_active = c + t_ + r_
        label = f"conv={c} temp={t_} ret={r_}"
        ok = out["logits"].shape == (6, 2)
        if n_active == 0:
            ok &= out["gates"] is None
            ok &= out["features"].shape[-1] == cfg.d_model  # MLP([t]) — RQ1 baseline
        else:
            ok &= out["gates"].shape == (6, n_active)
            ok &= torch.allclose(out["gates"].sum(-1), torch.ones(6), atol=1e-5)
            ok &= out["features"].shape[-1] == 2 * cfg.d_model
        check(label, bool(ok), f"{n_active} active channel(s)")

    print("\nM1 · Stage-4 gate is conditioned on the target embedding")
    cfg = Cfg(use_conv=True, use_temp=True, use_ret=True)
    model = Model(cfg).eval()
    b = batch()
    with torch.no_grad():
        g0 = model(b)["gates"]
    # same context, different target text -> gates must move
    b2 = {**b, "target": {**b["target"], "input_ids": torch.randint(1, 60, (6, 8))}}
    with torch.no_grad():
        g1 = model(b2)["gates"]
    delta = (g0 - g1).abs().max().item()
    check("gates respond to the target", delta > 1e-4, f"max |Δgate| = {delta:.2e}")
    spread = g0.std(dim=0).max().item()
    check("gates vary across instances", spread > 1e-5, f"max per-channel std = {spread:.2e}")
    check("one gate per active channel",
          isinstance(model.gate, nn.ModuleList) and len(model.gate) == 3
          and model.gate[0].in_features == 2 * cfg.d_model,
          "nn.ModuleList of Linear(2·d → 1)")

    print("\n§7.3 · the gate renormalises over active channels, never zero-fills")
    m2 = Model(Cfg(use_conv=True, use_ret=True)).eval()
    with torch.no_grad():
        g = m2(batch())["gates"]
    check("2-channel gate sums to one", g.shape == (6, 2)
          and torch.allclose(g.sum(-1), torch.ones(6), atol=1e-5), f"shape {tuple(g.shape)}")
    check("channel order matches model.active", m2.active == ["conv", "ret"], str(m2.active))

    print("\nM2 · temporal decay")
    mt = Model(Cfg(use_temp=True)).eval()
    check("λ is learnable by default", mt.temporal_lambda_raw.requires_grad)
    check("λ initialises to the configured value",
          abs(float(mt.temporal_lambda.detach()) - Cfg.temporal_lambda_init) < 1e-4,
          f"λ = {float(mt.temporal_lambda.detach()):.4f} per hour")
    # Softmax is shift-invariant, so a decay applied uniformly across a row is
    # a no-op by construction: recency can only be tested WITHIN a row.  Two
    # items per row, A and B; swap which one is recent and check c_temp moves
    # toward that one's single-item output.
    torch.manual_seed(7)
    n, ids = 4, torch.randint(1, 60, (2, 8))   # fixed A/B item texts
    fixed = batch(n=n)

    def temp_vec(delta_a, delta_b, only=None):
        keep = [0, 1] if only is None else [only]
        part = {"input_ids": ids[keep].repeat(n, 1),
                "attention_mask": torch.ones(n * len(keep), 8, dtype=torch.long),
                "batch_idx": torch.arange(n).repeat_interleave(len(keep)),
                "delta_hours": torch.tensor([delta_a, delta_b])[keep].repeat(n).float()}
        b = {**fixed, "temp": part}
        with torch.no_grad():
            return mt(b)["c_temp"]

    only_a, only_b = temp_vec(0.0, 0.0, only=0), temp_vec(0.0, 0.0, only=1)
    a_recent = temp_vec(0.0, 480.0)   # A is 0 h old, B is 20 days old
    b_recent = temp_vec(480.0, 0.0)
    d_a = (a_recent - only_a).norm() < (a_recent - only_b).norm()
    d_b = (b_recent - only_b).norm() < (b_recent - only_a).norm()
    check("attention shifts toward the more recent post", bool(d_a and d_b),
          f"Δt=(0h,480h)→A {bool(d_a)}, Δt=(480h,0h)→B {bool(d_b)}")
    check("a uniform Δt is a no-op (softmax shift-invariance)",
          torch.allclose(temp_vec(0.0, 0.0), temp_vec(96.0, 96.0), atol=1e-5),
          "decay ranks items, it does not shrink the channel")
    check("the collator key is delta_hours",
          "delta_hours" in "".join(
              "".join(c["source"]) for c in
              json.loads((ROOT / "leische_pipeline.ipynb").read_text(encoding="utf-8"))["cells"])
          and "delta_days" not in "".join(
              "".join(c["source"]) for c in
              json.loads((ROOT / "leische_pipeline.ipynb").read_text(encoding="utf-8"))["cells"]),
          "no delta_days left in the notebook")

    print("\nempty channels")
    me = Model(Cfg(use_conv=True, use_temp=True, use_ret=True)).eval()
    be = batch()
    for k in ("conv", "temp", "ret_sarc", "ret_nonsarc"):
        be[k] = {"input_ids": torch.zeros(0, 8, dtype=torch.long),
                 "attention_mask": torch.zeros(0, 8, dtype=torch.long),
                 "batch_idx": torch.zeros(0, dtype=torch.long)}
    be["conv"]["role"] = torch.zeros(0, dtype=torch.long)
    be["conv"]["is_submitter"] = torch.zeros(0, dtype=torch.long)
    be["temp"]["delta_hours"] = torch.zeros(0)
    with torch.no_grad():
        oe = me(be)
    check("all-empty channels still produce finite logits",
          bool(torch.isfinite(oe["logits"]).all()) and bool(torch.isfinite(oe["gates"]).all()))
    check("empty channels yield zero context vectors",
          all(float(oe[f"c_{c}"].abs().max()) == 0.0 for c in ("conv", "temp", "ret")))

    print("\n" + ("=" * 60))
    if failures:
        print(f"FAILED: {len(failures)} check(s) — {', '.join(failures)}")
        return 1
    print("all architecture checks PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
