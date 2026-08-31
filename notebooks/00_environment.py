# %% [markdown]
# # 00 — Environment
#
# Verifies the pinned environment, GPU, determinism, data presence, and model
# downloads before anything else runs. Guide §3 / §12 step 1.
#
# **This repo is in SMOKE phase** — the §10 readiness gate fails on the
# `dataset-v1` pilot, so every downstream number is a harness check, not a result.

# %%
import platform
import sys

import matplotlib
import numpy as np
import pandas as pd
import sklearn
import torch
import transformers

print(f"python        {sys.version.split()[0]} on {platform.system()} {platform.release()}")
for mod in (torch, transformers, sklearn, pd, np, matplotlib):
    print(f"{mod.__name__:<13} {mod.__version__}")

# %% [markdown]
# ## GPU / fp16 check (8 GB VRAM → xlm-roberta-base + fp16, guide §3)

# %%
assert torch.cuda.is_available(), "CUDA GPU required for the smoke training runs"
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / 1024**3
print(f"GPU: {props.name}  |  VRAM: {vram_gb:.1f} GB  |  capability: {props.major}.{props.minor}")
print(f"cuda runtime: {torch.version.cuda}  |  fp16 autocast available: True")
if vram_gb < 10:
    print("→ 8 GB class GPU confirmed: stick to xlm-roberta-base + fp16 "
          "(xlm-roberta-large only via LoRA, guide §9.4)")

# %% [markdown]
# ## Determinism check
#
# Same seed → bit-identical forward pass, twice. The training loop seeds
# python/numpy/torch the same way (`train.set_seed`).

# %%
from leische.train import set_seed


def seeded_forward() -> torch.Tensor:
    set_seed(13)
    layer = torch.nn.Linear(64, 8)
    x = torch.randn(4, 64)
    return layer(x)


a, b = seeded_forward(), seeded_forward()
assert torch.equal(a, b), "seeded forwards differ — determinism broken"
print("determinism check PASSED (two seeded forwards are bit-identical)")

# %% [markdown]
# ## Data presence + §10 readiness gate
#
# `data/` is gitignored; `scripts/sync_data.py` copies the uyam exports.

# %%
from leische.nbsupport import bootstrap

cfg, df, corpus, card, gate = bootstrap(need_corpus=True)
assert not gate.passed, "gate unexpectedly passed on the pilot — check the card"

# %% [markdown]
# ## Model downloads
#
# Fetches (or reuses from the HF cache) the two frozen/base models so later
# notebooks never stall on downloads: the shared encoder and the retrieval
# embedder.

# %%
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained(cfg.encoder_name)
enc = AutoModel.from_pretrained(cfg.encoder_name)
n_params = sum(p.numel() for p in enc.parameters()) / 1e6
print(f"{cfg.encoder_name}: {n_params:.0f}M params, hidden={enc.config.hidden_size}")
del enc

ret = SentenceTransformer(cfg.retrieval_model)
demo = ret.encode(["ang ganda naman ng serbisyo nila, sobrang bagal lang"])
print(f"{cfg.retrieval_model}: embedding dim {demo.shape[1]}")
del ret
print("model downloads OK")
