# leische

Model repository for the undergraduate thesis **Context-Aware Sarcasm Detection
in Code-Switching Social Media Posts**. The data half (collection + LLM-ensemble
annotation) is the separate [uyam](../uyam) repository; the handoff contract is
`uyam/docs/dataset-contract-leische.md` and the authoritative build plan is
[docs/MODEL_PLAN.md](docs/MODEL_PLAN.md) (copied from uyam).

> **STATUS: SMOKE PHASE — do not train for results.**
> The current export (`dataset-v1`) is a 100-item pilot with 8 sarcastic rows
> and no human gold subset. The §10 data-readiness gate FAILS, and the code
> enforces it: `train.py` refuses any non-smoke run while the gate fails, and
> every pilot-derived number is prefixed `SMOKE`. When `dataset-v2` lands,
> re-run `scripts/sync_data.py --version v2`, set `dataset_version: v2`, and
> the same notebooks become the real pipeline.

## Quickstart

```bash
uv sync --all-extras            # pinned env (torch 2.6.0+cu124, Python 3.12)
uv run python scripts/sync_data.py   # copy uyam exports into data/ (gitignored)
uv run pytest                   # pitfall-checklist properties (§11)
```

Notebooks run in order (00 → 06). Each `.ipynb` is committed with executed
outputs; the paired `.py` files (jupytext percent format) are the diff-friendly
sources. To re-execute headless:

```bash
uv run jupytext --to ipynb notebooks/01_data_eda.py
uv run jupyter nbconvert --to notebook --execute --inplace \
    notebooks/01_data_eda.ipynb --ExecutePreprocessor.timeout=-1
```

## Layout

```
data/                copied uyam exports (gitignored — sync script recreates)
src/leische/         data.py contexts.py encoders.py model.py train.py evaluate.py config.py
notebooks/           00 environment · 01 EDA · 02 context assembly · 03 baseline
                     04 context model · 05 ablation matrix · 06 RQ3 sentiment
configs/             yaml per experiment (smoke profiles now, real profiles later)
results/             frozen folds + SMOKE run artifacts (committed)
tests/               §11 pitfalls as executable properties
```

## Architecture (thesis-committed baseline, guide §4)

Five stages behind one model class: (1) shared `xlm-roberta-base` encoder with
mean pooling + projection for every text unit; (2) target-conditioned attention
over conversational items (role + is_submitter embeddings) and over author
history with `exp(−λ·Δt)` decay; (3) retrieval attention over per-fold
sarcastic/non-sarcastic exemplar banks; (4) GMU-style gated fusion over the
ACTIVE channels (per-instance gates, logged); (5) `MLP([t ; c_fused])`.
`use_conv/use_temp/use_ret = False` reduces it exactly to the RQ1
context-agnostic baseline. Every §9 upgrade (label-quality weighting, focal
loss, auxiliary cue heads, freezing/LoRA-style options, retrieval variants,
calibration) is a config flag, **off by default**.

## Hard rules encoded in the codebase

- Folds: `StratifiedGroupKFold`, stratified on `sarcastic×language`, **grouped
  by `submission_fullname`**; frozen to `results/folds-v{N}.json` with the
  dataset identity, refused on mismatch (§11.2, §11.8).
- Retrieval banks: training-fold rows only, same-thread neighbors excluded,
  rebuilt per fold, leakage asserted at build time (§11.1).
- Conversational context comes only from the embedded snapshot the annotators
  saw — never rebuilt from the corpus dump (§11.7).
- `keyword_oversampled` rows never enter natural-distribution metrics (§7.2).
- `score`/replies are post-hoc signals; the thesis frames the task as post-hoc
  thread analysis — state that framing in the manuscript (§11.5).

## §10 readiness gate (all must pass before real training)

1. Full corpus annotated under sarc-v2, exported as `dataset-v2`
2. ≥ 400–500 sarcastic positives
3. ~300-item human gold subset labeled, human-vs-ensemble κ in the card
4. Every `language × sarcastic` cell ≥ 10
5. Fold file frozen and committed
