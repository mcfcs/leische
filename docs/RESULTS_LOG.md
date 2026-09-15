# Results log

Dated, stage-by-stage record of what was run, what came out, which figures
back it, and what was concluded — per [FABILE_BRIEF.md](FABILE_BRIEF.md) §3.
**Every metric here is agreement with the LLM-ensemble labels, not with human
judgement** (inter-annotator Fleiss κ 0.376 on sarcasm; human-vs-ensemble
Cohen κ −0.148 on 31 gold items — [ANNOTATION_PROVENANCE.md](ANNOTATION_PROVENANCE.md)).

Runs were driven headlessly with `tools/run_notebook.py`; logs are in
`results/logs/`. Hardware: RTX 5090 Laptop (24 GB, sm_120), torch 2.9.1+cu128,
transformers 4.57.6, Python 3.12.

---

## 2026-09-15 · Recheck (brief §1)

`uv sync`, `tools/check_model_contract.py` (all architecture checks PASSED —
M1 gate target-conditioning, M2 hours/learnable λ, §7.3 renormalisation, empty
channels → zeros), then notebook §1–§6:

| check | observed |
|---|---|
| §10 gate | **TRAIN-READY: all 5 PASS** (sarc-v2 / 15,000 rows; 1,601 positives, 10.7%; min language×sarcastic cell 426; `context.source == {'annotator_snapshot'}`; `folds-v1-a419a4bc95.json` loaded, identity verified) |
| claim tier | gold 31 items WARN · human-vs-ensemble κ −0.148 WARN · Fleiss κ 0.376 WARN · LID validated PASS (62.5%) |
| rows / positives / threads | 15,000 / 1,601 / 1,393 |
| `LABEL_AUTHORITY` | `llm_ensemble (gemma3+qwen3+sealion → qwen3-32b)` |
| fold composition (rows/positives) | test folds 3,001/321 · 3,000/320 · 3,000/321 · 3,000/319 · 2,999/320; val 1,218–1,450 rows with 104–135 positives |
| leakage assertions | PASSED for all 15,000 retrieval queries (banks ⊆ 10,781 fold-0 train rows, same-thread excluded) |
| temporal coverage (k=5, 48 h) | 12,179 items; ≥1 item on **36.7%** of rows |
| conversational items | 28,117 over 15,000 rows |
| annotator-majority label (derived from `reliability.annotators[].sarcastic`) | **2,747 positives**, 1,712 rows differ from the shipped adjudicated label — matches ANNOTATION_PROVENANCE §3 exactly |

Environment notes: the Hugging Face Python downloader stalled at 0 bytes on this
network (CDN read timeouts); the two model weight files were fetched with curl
via a mirror and their SHA-256 verified against the huggingface.co ETags. Another
process (ComfyUI) held ~7 GB VRAM at session start and released it later —
worth checking `nvidia-smi` before every stage.

### Harness changes made before any real run (all in the notebook; no fold file touched)

- `run_cv` now also records, per fold: the F1-optimal threshold chosen on the
  **validation** rows (`threshold`, `*_tuned` metrics, `pred_tuned`), a
  temperature fitted on validation margins (`temperature`, `prob_cal`,
  `ece_cal`), AUPRC/AUROC, best epoch / epochs run, wall-clock; writes
  `val_predictions.csv`; optionally saves the fold-0 weights to
  `cache/checkpoints/` (gitignored). The 0.5 decision stays the committed one.
- `run_or_load`: a finished run (same result-affecting config + dataset
  identity) is re-loaded from `results/` instead of retrained.
- `build_fold_retrieval` vectorised (15,000 queries in ~1.5 s instead of
  minutes) and memoised; identical leakage assertions.
- `scatter_items` vectorised (was one kernel launch per context item).
- `Collator` gives the submission item the §4.1 selftext budget (128) — the
  original tokenised every conversational item at 96, so `max_len_selftext`
  was dead config.
- `label_source` config knob (`adjudicated` | `annotator_majority`) for the §4
  robustness row; the annotator-majority map reproduces the documented 2,747.
- `real_cfg()` profile: `smoke=False, batch_size=16, grad_accum=2` (effective 32, see the probe below), one seed
  per call; `Config.smoke` now follows the gate.
- Overfit-16 smoke test fixed to 8 positives + 8 negatives (with 1,601
  positives its "first 16 rows" were all positive, so a constant predictor
  passed).
- §12 is staged (A/B/C, `LEISCHE_STAGES`), §12b is stage D, §13 uses the real
  fold-0 checkpoint when present and reports an oracle-flag bound.

### Smoke tests (§9–§11, brief §2)

- overfit-16, baseline: final loss 0.0281 in 23 steps → **PASS**
- overfit-16, full model (conv+temp+ret): final loss 0.0463 in 23 steps → **PASS**
- §10 tiny-settings 5-fold dry run and both pooling paths: ran end to end
  (F1 is 0 at 8 steps/epoch by construction — a harness check, not a result).
- §11 smoke full model: checkpoint round-trip OK; features (15,000 × 512)
  cached; gate std over fold-0 test rows 0.006–0.012 at smoke scale (16
  optimiser steps — not evidence either way).

### Batch-size probe (real settings, fold 0, 12 timed steps)

| model | batch | s/step | steps/epoch | ≈ min/epoch (train + val) | peak alloc / reserved |
|---|---|---|---|---|---|
| 8_full | 32 | 0.46 | 337 | 2.6 + 0.1 | **17.3 / 19.1 GB** (85% of the 20.3 GB cap) |
| 8_full | 48 | — | — | — | OOM (caught by the cap) |
| 8_full | 64 | — | — | — | OOM (caught by the cap) |
| 1_baseline | 32 | 0.11 | 337 | 0.6 + 0.0 | 5.7 / 7.6 GB |
| 1_baseline | 64 | 0.18 | 169 | 0.5 + 0.0 | 8.0 / 9.2 GB |

Decision: **batch 16 × grad_accum 2 (effective 32) for every condition** —
batch 32 for the full model sits above the brief's 80%-of-cap line with no
margin for a long batch, and batch 64 buys the baseline nothing. Expected
cost: full model ≈ 3 min/epoch → ~15–25 min per fold with early stopping;
baseline ≈ 0.7 min/epoch. Stage A projected at ~2–2.5 h.
