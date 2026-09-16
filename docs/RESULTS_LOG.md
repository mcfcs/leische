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

---

## 2026-09-16 · Stage A — conditions 1 (baseline) and 8 (full), 5 folds × seed 13

Config: `real_cfg()` — thesis defaults (xlm-roberta-base, mean pooling, d=256,
lr 2e-5 / 1e-4, warmup 10%, ≤10 epochs, patience 3, fp16, inverse-frequency
class weights per fold, retrieval k=3 / MiniLM, temporal k=5 / 48 h / learnable
λ), batch 16 × accum 2. Runs: `results/ablation-1_baseline/`,
`results/ablation-8_full/`; log `results/logs/stage-A.log`; wall clock 22.6 min
(baseline) + 113.8 min (full); peak VRAM 6.6 / 13.3 GB. Checkpoints for fold 0
under `cache/checkpoints/` (not committed).

**All numbers are agreement with the LLM-ensemble labels, not with human judgement.**

### RQ1 headline (test folds, mean ± std over 5 folds)

| condition | F1 @0.5 | F1 @val-thr | precision @0.5 | recall @0.5 | AUPRC | AUROC | ECE raw / temp-scaled |
|---|---|---|---|---|---|---|---|
| 1_baseline | **0.353 ± 0.034** | 0.338 ± 0.045 | 0.298 | 0.497 | 0.308 ± 0.049 | 0.775 ± 0.029 | 0.177 / 0.165 |
| 8_full | **0.356 ± 0.014** | 0.349 ± 0.017 | 0.317 | 0.433 | 0.312 ± 0.018 | 0.776 ± 0.015 | 0.125 / 0.111 |

Significance, condition 8 − condition 1 (`results/significance.csv`,
`figures/bootstrap.png`): ΔF1 **+0.013**, paired-bootstrap 95% CI
**[−0.004, +0.030]**, p = 0.13; approximate randomization p = 0.13. At the
val-tuned decision ΔF1 +0.011, CI [−0.008, +0.031], p = 0.25. McNemar on the 0.5
decision gives p < 0.001 but it counts *correctness*, and at a 10.7% base rate it
mainly rewards the model that flags fewer rows (full 2,293 vs baseline 3,034
positives predicted, 1,601 true) — not an F1 statement.

**Conclusion: on the headline pair the context model does not beat the
target-only baseline.** The gap is inside the fold spread (`figures/fold-spread-f1_tuned.png`);
what the full model does change is the *shape* of the errors — higher precision,
lower recall, a tighter fold spread (std 0.014 vs 0.034) and better calibration
(ECE 0.125 vs 0.177).

### What the diagnostics say (brief §3, in order)

1. **Threshold** (`figures/threshold-sweep-*.png`) — the expected free win did
   not materialise. Per-fold validation-chosen thresholds ranged 0.45–0.87 and
   *lowered* mean test F1 (−0.014 baseline, −0.007 full); the pooled test sweep
   peaks at 0.54 / 0.47, i.e. at 0.5. With inverse-frequency class weights the
   0.5 decision is already near F1-optimal, and a threshold picked on 104–135
   validation positives is noise. **0.5 stays the reported decision**; the
   val-tuned columns are kept as an honest row.
2. **Training curves** (`figures/training-curves-condition.png`) — both learn;
   best epochs 1–7 (baseline) and 3–7 (full). Baseline fold 2 collapsed to
   all-negative at epoch 1 and never recovered (early stopping restored epoch 1:
   F1 0.295, AUPRC 0.230, fitted temperature pinned at the grid max 8.0). That is
   optimisation instability at lr 2e-5 with a 4.6× positive weight, and it is
   most of the baseline's fold variance. Val F1 runs ~0.04 above test for the
   full model (0.395 vs 0.356).
3. **Fold spread** — per-fold F1@0.5: baseline 0.369 / 0.371 / 0.295 / 0.379 /
   0.349; full 0.363 / 0.350 / 0.369 / 0.361 / 0.334. The full model is more
   consistent; neither is better fold by fold (3–2).
4. **Calibration** (`figures/calibration-*.png`) — full model ECE 0.125 raw,
   0.111 after temperature scaling (T ≈ 1.0–3.4 per fold). The full model flags
   15.3% of rows at 0.5 (13.1% at the val threshold) against a 10.7% base rate;
   the baseline flags 20.2%. Usable for RQ3 stage 2, with the caveat that ~half
   of the flags are false positives (precision 0.32).
5. **Gates** (`figures/gates.png`) — the M1 gate now varies per instance:
   conv 0.16 ± 0.11 (range 0.01–0.54), temp 0.33 ± 0.09, ret 0.51 ± 0.09.
   Retrieval carries half the weight on average; conversational context is
   systematically the least trusted channel. The means barely move by language
   (conv 0.15–0.18) or record type, and — notably — gate_conv does **not** rise
   with the number of conversational items (0.155 with none, 0.149 with 3+), so
   the variation is not an availability signal. It is higher on true positives
   (0.21 vs 0.16) and on adjudicated rows (0.18). Per-instance gating is
   mechanically supported; a per-instance *benefit* is not.
6. **Slices** (`figures/slices-*.png`, F1@0.5 full / baseline) — language:
   Taglish 0.46 / 0.45, English 0.34 / 0.34, Tagalog 0.26 / 0.25. resolved_by:
   majority 0.46 / 0.46, unanimous 0.40 / 0.34, **adjudicator 0.28 / 0.26**.
   sarcasm votes: 3-0 rows 0.43, 2-1 rows 0.26. Label noise dominates: the
   rows where the annotators disagreed (4,544 rows, 711 positives) are close
   to unlearnable for either model, and they hold 44% of the positives.
   Submissions have 4 positives — the slice is not interpretable.

Other observations: the two models agree on 84.6% of decisions with a
probability correlation of only 0.55 — different predictions, same F1. No
`keyword_oversampled` rows exist, so natural-only and all-rows metrics coincide.
The retrieval-embedding cache used by every stage was encoded on CPU by the
demo build and reused by the notebook (same MiniLM weights; identical across
stages).

### Decisions

- Stage B, C, D and the §4 rows (annotator-majority label, aux cue heads,
  label-quality weighting) queued as approved, in the order B → C → E → D.
- Committed baseline unchanged. A lower-encoder-LR / layer-wise-decay variant
  is the tuning row most likely to matter (the fold-2 collapse), to be added as
  a separate row after the approved stages.
- The RQ3 evaluation runs on the fold-0 full-model checkpoint with its
  validation-chosen threshold (0.65) — see the next entry.

---

## 2026-09-16 · RQ3 on the stage-A fold-0 checkpoint (§13)

Sarcasm model: condition 8, fold 0, seed 13, validation-chosen threshold 0.65,
temperature 1.41 → 351 of 3,001 fold-0 test rows flagged (precision 0.35,
recall 0.38, flag ECE 0.112). Heads: logistic regression fit on fold-0
**training** rows only — stage 1 on the target-only embedding with
`literal_sentiment`, stage 2 on `[t ; c_fused]` with `intended_sentiment`.
Ground truth `intended_sentiment` (κ 0.481 vs the human on the gold subset).
Output `results/rq3-fold0.csv`, `figures/rq3-macro_f1.png`.

| run (fold-0 test, macro-F1 / accuracy) | overall (n=3,001) | gold-sarcastic (321) | sarcastic ∧ literal≠intended (208) |
|---|---|---|---|
| stage 1 external (`aux.tx_sentiment`, cardiffnlp) | 0.625 / 0.631 | 0.243 / 0.486 | 0.198 / 0.341 |
| stage 1 only (ours, literal head) | 0.597 / 0.602 | 0.182 / 0.299 | 0.129 / 0.173 |
| **two-stage** (stage 2 on flagged rows) | 0.604 / 0.610 | 0.248 / 0.477 | **0.209 / 0.351** |
| two-stage, oracle flag (upper bound) | 0.619 / 0.628 | 0.254 / 0.536 | 0.216 / 0.404 |
| stage 2 everywhere (no flag) | 0.612 / 0.618 | 0.254 / 0.536 | 0.216 / 0.404 |

Reading, honestly:

- The thesis's claim holds in direction: re-reading flagged rows with the
  intended-sentiment head doubles accuracy on the slice that matters
  (sarcastic ∧ literal≠intended: 0.17 → 0.35; 0.40 with a perfect flag) at a
  cost of 0.01 on non-sarcastic rows.
- But the flag is not what carries it. Applying the stage-2 head to *every*
  row scores higher overall (0.612 vs 0.604) and equals the oracle on the
  sarcastic slice — the gain comes from training on `intended_sentiment`,
  not from conditioning on the sarcasm decision. With a 0.35-precision flag,
  gating stage 2 loses more on the false positives than it protects.
- Our stage-1 head (logistic regression on a 256-d frozen projection) is
  weaker than the shipped external sentiment model on every slice; the
  external model is the fairer stage-1 baseline. A "external stage 1 + our
  stage 2 on flagged rows" variant was added to §13 for the next run.
- Tagalog is the weakest language for sentiment as well (0.55 macro-F1).

---

## 2026-09-16 · Stage B — the remaining six RQ2 conditions, 5 folds × seed 13

Same `real_cfg()` as stage A. Runs `results/ablation-{2_conv,3_temp,4_ret,5_conv_temp,6_conv_ret,7_temp_ret}/`;
log `results/logs/stage-BCDE.log`; wall clock 50 / 40 / 76 / 61 / 105 / 98 min;
peak VRAM 7.9–12.6 GB. **Agreement with the LLM-ensemble labels, not human judgement.**

### RQ2 matrix (test folds, mean ± std over 5 folds, seed 13)

| condition | conv | temp | ret | F1 @0.5 | F1 @val-thr | P / R @0.5 | AUPRC | AUROC | ECE | ΔF1 vs 1 (paired bootstrap, 0.5) |
|---|---|---|---|---|---|---|---|---|---|---|
| 1_baseline | – | – | – | 0.353 ± 0.034 | 0.338 ± 0.044 | 0.30 / 0.50 | 0.308 ± 0.049 | 0.775 | 0.177 | — |
| 2_conv | ✓ | – | – | 0.355 ± 0.023 | 0.362 ± 0.019 | 0.31 / 0.48 | 0.327 ± 0.010 | 0.787 | 0.151 | +0.017 [−0.000, +0.032] p=0.06 |
| 3_temp | – | ✓ | – | 0.356 ± 0.043 | 0.355 ± 0.045 | 0.32 / 0.45 | 0.316 ± 0.012 | 0.784 | 0.131 | +0.019 [+0.005, +0.035] p=0.008 |
| 4_ret | – | – | ✓ | 0.350 ± 0.032 | 0.356 ± 0.029 | 0.33 / 0.41 | 0.316 ± 0.025 | 0.776 | 0.115 | +0.012 [−0.006, +0.031] p=0.20 |
| **5_conv_temp** | ✓ | ✓ | – | **0.378 ± 0.016** | 0.372 ± 0.016 | 0.32 / 0.50 | **0.334 ± 0.018** | **0.792** | 0.129 | **+0.036 [+0.021, +0.052] p<0.001** |
| 6_conv_ret | ✓ | – | ✓ | 0.355 ± 0.035 | 0.372 ± 0.019 | 0.34 / 0.43 | 0.320 ± 0.015 | 0.776 | 0.131 | +0.019 [+0.002, +0.037] p=0.04 |
| 7_temp_ret | – | ✓ | ✓ | 0.357 ± 0.023 | 0.358 ± 0.022 | 0.32 / 0.42 | 0.303 ± 0.020 | 0.776 | 0.118 | +0.015 [−0.004, +0.034] p=0.11 |
| 8_full | ✓ | ✓ | ✓ | 0.355 ± 0.014 | 0.349 ± 0.017 | 0.32 / 0.43 | 0.312 ± 0.018 | 0.776 | 0.125 | +0.013 [−0.003, +0.030] p=0.11 |

Per-fold F1@0.5 (folds 0–4): baseline 0.369 0.371 **0.295** 0.379 0.349 ·
conv+temp 0.353 0.377 0.388 0.375 0.395 · full 0.363 0.350 0.369 0.361 0.334.
Bootstraps are on the pooled 15,000 seed-13 test predictions
(`results/significance.csv` will carry the notebook's own version once stage C
finishes); figures `ablation-f1.png`, `ablation-auprc.png`, `ablation-recall_tuned.png`.

### Reading

- **Every context condition is at or above the baseline** on F1, AUPRC and
  AUROC, and every one is better calibrated (ECE 0.115–0.151 vs 0.177).
- **Conversational + temporal (condition 5) is the best model**, not the
  full model: +0.036 F1 and +0.026 AUPRC over the baseline, the tightest fold
  spread of the matrix, and the only condition whose pooled CI clears zero
  comfortably. Adding retrieval to it (→ condition 8) gives the gain back.
- **Retrieval does not help.** The four retrieval conditions sit at
  0.350–0.357, retrieval-only is the weakest row, and yet the 3-channel gate
  puts 0.51 of its weight on retrieval (0.69 / 0.67 in the two-channel pairs
  with conv / temp). The MiniLM top-3 neighbours are topically similar rows,
  not pragmatically similar ones, and the gate trusts them more than they
  deserve. Stage D (k=5, k=10, XLM-R `[CLS]`) is the direct test.
- **Caveat that must travel with the table:** the baseline's fold-2 collapse
  (0.295) inflates every gain. Excluding fold 2, baseline vs conv+temp is
  0.367 vs 0.375 (+0.008), and conv+temp wins only 3 of 5 folds outright
  (folds 1, 2, 4). Stage C adds seeds 42 and 7 for conditions 1 and 8, and a
  new stage C2 does the same for condition 5, so the comparison can be made on
  15 runs each before anything is called a result.
- Temporal context on its own (condition 3) is the single channel with the
  clearest effect (+0.019, p=0.008) despite being present on only 36.7% of
  rows — consistent with the brief's warning that the temporal condition
  measures data availability; more history per author would be the cheapest
  lever.
- Slices (F1@0.5): Taglish gains most from context (0.446 → 0.513 for
  conv+temp); Tagalog stays at 0.25–0.27 for every condition; English 0.31–0.35.
  Unanimous rows gain most (0.337 → 0.436 for conv+temp); adjudicated rows
  stay at 0.26–0.29 for all eight — label noise on the adjudicated 33% caps
  every condition alike.
