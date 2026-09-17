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

---

## 2026-09-16 · Stage C — seeds 42 and 7 on the headline pair (15 runs each)

Runs `results/ablation-{1_baseline,8_full}-seed{42,7}/`; log
`results/logs/stage-C-C2-E-D.log`; 26 + 117 + 27 + 117 min. Same config as
stage A. **Agreement with the LLM-ensemble labels, not human judgement.**

| condition | seed 13 | seed 42 | seed 7 | **pooled F1 @0.5 (15 runs)** | P / R @0.5 | AUPRC | AUROC | ECE |
|---|---|---|---|---|---|---|---|---|
| 1_baseline | 0.353 ± 0.034 | 0.369 ± 0.012 | 0.382 ± 0.021 | **0.368 ± 0.026** | 0.31 / 0.49 | 0.316 ± 0.031 | 0.786 | 0.148 |
| 8_full | 0.356 ± 0.014 | 0.357 ± 0.030 | 0.353 ± 0.031 | **0.355 ± 0.024** | 0.33 / 0.42 | 0.312 ± 0.015 | 0.778 | 0.115 |

Paired over the 15 (fold, seed) runs: **ΔF1 (full − baseline) = −0.013 ± 0.036,
full wins 4 of 15**, paired t-test p = 0.19, Wilcoxon p = 0.19. Pooled paired
bootstrap over the 45,000 test predictions: Δ = −0.006, 95% CI [−0.016, +0.004],
p = 0.25. Per-fold means over seeds — baseline 0.360 / 0.377 / 0.353 / 0.377 /
0.372, full 0.363 / 0.367 / 0.352 / 0.353 / 0.341.

### Reading

- **RQ1, honestly: the full context-aware model does not outperform the
  target-only XLM-R baseline on this data.** The point estimate is slightly
  *negative* and the CI straddles zero. What the full model changes is the
  error profile (precision 0.33 vs 0.31, recall 0.42 vs 0.49) and calibration
  (ECE 0.115 vs 0.148); it is not more accurate.
- The seed-13 baseline was the weak one: its fold-2 collapse was seed-specific
  (1 collapsed run in 15 for each model), and seeds 42 / 7 put the baseline at
  0.369 / 0.382. Every stage-B "gain" was measured against that weak seed, so
  the stage-B deltas overstate context by roughly +0.015.
- Against the 3-seed baseline (0.368), the single-seed conv+temp result
  (0.378) is a +0.010 point estimate, inside one fold std. Stage C2 (running)
  adds seeds 42 and 7 for condition 5; until it lands, "conv+temp helps" is a
  hypothesis, not a finding.
- The significance machinery in §12 (cell 40) recomputes the bootstrap /
  randomization / McNemar over all available seeds at the end of the chain
  and writes `results/significance.csv`; the pooled-prediction bootstrap
  above is the same test run from the CSVs.

---

## 2026-09-16 · Stage C2 — seeds 42 and 7 for condition 5 (conv + temp)

Runs `results/ablation-5_conv_temp-seed{42,7}/`, 62 min each; same log.
**Agreement with the LLM-ensemble labels, not human judgement.**

| condition | seed 13 | seed 42 | seed 7 | **pooled F1 @0.5 (15 runs)** | P / R @0.5 | AUPRC | AUROC | ECE |
|---|---|---|---|---|---|---|---|---|
| 1_baseline | 0.353 ± 0.034 | 0.369 ± 0.012 | 0.382 ± 0.021 | 0.368 ± 0.026 | 0.31 / 0.49 | 0.316 ± 0.031 | 0.786 | 0.148 |
| 5_conv_temp | 0.378 ± 0.016 | 0.368 ± 0.027 | 0.364 ± 0.033 | **0.370 ± 0.025** | 0.33 / 0.45 | **0.330 ± 0.021** | 0.789 | 0.116 |
| 8_full | 0.356 ± 0.014 | 0.357 ± 0.030 | 0.353 ± 0.031 | 0.355 ± 0.024 | 0.33 / 0.42 | 0.312 ± 0.015 | 0.778 | 0.115 |

Paired over 15 (fold, seed) runs, conv+temp − baseline: **ΔF1 = +0.002 ± 0.039,
7 wins of 15**, paired t-test p = 0.86; pooled paired bootstrap Δ = +0.009,
95% CI [−0.001, +0.019], p = 0.07. AUPRC +0.014 is the only metric where the
edge survives seeds, and it is small.

### Reading

- **The stage-B "conv+temp is the best model" result was a seed-13 artefact**
  — a strong seed for the context model against the baseline's weakest seed.
  With 15 runs each, conditions 1 and 5 are indistinguishable on F1, and
  condition 8 is a point below both.
- The stable pattern across everything run so far: context models are **better
  calibrated** (ECE 0.115–0.116 vs 0.148) and trade recall for precision; they
  are not more accurate against these labels. Per-instance gating works
  mechanically (gate std ≈ 0.1) but buys nothing on F1.
- Conclusion for the manuscript, as the data stands: RQ1 is negative (no
  significant gain from context), RQ2 shows no channel or combination that
  beats the target-only model beyond seed noise, with retrieval the weakest
  addition. The label-noise ceiling (F1 0.26–0.29 on adjudicated rows for every
  condition, 0.43+ on unanimous rows) is the dominant effect in the data and
  should be reported ahead of any architecture comparison.
- Stage E (annotator-majority label; aux cue heads; label-quality weighting)
  is running: the majority-label rows are the test of whether these
  conclusions hinge on the adjudicator.

---

## 2026-09-16 · Stage E — improvement rows (brief §4) and the seed-pooled matrix

Runs `results/impr-{majority-1_baseline,majority-8_full,aux-cues-8_full,label-weighting-8_full}/`
(20 / 102 / 100 / 91 min); `results/improvement-rows.csv`; log `results/logs/stage-C-C2-E-D.log`.
Each row is one flag turned on against an unchanged reference (seed 13).

### Annotator-majority label (the robustness row from ANNOTATION_PROVENANCE §6.4)

Trained and scored on the pre-adjudication majority vote (2,747 positives,
18.3%), then cross-scored against the other label. Same folds.

| trained on ↓ / scored against → | adjudicated label | majority label |
|---|---|---|
| **adjudicated** (stage A, seed 13) — baseline / full | 0.338 / 0.349 | 0.394 / 0.393 |
| **majority** (this stage) — baseline / full | **0.359 / 0.367** | **0.512 ± 0.020 / 0.520 ± 0.034** |

(F1 at the validation-chosen threshold; F1@0.5 for the majority rows is
0.512 / 0.519, AUPRC 0.490 / 0.507, ECE 0.118 / 0.121; the two models agree on
81–82% of decisions.)

- **The RQ1 / RQ2 conclusion does not hinge on the adjudicator.** Under the
  majority label the full model gains +0.008 over the baseline — the same
  "no material gain from context" as under the adjudicated label.
- **The majority label is the more learnable target by a wide margin** (F1 0.51
  vs 0.35, AUPRC 0.49 vs 0.31 — a random classifier would score 0.18 vs 0.11,
  so the lift is 2.7× vs 2.9× but the absolute agreement is far higher). And a
  model trained on the majority label agrees *better* with the adjudicated
  label (0.359 / 0.367) than a model trained on the adjudicated label does
  (0.338 / 0.349): the adjudicator's strict "must invert the literal meaning"
  criterion is a noisier training signal than the annotators' vote even for
  predicting the adjudicator itself.
- **The gate profile flips.** Under the adjudicated label the full model puts
  0.16 / 0.34 / 0.50 on conv / temp / ret; under the majority label it puts
  **0.34 / 0.26 / 0.40** — conversational context is trusted twice as much
  when the target label is the one the annotators (who saw that context)
  produced without the adjudicator's override. `figures/gates.png` (adjudicated)
  vs the majority-label gate panel drawn by §12.

### Other §4 rows (full model, seed 13, vs the seed-13 reference 0.356 / 0.349 / 0.312)

| row | F1 @0.5 | F1 @val-thr | AUPRC | ECE | per-fold F1@0.5 |
|---|---|---|---|---|---|
| aux_cue_heads | **0.372 ± 0.014** (+0.016) | 0.372 ± 0.008 (+0.022) | 0.314 (±0) | 0.138 | 0.377 0.377 0.389 0.363 0.353 |
| sample_weighting + soft_labels | 0.366 ± 0.020 (+0.010) | 0.366 ± 0.019 (+0.017) | 0.328 (+0.017) | 0.146 | 0.370 0.385 0.383 0.340 0.351 |

Both gains are inside the full model's own seed spread (0.353–0.357 @0.5
across seeds, fold std 0.014–0.031), so neither is claimed. The cue heads are
the more interesting of the two: the tightest fold spread of any run so far,
at no AUPRC cost — worth seeds if compute allows. Flags stay OFF in the
reported baseline.

### Seed-pooled RQ2 matrix (the notebook's `results/ablation-matrix.csv`, `figures/ablation-*.png`)

| condition | runs | F1 @0.5 | F1 @val-thr | AUPRC | AUROC |
|---|---|---|---|---|---|
| 1_baseline | 15 (3 seeds) | **0.368 ± 0.026** | 0.357 ± 0.032 | 0.316 ± 0.031 | 0.786 |
| 2_conv | 5 | 0.355 ± 0.023 | 0.362 ± 0.019 | 0.327 ± 0.010 | 0.787 |
| 3_temp | 5 | 0.356 ± 0.043 | 0.355 ± 0.045 | 0.316 ± 0.012 | 0.784 |
| 4_ret | 5 | 0.350 ± 0.032 | 0.356 ± 0.029 | 0.316 ± 0.025 | 0.776 |
| 5_conv_temp | 15 (3 seeds) | **0.370 ± 0.025** | 0.370 ± 0.022 | 0.330 ± 0.021 | 0.789 |
| 6_conv_ret | 5 | 0.355 ± 0.035 | 0.372 ± 0.019 | 0.320 ± 0.015 | 0.776 |
| 7_temp_ret | 5 | 0.357 ± 0.023 | 0.358 ± 0.022 | 0.303 ± 0.020 | 0.776 |
| 8_full | 15 (3 seeds) | 0.355 ± 0.024 | 0.355 ± 0.019 | 0.312 ± 0.015 | 0.778 |

Notebook significance over all seeds (`results/significance.csv`,
`figures/bootstrap.png`), condition 8 − condition 1: ΔF1 −0.006, 95% CI
[−0.015, +0.004], p = 0.23 at 0.5; −0.002, CI [−0.012, +0.008], p = 0.74 at the
val-tuned decision. The three-seed gate profile of the full model: conv
0.156 ± 0.098, temp 0.345 ± 0.087, ret 0.500 ± 0.078.

---

## 2026-09-17 · Stage T — hyperparameter rows (brief §3), one axis at a time

Runs `results/tune-*/` (21 / 25 / 16 / 51 / 123 min); table in
`results/improvement-rows.csv` (stage T rows); log `results/logs/stage-D-T.log`.
Same frozen folds, epochs selected on validation, seed 13, compared with the
unchanged seed-13 reference. **Agreement with the LLM-ensemble labels.**

| row | reference (seed 13) | F1 @0.5 | ΔF1 | AUPRC | collapsed folds | note |
|---|---|---|---|---|---|---|
| baseline, lr_encoder 1e-5 | 0.353 ± 0.034 | 0.366 ± 0.023 | +0.013 | 0.308 | 0 | no collapse; equal to the 3-seed baseline mean (0.368) |
| baseline, lr_encoder 3e-5 | 0.353 ± 0.034 | **0.314 ± 0.069** | −0.039 | 0.283 | **2** | unstable — two folds collapsed |
| baseline, freeze 4 layers + layer-wise decay 0.9 | 0.353 ± 0.034 | 0.349 ± 0.042 | −0.004 | 0.307 | 2 | no help; 3.7 GB peak |
| conv+temp, lr_encoder 1e-5 | 0.378 ± 0.016 | 0.373 ± **0.003** | −0.005 | 0.333 | 0 | same mean as its 3-seed average (0.370), tightest spread of any run |
| full, lr_encoder 1e-5 | 0.356 ± 0.014 | 0.363 ± 0.019 | +0.008 | 0.323 | 0 | still below the baseline |

### Reading

- **No tuned row beats the committed configuration**; the reported baseline
  stays lr 2e-5 / 1e-4, batch 16 × 2. The encoder learning rate is the one
  axis that matters, and it matters for *stability*, not for the mean: 3e-5
  collapses two folds, 1e-5 collapses none and shrinks the fold spread (the
  conv+temp row has std 0.003), 2e-5 sits between (one collapsed run in 15
  for both headline models across seeds).
- Freezing the bottom four layers with layer-wise decay (the §9.4 small-data
  hygiene) does not help here and collapsed two folds — the head appears to
  need the full encoder to move.
- Taken with stages C/C2/E, the tuning stage does not change the conclusion:
  the target-only baseline and the context models are separated by less than
  seed noise on these labels, whichever learning rate is used.
- Stage D (retrieval k=10, XLM-R `[CLS]` retrieval, unbounded temporal window,
  fixed λ) is running; k=5 already matched k=3 (F1@val-thr 0.374, AUPRC 0.315).

---

## 2026-09-17 · Stage D — specification variants (§12b), full model, seed 13

Runs `results/spec-*/`, table `results/spec-variants.csv`; 127 / 247 / 104 /
105 / 93 min. k=10 needed batch 8 × accum 4 (k=5 already peaked at 15.9 GB
allocated at batch 16; k=10 at batch 16 hit the cap and was caught cleanly).
**Agreement with the LLM-ensemble labels.**

| variant (thesis §3.4 wording vs pipeline) | F1 @0.5 | F1 @val-thr | AUPRC | AUROC | temporal coverage |
|---|---|---|---|---|---|
| manuscript-default (= condition 8: k=3, MiniLM, 48 h, learnable λ) | 0.356 ± 0.014 | 0.349 | 0.312 | 0.776 | 36.7% |
| retrieval k=5 | 0.376 ± 0.023 | 0.374 | 0.315 | 0.784 | 36.7% |
| retrieval k=10 | 0.348 ± 0.027 | 0.353 | 0.318 | 0.778 | 36.7% |
| retrieval index = XLM-R `[CLS]` (§3.4(3) literally) | 0.374 ± 0.014 | 0.369 | 0.319 | 0.789 | 36.7% |
| temporal unbounded (≤10 posts, no 48 h rule) | 0.370 ± 0.019 | 0.366 | **0.333** | 0.790 | **53.5%** |
| temporal λ fixed at init | 0.368 ± 0.012 | 0.365 | 0.315 | 0.781 | 36.7% |

### Reading

- Every variant lands in 0.348–0.376 F1 — inside the full model's own seed
  spread (0.353–0.357 across three seeds, fold std 0.012–0.027). None of the
  three manuscript-vs-pipeline disagreements (M2 window, M3 encoder, M2 λ)
  changes the result, so each can be reported as "no measurable effect"
  rather than argued.
- **Retrieval k:** k=5 ≈ k=3 ≈ k=10 on AUPRC (0.312–0.318). More exemplars
  do not make the retrieval channel useful; they only cost memory and time.
- **Retrieval encoder:** the literal XLM-R `[CLS]` index is not worse than the
  sentence encoder (AUPRC 0.319 vs 0.312). The M3 swap can stay for the
  stationarity argument, but it is not what limits the channel.
- **Temporal window:** lifting the 48 h rule raises coverage from 36.7% to
  53.5% of rows and gives the best AUPRC of any full-model run (0.333) — the
  only variant with a directional story, and it is a data-availability
  story, exactly as METHODOLOGY_REVIEW M2 predicted. The manuscript's 48 h
  rule should be reported as a constraint the 24-day collection window
  imposes, with this row beside it.
- **λ:** learnable vs fixed makes no difference (0.368 vs 0.356 / 0.315 vs
  0.312) — the decay ranks an author's few posts against each other and most
  rows have one or none.

---

## 2026-09-17 · RQ3 update — external stage 1 + our stage 2 (§13 re-run)

Same fold-0 checkpoint and heads as the earlier RQ3 entry; new row.
`results/rq3-fold0.csv`, `figures/rq3-macro_f1.png`, `figures/rq3-accuracy.png`.

| run (fold-0 test, macro-F1 / accuracy) | overall (3,001) | gold-sarcastic (321) | sarcastic ∧ literal≠intended (208) |
|---|---|---|---|
| stage 1 external only (`aux.tx_sentiment`) | 0.625 / 0.631 | 0.243 / 0.486 | 0.198 / 0.341 |
| **two-stage, external stage 1 + our stage 2 on flagged rows** | 0.621 / **0.631** | **0.276 / 0.589** | **0.247 / 0.471** |
| two-stage, our stage 1 (previous entry) | 0.604 / 0.610 | 0.248 / 0.477 | 0.209 / 0.351 |
| two-stage, oracle flag, our stage 1 | 0.619 / 0.628 | 0.254 / 0.536 | 0.216 / 0.404 |

This is the RQ3 claim in its defensible form: **with a strong context-free
stage 1, re-reading only the rows the sarcasm model flags costs nothing
overall (0.631 → 0.631 accuracy) and raises accuracy on the slice where
sarcasm inverts the sentiment from 0.34 to 0.47** (+0.10 on all gold-sarcastic
rows). The flag's 0.35 precision is what caps it; the earlier finding that
"stage 2 everywhere" beats gating still holds for our weak stage-1 head, not
for the external one.

---

## 2026-09-17 · Closing summary (brief §6)

Roughly 37 GPU-hours over 2026-09-15 → 17: stages A–E, C2, T and D, 9
improvement/tuning rows, 6 specification variants, 3 seeds on the headline
pair and on conv+temp, RQ3 on fold 0. Every run is in `results/<run_name>/`
with `fold_metrics.csv`, `predictions.csv`, `val_predictions.csv`,
`histories.json`, `run.json` (config, dataset identity, commit, wall-clock,
peak VRAM, `LABEL_AUTHORITY`); all on `folds-v1-a419a4bc95.json`, never
edited. Figures in `results/figures/`.

### What the numbers are

- **RQ1 — negative.** Target-only XLM-R baseline F1 0.368 ± 0.026 (15 runs)
  vs the full context model 0.355 ± 0.024; ΔF1 −0.013 ± 0.036, 4 wins of 15,
  p ≈ 0.2; pooled bootstrap CI [−0.016, +0.004]. The context model is better
  calibrated (ECE 0.115 vs 0.148) and more precise (0.33 vs 0.31) at lower
  recall (0.42 vs 0.49); it is not more accurate against these labels.
- **RQ2 — no channel or combination separates from the baseline beyond seed
  noise.** Conv+temp is the best point estimate (0.370 ± 0.025 over 15 runs,
  ΔF1 +0.002, AUPRC +0.014 at p 0.07); every retrieval condition is at or
  below the baseline, and the gate nevertheless assigns retrieval half its
  weight. Per-instance gating works mechanically (gate std ≈ 0.1; the profile
  flips to favour conversation under the majority label) but buys no F1.
- **RQ3 — positive in its narrow form.** Re-reading flagged rows with the
  intended-sentiment head lifts accuracy on sarcastic ∧ literal≠intended from
  0.34 to 0.47 with no overall cost, given the external stage 1.
- **The dominant effect in the data is the label.** Rows the annotators
  disagreed on (2-1 votes; 44% of the positives) score F1 ≈ 0.26 for every
  model; unanimous rows ≈ 0.43. Training on the pre-adjudication majority
  label gives F1 0.51 / AUPRC 0.49 and *better* agreement with the adjudicated
  label than training on it directly. All RQ1/RQ2 conclusions hold under both
  labels.

### What backs them

`results/ablation-matrix.csv` (seed-pooled), `results/significance.csv`
(notebook bootstrap / randomization / McNemar over all seeds),
`results/improvement-rows.csv` (E + T rows with cross-label scoring),
`results/spec-variants.csv`, `results/rq3-fold0.csv`, and the figures listed
in each entry above. Per-fold and per-seed numbers are in each run's
`fold_metrics.csv`.

### What changed (all committed on `single-notebook`, plain messages)

Headless driver `tools/run_notebook.py`; `run_cv` with validation-chosen
threshold + temperature, AUPRC/AUROC/ECE, val predictions, fold-0
checkpoints; `run_or_load`; vectorised retrieval banks and item scatter;
selftext budget honoured; `label_source`; staged §12 (A/B/C/C2/E/T), §12b
(D), §13 on the real checkpoint with the external-stage-1 row; fixed
overfit-16 smoke test; batch 16 × 2 after a VRAM probe; `demo/` (FastAPI +
static page, verified on CPU against the real checkpoint). No fold file, no
checkpoint and no post text in git.

### What could not be verified

- Nothing here says anything about human judgement: 31 gold items, κ −0.148.
- Single-seed rows (stage B's six conditions, all of D, E and T) are
  compared to a reference that itself varies ±0.015 across seeds; only the
  headline pair and conv+temp have three seeds.
- The demo's page was exercised through its API and a headless DOM, never
  in a real browser.
- RQ3 is fold 0 only (the heads and the flag come from one checkpoint).

### What to do next, with more compute

1. Seeds for aux-cue heads (the tightest single-seed row, +0.016) and for
   the unbounded temporal window (best AUPRC) — 4 h each.
2. RQ3 across all five folds with each fold's own checkpoint — 5 × 3 min
   once `save_checkpoint` is on for every fold.
3. The label is the lever, not the model: grow the gold subset (H1), settle
   the adjudicator's remit (H2), and re-export; every number above moves
   with it. Back-filling author history (H6) is the only cheap data change
   that would let the temporal channel be tested rather than reported as
   36.7% coverage.
4. A zero-shot LLM row from a model family outside the annotator pool (§9.7)
   to contextualise F1 ≈ 0.37 against these labels.

---

## 2026-09-17 · Exploration — a context model that beats the baseline (separate notebook)

`leische_explore_context.ipynb`, 3.4 GPU-hours, artifacts under
`results/explore/`; full write-up with literature and citations in
[suggestions/IMPROVEMENTS.md](../suggestions/IMPROVEMENTS.md). Same folds, same
metrics, same optimiser; the target-only baseline re-run in the same loop.
**Agreement with the LLM-ensemble labels.**

| 5 folds × seed 13 | F1 @0.5 | AUPRC | AUROC | ECE | paired ΔF1 vs target-only |
|---|---|---|---|---|---|
| target-only (same loop) | 0.369 ± 0.016 | 0.319 | 0.777 | 0.114 | — |
| early fusion + annotator heads + vote-share labels (x10) | **0.405 ± 0.010** | **0.376** | **0.826** | **0.069** | +0.037 ± 0.017, 5/5; bootstrap CI [+0.016, +0.059], p = 0.002 |
| same + kNN/author priors (x5) | 0.403 ± 0.015 | 0.368 | 0.826 | 0.087 | +0.034 ± 0.025, 5/5; CI [+0.015, +0.054] |

On fold 0 the recipe on xlm-roberta-large reaches F1 0.438 / AUPRC 0.398 /
AUROC 0.842 (single seed). The gain lives on the rows the three annotators
agreed on (3-0 votes: F1 0.467 → 0.550) and is absent on contested rows;
retrieval and author priors do not contribute; the rendering order matters.
