# Consultation notes — the sarcasm model from start to end (as of 2026-09-18)

For the adviser / panel meeting. It explains how the pipeline works in plain
terms, what every training run produced and how long it took, which versions
of the gated-fusion mechanism were tried, and what is worth deciding next.
Deeper detail lives in [docs/RESULTS_LOG.md](../docs/RESULTS_LOG.md) (dated
entries per stage), [SUMMARY.md](SUMMARY.md), [IMPROVEMENTS.md](IMPROVEMENTS.md)
(literature and references) and [GATED_FUSION_V2.md](GATED_FUSION_V2.md).

> Every accuracy number in this document is **agreement with the LLM-ensemble
> labels** (three annotator LLMs plus an adjudicator LLM; Fleiss κ 0.376 on
> sarcasm). Only 31 rows have a human label, and on those the human and the
> ensemble disagree (κ = −0.148). No number here says how well the model agrees
> with people.

---

## 0 · In five sentences

1. The pipeline is complete, gated, leak-checked and reproducible: 15,000 rows,
   1,601 sarcastic, five thread-grouped folds, every run stamped with who
   produced the labels.
2. The thesis's context-aware model does **not** beat the target-only XLM-R
   baseline on these labels (F1 0.355 vs 0.368 over 15 runs); none of the eight
   channel combinations does, and no tuning changes that.
3. A different way of using context *does* beat the baseline on every fold
   (F1 0.405 vs 0.369): render the post and the comment into one encoder input
   and train one prediction head per annotator. The gain sits entirely on rows
   the annotators agreed on.
4. The ceiling is the label, not the model: the annotators themselves agree
   with the final verdict at F1 0.49–0.58 (majority vote 0.606). An F1 of 0.6
   against the current labels would mean matching the labelling process, not
   beating it.
5. Two decisions for the consultation: whether the thesis model may change
   (add the early-fusion + annotator-heads row, and/or the redesigned gate),
   and whether the label definition may be revised (adjudicator remit, gold
   subset).

---

## 1 · How the notebook works, start to end

Everything lives in one notebook, `leische_pipeline.ipynb`, run from the top.
Long stages are driven from a terminal with `tools/run_notebook.py`, which
executes the same cells and logs to a file. Nothing is duplicated in a package.

| section | what happens | why it is there |
|---|---|---|
| **§1 Setup** | Checks the GPU and the torch build, fixes seeds, caps the process at 85% of VRAM so a spike raises a catchable error instead of freezing the laptop | reproducibility and not crashing the machine |
| **§2 Configuration** | One `Config` object holds every switch: encoder, budgets, which context channels are on, learning rates, all §9 upgrades (off by default) | every thesis claim stays reproducible with upgrades off |
| **§3 Data + gate** | Loads the uyam export, validates every row against the contract, prints where the labels came from (`LABEL_AUTHORITY`), and runs a two-tier readiness gate: *train-ready* (enough data, correct structure, context is the annotator snapshot, folds frozen) blocks training in code; *claim-ready* (gold subset, κ) never blocks but stamps every result | so no number can be quoted as human agreement when it is ensemble agreement |
| **§4 EDA** | Label counts by language, vote patterns, text lengths against token budgets, context coverage (how many rows have parents / replies / author history), a manual read of 20 sarcastic rows with all rationales | tells you whether a channel is measurable at all before modelling it |
| **§5 Frozen folds** | Five folds, stratified on sarcasm × language, **grouped by thread** so a comment is never scored against a model that saw its own thread; frozen to a file keyed on the dataset identity | 96% of rows share a thread with another row; ungrouped folds would leak and inflate exactly the effect the thesis claims |
| **§6 Context channels** | Builds the three context sources per row: *conversational* (the post, parent comments, replies — exactly the snapshot the annotators saw), *temporal* (the author's earlier posts, hours apart), *retrieval* (similar labelled training rows, banks rebuilt per fold from training rows only, same-thread excluded, leakage asserted) | the three inputs of RQ2 |
| **§7 Model** | The five-stage architecture: one shared XLM-R encodes every text; the target attends over conversational and temporal items (with time decay on the scores); retrieval attends over sarcastic and non-sarcastic exemplars; **Stage 4** computes one gate per channel from `[target ; channel]`, normalised to sum to one, and mixes the channels; a small MLP classifies `[target ; mixed context]`. With all channels off it is exactly the target-only baseline | RQ1 and RQ2 in one class, so the comparison cannot drift |
| **§8 Training** | For each fold (× seed): class weights from the training fold, AdamW with separate encoder / head learning rates, early stopping on validation F1, best weights restored; then on the **validation** rows only it picks the F1-optimal threshold and fits a temperature; writes `fold_metrics.csv`, `predictions.csv`, `val_predictions.csv`, `histories.json`, `run.json` (config, dataset identity, git commit, wall-clock, peak VRAM, label authority). Finished runs are reloaded, never repeated | the artifacts are the evidence; the validation-only decisions keep the test fold clean |
| **§8b Plots** | Eleven panels: training curves, fold spread, threshold sweep, PR/ROC, confusion, calibration, ablation bars, slices, gates, bootstrap, RQ3 | figures the manuscript can cite, written as they are drawn |
| **§9–§11 Smoke tests** | Overfit 16 rows (8 + 8) to near-zero loss for the baseline and the full model; a tiny 5-fold dry run; a checkpoint round-trip | a wiring bug is found before a GPU-hour is spent |
| **§12 Ablation** | The eight conditions × five folds, staged (A: baseline and full; B: the other six; C: extra seeds; E: improvement rows; T: tuning rows), then paired bootstrap, approximate randomisation and McNemar between condition 8 and condition 1 | RQ2, with the significance test the manuscript does not specify but the examiner asks for |
| **§12b Variants** | The places where the manuscript and the pipeline disagreed (retrieval k, retrieval index, 48 h temporal window, learnable λ), each reported with its temporal coverage | resolves the disagreements by reporting both, not by arguing |
| **§13 RQ3** | Two-stage sentiment: a literal-sentiment head on the target alone, then an intended-sentiment head re-reading only the rows the sarcasm model flags; sliced by language and by "sarcastic ∧ literal ≠ intended" | the thesis's sentiment claim, with an oracle bound and an external stage-1 cross-check |
| **§14 Verdict** | Re-prints the gate and the smoke outcomes | |

Two separate notebooks were added later and pull §1–§8b in rather than copying
them: `leische_explore_context.ipynb` (16 alternative ways to integrate
context, §3.3 below) and `leische_prototype_v3.ipynb` (the redesigned gate,
§2.3 below). A small FastAPI demo (`demo/`) loads a trained checkpoint and shows
the probability, gates, exemplars and sentiment for any typed comment with
editable context.

---

## 2 · The gated-fusion mechanism: versions tried

The thesis's central mechanism is Stage 4: per-instance gates that decide how
much of each context source to trust for *this* comment.

| version | gate | status | what was measured |
|---|---|---|---|
| **v0** (pipeline before the methodology review) | one softmax over the three *context* vectors concatenated — the target never entered the gate | replaced (M1) | gates nearly constant across rows (std 0.017 – 0.050 on the pilot); could not be per-instance by construction |
| **v1** (committed, manuscript §3.4.1 as written) | one `sigmoid(W [target ; channel])` per active channel, normalised to sum to one; disabled channels leave the gate | trained in every stage | gates vary per row (std ≈ 0.09–0.11): conv 0.16, temporal 0.33, retrieval 0.50 (3 seeds). But the conversational gate does **not** rise with the number of thread turns (0.155 with none, 0.149 with 3+), retrieval gets half the weight while every retrieval condition is at or below the baseline, and the profile flips to conv 0.34 / temp 0.26 / ret 0.40 when the model is trained on the annotator-majority label. Per-instance gating is mechanically real; a per-instance *benefit* is not |
| v1, two-channel pairs (conditions 5–7) | same gate over two channels | trained (stage B) | conv+temp: 0.45 / 0.55; conv+ret: 0.31 / 0.69; temp+ret: 0.33 / 0.67 — retrieval dominates whenever present |
| v1, specification variants (stage D) | same gate; retrieval k = 5 / 10, XLM-R `[CLS]` index, unbounded window, fixed λ | trained | all inside seed noise (F1 0.348–0.376); the unbounded window gives the best AUPRC (0.333) by raising temporal coverage from 36.7% to 53.5% |
| **no gate: early fusion** (exploration x1–x15) | context and target rendered into one encoder input; no channels, no gate | trained (3.7 GPU-h) | beats the baseline only when combined with per-annotator heads (F1 0.405 ± 0.010 on 5 folds); early fusion alone raises AUROC (0.73 → 0.80) but not F1 |
| **v2** (`GATED_FUSION_V2.md`, `leische_prototype_v3.ipynb`) | keeps three channels + per-instance gate, but: each item is encoded **with** the target; gate input `[t ; c ; t⊙c ; |t−c| ; self-report scalars]`; a fourth **null** channel ("no context helps"); channel dropout, learned temperature, entropy term; classifier sees the mismatch; retrieval as a zero-pass prototype contrast; per-annotator heads and **cue-supervised gates** (`contextual_incongruity → conv gate`) | drafted, stub-checked; small-data test in §2.3 | see §2.3 |

### 2.3 Small test of v2 (500 sarcastic + 1,000 non-sarcastic training rows)

Run on 2026-09-18 with `leische_prototype_v3.ipynb` §5b (14 GPU-minutes in
total). Training rows: a seeded sample of 500 sarcastic + 1,000 non-sarcastic
rows from fold 0's training half (thread grouping inherited, so nothing in the
test fold shares a thread with them); every prior, retrieval bank and class
weight built from those 1,500 rows only; early stopping on fold 0's own
validation split; **scored on the full fold-0 test set** (3,001 rows, 321
sarcastic, the natural 10.7% base rate). Single seed. A functional test, not a
result — absolute numbers are lower than the 10.8k-row runs by design.

| row (trained on 1,500 rows) | F1 @0.5 | P / R | AUPRC | AUROC | ECE | mean gates conv / temp / ret / null | min | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| target-only (reference) | 0.302 | 0.25 / 0.39 | 0.220 | 0.742 | 0.156 | — | 0.9 | 6.6 GB |
| **v1** committed gate (conv + temp + retrieval) | 0.317 | 0.22 / 0.57 | 0.211 | 0.728 | 0.185 | 0.34 / 0.34 / 0.32 (flat) | 2.6 | 9.6 GB |
| v2-a: matching gate + null channel + channel dropout | 0.295 | 0.22 / 0.45 | 0.230 | 0.741 | 0.181 | 0.20 / 0.38 / 0.14 / 0.28 | 3.1 | 9.7 GB |
| v2-b: v2-a + target-aware item encoding | 0.294 | 0.21 / 0.51 | 0.198 | 0.710 | 0.200 | 0.19 / 0.31 / 0.23 / 0.27 | 3.2 | 14.0 GB |
| **v2-c: v2-b + per-annotator heads + vote-share labels + cue-supervised gates** | **0.362** | 0.28 / 0.51 | **0.275** | **0.758** | **0.131** | 0.12 / 0.45 / 0.42 / 0.02 | 4.0 | 14.1 GB |

What the gates did (mean gate by how many thread turns the row has, and on
rows the annotators flagged `contextual_incongruity`):

| row | conv gate: post only → 1 turn → 2+ turns | null gate: post only → 2+ turns | conv gate on incongruity-flagged vs other rows |
|---|---|---|---|
| v2-a | 0.22 → 0.18 → 0.16 | 0.28 → 0.29 | **0.26 vs 0.19** (unsupervised) |
| v2-b | 0.21 → 0.18 → 0.16 | 0.27 → 0.28 | 0.21 vs 0.19 |
| v2-c | 0.14 → 0.11 → 0.10 | 0.02 → 0.02 | **0.23 vs 0.11** (supervised) |

Reading, honestly:

- **The votes carry the gain again.** With 1,500 training rows the committed
  gate (v1) is at the target-only level; the redesigned gate alone (v2-a) and
  the target-aware encoding (v2-b) do not move F1; adding the per-annotator
  heads, vote-share labels and cue-supervised gates (v2-c) lifts every metric:
  +0.06 F1, +0.055 AUPRC and the best calibration of the five. The pattern is
  the same one the full-data exploration found, now reproduced inside the
  thesis's channel-and-gate framework.
- **The gate redesign changes the gate's behaviour, not (yet) the accuracy.**
  The matching-feature gate stops trusting retrieval (0.32 → 0.14) and its
  conversational gate opens more on rows the annotators flagged as
  context-incongruent even without supervision (0.26 vs 0.19); with the cue
  loss the separation doubles (0.23 vs 0.11). That is the first evidence in
  the project of a gate responding to *why* a comment is sarcastic.
- **Two things did not work as designed.** The null gate does not rise when a
  row has no thread turns (0.28 → 0.29, flat), so "no context helps" is not
  being learned from availability at this data size; and in v2-c the null
  gate collapsed to 0.02 because its supervision target (polarity inversion
  without incongruity) is positive on only 4% of rows, so the loss teaches it
  to stay shut. The fix is to supervise the null gate with "no thread turn
  available" rows instead, or not at all; this is corrected in
  `GATED_FUSION_V2.md` §2.7 and is a one-line change in the notebook.
- **Cost.** Target-aware item encoding doubles the item tokens (14 GB peak at
  batch 16 vs 9.6 GB) for no gain at this size; on the full data it should be
  re-tested (v2-b vs v2-a) before being kept.
- **Next step, if the consultation agrees:** run rows v2-a → v2-c on the full
  fold-0 training set (~30 min), then seeds and five folds for whichever row
  beats the exploration's 0.405 (`NEXT_SESSION_BRIEF.md` T1).

---

## 3 · Results and runtime

### 3.1 The thesis matrix (RQ1, RQ2) — `results/ablation-matrix.csv`

| condition | conv | temp | ret | F1 @0.5 | AUPRC | runs | minutes / 5 folds | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| 1 baseline | – | – | – | **0.368 ± 0.026** | 0.316 | 15 | 23 | 6.6 GB |
| 2 | ✓ | – | – | 0.355 ± 0.023 | 0.327 | 5 | 50 | 9.1 GB |
| 3 | – | ✓ | – | 0.356 ± 0.043 | 0.316 | 5 | 40 | 7.9 GB |
| 4 | – | – | ✓ | 0.350 ± 0.032 | 0.316 | 5 | 76 | 10.1 GB |
| 5 | ✓ | ✓ | – | 0.370 ± 0.025 | 0.330 | 15 | 61 | 10.0 GB |
| 6 | ✓ | – | ✓ | 0.355 ± 0.035 | 0.320 | 5 | 105 | 12.6 GB |
| 7 | – | ✓ | ✓ | 0.357 ± 0.023 | 0.303 | 5 | 98 | 11.4 GB |
| 8 full | ✓ | ✓ | ✓ | 0.355 ± 0.024 | 0.312 | 15 | 114 | 13.3 GB |

Condition 8 − condition 1 over 15 paired runs: ΔF1 −0.013 ± 0.036, 4 wins of
15, p ≈ 0.2; bootstrap CI [−0.016, +0.004]. The context model is better
calibrated (ECE 0.115 vs 0.148) and more precise at lower recall; it is not
more accurate.

### 3.2 Added rows

| row | F1 @0.5 | AUPRC | note | minutes |
|---|---|---|---|---|
| annotator-majority label, baseline / full | 0.512 / 0.519 | 0.490 / 0.507 | trained and scored on the pre-adjudication majority; same conclusion (full − baseline +0.008) | 20 / 102 |
| aux cue heads (full) | 0.372 ± 0.014 | 0.314 | +0.016 vs 0.356, tightest spread, inside seed noise | 100 |
| label-quality weighting (full) | 0.366 ± 0.020 | 0.328 | inside seed noise | 91 |
| encoder lr 1e-5 / 3e-5 (baseline) | 0.366 / **0.314** | 0.308 / 0.283 | 3e-5 collapses two folds; 1e-5 only tightens spread | 21 / 25 |
| freeze 4 + layer-wise decay (baseline) | 0.349 | 0.307 | no help | 16 |
| retrieval k=5 / k=10 / XLM-R index (full) | 0.376 / 0.348 / 0.374 | 0.315 / 0.318 / 0.319 | all inside seed noise | 127 / 247 / 104 |
| temporal unbounded / λ fixed (full) | 0.370 / 0.368 | **0.333** / 0.315 | coverage 36.7% → 53.5% | 105 / 93 |
| RQ3 two-stage sentiment (fold 0) | accuracy 0.631 overall; 0.34 → 0.47 on sarcastic ∧ literal ≠ intended (external stage 1 + our stage 2) | | the claim in its defensible form | 1.4 |

### 3.3 The exploration (separate notebook; same folds, metrics, optimiser)

| variant (fold 0, seed 13 unless noted) | F1 | AUPRC | AUROC | minutes |
|---|---|---|---|---|
| x0 target-only (re-run in this loop) | 0.367 | 0.328 | 0.733 | 3.6 |
| x1 early fusion (post → parents → target → replies in one input) | 0.362 | 0.322 | 0.795 | 5.8 |
| x2 + kNN / author priors | 0.359 | 0.287 | 0.763 | 5.0 |
| x3 + one head per annotator | 0.349 | 0.319 | 0.778 | 5.8 |
| x4 + vote-share soft labels | 0.373 | 0.288 | 0.769 | 4.3 |
| **x10 early fusion + annotator heads + soft labels** | **0.406** | 0.341 | 0.815 | 6.6 |
| x12 + priors + heads | 0.401 | 0.374 | 0.807 | 5.7 |
| x7 early fusion on xlm-roberta-large | 0.401 | 0.347 | 0.818 | 15.1 |
| **x13 large + priors + heads + soft** | **0.438** | **0.398** | **0.842** | 15.4 |
| x8 mDeBERTa-v3-base | 0.385 | 0.363 | 0.815 | 9.3 |
| x9 RoBERTa-Tagalog | 0.337 | 0.265 | 0.750 | 3.6 |
| x14 / x15 the exact uyam prompt rendering | 0.378 / 0.358 | 0.309 / 0.307 | | 5.4 / 6.5 |
| **x10, 5 folds** | **0.405 ± 0.010** vs 0.369 ± 0.016 | 0.376 vs 0.319 | 0.826 vs 0.777 | 6–7 per fold |

x10 − target-only over 5 folds: ΔF1 +0.037 ± 0.017, 5 wins of 5, bootstrap
CI [+0.016, +0.059], p = 0.002. The gain is on unanimous rows only (3-0 votes:
F1 0.467 → 0.550; 2-1 votes: flat). Removing the annotator heads (x11) drops
back to 0.347; the priors are dispensable; the rendering order matters
(target first is worse); copying the annotators' exact markers does not help.

### 3.4 The label ceiling

| scorer | F1 vs the final verdict |
|---|---|
| gemma3 27B / SEA-LION 9B / qwen3 8B (the annotators) | 0.576 / 0.560 / 0.494 |
| majority of the three | **0.606** (0.971 on unanimous rows, 0.345 on contested rows) |
| best student model (x10, 5 folds) | 0.405 |

### 3.5 Runtime summary (RTX 5090 Laptop, 24 GB, batch 16 × 2, fp16 / bf16)

| block | GPU time |
|---|---|
| stage A (conditions 1, 8) | 2.3 h |
| stage B (conditions 2–7) | 7.2 h |
| stages C + C2 (seeds) | 6.9 h |
| stage E (improvement rows) | 5.2 h |
| stage T (tuning rows) | 3.9 h |
| stage D (specification variants) | 11.3 h |
| RQ3 | minutes |
| exploration (three passes, 39 runs) | 3.7 h |
| **total** | **≈ 41 h** |

Each full-model fold costs ~23 min at 13.3 GB; a baseline fold ~4.5 min; an
early-fusion fold 6–7 min at 8.6 GB; the large encoder ~15 min per fold at
15.2 GB.

---

## 4 · What the evidence says (short)

- Late fusion of separately-pooled context vectors adds nothing the target-only
  encoder lacks; token-level interaction between target and post does, but
  only once the model is also trained on the three annotators' votes.
- Retrieval is a distractor in every form; the gate over-trusts it.
- Temporal context is data-limited (36.7% coverage); the channel measures
  availability.
- The contested 44% of positives are unlearnable for every model, including
  the annotators' own majority (0.345). The label, not the architecture, sets
  the ceiling.

---

## 5 · Suggestions, and the decisions the consultation should make

| # | suggestion | expected effect | cost | decision needed |
|---|---|---|---|---|
| 1 | Add **condition 9** — early fusion + per-annotator heads + vote-share labels — to the RQ2 matrix, 5 folds × 3 seeds, beside conditions 1–8 | +0.03–0.04 F1, confirmed on 5 folds | 1.5 h | may the thesis model include a row that departs from Chapter III's late fusion? |
| 2 | Test the **redesigned gate (v2)** as rows v2-a … v2-e; keep it if the null gate tracks context availability and F1 ≥ condition 9 | +0.02–0.04 on top of (1); a gate whose weights mean something | 4–6 h | is the per-instance gating claim worth defending with a mechanism, or reported as an ablation? |
| 3 | Report results against **both** verdicts (adjudicated and annotator-majority) and the 3-0 / 2-1 slices as a first-class table | changes what the numbers mean; 0.51 on the majority label | none | is a documented label revision acceptable? |
| 4 | Data side (`UYAM_CHANGE_BRIEF.md`): ship both labels, settle the adjudicator's remit (it rejects mock praise and hyperbole the human counts), guideline, gold subset to ~300, author-history back-fill | the only route to numbers that mean human agreement | annotation time | who labels the gold subset, and by when? |
| 5 | xlm-roberta-large as a "capacity" row | +0.03 (fold 0) | 15 min / fold | keep XLM-R base as the thesis architecture? |
| 6 | Drop retrieval exemplar texts (keep at most a kNN label rate) | −60% compute, no loss | none | may the manuscript report retrieval as a negative result? |
| 7 | Rationale distillation from the 45,000 annotator rationales | unknown, largest unused signal | days | out of scope for this thesis? |

What I would not spend more time on: thresholds, learning rates, freezing,
retrieval k, the temporal decay form, the exact prompt rendering — all
measured, all flat.

---

## 6 · Questions worth answering at the consultation

1. Is F1 against the *ensemble* verdict an acceptable headline metric for the
   manuscript, given κ = −0.148 on the 31 gold items? If not, what is the plan
   for the gold subset?
2. Chapter III specifies late gated fusion. Can the results chapter present
   the early-fusion + annotator-heads model as the thesis's best model, or
   only as an extension?
3. Is the adjudicator's definition of sarcasm ("must invert the literal
   meaning") the thesis's definition? The human annotator's is wider. This
   single sentence decides 1,146 labels.
4. Should RQ2 be reframed as "which sources are *measurable* on this corpus"
   (temporal coverage 36.7%) rather than "which sources help"?
