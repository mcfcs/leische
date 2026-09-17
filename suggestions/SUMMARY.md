# Leische in one document — what was tested, where it stands, what would move it

A plain-language summary for the thesis team. The detailed logs are
[docs/RESULTS_LOG.md](../docs/RESULTS_LOG.md) (every training stage, dated) and
[IMPROVEMENTS.md](IMPROVEMENTS.md) (literature, proposals, evidence, 23
references); the design draft for a stronger fusion mechanism is
[GATED_FUSION_V2.md](GATED_FUSION_V2.md).

> **The one caveat that travels with every number.** All labels were produced
> by an ensemble of three LLM annotators (gemma3 27B, qwen3 8B, SEA-LION 9B)
> plus an LLM adjudicator (qwen3 32B). Only 31 items have a human label, and on
> those the human and the ensemble disagree (Cohen's κ = −0.148). Every F1 below
> is *agreement with the ensemble's verdict*, not with human judgement.

---

## 1 · How every number is measured

- **The task.** For each Reddit comment (the *target*), predict whether it is
  sarcastic. The model may also read the thread around it (the post, the parent
  comments, the replies), the author's earlier posts, and similar training
  examples, depending on the condition.
- **The truth it is scored against.** The row's final verdict,
  `labels.sarcastic`: the three annotators' majority when they agreed, the
  adjudicator's call when they did not (33% of rows). 1,601 of 15,000 rows are
  sarcastic (10.7%).
- **F1.** Precision is "of the rows the model called sarcastic, how many the
  verdict also calls sarcastic"; recall is "of the rows the verdict calls
  sarcastic, how many the model found"; F1 is their harmonic mean. With 10.7%
  positives, always saying "not sarcastic" scores F1 = 0, and a random guess
  scores about 0.19. AUPRC (area under the precision–recall curve) measures the
  ranking quality without picking a threshold; chance is 0.107.
- **Folds and seeds.** The 15,000 rows are split five ways so that whole Reddit
  threads stay on one side (no target is ever scored against a model that saw
  its own thread). Every model is trained five times (one per fold) and the
  mean ± std is reported; the headline models were also re-run with three
  random seeds, because a single run can move by ±0.03 F1 on this data.

---

## 2 · What was tested (about 41 GPU-hours on the RTX 5090, 2026-09-15 → 17)

| stage | what | why |
|---|---|---|
| Recheck + smoke | data gate (5/5 PASS), leakage checks, overfit-16 wiring tests | training was legitimate before any result was produced |
| A | baseline (target only) and full model (all three context channels), 5 folds | the RQ1 headline |
| B | the other six channel combinations | the RQ2 ablation matrix |
| C, C2 | three seeds for the headline pair and for the best-looking condition | separate signal from seed noise |
| D | retrieval k = 5 / 10, XLM-R `[CLS]` retrieval index, unbounded temporal window, fixed λ | the manuscript-vs-pipeline disagreements (M2, M3) |
| E | annotator-majority label; auxiliary cue heads; label-quality weighting | the brief's §4 improvement rows |
| T | encoder learning rate 1e-5 / 3e-5; freeze 4 layers + layer-wise decay | the brief's §3 tuning rows |
| RQ3 | two-stage sentiment on the fold-0 checkpoint | pre- vs post-sarcasm sentiment reading |
| Exploration | a different way to integrate context (16 variants, 3 seeds, 5 folds) in a separate notebook | can *anything* beat the baseline on these labels? |

---

## 3 · Current results

### 3.1 The thesis pipeline (committed architecture)

| model | F1 (15 runs: 5 folds × 3 seeds) | precision / recall | AUPRC | calibration (ECE, lower is better) |
|---|---|---|---|---|
| target-only XLM-R baseline | **0.368 ± 0.026** | 0.31 / 0.49 | 0.316 | 0.148 |
| full context model (conv + temporal + retrieval, gated) | 0.355 ± 0.024 | 0.33 / 0.42 | 0.312 | 0.115 |
| conv + temporal (best of the eight conditions) | 0.370 ± 0.025 | 0.33 / 0.45 | 0.330 | 0.116 |

- **RQ1 is negative:** the context model does not beat the target-only
  baseline (ΔF1 −0.013 over 15 paired runs, 4 wins of 15, p ≈ 0.2).
- **RQ2:** all eight conditions sit between 0.350 and 0.370; no channel or
  combination separates from the baseline beyond seed noise. Retrieval is the
  weakest addition in every form tried, yet the gate gives it half its weight.
- **RQ3 is positive in a narrow form:** re-reading only the rows the sarcasm
  model flags with an intended-sentiment head keeps overall sentiment accuracy
  at 0.631 and raises it on the slice where sarcasm inverts the sentiment from
  0.34 to 0.47 (fold 0).
- **Tuning did not move it:** the validation-chosen threshold, three learning
  rates, freezing, label weighting and cue heads all stay inside seed noise.
- **The label is the dominant effect:** on rows the annotators disagreed on
  (2-1 votes, 44% of the positives) every model scores F1 ≈ 0.26; on unanimous
  rows ≈ 0.43. Trained and scored on the *pre-adjudication* majority label,
  the same models reach F1 0.51.

### 3.2 The exploration (separate notebook, same folds and metrics)

| model (5 folds, seed 13) | F1 | precision / recall | AUPRC | AUROC | ECE |
|---|---|---|---|---|---|
| target-only, same loop | 0.369 ± 0.016 | 0.31 / 0.46 | 0.319 | 0.777 | 0.114 |
| **early fusion + one head per annotator + vote-share labels** | **0.405 ± 0.010** | 0.41 / 0.41 | **0.376** | **0.826** | **0.069** |
| same on xlm-roberta-large (fold 0 only, one seed) | 0.438 | 0.41 / 0.48 | 0.398 | 0.842 | 0.065 |

ΔF1 +0.037 ± 0.017 vs target-only, 5 wins of 5, bootstrap 95% CI
[+0.016, +0.059], p = 0.002. This is the first model in the project that beats
the baseline on every fold and every metric — and it does it in 6–7 minutes per
fold, cheaper than the committed full model. Three ingredients: the post and
the target rendered into **one** encoder input (early fusion); a prediction
head **per annotator** trained on that annotator's vote; and soft labels from
the vote share. The gain lives entirely on rows the annotators agreed on
(3-0 votes: F1 0.467 → 0.550); contested rows stay flat. Retrieval and author
priors did not contribute; copying the annotator prompt's exact rendering did
not help either.

---

## 4 · Why the committed context model did not help

1. **It sees the thread through a keyhole.** Every context item is encoded on
   its own and squeezed to one 256-d vector before the target ever meets it;
   target tokens and context tokens never share an attention map. All positive
   results in the literature on conversational sarcasm put them in one input.
2. **Half its attention goes to a distractor.** The nearest-neighbour
   exemplars are topically similar comments with noisy labels; the gate trusts
   them most (0.50), and every retrieval condition is at or below the baseline.
3. **It trains on a collapsed label.** The three votes and the adjudicator's
   override are the only per-row reliability signal in the data, and the
   single hard label throws them away. Using them (one head per annotator) is
   what made the exploration model work.

---

## 5 · Can it reach F1 0.6?

The honest way to answer is to ask how well the *annotators themselves* agree
with the verdict they helped produce. They are 8–27B-parameter LLMs, they read
the same thread, and their votes **are** the label on 67% of rows:

| scorer (15,000 rows) | F1 vs the final verdict | precision / recall | κ |
|---|---|---|---|
| gemma3 27B (annotator) | 0.576 | 0.45 / 0.81 | 0.51 |
| SEA-LION 9B (annotator) | 0.560 | 0.42 / 0.86 | 0.49 |
| qwen3 8B (annotator) | 0.494 | 0.38 / 0.70 | 0.41 |
| **majority of the three** | **0.606** | 0.48 / 0.82 | 0.55 |
| majority vote, unanimous rows only | 0.971 | | |
| majority vote, contested (2-1) rows only | 0.345 | | |
| two annotators predicting the third | 0.40–0.62 | | 0.27–0.45 |
| best student model so far (exploration, 5 folds) | 0.405 | 0.41 / 0.41 | |

So on the current labels, **F1 ≈ 0.6 is the agreement level of the labelling
process itself.** A 278-million-parameter encoder fine-tuned on 12,000 of
these labels is not going to agree with the verdict more consistently than the
27-billion-parameter model that helped write it. Three consequences:

- **With the current labels and the current framework, expect 0.40–0.45,
  not 0.6.** The committed design is at 0.35–0.37; the exploration recipe is at
  0.405; the large encoder adds +0.03 on fold 0; a stronger gate
  ([GATED_FUSION_V2.md](GATED_FUSION_V2.md)) is a plausible +0.02–0.04 on top,
  mostly through the same mechanisms. Anything above ~0.5 on the adjudicated
  label should be treated as a leak until proven otherwise.
- **The route to 0.6 runs through the label, not the model.** Against the
  three-annotator *majority* verdict the same models already score 0.51–0.52,
  and on the unanimous 70% of rows any decent model scores far higher. A thesis
  that (a) reports results against both verdicts, (b) settles the adjudicator's
  definition (its "must invert the literal meaning" rule is what the human
  annotator disagrees with), and (c) grows the gold subset to ~300 items, can
  legitimately report a 0.5–0.6 number *and* say what it means. That is a
  documented label revision, not a trick — but it is a decision for the thesis,
  not for the model code.
- **Against a human gold standard, nothing can be said yet.** 31 items with
  κ = −0.148 means the human and the ensemble currently disagree about what
  sarcasm is; until the gold subset is large enough, a higher F1 against the
  ensemble does not imply better agreement with people.

---

## 6 · Possible improvements, ranked by expected value

| # | change | expected effect on the current labels | cost | where it is described |
|---|---|---|---|---|
| 1 | Add the exploration recipe (early fusion + per-annotator heads + vote-share labels) as **condition 9** of the thesis matrix, 5 folds × 3 seeds | +0.03–0.04 F1 over the baseline, confirmed | ~1.5 h GPU | IMPROVEMENTS.md §6 |
| 2 | **Stage-4 v2**: target-aware item encoding, matching features in the gate, a "no context" option, channel dropout, cue-supervised gates — keeps the thesis's three channels and per-instance gate | +0.02–0.04 on top of (1); a gate whose weights mean something | ~3 h GPU for the ablation | GATED_FUSION_V2.md |
| 3 | xlm-roberta-large as the "capacity" row | +0.03 (fold 0 evidence) | 15 min per fold | IMPROVEMENTS.md §5 |
| 4 | Report the annotator-vote slices (3-0 vs 2-1) as a first-class table, and results against the majority verdict beside the adjudicated one | changes what the numbers *mean*, not the model | none | RESULTS_LOG stage E |
| 5 | Drop retrieval exemplar texts; keep at most the kNN label rate as a scalar | −60% compute, no loss | none | stage D, exploration x2/x11 |
| 6 | Unbounded temporal window (+17 points of coverage) | +0.02 AUPRC | none | stage D |
| 7 | Rationale distillation from the 45,000 annotator rationales | unknown, potentially the largest unused signal | days | IMPROVEMENTS.md P7 |
| 8 | Label side: gold subset to ~300, adjudicator remit, guideline for hyperbole-only cases | the only route to numbers that mean human agreement | annotation time | ANNOTATION_PROVENANCE.md §6 |

---

## 7 · Small glossary

- **Target** — the comment being judged. **Context** — the post, parent
  comments and replies around it (conversational), the author's earlier posts
  (temporal), and similar labelled training examples (retrieval).
- **Gate** — the thesis's Stage 4: per-instance weights that say how much of
  each context channel to trust for this comment.
- **Early fusion** — putting context and target into one encoder input so
  their tokens attend to each other, instead of encoding them separately and
  merging vectors afterwards (late fusion).
- **Annotator heads** — extra output layers that learn each annotator's own
  vote, so the model learns *where the annotators disagree* rather than a
  single collapsed answer.
- **κ (kappa)** — agreement corrected for chance; 0 is chance, 1 is perfect,
  negative is worse than chance.
- **Seed** — the random initialisation; on this data it moves F1 by ±0.03, so
  single runs are never reported as results.
