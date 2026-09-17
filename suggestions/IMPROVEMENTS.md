# How to get context to pay on these labels — diagnosis, literature, proposals, evidence

Written after the staged run logged in [docs/RESULTS_LOG.md](../docs/RESULTS_LOG.md)
(37 GPU-hours; 15,000 rows; the committed context model did not beat the
target-only baseline). The question here is narrower than "is the thesis
right": **given the data as it stands — 15,000 rows, LLM-ensemble labels,
31 gold items — what would make an integrated-context model outperform the
target-only XLM-R baseline, and does it?** The companion notebook
[leische_explore_context.ipynb](../leische_explore_context.ipynb) tests the
proposals on fold 0 and on all five folds for the best one; §5 below reports
what it found. Everything is agreement with the LLM-ensemble labels, never with
human judgement.

---

## 1 · What the results actually say (diagnosis)

| observation (RESULTS_LOG) | what it implies for the design |
|---|---|
| Baseline 0.368 ± 0.026 vs full 0.355 ± 0.024 F1 over 15 runs; every one of the eight conditions inside 0.350–0.370 | Late fusion of three *separately pooled* context vectors adds parameters but no information the target-only encoder lacks — or adds it in a form the classifier cannot use. |
| The gate gives retrieval 0.50 of its weight, yet every retrieval condition is at or below the baseline; k=3/5/10 and the encoder swap change nothing | The retrieval channel is a *distractor*: MiniLM top-k neighbours are topically similar rows, and their (noisy) labels do not generalise. Feeding 6–20 exemplar **texts** through the encoder is expensive noise. |
| Conversational context gets the **lowest** gate weight (0.16) — although the LLM annotators produced the label *while reading that very context* | The information is there, but the architecture hands the classifier a 256-d attention-pooled summary of parent/reply texts; it cannot align target tokens with context tokens. Under the annotator-majority label the same gate rises to 0.34 — the channel matters, the late-fusion route wastes it. |
| Temporal: 36.7% coverage; unbounded window → 53.5% coverage and the best AUPRC of any full-model run (0.333) | Author history is data-limited; what the channel really conveys is "how this author usually talks". A cheap *prior* may carry that as well as five encoder passes. |
| 2-1 vote rows: F1 ≈ 0.26 for every model; 3-0 rows ≈ 0.43; adjudicated rows ≈ 0.27 | ~44% of the positives are contested. A single hard label throws away the disagreement structure; the three annotator votes and the adjudicator's override are the *only* per-row reliability signal the data has. |
| Majority-label training: F1 0.51 vs 0.35, and *better* agreement with the adjudicated label than adjudicated-label training | The adjudicator's criterion is a noisier training target than the annotators' vote even for predicting the adjudicator. |
| Val-tuned thresholds hurt; LR 3e-5 collapses folds; freezing hurts | Optimisation is already near its ceiling for this architecture; the gain has to come from the input and the target, not the schedule. |

Two cheap diagnostics (notebook §2, fold 0, training rows only, same thread
excluded): the kNN-100 sarcasm rate scores AUPRC 0.147 / AUROC 0.63 (chance
0.107 / 0.50); the author's labelled-history rate scores AUPRC 0.146 on the 52%
of test rows whose author has any training rows (median 2 rows). Both are weak
but non-zero: worth a handful of scalar features, not a channel of encoder
passes.

**In one sentence:** the labels are a function of (target, rendered thread,
three LLM readers); the committed model sees the thread through a keyhole,
spends half its attention on a distractor, and trains on a collapsed label.

---

## 2 · What the literature says about each lever

**Conversation context works when it is in the same input as the target.**
Ghosh, Fabbri and Muresan (2018) showed with LSTMs that modelling the
conversation context together with the current turn improves sarcasm
detection and that attention can locate the triggering part of the context
[[1]](https://aclanthology.org/J18-4009/). In the FigLang 2020 shared task on
Twitter and Reddit threads [[2]](https://aclanthology.org/2020.figlang-1.1/),
the strongest systems concatenated the response with its context inside one
transformer: Dong, Li and Choi (2020) report +3.1 / +7.0 F1 over their
context-free baselines with multi-head attention over the target and the
thread [[3]](https://aclanthology.org/2020.figlang-1.38/), and Baruah et al.
(2020) found the *amount* of context matters — the last turn helped on
Twitter, while on Reddit adding more turns hurt [[4]](https://aclanthology.org/2020.figlang-1.12/).
The committed pipeline never puts context tokens and target tokens in one
attention map; it is the one design the field's positive results all share.

**Author identity is a strong, cheap prior — where authors repeat.** CASCADE
(Hazarika et al. 2018) combined discourse context with stylometric user
embeddings and gained substantially on the SARC Reddit corpus
[[5]](https://aclanthology.org/C18-1156/); earlier, Amir et al. (2016) and
Bamman and Smith (2015) established user embeddings and author features as
the largest single context effect on Twitter
[[6]](https://aclanthology.org/K16-1017/) [[7]](https://ojs.aaai.org/index.php/ICWSM/article/view/14655).
Our corpus is thin per author (median 2 labelled rows), so the prior is weak
here — but it costs nothing and it is exactly what the thesis's "temporal"
channel is groping for.

**Nearest neighbours as a prior, not as extra input.** Khandelwal et al.
(2020) showed that interpolating a model with a k-nearest-neighbour estimate
over a datastore helps without retraining [[8]](https://arxiv.org/abs/1911.00172).
For classification the analogue is the neighbourhood label rate as a feature —
which keeps the thesis's "expectation from similar examples" idea while
removing the 6–20 encoder passes and the noise of individual exemplar texts.

**Disagreement is signal; train on it.** Uma et al. (2021) survey the case
against collapsing multiple judgements into one gold label
[[9]](https://jair.org/index.php/jair/article/view/12752). Davani, Díaz and
Prabhakaran (2022) show that a multi-task model with one head per annotator
over a shared encoder matches or beats majority-label training on seven
subjective tasks and yields usable uncertainty
[[10]](https://aclanthology.org/2022.tacl-1.6/); Rodrigues and Pereira (2018)
give the general "crowd layer" that learns each annotator's reliability
end-to-end [[11]](https://ojs.aaai.org/index.php/AAAI/article/view/11506).
Lukasik et al. (2020) show label smoothing is competitive with loss correction
under label noise [[12]](https://proceedings.mlr.press/v119/lukasik20a.html) —
the annotator vote share is a *data-driven* smoothing target. Sarcasm in
particular is perceived differently by readers and authors (Oprea and Magdy
2020 [[13]](https://aclanthology.org/2020.acl-main.118/); SemEval-2022
iSarcasmEval [[14]](https://aclanthology.org/2022.semeval-1.111/)), so a
third-party (here: LLM) label is inherently a distribution, not a point.

**LLM labels are a legitimate but validated-per-task training signal.**
Wang et al. (2021) show GPT-3 labels train downstream models at a fraction of
human cost [[15]](https://aclanthology.org/2021.findings-emnlp.354/); Pangakis,
Wolken and Fasching (2023) show LLM annotation quality varies by task and must
be validated against humans [[16]](https://arxiv.org/abs/2306.00176); Zhou
(2023) finds current LLMs mediocre at sarcasm specifically
[[17]](https://arxiv.org/abs/2312.03706). Hsieh et al. (2023) go further and
distil the LLM's *rationales* into a small model, beating plain fine-tuning
with less data [[18]](https://aclanthology.org/2023.findings-acl.507/) — every
row here carries three annotator rationales and an adjudicator rationale that
nobody has used yet.

**Encoders.** XLM-R (Conneau et al. 2020) is the committed choice
[[19]](https://aclanthology.org/2020.acl-main.747/); mDeBERTa-v3-base reports
+3.6 XNLI over XLM-R base at the same size
[[20]](https://arxiv.org/abs/2111.09543); RoBERTa-Tagalog (Cruz and Cheng
2022) is the Tagalog-native alternative with +4.5 accuracy over prior Filipino
models on Filipino benchmarks [[21]](https://aclanthology.org/2022.lrec-1.703/),
though its English side is weaker and 45% of this corpus is Taglish. The
TweetTaglish resource (Herrera et al. 2022) is the closest published
code-switching dataset for the language pair
[[22]](https://aclanthology.org/2022.lrec-1.225/); no published Taglish
sarcasm corpus was found, which makes this dataset itself a contribution.
Majumder et al. (2019) show sentiment and sarcasm help each other in a
multi-task setting [[23]](https://arxiv.org/abs/1901.08014) — the pipeline's
`aux_polarity_shift` flag is that idea and is still untested at scale.

---

## 3 · Proposals, ordered by expected value on *this* data

| # | change | replaces | cost | why it should work here |
|---|---|---|---|---|
| P1 | **Early fusion**: one encoder input `POST … PARENT … COMMENT … REPLY …` rendered in the annotators' order, per-segment budgets, target never truncated | conv channel (separate item encodings + attention pooling + gate) | *cheaper* than the full model (one pass of ≤512 tokens vs ~10 passes) | the label was produced from exactly this rendering; token-level attention between target and context is what [1–4] rely on |
| P2 | **kNN prior features**: kNN-25/100 sarcasm rate + local density over training rows (same thread excluded), as 3 scalars into the classifier | retrieval channel (6–20 exemplar encoder passes) | seconds per fold | keeps the "expectation from similar examples" idea, drops the noisy texts and the gate that over-trusts them |
| P3 | **Author prior features**: leave-one-out author sarcasm rate and log-count from training rows; log-count of unlabelled prior posts (48 h / unbounded) | temporal channel (up to 5 encoder passes) | seconds | CASCADE-style user signal at zero encoder cost; honest about its 52% coverage |
| P4 | **Multi-annotator heads** (Davani et al.): one 2-way head per annotator model on the shared representation, summed CE, masked where an annotator failed | single collapsed label | free | uses the only per-row reliability signal; the main head still predicts the shipped label |
| P5 | **Vote-share soft labels**: target = ½ shipped label + ½ annotator vote share | hard label | free | data-driven smoothing; 2-1 rows stop being trained as if certain |
| P6 | Encoder rows: xlm-roberta-large, mdeberta-v3-base, roberta-tagalog-base under P1 | xlm-roberta-base | 1–4× | capacity and language fit; report as rows, keep XLM-R base as the thesis architecture |
| P7 | (not run) **Rationale distillation**: train the student to generate or score the annotators' rationales (Hsieh et al.) | — | large | the 45,000 rationales encode *why* each label was given; the cue heads (+0.016 in stage E) were a first, crude version of this |
| P8 | (not run) **Prompt-faithful rendering**: mirror the uyam `sarc-v2` prompt's context block verbatim (field names, order, which items) | assumed order | trivial once the prompt is available | the student should see what the teachers saw; `block_order` / `ROLE_PREFIX` in the notebook are the switches |

What is deliberately *not* proposed: more retrieval exemplars, a larger
temporal k, threshold or learning-rate tuning — all measured, all flat.

---

## 4 · How the exploration notebook tests P1–P6

`leische_explore_context.ipynb` executes the committed pipeline's §1–§8b in
place (same data, same frozen folds, same `LABEL_MAPS`, same metrics), then:

- **Inputs** — `BlockEncoder` assembles token ids per segment with per-segment
  budgets (post 128, parent 96 × 2, target 192, reply 64 × 3) joined by the
  encoder's separator; on overflow the longest non-target segment is trimmed
  first. `scalar_features(fold)` builds the seven priors from training rows
  only, leave-one-out for training rows, standardised on the training fold.
  `annotator_votes(row)` returns the three votes and their share.
- **Model** — `FusedModel`: backbone → mean pooling → [+ 32-d projection of the
  priors] → MLP; optional per-annotator heads. bf16 autocast, length-bucketed
  batches, early stopping on validation F1@0.5 with best-weight restore, the
  pipeline's optimiser/scheduler/class-weight code reused as is.
- **Protocol** — fold 0, seed 13, ten variants (`x0` re-runs the target-only
  baseline inside this loop so the comparison is like-for-like); seeds 42 and
  7 for the two base-encoder variants with the best **validation** F1 plus
  `x0`; a 5-fold confirmation of the best variant against `x0`, with the same
  paired bootstrap the pipeline uses. Outputs under `results/explore/`, finished
  variants reload.

---

## 5 · Evidence

Two passes of `leische_explore_context.ipynb` on 2026-09-17, 3.4 GPU-hours
total (logs `results/logs/explore-fold0*.log`; artifacts `results/explore/`,
37 runs). bf16 autocast and length-bucketed batches; the committed pipeline's
optimiser, scheduler, class weights, folds and metrics; `x0` re-runs the
target-only baseline inside this loop so every comparison is like-for-like
(x0 on fold 0 / seed 13 scores 0.367 F1 / 0.328 AUPRC here vs 0.369 / 0.328 in
the committed pipeline). **Agreement with the LLM-ensemble labels.**

### 5.1 Fold 0, seed 13 — the grid (`results/explore/grid-fold0-seed13.csv`)

| run | what changes vs x0 | F1 @0.5 | F1 @val-thr | AUPRC | AUROC | ECE | min |
|---|---|---|---|---|---|---|---|
| x0 target-only | — (XLM-R base, 192 tokens) | 0.367 | 0.372 | 0.328 | 0.733 | 0.101 | 3.6 |
| x1 early fusion | P1: one rendered block, post→parents→target→replies | 0.362 | 0.364 | 0.322 | **0.795** | 0.093 | 5.8 |
| x2 + priors | P1 + P2/P3 scalar priors | 0.359 | 0.367 | 0.287 | 0.763 | 0.094 | 5.0 |
| x3 + annotator heads | P1 + P4 | 0.349 | 0.366 | 0.319 | 0.778 | 0.089 | 5.8 |
| x4 + soft votes | P1 + P5 | 0.373 | 0.378 | 0.288 | 0.769 | 0.134 | 4.3 |
| **x5 + priors + heads + soft** | P1 + P2–P5 | **0.392** | 0.392 | 0.338 | 0.811 | 0.089 | 6.5 |
| x6 target-first order | P1 with the target rendered first | 0.362 | 0.363 | 0.273 | 0.765 | 0.140 | 4.4 |
| x7 xlm-roberta-large | P1 on the large encoder (lr 1e-5, batch 8×4) | 0.401 | 0.403 | 0.347 | 0.818 | 0.103 | 15.1 |
| x8 mdeberta-v3-base | P1 on mDeBERTa | 0.385 | 0.365 | **0.363** | 0.815 | 0.183 | 9.3 |
| x9 roberta-tagalog-base | P1 on RoBERTa-Tagalog | 0.337 | 0.325 | 0.265 | 0.750 | 0.159 | 3.6 |
| **x10 + heads + soft** | P1 + P4 + P5 (no priors) | **0.406** | **0.430** | 0.341 | 0.815 | **0.073** | 6.6 |
| x11 + priors + soft | P1 + P2/P3 + P5 (no heads) | 0.347 | 0.320 | 0.287 | 0.774 | 0.111 | 5.0 |
| x12 + priors + heads | P1 + P2/P3 + P4 (no soft) | 0.401 | **0.443** | **0.374** | 0.807 | 0.076 | 5.7 |
| **x13 large + priors + heads + soft** | x5's recipe on xlm-roberta-large | **0.438** | 0.436 | **0.398** | **0.842** | **0.065** | 15.4 |

### 5.2 Seeds on fold 0 (`results/explore/grid-fold0-seeds.csv`)

| run | seeds | F1 @0.5 | AUPRC | paired vs x0 (same seeds) |
|---|---|---|---|---|
| x0 target-only | 13 / 42 / 7 | 0.358 ± 0.008 | 0.318 ± 0.012 | — |
| x4 + soft votes | 13 / 42 / 7 | 0.376 ± 0.021 | 0.302 ± 0.033 | ΔF1 +0.018 ± 0.024 (3/3 wins), ΔAUPRC −0.016 |
| x5 + priors + heads + soft | 13 / 42 / 7 | 0.394 ± 0.004 | 0.308 ± 0.028 | ΔF1 +0.036 ± 0.010 (3/3), ΔAUPRC −0.011 |
| x10 + heads + soft | 13 / 42 / 7 | 0.393 ± 0.016 | 0.328 ± 0.022 | **ΔF1 +0.035 ± 0.012 (3/3), ΔAUPRC +0.010** |

### 5.3 Five folds, seed 13 — the number that compares with the thesis matrix

| run | F1 @0.5 | F1 @val-thr | P / R @0.5 | AUPRC | AUROC | ECE | paired ΔF1 vs x0 | bootstrap (15,000 preds) |
|---|---|---|---|---|---|---|---|---|
| x0 target-only (this loop) | 0.369 ± 0.016 | 0.364 | 0.31 / 0.46 | 0.319 ± 0.027 | 0.777 | 0.114 | — | — |
| committed baseline (RESULTS_LOG) | 0.353 ± 0.034 (seed 13), 0.368 ± 0.026 (3 seeds) | | 0.31 / 0.49 | 0.316 | 0.786 | 0.148 | | |
| committed full model (RESULTS_LOG) | 0.355 ± 0.024 (3 seeds) | | 0.33 / 0.42 | 0.312 | 0.778 | 0.115 | | |
| **x5 + priors + heads + soft** | **0.403 ± 0.015** | 0.405 | 0.35 / 0.48 | **0.368 ± 0.019** | **0.826** | 0.087 | **+0.034 ± 0.025, 5/5** | +0.034, CI [+0.015, +0.054], p < 0.001 |
| **x10 + heads + soft** | **0.405 ± 0.010** | **0.410** | 0.41 / 0.41 | **0.376 ± 0.027** | **0.826** | **0.069** | **+0.037 ± 0.017, 5/5** | +0.037, CI [+0.016, +0.059], p = 0.002 |

Per-fold ΔF1 for x10: +0.040 / +0.010 / +0.043 / +0.035 / +0.056; for x5:
+0.026 / +0.020 / +0.019 / +0.029 / +0.077. Both variants beat the target-only
baseline on every fold and on every metric, and they do it in 6–7 minutes per
fold at 8.6 GB — cheaper than the committed full model (23 min, 13.3 GB).

### 5.4 Where the gain comes from (x10 vs x0, 15,000 pooled test rows)

| slice | n (pos) | F1 x0 → x10 | AUPRC x0 → x10 |
|---|---|---|---|
| 3-0 annotator votes | 10,454 (890) | 0.467 → **0.550** | 0.443 → **0.580** |
| 2-1 annotator votes | 4,544 (711) | 0.272 → 0.265 | 0.201 → 0.218 |
| resolved by majority | 6,759 (630) | 0.496 → **0.582** | 0.482 → **0.641** |
| resolved by unanimous | 3,355 (218) | 0.415 → **0.492** | 0.394 → 0.515 |
| resolved by adjudicator | 4,886 (753) | 0.276 → 0.273 | 0.207 → 0.223 |
| English | 4,171 (460) | 0.338 → 0.346 | 0.282 → 0.334 |
| Tagalog | 4,078 (426) | 0.255 → 0.286 | 0.177 → 0.213 |
| Taglish | 6,751 (715) | 0.503 → **0.544** | 0.453 → **0.577** |
| post only, no turns | 7,218 (813) | 0.376 → 0.412 | 0.308 → 0.370 |
| one thread turn | 4,223 (414) | 0.351 → 0.406 | 0.293 → 0.371 |
| two or more turns | 3,559 (374) | 0.370 → 0.388 | 0.307 → 0.370 |

x10 predicts 1,602 positives to x0's 2,397 (1,601 true); the two agree on 87%
of decisions. Scored against the annotator-majority label instead, x10 is
level with x0 (0.456 vs 0.459): it has become a better model of the *shipped*
(adjudicated) label without drifting toward the majority vote.

### 5.5 What this says

1. **The context model can beat the baseline — by changing how context and
   labels enter, not by adding channels.** Early fusion alone (x1) buys
   ranking quality (AUROC 0.73 → 0.80) but no F1; the annotator heads alone
   (x3) buy nothing; soft votes alone (x4) buy F1 at the price of AUPRC. Put
   two of them together (x10, x12, x5) and every metric moves: F1 +0.035, AUPRC
   +0.05, AUROC +0.05, ECE halved, on five folds and three seeds. The
   attribution rows (x10–x12) say the **annotator heads** are the necessary
   ingredient and the priors are dispensable (x11, heads removed, falls to
   0.347).
2. **The gain is concentrated on the rows the annotators agreed on** (3-0
   votes: +0.08 F1, +0.14 AUPRC) and absent on contested rows (2-1: flat).
   Multi-annotator supervision lets the encoder learn the consistent part of
   the label and stop fitting the contested part as if it were certain — which
   is exactly what Davani et al. predict. It also means the ceiling is still the
   label: the 2-1 rows, 44% of the positives, are as unlearnable as before.
3. **The gain is uniform across context availability** (+0.03–0.05 whether a
   row has no thread turns, one, or several). So the early-fusion block's value
   is mostly the *post* (present on nearly every row) and the rendering
   (target-first, x6, is worse than the annotators' order), not the parent and
   reply turns — consistent with 48% of rows having neither.
4. **Retrieval and author priors do not help** in any form tried (x2, x11),
   confirming the stage-D finding from a different direction: the neighbourhood
   and author signals are real but too weak (AUPRC 0.15 as priors) to add to a
   fine-tuned encoder.
5. **Capacity is a second, independent lever.** The large encoder adds +0.03–0.04
   F1 on top of the recipe (x7 0.401 → x13 0.438 on fold 0, AUPRC 0.398, AUROC
   0.842, ECE 0.065) at 15 min per fold; mDeBERTa-base gives the best base-size
   AUPRC (0.363) but is badly calibrated out of the box; RoBERTa-Tagalog is
   worse everywhere (the corpus is 45% Taglish, 28% English).
6. **Caveats.** Single seed for most grid rows (seed spread on fold 0 is
   ±0.02 F1, as x4 and x10 show); the 5-fold confirmations are one seed each;
   the block order is an assumption until the uyam prompt is mirrored (P8);
   the gain is agreement with an LLM ensemble, and the multi-annotator recipe
   exists *because* the labels came from three readers — with a single human
   label it would have nothing to train on.

---

## 6 · What to change in the thesis pipeline, in order

1. **Add condition 9, "early fusion + multi-annotator heads + vote-share
   labels", as a ninth row of the RQ2 matrix** rather than replacing anything:
   `Tokenize` already supports per-segment budgets; the block builder,
   `FusedModel` and the loss in the exploration notebook are ~150 lines and
   plug into `run_cv` (the model returns the same `logits / features /
   target_emb / gates=None` dict). Run it 5 folds × 3 seeds so it sits in the
   same table as conditions 1–8.
2. **Report the annotator-vote slices as a first-class table.** The 3-0 / 2-1
   split explains more variance than any architecture decision in the thesis;
   the manuscript's results chapter should lead with it.
3. **Mirror the uyam `sarc-v2` prompt in the block renderer** (`block_order`,
   `ROLE_PREFIX`, which items and how many) and re-run x10 — the one untested
   lever with a clear mechanism.
4. **Keep the retrieval channel as a negative result.** Three encoders, three
   k values, two fusion styles and two prior forms all say the same thing;
   that is a finding, and it saves 60% of the compute.
5. **Offer xlm-roberta-large as the "capacity" row** (x13: +0.07 F1 over the
   baseline on fold 0) with the base model kept as the thesis architecture.
6. **Next, with more compute:** 3 seeds × 5 folds for x10 and x13; the
   rationale-distillation row (P7) — the 45,000 annotator rationales are the
   largest unused supervision in the dataset; and RQ3 re-run on the x10
   checkpoint, since its flag is far better calibrated (ECE 0.069 vs 0.11).

---

## References

1. Ghosh, D., Fabbri, A. R., & Muresan, S. (2018). Sarcasm Analysis Using Conversation Context. *Computational Linguistics*, 44(4), 755–792. https://aclanthology.org/J18-4009/
2. Ghosh, D., Vajpayee, A., & Muresan, S. (2020). A Report on the 2020 Sarcasm Detection Shared Task. *Proceedings of the Second Workshop on Figurative Language Processing (FigLang 2020)*, ACL. https://aclanthology.org/2020.figlang-1.1/
3. Dong, X., Li, C., & Choi, J. D. (2020). Transformer-based Context-aware Sarcasm Detection in Conversation Threads from Social Media. *FigLang 2020*. https://aclanthology.org/2020.figlang-1.38/
4. Baruah, A., Das, K., Barbhuiya, F., & Dey, K. (2020). Context-Aware Sarcasm Detection Using BERT. *FigLang 2020*, 83–87. https://aclanthology.org/2020.figlang-1.12/
5. Hazarika, D., Poria, S., Gorantla, S., Cambria, E., Zimmermann, R., & Mihalcea, R. (2018). CASCADE: Contextual Sarcasm Detection in Online Discussion Forums. *Proceedings of COLING 2018*. https://aclanthology.org/C18-1156/
6. Amir, S., Wallace, B. C., Lyu, H., Carvalho, P., & Silva, M. J. (2016). Modelling Context with User Embeddings for Sarcasm Detection in Social Media. *Proceedings of CoNLL 2016*, 167–177. https://aclanthology.org/K16-1017/
7. Bamman, D., & Smith, N. A. (2015). Contextualized Sarcasm Detection on Twitter. *Proceedings of ICWSM 2015*, 574–577. https://doi.org/10.1609/icwsm.v9i1.14655
8. Khandelwal, U., Levy, O., Jurafsky, D., Zettlemoyer, L., & Lewis, M. (2020). Generalization through Memorization: Nearest Neighbor Language Models. *ICLR 2020*. https://arxiv.org/abs/1911.00172
9. Uma, A. N., Fornaciari, T., Hovy, D., Paun, S., Plank, B., & Poesio, M. (2021). Learning from Disagreement: A Survey. *Journal of Artificial Intelligence Research*, 72, 1385–1470. https://doi.org/10.1613/jair.1.12752
10. Davani, A. M., Díaz, M., & Prabhakaran, V. (2022). Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations. *Transactions of the ACL*, 10, 92–110. https://aclanthology.org/2022.tacl-1.6/
11. Rodrigues, F., & Pereira, F. C. (2018). Deep Learning from Crowds. *Proceedings of AAAI-18*. https://ojs.aaai.org/index.php/AAAI/article/view/11506
12. Lukasik, M., Bhojanapalli, S., Menon, A. K., & Kumar, S. (2020). Does Label Smoothing Mitigate Label Noise? *Proceedings of ICML 2020*, PMLR 119, 6448–6458. https://proceedings.mlr.press/v119/lukasik20a.html
13. Oprea, S., & Magdy, W. (2020). iSarcasm: A Dataset of Intended Sarcasm. *Proceedings of ACL 2020*. https://aclanthology.org/2020.acl-main.118/
14. Abu Farha, I., Oprea, S. V., Wilson, S., & Magdy, W. (2022). SemEval-2022 Task 6: iSarcasmEval, Intended Sarcasm Detection in English and Arabic. *Proceedings of SemEval-2022*. https://aclanthology.org/2022.semeval-1.111/
15. Wang, S., Liu, Y., Xu, Y., Zhu, C., & Zeng, M. (2021). Want To Reduce Labeling Cost? GPT-3 Can Help. *Findings of EMNLP 2021*. https://aclanthology.org/2021.findings-emnlp.354/
16. Pangakis, N., Wolken, S., & Fasching, N. (2023). Automated Annotation with Generative AI Requires Validation. *arXiv:2306.00176*. https://arxiv.org/abs/2306.00176
17. Zhou, J. (2023). An Evaluation of State-of-the-Art Large Language Models for Sarcasm Detection. *arXiv:2312.03706*. https://arxiv.org/abs/2312.03706
18. Hsieh, C.-Y., Li, C.-L., Yeh, C.-K., Nakhost, H., Fujii, Y., Ratner, A., Krishna, R., Lee, C.-Y., & Pfister, T. (2023). Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes. *Findings of ACL 2023*. https://aclanthology.org/2023.findings-acl.507/
19. Conneau, A., Khandelwal, K., Goyal, N., Chaudhary, V., Wenzek, G., Guzmán, F., Grave, E., Ott, M., Zettlemoyer, L., & Stoyanov, V. (2020). Unsupervised Cross-lingual Representation Learning at Scale. *Proceedings of ACL 2020*. https://aclanthology.org/2020.acl-main.747/
20. He, P., Gao, J., & Chen, W. (2021). DeBERTaV3: Improving DeBERTa using ELECTRA-Style Pre-Training with Gradient-Disentangled Embedding Sharing. *arXiv:2111.09543*. https://arxiv.org/abs/2111.09543
21. Cruz, J. C. B., & Cheng, C. (2022). Improving Large-scale Language Models and Resources for Filipino. *Proceedings of LREC 2022*. https://aclanthology.org/2022.lrec-1.703/
22. Herrera, M., Aich, A., & Parde, N. (2022). TweetTaglish: A Dataset for Investigating Tagalog-English Code-Switching. *Proceedings of LREC 2022*, 2090–2097. https://aclanthology.org/2022.lrec-1.225/
23. Majumder, N., Poria, S., Peng, H., Chhaya, N., Cambria, E., & Gelbukh, A. (2019). Sentiment and Sarcasm Classification with Multitask Learning. *IEEE Intelligent Systems*, 34(3), 38–43. https://arxiv.org/abs/1901.08014
