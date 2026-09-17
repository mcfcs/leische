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

*(filled in from the notebook run — see the tables in `results/explore/`)*
