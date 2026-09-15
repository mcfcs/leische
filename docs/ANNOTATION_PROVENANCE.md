# Annotation provenance — how the labels were actually produced

**Every label in this corpus was produced by an LLM ensemble. No human
annotated the training data.** The thesis manuscript (§3.3) describes a single
human annotator with a two-pass intra-annotator κ; that is not what happened,
and the manuscript text is being revised. This document is the account of what
*did* happen — for the methodology rewrite, and for anyone assessing the
pipeline before training.

| | manuscript §3.3 | reality |
|---|---|---|
| annotator | one human, native Tagalog + English | three LLMs + an LLM adjudicator |
| reliability statistic | Cohen's κ between two passes, ≥2 weeks apart | Fleiss' κ across three models |
| validation | — | 31 human-labelled gold items (target ~300) |
| labels per item | sarcasm, language, sentiment | + literal *and* intended sentiment, + 4 cue booleans |

Dataset: `data/dataset-v1.jsonl`, `prompt_version = sarc-v2`,
`uyam_commit = 039c3aa8…`, **15,000 rows / 1,601 sarcastic positives (10.7%)**.

---

## 1 · The annotation pipeline

Four stages, in order. Each row records which stage decided it
(`reliability.resolved_by`, and `reliability.per_label` for each label
separately).

1. **Three independent annotators** — `gemma3`, `qwen3`, `sealion` — each see
   the target *and* its conversational context block, and return sarcasm,
   language, literal sentiment, intended sentiment, four cue booleans, a
   confidence, and a free-text rationale.
2. **Unanimity-or-escalate on sarcasm**; majority on the three-class labels.
   A row where all three agree is `resolved_by = "unanimous"` (3,355 rows) or
   `"majority"` (6,759).
3. **Blind adjudication** by a larger model, `qwen3-32b`, on rows that
   escalated — 4,886 rows (33%). **Its call is final on every row it sees.**
4. **Human final call** on low-confidence items — `reliability.needs_human`.
   This stage has **never fired**: it is `False` on all 15,000 rows.

### Model provenance

Recorded per row in `reliability.annotators[].model_digest` and in
`dataset_card.json → annotator_provenance`. Temperature-0 decoding is near- but
not bit-reproducible across Ollama versions and GPUs, so the digests plus
`prompt_version` are the reproducibility contract, not the decoding settings.

| model | role | digest (short) | rows | Ollama |
|---|---|---|---|---|
| `gemma3` | annotator | `a418f583…` | 15,000 | 0.32.6 / 0.34.0 |
| `qwen3` | annotator | `500a1f06…` | 14,998 | 0.32.11 / 0.33.3 |
| `sealion` | annotator | `79922b43…` | 15,000 | 0.32.11 / 0.33.3 |
| `qwen3-32b` | **adjudicator** | `030ee887…` | 4,886 | 0.34.0 |

Two rows lost one annotator (`qwen3`) to a call failure; they carry
`n_annotators = 2`.

---

## 2 · Agreement between the annotators

Fleiss' κ / Krippendorff's α across the three annotator models, n = 15,000:

| label | Fleiss' κ | Krippendorff's α | reading |
|---|---|---|---|
| **`sarcastic`** | **0.376** | 0.376 | fair |
| `language` | 0.439 | 0.439 | moderate |
| `literal_sentiment` | 0.637 | 0.637 | substantial |
| `intended_sentiment` | 0.579 | 0.579 | moderate |

κ = 0.376 on the primary target label is low, and it is the honest headline
number for annotation reliability. It is not unusual for sarcasm — the
manuscript's own §2.1 cites third-party annotators recovering author-intended
sarcasm at F = 0.616 — but it must be reported, not buried.

---

## 3 · The gold subset — and what it found

31 items carry a human label in `human_gold` (target ~300). They are
evaluation-only and are never trained on.

### Cohen's κ, human vs. the shipped ensemble label

| label | κ |
|---|---|
| **`sarcastic`** | **−0.148** |
| `language` | 0.466 |
| `literal_sentiment` | 0.248 |
| `intended_sentiment` | 0.481 |

A *negative* κ means the two raters agree less than chance would predict. The
confusion matrix shows why:

```
               ensemble=False   ensemble=True
human=False          22               3
human=True            6               0
```

**Zero overlap on the positive class.** The ensemble found 3 sarcastic items;
the human said none of those 3 were. The human found 6; the ensemble found none
of them. Raw agreement is 71%, while always predicting "not sarcastic" would
score 80.6% on this sample.

### The cause: the adjudicator, not the annotators

Re-scoring the same 31 rows against the **annotator majority** — i.e. what the
label would have been without stage 3 — gives a completely different picture:

| labelling rule | gold confusion | raw agreement | Cohen's κ |
|---|---|---|---|
| shipped label (adjudicated) | 22 / 3 / 6 / 0 | 71.0% | **−0.148** |
| annotator majority (no adjudication) | 21 / 4 / 2 / 4 | 80.6% | **+0.450** |

The annotator majority recovers 4 of the human's 6 positives. The adjudicated
label recovers 0 of 6.

Measured across the whole corpus, the adjudicator is systematically
conservative about sarcasm:

- it sees 4,886 rows and **overrides the annotator majority on 1,712 of them (35%)**;
- 52 of those overrides reverse a **unanimous 3-0** annotator vote — 43 from
  sarcastic to not-sarcastic, 9 the other way;
- net, it removes **1,146 positives**. Without stage 3 the corpus would carry
  2,747 positives (18.3% base rate) instead of 1,601 (10.7%).

Reading the rationales makes the mechanism plain. The adjudicator applies a
strict criterion — *"no evidence of saying one thing while meaning another"*,
*"does not invert or mask the literal meaning"* — and rejects playful
hyperbole, mock-praise and self-deprecation. The human annotator counts those
as sarcasm. Two examples:

- *"sosyal na mga lizard ngayon, human food na mostly ang gustong makain"* —
  mock-praise. Human: sarcastic. Adjudicator: "humorous observation… does not
  invert or mask the literal meaning."
- *"sorry we're just peasants here, we wouldn't waste that much for an event"* —
  human: sarcastic. Adjudicator: "self-deprecating remark… no indication of
  sarcasm or irony."

And at least one genuine ensemble error runs the other way: on *"He's still
here sa PH? Broke? Jobless?"* all three annotators said not-sarcastic and the
adjudicator overrode them to sarcastic; the human agreed with the annotators.

**This is a definitional gap, not random noise** — which means it is fixable in
the guideline rather than in the model. Regenerate the full report any time
with:

```bash
uv run python tools/gold_disagreements.py --out disagreements.md
```

> The report embeds raw post text, so it is written on demand and not committed.

n = 31 with 6 human positives is far too small to be a verdict, and the gold
rows over-sample hard cases (48% adjudicated vs 33% of the corpus), which
biases κ downward. The *direction* is nonetheless consistent and quantified.

---

## 4 · What this means for the results

The model is trained to reproduce these labels. So:

- **Reported F1 measures agreement with the LLM ensemble, not with human
  judgement.** Until the gold subset validates the labels, every number carries
  that caveat. The pipeline enforces it: `LABEL_AUTHORITY` is stamped into every
  `run.json` and prepended as columns to every `fold_metrics.csv`, and the §10
  gate prints a CLAIM-READY tier that never blocks but always warns.
- **The 10.7% base rate is an artefact of the adjudicator's strictness**, not a
  property of the subreddits. If stage 3 is revised, the base rate moves and
  every previous run becomes incomparable — which is why the frozen fold file is
  keyed on `uyam_commit`.
- **RQ1/RQ2 are less exposed than RQ3.** The ablation compares conditions
  against *the same* labels, so a consistent label bias largely cancels; RQ3
  compares predicted sentiment against `intended_sentiment` as ground truth, and
  that label has κ = 0.481 against the human.
- `literal_sentiment` (κ = 0.637 inter-annotator) is the most reliable label in
  the set; `sarcastic` is the least.

---

## 5 · For the manuscript revision

§3.3 must be rewritten. What it needs to say, at minimum:

1. **Annotation was performed by an ensemble of three open-weight LLMs**
   (gemma3, qwen3, sealion) conditioned on the target *and* its conversational
   context, with unanimity-or-escalate resolution and blind adjudication by a
   larger model (qwen3-32b).
2. **Reliability is Fleiss' κ across the three annotators**, reported per label
   (sarcasm 0.376, language 0.439, literal sentiment 0.637, intended sentiment
   0.579) — replacing the intra-annotator Cohen's κ the current text promises.
3. **Validation is a human-labelled gold subset**, evaluation-only, with
   human-vs-ensemble Cohen's κ reported. State n and κ honestly, including the
   adjudicator finding above if the pipeline is not revised first.
4. **The limitation paragraph** should say that labels are model-generated,
   that sarcasm is the least reliable of the four, and that reported performance
   is therefore agreement with an ensemble rather than with human judgement.
5. §3.2.1's promised LID validation is now reportable: fastText agrees with the
   ensemble language label on **62.5%** of 15,000 rows and with the human gold
   on **71.0%** of 31.
6. §3.3's two-pass protocol can be retained honestly if reframed as the *gold
   subset* protocol rather than the corpus protocol.

Nothing here requires claiming the LLM ensemble is as good as a human. It is a
legitimate low-resource annotation strategy provided its reliability is
reported and its labels are validated against a human sample — which is exactly
what the gold subset is for.

---

## 6 · Open actions

Ordered by how much they affect what the results mean.

1. **Grow the gold subset to ~300**, stratified over `language × sarcastic` and
   deliberately including 2-1 splits. At n = 31 the κ confidence interval spans
   most of the possible range.
2. **Decide the adjudicator's remit.** Two concrete options, both defensible:
   (a) forbid it from overriding a unanimous 3-0 annotator vote (52 rows), or
   (b) widen its sarcasm definition to include mock-praise, hyperbolic banter
   and self-deprecation, which is what the human annotator is counting.
   Either way, re-export and re-freeze folds.
3. **Write the annotation guideline down.** The disagreements are a definition
   gap between the adjudicator prompt and the human annotator. The thesis
   already names four cues (polarity inversion, rhetorical intent, contextual
   incongruity, hyperbole) — the guideline should state whether hyperbole
   *alone*, with no polarity inversion, counts as sarcasm. Right now the
   adjudicator says no and the human says yes.
4. **Report the annotator-majority label as a robustness row.** The pipeline can
   train on both label sets; a model that behaves the same under either is
   evidence the finding does not hinge on stage 3.
5. **Wire up `needs_human`.** The stage exists and has never fired; low-model-
   confidence rows are exactly the ones a human pass would most improve.

---

## 7 · What the export got right

For the record, and against the earlier [UYAM_HANDOFF.md](UYAM_HANDOFF.md)
list — this export closed nearly all of it:

- `context.source == "annotator_snapshot"` on all 15,000 rows: the model now
  consumes exactly the context block the annotators saw (was H2, the one
  §11.7 violation).
- Adjudication ran; `skipped_unresolved = 0` (was H4).
- Cue labels, `aux.tx_sentiment`, `aux.lid` all populated (were H5, H6).
- `author_hash` is a real one-way hash (was H7).
- Bot / too-short / link-post / image-only filtering applied, `is_text_only`
  true on every row (was H10).
- `uyam_commit`, `created_at`, per-annotator digests and confidences all
  stamped (were H11, H12).
- Per-label resolution in `reliability.per_label`, so sarcasm's resolution is
  no longer conflated with the joint resolution of all four labels.

Two items remain open from that list: the human gold subset (H1, now the
CLAIM-READY blocker) and keyword oversampling, which still never ran —
`sampling_strategy` is `natural` on all 15,000 rows (H9).
