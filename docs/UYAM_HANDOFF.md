# uyam → leische handoff

What the model repository needs from the data repository, ordered by how much
it blocks.

> **Most of this list is closed.** The `sarc-v2` export at
> `uyam_commit = 039c3aa8…` (15,000 rows / 1,601 positives) shipped the
> annotator context snapshot, adjudication, cue labels, LID, hashed authors,
> text-only filtering and full model provenance. What follows is what is left,
> plus a new finding about the adjudicator that outranks everything else.

Readiness gate on the current export:

```
TRAIN-READY (blocks run_cv)
  [PASS] full corpus under sarc-v2+        prompt_version=sarc-v2, 15000 rows
  [PASS] ≥400 sarcastic positives          1601 positives (10.7% base rate)
  [PASS] every language×sarcastic cell ≥10 min cell 426
  [PASS] conversational context is the annotator snapshot
  [PASS] fold file frozen                  folds-v1-a419a4bc95.json

CLAIM-READY (stamps results, never blocks)
  [WARN] gold subset ≥250 items            31 labelled                     ← H1
  [WARN] human-vs-ensemble sarcasm κ ≥0.60 κ=-0.1481 on n=31               ← H2
  [WARN] inter-annotator sarcasm κ ≥0.60   Fleiss κ=0.3761
  [PASS] automatic LID validated           62.5% agreement on n=15000
```

Training is unblocked. What is left decides what the numbers *mean*.

---

## Blocking the claims

### H1 — Gold subset is at 31 of ~300

Evaluation-only human labels. At n = 31 (6 human positives) the κ confidence
interval spans most of the possible range, so neither a good nor a bad number
would be trustworthy yet.

**Ask:** label ~300 items, stratified over `language × sarcastic`, deliberately
over-sampling 2-1 splits — that is where the disagreement lives and where each
label buys the most information. Keep writing them to `human_gold` and the
resulting Cohen's κ to `dataset_card.json`.

**Hard rule:** gold rows are evaluation-only, forever. The pipeline never trains
on them.

### H2 — The adjudicator is removing most of the positive class

**This is the highest-value fix available, and it is a prompt decision, not a
modelling one.** Full analysis in
[ANNOTATION_PROVENANCE.md §3](ANNOTATION_PROVENANCE.md); the short version:

- `qwen3-32b` adjudicates 4,886 rows and **overrides the annotator majority on
  1,712 (35%)**, including 52 reversals of a **unanimous 3-0** vote (43 of them
  sarcastic → not-sarcastic).
- Net it removes **1,146 positives**: 2,747 (18.3%) → 1,601 (10.7%).
- On the 31 gold rows, the shipped adjudicated label scores Cohen's
  κ = **−0.148** against the human. The **annotator majority** on the same rows
  scores **+0.450**.

The adjudicator applies a strict "must invert or mask the literal meaning"
criterion and rejects mock-praise, hyperbolic banter and self-deprecation,
which the human annotator counts as sarcasm.

**Ask — pick one:**

1. Forbid the adjudicator from overriding a unanimous 3-0 annotator vote
   (52 rows), or
2. widen its sarcasm definition to match the human annotator's, or
3. ship **both** label sets (`labels.sarcastic` and
   `labels.sarcastic_annotator_majority`) so the model repo can report the
   robustness row without a re-export.

Option 3 is the cheapest and is strictly more informative. Whichever is chosen,
re-export and the fold file re-freezes automatically on the new `uyam_commit`.

### H3 — Write the annotation guideline down

The disagreements are a definition gap, not noise. The thesis names four cues
(polarity inversion, rhetorical intent, contextual incongruity, hyperbole) but
never says whether **hyperbole alone**, with no polarity inversion, is sarcasm.
The adjudicator says no; the human says yes. That single sentence would resolve
most of the 9 gold disagreements.

Worth noting from the corpus: `hyperbole` fires on 825 rows but only 194 of
them are labelled sarcastic, while `polarity_inversion` fires on 641 rows of
which **all 641** are sarcastic. The cue set is already telling you where the
boundary is being drawn.

---

## Still open, lower priority

### H4 — Keyword oversampling never ran

Thesis §3.1 promises "keyword- and heuristic-based oversampling of potentially
sarcastic threads". `sampling_strategy` is `natural` on all 15,000 rows, so the
§7.2 test-fold hygiene path runs but has nothing to separate.

**Ask:** run it, or drop the claim from §3.1. With the adjudicator question
(H2) open, oversampling would be the second-best lever on positive-class size.

### H5 — `needs_human` has never fired

The stage exists in the contract and is `False` on all 15,000 rows. Low-model-
confidence rows are exactly where a human pass buys the most.

**Ask:** either wire it up with a confidence threshold, or remove it from the
contract so it does not read as a stage that ran and found nothing.

### H6 — Collection window is ~5 weeks

The corpus spans 2026-08-01 → 2026-09-04; the exported rows 2026-08-01 →
2026-09-03. Thesis §1.4 scopes the study to "at least three months".

This is the direct cause of the thin temporal channel — **36.7%** of rows have
≥1 same-author post within the 48 h window, 11.9% have ≥3, 5.5% have the full
5. Two thirds of the dataset presents an empty temporal channel, so RQ2's
temporal condition currently measures data availability rather than whether
temporal context helps.

**Ask:** back-fill `corpus-*.jsonl` with more history **per already-collected
author**. That thickens the temporal channel for the existing labelled rows
without any new annotation — the cheapest single improvement available to RQ2.

### H7 — Language is the weakest of the four labels

Inter-annotator Fleiss' κ = 0.439, fastText LID agrees with the ensemble on
only 62.5% of rows, and language is the **stratification key** for every fold
and the disaggregation axis for every research question.

**Ask:** consider resolving `language` from the LID ratios (which are
deterministic) rather than from a model vote, or adjudicate it separately.
`aux.lid.en_ratio` / `tl_ratio` are already exported, so the thesis §3.2.1
rule (≥90% → monolingual, else Taglish) can be applied directly.

---

## Closed by the sarc-v2 export

| was | item | now |
|---|---|---|
| H2 | conversational context snapshot | `context.source == "annotator_snapshot"` on all rows |
| H3 | sealion failures | all three annotators on ~15,000 rows |
| H4 | unadjudicated splits | adjudication ran; `skipped_unresolved = 0` |
| H5 | cue labels | `labels.cues` populated, plus per-annotator cues |
| H6 | LID output | `aux.lid` with ratios + `auto_label`; agreement reported |
| H7 | plaintext author handles | `author_hash` is a real one-way hash |
| H10 | image-only posts | filtered; `is_text_only` true on every row |
| H11 | export identity | `uyam_commit`, `created_at`, prompt version stamped |
| H12 | per-annotator confidence | present on every annotator record |
| — | joint vs per-label resolution | `reliability.per_label` added |

`tools/build_uyam_export.py` — the adapter that derived the contract from the
older CSV export — is superseded by this export and kept only for reference.
