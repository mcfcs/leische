# Brief for a session in the `uyam` repository — what leische needs from the data side

Paste §0–§5 as the prompt in a session opened in the uyam repo
(`https://github.com/mcfcs/uyam`). It is written the way `docs/FABILE_BRIEF.md`
was written for leische: ground rules, what was learned, what to change in
priority order, how to know it worked, what not to touch.

---

## 0 · Ground rules

- **Never modify or re-export `sarc-v2` in place.** Every change below that
  touches a prompt, the adjudicator, or the label definition bumps
  `PROMPT_VERSION` (to `sarc-v3`) and produces a *new* export with a new
  `uyam_commit`; leische keys its frozen folds on that commit and will re-freeze
  automatically. Old runs must stay comparable to old exports.
- **Do not change the three annotator prompts** unless explicitly asked in §2
  — the 45,000 annotator votes and rationales are the asset; re-annotating
  costs ~3 days of GPU. The runner's content-hash check exists for this reason;
  keep it.
- **Never rebuild conversational context from the corpus dump** for existing
  rows; the persisted `context_json` / `context_text` per item is exactly what
  the annotators saw and the model must train on that.
- Run the existing test suite (`tests/test_annotate_*.py`) before and after;
  add a test for every new export field.
- Commit with plain messages; no attribution trailers.

## 1 · What leische found (why these changes)

Full detail: `leische/docs/RESULTS_LOG.md`, `leische/suggestions/SUMMARY.md`,
`leische/suggestions/IMPROVEMENTS.md`.

- On the shipped (adjudicated) label the best model reaches F1 0.405; the
  three annotator LLMs themselves score 0.49–0.58 against that label and their
  majority vote 0.606. **The label is the ceiling, not the model.**
- Rows where the annotators split 2-1 (4,544 rows, 44% of the positives) are
  unlearnable for every model (F1 ≈ 0.27), and the annotators' own majority
  scores 0.345 on them. Unanimous rows score 0.55–0.97.
- The adjudicator overrides the majority on 1,712 of 4,886 escalated rows and
  removes 1,146 positives; on the 31 gold items it drives human-vs-ensemble
  κ from +0.45 (majority) to −0.15 (shipped). Its rationales apply a strict
  "must invert the literal meaning" rule that rejects mock-praise, hyperbole
  and self-deprecation — which the human annotator counts as sarcasm.
- Training on the majority label gives F1 0.51 and *better* agreement with the
  adjudicated label than training on the adjudicated label itself.
- Temporal context covers only 36.7% of rows within 48 h (53.5% unbounded);
  the channel measures data availability.
- Per-annotator votes turned out to be the single most useful training signal
  (one prediction head per annotator: +0.037 F1). The export already carries
  them; §2 asks to make them first-class and stable.

## 2 · Changes, in priority order

### U1 · Ship both label sets and the vote share (no re-annotation)

Add to `labels`: `sarcastic_majority` (majority of the annotator votes; ties →
the shipped label) and `sarcasm_vote_share` (fraction of annotators voting
sarcastic, 0–1). Keep `labels.sarcastic` as the adjudicated verdict. Add to
`dataset_card.json` the counts for both and Cohen's κ between them. leische
currently derives these itself (`LABEL_MAPS["annotator_majority"]` — 2,747
positives); shipping them makes the robustness row reproducible from the
export alone. Acceptance: leische's §3 validator passes; the derived and
shipped majority labels agree on all 15,000 rows.

### U2 · Adjudicator remit — `sarc-v3` for stage 3 only

Two options; pick one and document it in the guideline (U3):

1. **Forbid overriding a unanimous 3-0 vote** (52 rows) — minimal, keeps the
   current definition.
2. **Widen the adjudicator's sarcasm definition** to what the human annotator
   applies: mock praise, hyperbolic banter and self-deprecation count as
   sarcasm when the intended meaning departs from the literal one, even
   without a strict polarity inversion. Re-run the adjudicator on the 4,886
   escalated rows only.

Either way: bump `PROMPT_VERSION` for the adjudicator prompt, keep the
annotator prompt hash unchanged (verify the runner does not force annotator
re-runs), export as a new dataset version, and report in the card how many
rows changed and the new gold κ. Acceptance: gold-vs-ensemble κ on the 31
items moves toward the +0.45 the majority label already has.

### U3 · Write the annotation guideline down, and grow the gold subset to ~300

The disagreements are a definition gap. Write `docs/GUIDELINE.md` stating,
with examples from the gold disagreements (`leische/tools/gold_disagreements.py`),
whether hyperbole alone, mock praise and self-deprecation are sarcasm. Then
label the gold subset to the configured 300 (`review.gold_size`), stratified
on `language × sarcastic` with 2-1 splits oversampled (`gold_split_oversample`
already exists), and store the human label in `human_gold` with the labeller's
id. Compute κ against **both** label sets and per cue. Acceptance:
`n_gold_items ≥ 250` in the card; leische's claim-ready tier clears its first
warning.

### U4 · Export the rendered context text verbatim

`context.source == "annotator_snapshot"` ships the structured snapshot; also
ship `context.rendered` — the exact `context_text` string the DB persisted
(`=== THREAD CONTEXT === … [PARENT depth=… (OP)] … [REPLY i] …`). leische
re-implemented the rendering from `context.py`; shipping it removes the
re-implementation and makes prompt-faithful students trivially exact.
Acceptance: `context.rendered` present on all rows and equal to what
`build_context` produces for the same snapshot (test).

### U5 · Back-fill author history for existing authors

Two thirds of rows have no same-author post within 48 h. Collect earlier posts
*per already-annotated author* (no new annotation needed), append to
`corpus-*.jsonl`, and report the new coverage in the card
(`temporal_context_coverage`). Target ≥ 60% of rows with ≥ 1 prior post
within 48 h. Acceptance: card coverage numbers; leische's §12b temporal rows
re-run automatically on the new commit.

### U6 · Make `needs_human` fire

`review.adjudicator_confidence_floor = 0.6` exists but `needs_human` is False
on all 15,000 rows. Find out why (is the adjudicator's confidence ever below
the floor? is the flag written?), fix or remove the stage from the contract,
and route the flagged rows into the Streamlit review queue. Acceptance: the
card reports how many rows were queued and how many a human resolved.

### U7 · A deterministic language label

Inter-annotator κ on `language` is 0.44 and fastText agrees with the vote on
only 62.5%. Add `labels.language_lid` from the existing LID ratios with the
thesis §3.2.1 rule (≥ 90% one language → monolingual, else Taglish), keep the
voted label, and report both in the card. leische can then stratify and slice
on either. Acceptance: field present; agreement between the two reported.

### U8 · Decide keyword oversampling

`sampling_strategy` is `natural` on every row while the manuscript §3.1
promises keyword/heuristic oversampling. Either run it (flagging rows
`keyword_oversampled`; leische's §7.2 hygiene path already excludes them from
natural-distribution metrics) or record that the claim is dropped.

## 3 · Export contract additions (summary)

| field | type | source |
|---|---|---|
| `labels.sarcastic_majority` | bool | majority of `reliability.annotators[].sarcastic` |
| `labels.sarcasm_vote_share` | float | share of sarcastic votes |
| `labels.language_lid` | enum | `aux.lid` ratios, 90% rule |
| `context.rendered` | str | persisted `context_text` |
| `human_gold.labeller` | str | reviewer id |
| card: counts, κ (both label sets, per cue), temporal coverage, needs_human counts | | |

Everything else stays as in `docs/dataset-contract-leische.md`.

## 4 · How to know it worked

1. `tests/` green; new tests for U1, U4, U7.
2. `dataset_card.json` carries the new counts and κ values.
3. In leische: drop the export into `data/`, run
   `uv run python tools/run_notebook.py --until 6` — the gate re-freezes folds
   on the new `uyam_commit`, the validator reports zero violations, and the
   printed `LABEL_AUTHORITY` names the new prompt version.
4. For U2: the majority-label robustness row and the adjudicated row converge.

## 5 · What not to do

- Do not re-annotate with the three annotator models unless U2 option 2
  requires it (it should not).
- Do not "fix" mojibake or text in existing rows — labels were conditioned on
  the text as is.
- Do not change fold-relevant fields (`submission_fullname`, `reddit_fullname`).
- Do not put raw post text into any committed report; `gold_disagreements.py`
  writes on demand for this reason.
