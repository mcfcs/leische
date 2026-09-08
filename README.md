# leische

Model repository for the undergraduate thesis **Context-Aware Sarcasm Detection
in Code-Switching Social Media Posts**. The data half (collection + LLM-ensemble
annotation) is the separate [uyam](../uyam) repository; the methodology
reference is [docs/MODEL_PLAN.md](docs/MODEL_PLAN.md).

> **STATUS: the pipeline runs on the real export, but the §10 readiness gate
> still fails on one check.**
> `dataset-v2` (sarc-v2) gives **4,416 rows / 403 sarcastic positives / 679
> threads**. Four of the five gate checks pass; the blocker is that **no human
> gold subset exists**, so nothing validates the LLM-ensemble labels — and
> ensemble agreement on sarcasm is only Fleiss' κ = 0.386. Until a gold subset
> lands, `run_cv` refuses non-smoke runs in code and every number is prefixed
> `SMOKE`.

Read before running:

- **[docs/PIPELINE_SPEC.md](docs/PIPELINE_SPEC.md)** — what the notebook does,
  section by section, with every choice traced to a manuscript clause.
- **[docs/METHODOLOGY_REVIEW.md](docs/METHODOLOGY_REVIEW.md)** — manuscript
  Chapter III vs. this pipeline; the five departures (M1–M5) and why.
- **[docs/UYAM_HANDOFF.md](docs/UYAM_HANDOFF.md)** — what uyam still owes the
  model repo, ordered by how much it blocks.

## Layout — one notebook

The entire pipeline lives in **[leische_pipeline.ipynb](leische_pipeline.ipynb)**:
environment checks → data loading + contract validation + readiness gate → EDA
→ frozen thread-grouped folds → the three context channels with leakage
assertions → the 5-stage XLM-R model behind RQ2 ablation flags → training
harness → the two §10 smoke tests (overfit-16, tiny-settings 5-fold dry run) →
the 8×5 ablation matrix with significance tests → the RQ3 two-stage sentiment
evaluation.

```
leische_pipeline.ipynb        the whole pipeline (all functions inline)
tools/build_uyam_export.py    uyam CSVs → the dataset contract
tools/check_model_contract.py architecture checks (no GPU, no model download)
docs/PIPELINE_SPEC.md         what the notebook does, clause by clause
docs/MODEL_PLAN.md            methodology reference (from uyam)
docs/METHODOLOGY_REVIEW.md    manuscript vs. pipeline; the M1–M5 decisions
docs/UYAM_HANDOFF.md          data-side gaps and what to fix in uyam
results/                      frozen folds + run artifacts (committed)
data/                         the uyam export + derived contract (gitignored)
cache/                        embeddings / checkpoints (gitignored)
```

The `main` branch keeps the alternative layout (importable `src/leische`
package + per-stage notebooks + pytest suite).

## Data

uyam ships two flat CSVs into `data/`:

```
data/annotated-review.csv     6,650 annotated targets
data/uyam_export.csv          20,573-row scraped corpus
```

`tools/build_uyam_export.py` derives the nested dataset contract the notebook
consumes, and never fabricates a field uyam does not collect:

```bash
uv run python tools/build_uyam_export.py
# → data/dataset-v2.jsonl, data/corpus-v2.jsonl, data/dataset_card.json
```

It drops 1,795 rows whose sarcasm vote never resolved (uyam ran no adjudicator
pass) and 439 whose language vote never resolved (language is the
stratification key). Fields uyam does not collect — `labels.cues`,
`aux.tx_sentiment`, `aux.lid`, `human_gold` — are emitted as `null`, and the
code paths that need them stay off.

**Conversational context is currently a corpus rebuild, not the snapshot the
annotators saw** (`context.source == "corpus_rebuild"`, MODEL_PLAN §11.7). The
notebook prints this warning on every run; it clears itself once uyam exports
the real snapshot ([handoff H2](docs/UYAM_HANDOFF.md)).

The notebook also still reads a sibling `../uyam/data/annotated/` export if one
is present.

## Quickstart

```bash
uv sync                                      # pinned env (torch 2.6.0+cu124, Python 3.12)
uv run python tools/build_uyam_export.py     # build the contract from the CSVs
uv run python tools/check_model_contract.py  # architecture checks (seconds, CPU only)
uv run jupyter lab                           # open leische_pipeline.ipynb, run top-to-bottom
```

Headless re-execution:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    leische_pipeline.ipynb --ExecutePreprocessor.timeout=-1
```

## Hard rules encoded in the notebook (runtime assertions)

- Folds: `StratifiedGroupKFold`, stratified on `sarcastic×language`, **grouped
  by `submission_fullname`**, frozen to `results/folds-v{N}.json` with the
  dataset identity and refused on mismatch. 96.3% of rows share a thread with
  another annotated row, so this is load-bearing, not cosmetic.
- Retrieval banks: training-fold rows only, same-thread excluded, rebuilt per
  fold, leakage asserted at build time.
- Conversational context comes from one source per run, and which source it is
  is printed every run.
- `keyword_oversampled` rows never enter natural-distribution metrics (the
  current export has none — [handoff H9](docs/UYAM_HANDOFF.md)).
- Aux losses mask rows whose labels uyam never collected rather than training
  on fabricated values.
- The Stage-4 gate is `sigmoid(W[hₜ ; cᵢ])` per active channel, normalised —
  the manuscript's §3.4.1 formulation, not a joint softmax; the target must
  reach the gate for per-instance gating to mean anything.
- Temporal decay modulates the attention **scores** (`−λ·Δt` pre-softmax, Δt in
  hours, λ learnable), which is what §3.4.1 specifies.
- Thesis architecture is the committed baseline; every §9 upgrade is a config
  flag defaulting to OFF.

## Readiness gate — current status

```
[PASS] dataset-v2 under sarc-v2           dataset_version=v2, prompt_version=sarc-v2
[PASS] ≥400 sarcastic positives           403 positives
[FAIL] gold subset labeled + κ reported   n_gold_items=0, sarcastic κ=None
[PASS] every language×sarcastic cell ≥10  min cell 58
[PASS] fold file frozen                   results/folds-v2.json
```

To clear the last check: label ~300 items in the uyam review tab and report
human-vs-ensemble Cohen's κ in `dataset_card.json`
([handoff H1](docs/UYAM_HANDOFF.md)). Then `Config(smoke=False, ...)` for real
runs.
