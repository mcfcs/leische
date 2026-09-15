# leische

Model repository for the undergraduate thesis **Context-Aware Sarcasm Detection
in Code-Switching Social Media Posts**. The data half (collection + LLM-ensemble
annotation) is the separate [uyam](../uyam) repository; the methodology
reference is [docs/MODEL_PLAN.md](docs/MODEL_PLAN.md).

> **STATUS: TRAIN-READY.** The full `sarc-v2` export (`uyam_commit 039c3aa8…`)
> gives **15,000 rows / 1,601 sarcastic positives / 1,393 threads**, with the
> annotator context snapshot, adjudication, cue labels and LID all present. All
> five train-ready gate checks pass and `run_cv` is unblocked.
>
> **Labels are not yet validated.** Every label was produced by an LLM ensemble,
> not a human (see [docs/ANNOTATION_PROVENANCE.md](docs/ANNOTATION_PROVENANCE.md)),
> the gold subset stands at 31 of ~300 items, and human-vs-ensemble sarcasm
> κ is **−0.148** — traced to the adjudicator pass, not the annotators. So every
> metric measures agreement with the ensemble, not with human judgement, and the
> pipeline stamps `LABEL_AUTHORITY` onto every artifact saying so.

Read before running:

- **[docs/ANNOTATION_PROVENANCE.md](docs/ANNOTATION_PROVENANCE.md)** — how the
  labels were actually produced (LLM ensemble, not the human annotator §3.3
  describes), what the gold subset found, and what the manuscript must say.
- **[docs/PIPELINE_SPEC.md](docs/PIPELINE_SPEC.md)** — what the notebook does,
  section by section, with every choice traced to a manuscript clause.
- **[docs/METHODOLOGY_REVIEW.md](docs/METHODOLOGY_REVIEW.md)** — manuscript
  Chapter III vs. this pipeline; the five departures (M1–M5) and why.
- **[docs/UYAM_HANDOFF.md](docs/UYAM_HANDOFF.md)** — what uyam still owes the
  model repo, ordered by how much it blocks.
- **[docs/FABILE_BRIEF.md](docs/FABILE_BRIEF.md)** — the brief for the training
  run: what to check, what to tune, what to look at, and the demo spec.

## Layout — one notebook

The entire pipeline lives in **[leische_pipeline.ipynb](leische_pipeline.ipynb)**:
environment checks → data loading + contract validation + readiness gate → EDA
→ frozen thread-grouped folds → the three context channels with leakage
assertions → the 5-stage XLM-R model behind RQ2 ablation flags → training
harness → **plot helpers (§8b)** → the two §10 smoke tests (overfit-16,
tiny-settings 5-fold dry run) → the 8×5 ablation matrix with significance tests
→ specification variants (§12b) → the RQ3 two-stage sentiment evaluation.

Every figure is written to `results/figures/` as it is drawn, so the manuscript
can cite them directly.

```
leische_pipeline.ipynb        the whole pipeline (all functions inline)
tools/check_model_contract.py architecture checks (no GPU, no model download)
docs/FABILE_BRIEF.md          brief for the training / tuning / demo run
tools/gold_disagreements.py   every human-vs-ensemble disagreement, in full
tools/build_uyam_export.py    CSV-era adapter (superseded by the uyam export)
docs/ANNOTATION_PROVENANCE.md how the labels were produced; the gold finding
docs/PIPELINE_SPEC.md         what the notebook does, clause by clause
docs/MODEL_PLAN.md            methodology reference (from uyam)
docs/METHODOLOGY_REVIEW.md    manuscript vs. pipeline; the M1–M5 decisions
docs/UYAM_HANDOFF.md          data-side gaps and what to fix in uyam
results/                      frozen folds + run artifacts (committed)
results/figures/              every plot §8b draws, written as it runs
data/                         the uyam export + derived contract (gitignored)
cache/                        embeddings / checkpoints (gitignored)
```

The `main` branch keeps the alternative layout (importable `src/leische`
package + per-stage notebooks + pytest suite).

## Data

uyam ships the dataset contract directly — no adapter step:

```
data/dataset-v1.jsonl    15,000 annotated targets (sarc-v2)
data/corpus-v1.jsonl     20,573-row scraped corpus (temporal context source)
data/dataset_card.json   identity, counts, agreement, provenance
```

Drop a new export in `data/` and re-run. Fold identity keys on
`uyam_commit + prompt_version + row count`, not the version string, so a new
export freezes its own fold file (`results/folds-{version}-{hash}.json`) and can
never be confused with an older one. The notebook also still reads a sibling
`../uyam/data/annotated/` export if one is present.

Conversational context is the **annotator snapshot** — exactly the block the
annotator models saw (MODEL_PLAN §11.7). The notebook asserts this at the gate
and refuses to train on a corpus rebuild.

## Quickstart

```bash
uv sync                                      # pinned env (torch 2.6.0+cu124, Python 3.12)
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
- Conversational context must be the annotator snapshot; a corpus rebuild
  blocks training at the gate (§11.7).
- `keyword_oversampled` rows never enter natural-distribution metrics (the
  current export has none — [handoff H4](docs/UYAM_HANDOFF.md)).
- Every results artifact carries `LABEL_AUTHORITY`: who produced the labels,
  the inter-annotator κ, and the gold-subset status.
- Aux losses mask rows whose labels uyam never collected rather than training
  on fabricated values.
- The Stage-4 gate is `sigmoid(W[hₜ ; cᵢ])` per active channel, normalised —
  the manuscript's §3.4.1 formulation, not a joint softmax; the target must
  reach the gate for per-instance gating to mean anything.
- Temporal decay modulates the attention **scores** (`−λ·Δt` pre-softmax, Δt in
  hours, λ learnable), which is what §3.4.1 specifies.
- Thesis architecture is the committed baseline; every §9 upgrade is a config
  flag defaulting to OFF.

## Readiness gate — two tiers

**Train-ready** blocks `run_cv` in code. **Claim-ready** never blocks; it stamps
every artifact so no metric can be quoted as agreement with human judgement when
it is agreement with an LLM ensemble.

```
TRAIN-READY (blocks run_cv)
  [PASS] full corpus under sarc-v2+        prompt_version=sarc-v2, 15000 rows
  [PASS] ≥400 sarcastic positives          1601 positives (10.7% base rate)
  [PASS] every language×sarcastic cell ≥10 min cell 426
  [PASS] conversational context is the annotator snapshot
  [PASS] fold file frozen                  folds-v1-a419a4bc95.json
  => training is legitimate.

CLAIM-READY (stamps results, never blocks)
  [WARN] gold subset ≥250 items            31 labelled
  [WARN] human-vs-ensemble sarcasm κ ≥0.60 κ=-0.1481 on n=31
  [WARN] inter-annotator sarcasm κ ≥0.60   Fleiss κ=0.3761
  [PASS] automatic LID validated           62.5% agreement on n=15000
```

Run with `Config(smoke=False, ...)` for real training. To clear the claim tier,
see [handoff H1 and H2](docs/UYAM_HANDOFF.md) — the negative gold κ traces to
the adjudicator pass, and `tools/gold_disagreements.py` prints the evidence.
