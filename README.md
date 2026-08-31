# leische

Model repository for the undergraduate thesis **Context-Aware Sarcasm Detection
in Code-Switching Social Media Posts**. The data half (collection + LLM-ensemble
annotation) is the separate [uyam](../uyam) repository; the methodology
reference is [docs/MODEL_PLAN.md](docs/MODEL_PLAN.md).

> **STATUS: SMOKE PHASE — do not train for results.**
> The current export (`dataset-v1`) is a 100-item pilot (8 sarcastic rows, no
> human gold subset). The §10 data-readiness gate FAILS and the notebook
> enforces it: non-smoke runs are refused in code, and every pilot-derived
> number is prefixed `SMOKE`.

## Layout — one notebook

The entire pipeline lives in **[leische_pipeline.ipynb](leische_pipeline.ipynb)**
(committed with executed outputs): environment checks → data loading + contract
validation + readiness gate → EDA → frozen thread-grouped folds → the three
context channels with leakage assertions → the 5-stage XLM-R model behind
RQ2 ablation flags → training harness → the two §10 smoke tests (overfit-16,
tiny-settings 5-fold dry run) → the 8×5 ablation matrix with significance
tests → the RQ3 two-stage sentiment evaluation.

```
leische_pipeline.ipynb   the whole pipeline (all functions inline)
docs/MODEL_PLAN.md       methodology reference (from uyam)
results/                 frozen folds + SMOKE artifacts (committed)
data/                    optional local copy of the uyam export (gitignored)
cache/                   embeddings / checkpoints (gitignored)
```

The `main` branch keeps the alternative layout (importable `src/leische`
package + per-stage notebooks + pytest suite).

## Data

No sync step. The notebook looks for the export in this order:

1. `./data/dataset-v1.jsonl` — a copy you downloaded from uyam
2. `../uyam/data/annotated/` — read directly when uyam is cloned next to this repo

## Quickstart

```bash
uv sync                        # pinned env (torch 2.6.0+cu124, Python 3.12)
uv run jupyter lab             # open leische_pipeline.ipynb, run top-to-bottom
```

Headless re-execution:

```bash
uv run jupyter nbconvert --to notebook --execute --inplace \
    leische_pipeline.ipynb --ExecutePreprocessor.timeout=-1
```

## Hard rules encoded in the notebook (runtime assertions)

- Folds: `StratifiedGroupKFold`, stratified on `sarcastic×language`, **grouped
  by `submission_fullname`**, frozen to `results/folds-v{N}.json` with the
  dataset identity and refused on mismatch.
- Retrieval banks: training-fold rows only, same-thread excluded, rebuilt per
  fold, leakage asserted at build time.
- Conversational context comes only from the embedded snapshot the annotators
  saw — never rebuilt from the corpus dump.
- `keyword_oversampled` rows never enter natural-distribution metrics.
- Thesis architecture is the committed baseline; every §9 upgrade is a config
  flag defaulting to OFF.

## When dataset-v2 lands

1. Drop the new export in `./data/` (or just `git pull` uyam next door).
2. Set `Config.dataset_version = "v2"` and re-run top-to-bottom — folds
   re-freeze automatically for the new identity.
3. Once the §10 gate prints PASS: `Config(smoke=False, ...)` for real runs.
