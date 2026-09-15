# Demo website

A page that shows a panel what the model *does* (FABILE_BRIEF §5). Not a metrics
dashboard — the ablation table lives in the notebook.

## Run it

```bash
uv sync --extra demo
uv run --extra demo uvicorn demo.server:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>. Startup takes 10–20 s (dataset, XLM-R,
retrieval banks, preloaded examples); a prediction on CPU takes about 0.2 s for a
short comment with full context.

**While the GPU is training, force the demo onto the CPU:**

```bash
LEISCHE_DEMO_DEVICE=cpu uv run --extra demo uvicorn demo.server:app --host 127.0.0.1 --port 8000
```

## What it shows

1. **A text box** — paste a Taglish, Tagalog or English comment and get a sarcasm
   probability with a clear verdict, plus the predicted literal and intended
   sentiment (RQ3's two-stage rule: stage 2 only re-reads rows flagged sarcastic).
2. **A context editor** — parent post, parent comments, replies, and earlier posts
   by the same author with an hours-ago field. Three toggles (conversational /
   temporal / retrieval) re-score on every change and the page shows the previous
   probability as a ghost mark with the delta. Switching a channel off gives it
   **no items**; the channel stays in the architecture and still gets a gate
   weight, exactly as on a real row with no thread around it.
3. **The gate values** as three bars that sum to one — the per-instance weight the
   model puts on each context source for this input.
4. **The retrieved exemplars** — nearest sarcastic and nearest non-sarcastic
   training rows with cosine similarities, side by side.
5. **Preloaded corpus examples**, grouped English / Tagalog / Taglish plus every
   gold disagreement, each labelled with who said what. Clicking one fills the
   comment box and the whole context editor and scores it.
6. **The honesty line**, always visible in the sticky header, plus an unmissable
   banner when the weights are untrained.

## Environment

| variable | default | what it does |
|---|---|---|
| `LEISCHE_CHECKPOINT` | `cache/checkpoints/ablation-8_full-fold0-seed13.pt` | the trained checkpoint to serve |
| `LEISCHE_RQ3_HEADS` | `cache/checkpoints/rq3-heads-fold0.pkl` | the RQ3 sentiment heads |
| `LEISCHE_DEMO_DEVICE` | `cuda` if available, else `cpu` | where the classifier runs |

The demo never sets a VRAM cap, so it will not disturb the trainer's memory
budget — but it will still allocate on the GPU if you let it. Set
`LEISCHE_DEMO_DEVICE=cpu` while a run is in flight.

## Without a checkpoint

The server boots anyway, so the page can be built and shown before training
finishes:

- the weights are randomly initialised, `GET /api/meta` returns `trained: false`
  and the page shows an **UNTRAINED WEIGHTS — no checkpoint found** banner;
- threshold 0.5, temperature 1.0;
- the retrieval banks come from `results/folds-v1-a419a4bc95.json`, fold 0.

Without `rq3-heads-fold0.pkl`, `sentiment` comes back `null` and the page says
*sentiment heads not trained yet*. Both files appear on their own: `run_cv` writes
the checkpoint when `save_checkpoint=True` on a non-smoke fold-0 run, and §13
writes the heads pickle.

## How it stays in sync with the notebook

`demo/notebook_source.py` locates three cells in `leische_pipeline.ipynb` by a
content marker and execs them — `Config`, the model cell, and a slice of the
training harness holding `Collator` and `to_device`. Same technique as
`tools/check_model_contract.py`. Nothing about the architecture is re-implemented
here, so a change in the notebook reaches the demo on the next restart.

The one thing that is deliberately stubbed is `apply_memory_guard`: the Config
cell is exec'd against a torch shim reporting no CUDA, so the guard returns early
instead of calling `set_per_process_memory_fraction` on the trainer's GPU.

## API

| route | returns |
|---|---|
| `GET /` | the page (`demo/static/`) |
| `GET /api/meta` | checkpoint, thresholds, temperature, label authority, bank sizes, device |
| `GET /api/examples` | the preloaded corpus rows, pre-filled with their context |
| `POST /api/predict` | probability, verdict, margin, gates, exemplars, sentiment |
| `GET /api/docs` | FastAPI's own schema browser |

`POST /api/predict` clips its input: 2000 characters of comment text, 8 parents,
8 replies, 5 earlier posts (and drops history items outside the 48 h window,
saying so in `notes`).

## Data

Read at runtime, never copied and never committed: `data/dataset-v1.jsonl`,
`data/corpus-v1.jsonl`, `results/folds-v1-a419a4bc95.json`, the checkpoints under
`cache/`, and `cache/retrieval-v1-sentence-transformers__*.npz` (rebuilt in the
notebook's own format if missing — about 70 s on CPU).

Preloaded examples are chosen deterministically from rows **outside** the
checkpoint's `train_fullnames`, so the model has not seen them — except the gold
disagreements, which are included whether or not they were trained on, with the
note saying which.
