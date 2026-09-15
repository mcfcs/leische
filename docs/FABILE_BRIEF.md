# Brief for Fable — train, tune, assess, demo

Hand this file to Fable as the prompt (`Read docs/FABILE_BRIEF.md and execute it`),
or paste §0–§6 directly.

---

## 0 · Ground rules

- **You may commit directly to `single-notebook`. Do NOT add a `Co-Authored-By`
  trailer to any commit.** Plain messages, no attribution footer.
- **You may deploy subagents, and should** for anything parallelisable —
  literature checks, doc writing, the demo front-end, log triage. **Prefer
  non-Fable models for subagents** (Sonnet or Haiku for mechanical work, Opus
  for anything needing judgement); keep Fable itself on the training loop and
  the analysis.
- The notebook is the single source of truth. Functions live inline in it, not
  in a package. `tools/` holds only things that run *outside* a notebook session.
- **Never edit `results/folds-*.json`.** Every condition must be scored on
  identical splits or the ablation means nothing.
- Anything you cannot verify, say so plainly rather than reporting it as done.

## 0b · Hardware, and not crashing the machine

**RTX 5090 Laptop, 24 GB VRAM, compute capability 12.0 (sm_120, Blackwell).**
The model trains on *this* machine — the 8 GB 4070 in older notes is a
different laptop, and every "does not fit in 8 GB" caveat in `MODEL_PLAN.md`
is obsolete.

**The environment was re-pinned for this GPU.** `torch 2.6.0+cu124` wheels are
built only up to sm_90, so on Blackwell they fail at the *first kernel launch*
with an opaque `no kernel image is available for execution on the device` —
after the data has already loaded. `pyproject.toml` now pins **`torch 2.9.1` on
the cu128 index**. Run `uv sync` and let §1 of the notebook confirm it: the
setup cell checks `torch.cuda.get_arch_list()` against this GPU's arch and
raises with the exact fix if the wheel is wrong. If you ever change the torch
pin, keep it `>= 2.7` on cu128 or newer.

### Leave the machine usable

24 GB is comfortable, but this is a **laptop** — the display, the compositor and
your editor share that VRAM. A process that grabs all of it does not produce a
clean OOM; it can stall or kill the desktop. Three guards are already wired in,
and you should keep them on:

| guard | where | what it does |
|---|---|---|
| `Config.vram_fraction = 0.85` | applied by `apply_memory_guard(CFG)` | caps the process at ~20 GB, so a spike raises a catchable `torch.OutOfMemoryError` instead of starving the driver |
| `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | set in §1 before the first allocation | stops allocator fragmentation causing an OOM that `nvidia-smi` says should not have happened |
| `dataloader_workers = 0`, `pin_memory = False` | `Config`, used by `make_loaders` | workers fork a copy of `CONV`/`TEMP`/`EMB` per process, and pinned host memory is not swappable — both are system-RAM risks, not VRAM ones |

`cuda_report()` prints peak allocated vs reserved, and `run_cv` calls it at the
end of every run. **Read that number before you raise the batch size.** Raise
`batch_size` only while peak stays under ~80% of the capped budget; if you need
the headroom back, lower `vram_fraction` rather than removing the cap.

If you do hit an OOM: reduce `batch_size` and raise `grad_accum` to keep the
effective batch, or lower `max_len_target`. Do not disable the cap to make an
OOM go away — that converts a recoverable error into a machine crash.

Also watch **system RAM**, which no guard covers: `corpus-v1.jsonl` (20,573
rows), the 15,000-row dataframe, the cached retrieval embeddings and every
fold's predictions all live in the Python process. If you run conditions in a
loop inside one kernel, `del` the big intermediates between them.

### Model size

With 24 GB, `xlm-roberta-base` in fp16 runs comfortably at **batch 32–64**
(start at 32 and check `cuda_report()`), which is a large speed-up over the
batch-8 defaults those old notes assumed — tune `batch_size` upward early, it
is the cheapest wall-clock win available.

`xlm-roberta-large` (~560 M params) now **does** fit and is worth one comparison
run once the base model is measured — but it is a §9.6 encoder-ablation row, not
the thesis baseline. Keep XLM-R base as the reported architecture.

---

## 1 · Recheck the pipeline and the data

Before any GPU time:

```bash
uv sync
uv run python tools/check_model_contract.py     # architecture, CPU only, seconds
uv run jupyter lab                              # run §1–§6 top to bottom
```

Confirm, and report the actual numbers you see:

- the §10 gate prints **TRAIN-READY: all 5 PASS** (if not, stop and say why);
- `context.source == {'annotator_snapshot'}` — a corpus rebuild blocks training
  by design;
- 15,000 rows / 1,601 positives / 1,393 threads; fold file
  `results/folds-v1-a419a4bc95.json` loads with its identity verified;
- the leakage assertions pass for all 15,000 retrieval queries;
- `LABEL_AUTHORITY` prints and names the three annotator models plus the
  adjudicator.

**Then read [`ANNOTATION_PROVENANCE.md`](ANNOTATION_PROVENANCE.md) before you
interpret a single metric.** The short version: every label is LLM-generated,
inter-annotator Fleiss' κ on sarcasm is 0.376, and human-vs-ensemble κ on the
31-item gold subset is **−0.148** — traced to the adjudicator pass, which
removed 1,146 positives and drove the base rate from 18.3% to 10.7%. Your F1 is
**agreement with the ensemble**, not with human judgement. Say so in every
summary you write.

Also re-read [`METHODOLOGY_REVIEW.md`](METHODOLOGY_REVIEW.md) (the M1–M5
decisions) and [`PIPELINE_SPEC.md`](PIPELINE_SPEC.md) (what each section does).

---

## 2 · First training batch

Run the §9–§11 smoke tests first — overfit-16 must reach ~zero loss for both
the baseline and the full model. If it does not, it is a wiring bug; fix that
before anything else.

Then the first real run. **Budget your compute deliberately** — the full model
pushes ~9 encoder passes per example (target + conv items + temporal items +
2×k retrieval exemplars), so the 8×5×3 grid is still substantial even on 24 GB.
The constraint here is wall-clock, not VRAM: raise `batch_size` first (see §0b),
then plan the grid.

Staged plan:

| stage | what | why |
|---|---|---|
| A | condition 1 (baseline) and condition 8 (full), **5 folds × 1 seed**, full settings | the RQ1 headline, cheapest useful result |
| B | the remaining 6 conditions, **5 folds × 1 seed** | completes the RQ2 matrix |
| C | conditions 1 and 8 only, **seeds 42 and 7** | variance on the pair that carries the claim |
| D | §12b specification variants | retrieval k and encoder, temporal window, fixed vs learnable λ |

Report stage A before starting B. If stage A takes more than ~4 hours, say so
and propose a reduced plan rather than silently running for a day.

`run_cv` writes `fold_metrics.csv`, `predictions.csv`, `histories.json` and
`run.json` per run — all stamped with `LABEL_AUTHORITY` and the dataset
identity. Do not hand-edit them.

---

## 3 · Assess, retune, observe

### The plots are already built — use them

§8b of the notebook has every panel you need; they take the frames `run_cv`
returns and write PNGs to `results/figures/`:

`plot_training_curves` · `plot_fold_spread` · `plot_threshold_sweep` ·
`plot_pr_roc` · `plot_confusion` · `plot_calibration` · `plot_ablation` ·
`plot_slices` · `plot_gates` · `plot_bootstrap` · `plot_rq3`

If you need a panel that does not exist, add it to §8b in the same style
(validated palette, legend row below the axes, headroom for bar labels) rather
than inventing a new theme.

### What to actually look at, in order

1. **`plot_threshold_sweep` first.** At a 10.7% base rate 0.5 is almost
   certainly not F1-optimal. Pick the threshold **on validation folds only**,
   never on test, then re-report. This single change usually moves F1 more than
   any hyperparameter.
2. **`plot_training_curves`** — is it learning at all, and where did early
   stopping fire? Train loss falling while val F1 is flat means it is
   memorising; both flat means the lr is wrong or the head is starved.
3. **`plot_fold_spread`** — before claiming any condition beats another, check
   the per-fold points. With ~320 test positives per fold, a 0.02 F1 gap inside
   a 0.04 std is not a result.
4. **`plot_calibration`** — RQ3 stage 2 only fires on flagged rows, so a badly
   calibrated flag breaks the sentiment story before it starts.
5. **`plot_gates`** — the gate is now conditioned on the target (M1). If the
   per-channel std is still near zero, the per-instance gating claim is not
   supported and that is a finding worth reporting honestly.
6. **`plot_slices`** by `language`, `record_type`, `resolved_by`. If F1 is
   strong only on `unanimous` rows, label noise dominates.

### Hyperparameters worth tuning

Tune on **validation only**, folds frozen, one axis at a time. In rough order of
expected payoff on this data:

- decision threshold (see above — do this first, it is free);
- `lr_encoder` ∈ {1e-5, 2e-5, 3e-5}, `lr_heads` ∈ {1e-4, 3e-4};
- `batch_size` (24 GB allows 32–64 for the base encoder) × `grad_accum`;
- `class_weighting` vs `focal_loss` (γ=2) vs `weighted_sampler`
  (`target_pos_frac` 0.25–0.30) — the §9.2 flags already exist;
- `d_model` ∈ {256, 384}; `dropout` ∈ {0.1, 0.2, 0.3};
- `freeze_bottom_layers` ∈ {0, 4, 6} and `layerwise_lr_decay` = 0.9 (§9.4 —
  small-data hygiene, likely to help here);
- `max_len_target` — the median target is short, so 192 may be wasted compute.

Keep the thesis-committed configuration as the reported baseline and present
tuned variants as separate rows. Do not quietly replace the baseline.

### Observation write-up

After each stage, append to `docs/RESULTS_LOG.md` (create it): the config, the
numbers with mean ± std, which figures back the claim, and what you concluded.
Short entries, dated, honest about what is noise.

---

## 4 · Improvements, if the numbers ask for them

Only after the baseline is measured and logged. Each one is an existing config
flag — turn on one at a time and keep it as an added row, never as a silent
change to the baseline:

- `sample_weighting` / `soft_labels` (§9.1) — the dataset says how certain each
  label is; `reliability.n_annotators` and `resolved_by` are the real quality
  axis here.
- `aux_cue_heads` (§9.3) — the four cue labels are now populated and are free
  supervision. Note `polarity_inversion` fires on 641 rows of which **all 641**
  are sarcastic; `hyperbole` fires on 825 of which only 194 are.
- `aux_polarity_shift`, `aux_language_head`.
- `temperature_scaling` and `tune_threshold_on_val` (§9.8).
- **Robustness row worth its cost:** retrain on the *annotator-majority* sarcasm
  label instead of the adjudicated one (2,747 positives, 18.3%). If the model
  behaves the same under both, the finding does not hinge on the adjudicator —
  that is a strong thing to be able to say. Derive the label from
  `reliability.annotators[].sarcastic`; do not re-export.

If an improvement helps, say by how much and whether it survives the fold
spread. If it does not help, say that too and leave the flag off.

---

## 5 · Demo website

A page that shows a panel what the model *does*, not a dashboard of metrics.
Deploy a subagent for the front-end (non-Fable) while you keep training.

Must have:

- **A text box.** Type or paste a Taglish comment, get a sarcasm probability
  with a clear verdict, plus the predicted literal and intended sentiment.
- **Context toggles.** Let them add a parent post and replies, then flip
  conversational / temporal / retrieval on and off and watch the prediction
  move. This is the thesis in one interaction — it makes RQ2 tangible in a way
  the ablation table cannot.
- **The gate values, visualised.** Show the three per-instance weights so the
  panel can see the model deciding which context to trust for *this* input.
- **The retrieved exemplars.** Show the nearest sarcastic and non-sarcastic
  training examples — it makes the expectation-violation mechanism concrete.
- **Preloaded examples.** Include several from the corpus covering English,
  Tagalog and Taglish, and at least one from the gold disagreements so the
  panel sees an honest hard case.
- **An honesty line**, always visible: labels are LLM-ensemble generated,
  human-validated on 31 items so far, κ = −0.148. Do not ship a demo that
  implies the labels are human gold.

Practical: a small FastAPI or Gradio backend loading one checkpoint, with a
static front-end, is enough. Keep it runnable locally with one command and
document that command in the README. Do **not** put a checkpoint or any raw
post text into git.

---

## 6 · When you are done

Report: what the numbers are, what backs them, what you changed, what you could
not verify, and what you would do next with more compute. Push commits as you
go — no co-author trailer.

If something here is wrong or a better path exists, say so and explain why
rather than following it off a cliff.

---

## Known traps

| trap | why it bites here |
|---|---|
| Reporting F1 at threshold 0.5 | base rate is 10.7%; the optimal threshold is elsewhere |
| Tuning on the test fold | folds are frozen precisely so this cannot happen by accident |
| Treating F1 as "sarcasm accuracy" | it is agreement with an LLM ensemble; gold κ is −0.148 |
| Claiming a small ablation gap | check `plot_fold_spread` and the bootstrap CI first |
| Reading the temporal condition as "temporal context does not help" | only 36.7% of rows have ≥1 temporal item; it is data availability |
| Editing the fold file | every condition must share splits |
| Disabling `vram_fraction` to dodge an OOM | turns a catchable error into a desktop crash on a laptop GPU |
| Assuming the batch-8 defaults | they were written for an 8 GB machine; 24 GB wants 32–64 |
| A torch wheel without sm_120 | fails at the first kernel, long after load; §1 preflights it |
| Committing checkpoints or post text | `data/`, `cache/` and `*.pt` are gitignored — keep it that way |
