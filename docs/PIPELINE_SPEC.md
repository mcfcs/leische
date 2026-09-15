# Pipeline specification — what `leische_pipeline.ipynb` actually does

A section-by-section account of the notebook as committed, with every choice
traced back to a clause in the manuscript (`FinalManuscript_DISCS_UG_Latest.pdf`,
Chapter III) and marked **=** (implements it), **+** (goes beyond it), or **≠**
(deliberately departs, with the reason).

Companion documents:
[ANNOTATION_PROVENANCE.md](ANNOTATION_PROVENANCE.md) is how the labels were
produced; [METHODOLOGY_REVIEW.md](METHODOLOGY_REVIEW.md) is the decision record
for the departures; [UYAM_HANDOFF.md](UYAM_HANDOFF.md) is what the data side
still owes; [FABILE_BRIEF.md](FABILE_BRIEF.md) is the brief for the training run.

Verify without a GPU or a model download:

```bash
uv run python tools/check_model_contract.py  # architecture checks on the model cell
```

Current dataset: `data/dataset-v1.jsonl`, `prompt_version = sarc-v2`,
`uyam_commit = 039c3aa8…` — **15,000 rows / 1,601 positives / 1,393 threads**.

---

## Correspondence at a glance

| Manuscript clause | Pipeline | |
|---|---|---|
| §3.4.1 S1 · one shared XLM-R encodes every text unit separately | `SharedEncoder` — one `xlm-roberta-base`, mean-pool, `Linear(768→256)` | **=** |
| §3.4.1 S2 · target-conditioned attention over conversational items | `TargetAttention(t as query)`, + role & is-submitter embeddings | **=/+** |
| §3.4.1 S2 · temporal attention scores modulated by `exp(−λ·Δt)`, Δt in hours, λ learnable | `score_bias = −λ·Δt_hours` added pre-softmax; λ learnable via softplus | **=** |
| §3.4(2) · ≤5 author posts within 48 h before | `temporal_k=5`, `temporal_window_hours=48.0` | **=** |
| §3.4.1 S3 · attend to sarcastic and non-sarcastic exemplars, concat, project | `ret_attn_sarc` + `ret_attn_nonsarc` → `Linear(2d→d)` | **=** |
| §3.4(3) · top-k each bank, k=3 default, index rebuilt per fold | `retrieval_k=3`, banks rebuilt per fold, leakage asserted | **=** |
| §3.4(3) · index over XLM-R `[CLS]` | frozen multilingual sentence encoder; `[CLS]` available as `retrieval_encoder="xlmr_cls"` | **≠** M3 |
| §3.4.1 S4 · `gᵢ = sigmoid(W[hₜ ; cᵢ])`, normalised to sum to one | one `Linear(2d→1)` per active channel, sigmoid, normalised | **=** |
| §3.4.1 S5 · `MLP([hₜ ; c_fused])`, cross-entropy | `Linear→GELU→Dropout→Linear`, 2 logits | **=** |
| §3.5 · inverse-frequency class weights | computed **per training fold**, never globally | **=/+** |
| §3.5 · stratified 5-fold on sarcasm × language, 80/10/10 | `StratifiedGroupKFold` **grouped by thread**, val carved out under the same grouping | **≠** M4 |
| §3.5 · 8-condition full-factorial ablation on the same folds | `CONDITIONS` × frozen `FOLDS` | **=** |
| §3.5 · F1 primary, precision/recall/accuracy supporting, mean ± std over folds | `safe_metrics` + `fold_summary`, over folds **× seeds** | **=/+** |
| §3.5 · two-stage sentiment, pre- vs post-sarcasm, vs annotated sentiment | stage 1 on the target-only embedding, stage 2 on `[t ; c_fused]` | **=** |
| §3.5 · disaggregate by language and sarcasm label | `rq3_report` slices + the `literal≠intended` cell | **=/+** |
| §7.4 · per-fold confusion, gate distributions, calibration | plotted, not just printed (§8b) | **=/+** |
| §3.2.1 · fastText LID, report auto-vs-manual accuracy | `aux.lid` shipped; 62.5% vs ensemble, 71.0% vs gold | **=** |
| §3.3 · single human annotator, intra-annotator κ | **not implementable** — labels are a 3-LLM ensemble; see [ANNOTATION_PROVENANCE.md](ANNOTATION_PROVENANCE.md) | ✗ |
| §3.1 · keyword oversampling | **not present** — no `keyword_oversampled` rows exist | ✗ H4 |

---

## Section by section

### 1 · Setup & environment (cells 1–2)

Prints versions, asserts CUDA, checks that a fixed seed gives a bit-identical
forward pass, and pins `ROOT` / `cache/` / `results/`. **+** — the manuscript
says nothing about determinism; a reproducibility check belongs at the top.

### 2 · Configuration (cells 3–4)

One `Config` dataclass holding every switch. Defaults are the manuscript's
committed values; every MODEL_PLAN §9 upgrade is a flag defaulting to `OFF`, so
each manuscript claim stays reproducible with upgrades disabled.

Two throttled profiles, `smoke_cfg()` and `ablation_cfg()`, exist only to prove
the harness — they shorten sequences and cap steps per epoch, so nothing they
produce is a result.

Knobs that encode a manuscript clause:

| knob | default | clause |
|---|---|---|
| `encoder_name` / `pooling` / `d_model` | `xlm-roberta-base` / `mean` / 256 | §3.4.1 S1 |
| `max_len_target` / `_context` / `_selftext` | 192 / 96 / 128 | MODEL_PLAN §4.1 |
| `temporal_k` | **5** | §3.4(2) |
| `temporal_window_hours` | **48.0** (`None` lifts it) | §3.4(2) |
| `temporal_lambda_init` | 0.0289 = ln2/24, i.e. a one-day half-life **in hours** | — (λ is learnable, so this is a starting point) |
| `temporal_lambda_learnable` | **True** | §3.4.1 S2 |
| `retrieval_k` | **3** | §3.4(3) |
| `retrieval_encoder` | `sentence_transformer` (`xlmr_cls` = the literal §3.4(3)) | M3 |
| `class_weighting` | `inverse_freq`, per fold | §3.5 |
| `n_folds` | 5 | §3.5 |
| `seeds` | `[13, 42, 7]` | **+** |

### 3 · Data — load, validate, readiness gate (cells 5–8)

Loads `data/dataset-vN.jsonl` + `corpus-vN.jsonl` + `dataset_card.json`
straight from the uyam export, validates every row against the contract, and
runs the §10 readiness gate. The current export passes with **zero violations**.

The validator still treats `labels.cues`, `aux.*` and `human_gold` as optional —
an older export left them null — but all four are populated now.

Identity keys on `uyam_commit + prompt_version + row count`, not the version
string (uyam has used "v1" for two different corpora), and every run prints
which source the conversational channel came from.

**The gate has two tiers.** *Train-ready* blocks `run_cv` from any non-smoke run,
in code — including a check that conversational context is the annotator
snapshot rather than a corpus rebuild (§11.7), since a model trained on a
rebuild learns from context the labels never saw. *Claim-ready* never blocks;
it stamps `LABEL_AUTHORITY` — who produced the labels, the inter-annotator κ,
the gold-subset status — into every `run.json` and as columns on every
`fold_metrics.csv`. **+** — the manuscript has no such concept.

### 4 · EDA (cells 9–18)

Label counts and the `language × sarcastic` stratification cells (flagging any
cell < 10); vote and resolution distributions; token counts against the Stage-1
truncation budgets; **context-coverage panels** for the conversational and
temporal channels; agreement statistics echoed from the dataset card (including
human-vs-ensemble gold κ and the LID validation §3.2.1 promises); and a manual
read of 20 sarcastic rows with all three annotator rationales *and* the
adjudicator's — the adjudicator overrides the annotator majority on ~35% of the
rows it sees, so its reasoning belongs in the read.

The coverage panels are the ones to carry into the manuscript — they are what
tells you whether a context channel is measurable at all on this corpus.

### 5 · Frozen folds (cells 19–20) — **≠ M4**

`StratifiedGroupKFold(5)`, stratified on `sarcastic × language` per §3.5, but
**grouped by `submission_fullname`**, which §3.5 does not ask for. 96.3% of
rows share a thread with another annotated row, so ungrouped folds would leak
thread-mates between train and test through both the conversational channel and
the retrieval bank — inflating exactly the effect the thesis wants to measure.

Validation is carved from the train side under the same grouping (≈80/10/10 as
§3.5 requires). Runtime assertions refuse a fold whose threads span two splits,
and refuse (outside smoke mode) a split with zero positives. Folds are frozen to
`results/folds-{version}-{identity hash}.json`, so two exports can never collide
on one fold file and nothing needs deleting by hand when the data changes. The
gate re-evaluates here, after the file exists, so a first run is not blocked by
an artifact it is about to create. An author-overlap audit reports how many
authors appear in more than one test fold.

### 6 · Context channels (cells 21–24)

- **Conversational** — `[submission, ancestors…, replies…]`, each carrying a
  role embedding and an is-submitter flag. Built once; independent of fold and
  config.
- **Temporal** — `TemporalIndex.history()` returns up to `temporal_k` posts by
  the same `author_hash` strictly before the target and within
  `temporal_window_hours`, most recent first, with Δt **in hours**. Memoised
  per `(k, window)` so §12b can vary it.
- **Retrieval** — frozen embeddings for every target, cached to `cache/`; banks
  rebuilt per fold from **training rows only**, with the query's own thread
  excluded. `assert_no_leakage` re-checks all three conditions at build time:
  bank ⊆ training fold, no shared thread, no self-retrieval.

Cell 24 prints, for five random rows, the exact texts that entered each
channel — the audit MODEL_PLAN §6 asks for.

### 7 · Model (cells 25–26)

One `ContextAwareSarcasmModel` covering all eight ablation conditions.
`use_conv = use_temp = use_ret = False` reduces it *exactly* to the RQ1
context-agnostic baseline `MLP([t])`, so the RQ1 comparison cannot drift.

Stage 4 follows §3.4.1 verbatim: one `Linear(2d→1)` per **active** channel,
input `[hₜ ; cᵢ]`, sigmoid, then normalised to sum to one. Because only active
channels get a gate, §7.3's rule (disabled channels leave the gate rather than
being zero-filled) holds by construction.

Two details worth knowing:

- **Temporal decay modulates the attention scores**, not the key/value vectors.
  Adding `−λ·Δt` pre-softmax is identical to multiplying the softmax weights by
  `exp(−λ·Δt)` and renormalising, which is what §3.4.1 says. Scaling K/V (what
  MODEL_PLAN §4.2 proposed) is *not* the same: `w_k`/`w_v` are affine, so an
  item decayed to zero still contributes their bias and recency stops being
  monotone in Δt.
- **A uniform Δt across a row is a no-op**, because softmax is shift-invariant.
  The decay *ranks* an author's posts against each other; it does not shrink
  the channel for authors whose history is uniformly old. Suppressing a weak
  temporal channel is the Stage-4 gate's job, not the decay's.

Empty channels emit exact zeros (including the retrieval branch, whose
projection bias is zeroed for rows with no exemplars) so that
`missing_channel = "zeros"` and `"learned"` are genuinely different options.

### 8 · Training harness (cells 27–28)

`SarcasmDataset` / `Collator` flatten variable-length context items into a flat
batch plus an owner index, which `scatter_items` re-pads. AdamW with separate
encoder (2e-5) and head (1e-4) learning rates, linear warmup, fp16, grad
clipping, early stopping on validation F1 with best-weight restore.

`compute_loss` implements §3.5's inverse-frequency weighted cross-entropy, with
the §9 alternatives (soft labels, focal loss, sample weights, three auxiliary
heads) behind flags. Auxiliary losses **mask rows whose labels uyam never
collected** using a `-1` sentinel, rather than training on fabricated zeros.

`run_cv` loops folds × seeds, writes `fold_metrics.csv`, `predictions.csv` and
a `run.json` carrying the full config, the dataset identity, the git commit and
the gate status — and refuses to run non-smoke while the gate fails.

Reporting helpers: `fold_summary` (mean ± std), `natural_and_all` (§7.2 hygiene),
`slice_metrics` (disaggregation), `ece` (calibration).

### 8b · Plots (cells 29–30) — **+**

Eleven plot helpers covering everything MODEL_PLAN §7.4 asks for and the panels
a 10.7% base rate makes mandatory: training curves, fold spread, threshold
sweep, PR/ROC, confusion, calibration + ECE, the ablation matrix, slice
disaggregation, gate behaviour, the bootstrap ΔF1 distribution, and RQ3
pre/post sentiment. Each takes the frames `run_cv` returns and writes a PNG to
`results/figures/`, so any run can be plotted after the fact.

Chart rules, applied uniformly: a CVD-validated three-slot categorical palette
(blue/orange/aqua — all-pairs ΔE 9.2 deutan, 24.0 normal-vision); magnitude
charts use **one hue with emphasis** rather than eight identities; no dual-axis
chart anywhere; legends sit in a row below the axes so they can cover neither
the marks nor the title; adjacent bars are separated by a 2px surface gap, not
a border; bar-end labels get explicit headroom. `run_cv` now also returns
`histories` and `paired_bootstrap` returns its resamples, so the dynamics and
significance panels have something to draw.

### 9–11 · Smoke tests and the full model (cells 31–36)

Smoke test 1 overfits 16 rows to near-zero loss for both the baseline and the
full model — a wiring check, not a result. Smoke test 2 runs a 5-fold dry run at
tiny settings and exercises both pooling paths. Cell 34 trains the full model on
fold 0, logs the per-instance gate distribution overall and by language,
round-trips a checkpoint (asserting identical predictions), and caches the
frozen `[t ; c_fused]` features **and** the target-only embeddings that §13
needs. **+** — none of this is in the manuscript; all of it is what makes the
numbers trustworthy.

### 12 · Ablation matrix (cells 37–40)

The eight §3.5 conditions × 5 folds on the frozen folds, then significance
tests between condition 8 and condition 1: paired bootstrap with a 95% CI,
approximate randomization, and McNemar on the discordant pairs. **+** — §3.5
asks for the ablation but specifies no significance test, and this is the first
thing an examiner asks for.

### 12b · Specification variants (cells 41–42) — **+**

The three places where the manuscript and the pipeline disagreed are resolved by
*reporting both*: retrieval k ∈ {3, 5, 10}, sentence encoder vs XLM-R `[CLS]`,
the 48 h temporal bound vs unbounded, and learnable vs fixed λ. Each row carries
its temporal coverage, which is the column that belongs in the manuscript — the
48 h bound is a data-availability constraint, not a modelling result.

### 13 · RQ3 two-stage sentiment (cells 43–44)

Ground truth is `labels.intended_sentiment`; the 233 rows whose sentiment vote
never resolved are excluded rather than scored against a null.

- **Stage 1 (pre-sarcasm)** — a head trained on `literal_sentiment` over the
  **target-only** embedding. This is §3.5 as written ("applied directly to the
  original target post text"). The shipped `aux.tx_sentiment` (MODEL_PLAN §8's
  proposal) is now also reported as an external, model-independent cross-check,
  which bounds how much of any stage-1→2 gain is our own head rather than
  sarcasm awareness.
- **Stage 2 (post-sarcasm)** — rows the sarcasm model flags are re-predicted by
  a head over the frozen `[t ; c_fused]` features trained on
  `intended_sentiment`; unflagged rows keep their stage-1 prediction.

Both heads are fit on fold-0 **training** rows only, so RQ3 supervision never
touches the sarcasm encoder or the test fold. Results are disaggregated by
language and by gold sarcasm label, plus the *sarcastic ∧ literal≠intended*
slice — the cell where sarcasm-awareness has to show its value.

Sarcasm-flag calibration (ECE) is reported alongside, because which rows get
flagged is what determines whether stage 2 helps at all.

### 14 · Verdict (cells 45–46)

Re-prints the smoke-test outcomes and the readiness gate.

---

## Where the pipeline is stricter than the manuscript

Eight things Chapter III does not mention and an examiner will:

1. Thread-grouped folds with runtime leakage assertions (§5).
2. Retrieval banks rebuilt per fold, same-thread excluded, asserted (§6).
3. Folds frozen to a file and identity-checked, so conditions are comparable.
4. Mean ± std over folds **× seeds** — with ~80 test positives per fold a
   single-run F1 is noise.
5. Significance testing on the headline pair.
6. Natural-distribution vs all-rows metric hygiene.
7. Per-instance gate logging, calibration/ECE, and disaggregation by
   `language`, `record_type` and `resolved_by` — all of it plotted, not just
   printed (§8b).
8. A readiness gate that refuses to produce reportable numbers until the data
   supports them.

## Where the pipeline cannot follow the manuscript

Two Chapter III commitments remain blocked on the data, not the code — full
detail in [UYAM_HANDOFF.md](UYAM_HANDOFF.md):

- **§3.3 annotation.** The data is a three-LLM ensemble, not a single human
  annotator, so the promised intra-annotator κ cannot be reported.
  Inter-annotator Fleiss' κ on sarcasm is 0.376, and the 31-item gold subset
  scores human-vs-ensemble Cohen's κ = **−0.148** — traced to the adjudicator
  pass. See [ANNOTATION_PROVENANCE.md](ANNOTATION_PROVENANCE.md); the manuscript
  text is being revised.
- **§3.1 keyword oversampling.** No oversampled rows exist; the §7.2 hygiene
  path runs but has nothing to separate.

§3.2.1's LID validation, previously impossible, is now reportable: fastText
agrees with the ensemble language label on 62.5% of 15,000 rows and with the
human gold on 71.0% of 31. Conversational context is now the annotator snapshot
on every row, so the §11.7 caveat is closed — and the gate refuses to train on
anything else.
