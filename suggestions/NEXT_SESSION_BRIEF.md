# Brief for the next leische session — Stage 4 v2, condition 9, and the confirmations

Paste §0–§5 as the prompt (`Read suggestions/NEXT_SESSION_BRIEF.md and execute
it`). It continues from the state recorded in `docs/RESULTS_LOG.md` (the 41
GPU-hour run), `suggestions/SUMMARY.md` (where things stand) and
`suggestions/GATED_FUSION_V2.md` (the design to test). The prototype notebook
`leische_prototype_v3.ipynb` implements that design and has **not** been run.

---

## 0 · Ground rules (unchanged from `docs/FABILE_BRIEF.md`)

- Commit directly to `single-notebook`, plain messages, no attribution trailer.
  Push as you go.
- `leische_pipeline.ipynb` is the source of truth for the thesis model; the
  exploration and prototype notebooks pull it in (`tools/run_notebook.py
  --notebook …`) and never duplicate it. `tools/` holds only things that run
  outside a notebook.
- **Never edit `results/folds-*.json`.** Every row must share the frozen
  folds.
- Every number is agreement with the LLM-ensemble labels (Fleiss κ 0.376;
  human κ −0.148 on 31 items). Say so in every table you write.
- Keep the VRAM guards on (`vram_fraction 0.85`, workers 0). The full model
  peaks at 13.3 GB at batch 16 × 2; the large encoder at 15.2 GB at batch 8 × 4.
  Do not lift the cap to dodge an OOM.
- Anything you cannot verify, say so.

## 1 · What is already known (do not re-derive)

- Baseline 0.368 ± 0.026 F1 (15 runs); committed full model 0.355 ± 0.024; all
  eight conditions inside 0.350–0.370; retrieval never helps; tuning rows are
  flat. Label ceiling: the annotators score 0.49–0.58 against the shipped
  label, their majority 0.606. Do not expect 0.6.
- The exploration's recipe — one rendered block (post → parents → target →
  replies) + one head per annotator + vote-share soft labels — scores
  0.405 ± 0.010 on 5 folds (ΔF1 +0.037, CI [+0.016, +0.059]); the large
  encoder adds ~+0.03 on fold 0. Annotator heads are the necessary ingredient;
  priors are dispensable; the exact uyam prompt rendering does not help.
- Practicalities that bit: the Hugging Face downloader stalls on this machine
  (`tools/hf_fetch.py` is the fix); bf16 runs are not bit-reproducible (same
  seed, F1 0.313 vs 0.367 once) so seeds are mandatory; adding a config field
  invalidates skip-if-done unless missing keys default (already handled in
  the exploration notebook — copy that pattern).

## 2 · Tasks, in order

### T1 · Run the Stage 4 v2 plan (`leische_prototype_v3.ipynb`, §5 rows v2-a … v2-e)

Fold 0, seed 13 first; then seeds 42 / 7 on the best row; then 5 folds. Rows:

| row | change | keep if | otherwise |
|---|---|---|---|
| v2-a | committed model + matching features + null channel + channel dropout (no re-encoding) | null gate varies with context availability; F1 ≥ baseline | go to v2-b |
| v2-b | v2-a + target-aware item encoding + prototype retrieval | F1 ≥ baseline on ≥ 4/5 folds | report early fusion as the model |
| v2-c | v2-b + annotator heads + vote-share labels + cue-supervised gates | ≥ x10 (0.405) | the votes carry the gain, not the gate |
| v2-d | v2-c on xlm-roberta-large (batch 8 × 4, lr 1e-5) | ≥ x13 (0.438 fold 0) | not additive |
| v2-e | seeds + 5 folds + the §12 significance block on the best row | CI clears zero | seed noise |

Budget ≈ 4–6 GPU-hours. Before the first real row, run the notebook's stub
check cell (CPU, 4 rows) and fix anything it raises. Report after v2-c before
starting v2-d. Plot: `plot_gates` on the null channel vs context availability
(the figure the thesis's per-instance claim needs).

### T2 · Add condition 9 to the thesis matrix

Port the exploration's winning recipe (or v2-c if it wins) into
`leische_pipeline.ipynb` as `9_early_fusion` (flags, not a silent change):
`Config.input_mode = "early_fusion"`, `annotator_heads`, `vote_share_labels`;
`run_cv` unchanged. 5 folds × 3 seeds, then the significance block, then
`plot_ablation`. It must appear in `results/ablation-matrix.csv` beside
conditions 1–8.

### T3 · Confirmations still owed

- Seeds 42 / 7 for `impr-aux-cues-8_full` (tightest single-seed row, +0.016)
  and for `spec-temporal-unbounded` (best AUPRC).
- RQ3 (§13) on a condition-9 checkpoint: its flag is better calibrated
  (ECE 0.069 vs 0.11), so the two-stage sentiment slice should move; save the
  checkpoint with `save_checkpoint=True`.

### T4 · Documents

Append dated entries to `docs/RESULTS_LOG.md`; update `suggestions/SUMMARY.md`
§3 and §6 with the new rows; if v2 changes the picture, update
`GATED_FUSION_V2.md` §6 with what happened. Keep the honesty line.

## 3 · What to look at, in order (per row)

1. `fold_metrics.csv`: F1@0.5, AUPRC, ECE, best epoch — is it learning and
   is it calibrated?
2. Gate distribution: null-gate mass vs number of context turns; conv gate vs
   `contextual_incongruity`. If the null gate is flat across availability, the
   redesign has not worked, whatever F1 says.
3. Slices by `sarcasm_votes` (3-0 vs 2-1): any gain should be on 3-0 rows;
   a gain on 2-1 rows is suspicious.
4. Seeds before any claim.

## 4 · Known traps

| trap | why |
|---|---|
| judging a row from one seed | ±0.03 F1 seed spread on this data |
| a new `XConfig`/`V3Config` field | breaks skip-if-done unless compared at its default |
| retrieval exemplar texts | 60% of the compute for no gain — prototypes only |
| lifting the VRAM cap | turns an OOM into a desktop crash |
| any F1 > 0.5 on the adjudicated label | assume a leak (thread grouping, LOO author prior, bank ⊆ train) until proven otherwise |
| `Config.smoke` | follows the gate; `real_cfg()` is the real profile |

## 5 · Report

What the numbers are, what backs them (paths), what changed, what could not be
verified, what next. Tables, mean ± std, per-fold values, the caveat line.
