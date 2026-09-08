# Methodology review — manuscript vs. pipeline

Review of `FinalManuscript_DISCS_UG_Latest.pdf` Chapter III against
`docs/MODEL_PLAN.md` and `leische_pipeline.ipynb`, run against the real uyam
export (4,416 rows / 403 positives / 679 threads).

The manuscript is deliberately general — it was written before the data
existed. Where it is *specific*, it should win; where it is silent, the
pipeline is the spec of record. Four places are neither: the pipeline silently
contradicts a specific sentence in §3.4.1. Those are the ones to decide.

**Bottom line:** the pipeline is **better than the manuscript on protocol** and
**worse on two model specifics** — and the two it gets wrong are the ones that
touch the thesis's central claim.

---

## M1 · Gated fusion — the pipeline is wrong, change the code

**Manuscript §3.4.1 Stage 4:** "a scalar gate value `gᵢ` is computed by passing
**the concatenation of the target embedding hₜ and the context vector cᵢ**
through a sigmoid-activated linear layer. The three gate values are then
normalized to sum to one."

**Pipeline** (`ContextAwareSarcasmModel.forward`):

```python
g = F.softmax(self.gate(torch.cat(chans, dim=-1)), dim=-1)   # gate = Linear(3d → 3)
```

One linear layer over the concatenated **context** vectors. `hₜ` never enters
the gate.

**Verdict: manuscript is better. This is the one that matters.**

The claim the whole thesis rests on is *per-instance* gating — "the model can
suppress uninformative context sources **depending on the input instance**"
(§3.4). A gate that cannot see the target cannot condition on the target. It
can only ask "do these three context vectors look informative in general",
which is a much weaker mechanism than the one being claimed, and it is not the
GMU formulation the thesis cites [2].

It also shows up in the smoke output: the fold-0 gate values are nearly
constant across instances (`gate_conv` std 0.050, `gate_ret` std **0.017**
over the whole test fold). A per-instance gate that barely varies per instance
is the symptom.

**Fix** — roughly ten lines:

```python
self.gate = nn.ModuleList([nn.Linear(2 * d, 1) for _ in self.active])
...
raw = torch.cat([torch.sigmoid(gate(torch.cat([t, c], -1)))
                 for gate, c in zip(self.gate, chans)], dim=-1)
g = raw / raw.sum(-1, keepdim=True).clamp(min=1e-6)
```

Note this also preserves the §7.3 ablation rule for free: disabled channels are
simply absent from the list, so the normalisation is over active channels only
— no renormalising a softmax after the fact.

---

## M2 · Temporal context — split decision

| | manuscript §3.4(2) / §3.4.1 | pipeline |
|---|---|---|
| how many prior posts | **≤ 5** | `temporal_k = 10` |
| time window | **within 48 h before** | none — any prior post |
| Δt unit | **hours** | days |
| λ | **learnable** | `temporal_lambda_learnable = False` |

### λ — manuscript is right, and MODEL_PLAN has a factual error

`docs/MODEL_PLAN.md` §4.2 says "*thesis states fixed decay; learnable-λ is a
strict generalization*". That is wrong. §3.4.1 Stage 2 says plainly: "**λ is a
learnable parameter**." The default should flip to `True`, and the notebook's
comment "`(fixed — thesis baseline)`" is backwards.

### Units — cosmetic once λ is learnable, not before

With λ learnable, hours vs days is just a rescaling of λ. With λ *fixed* at
0.1 it is not remotely cosmetic: `exp(−0.1·Δt_hours)` is dead after a day,
`exp(−0.1·Δt_days)` still weights a month-old post at 0.05. Since the
manuscript says learnable, switch to hours anyway so the reported λ is
comparable to the text.

### The 48 h window — **the manuscript is worse on this data**

Measured on the current dataset:

| rule | ≥1 item | ≥3 items | full 5 |
|---|---|---|---|
| manuscript (≤5, within 48 h) | **34.5%** | 10.8% | 5.3% |
| pipeline (≤10, unbounded) | **45.7%** | 18.0% | 9.8% |

The 48 h rule discards a quarter of an already-scarce channel. It is in the
manuscript because §3.1 assumed "a collection window of at least three
months"; the actual annotated window is **24 days** (2026-08-01 → 08-24, see
`UYAM_HANDOFF.md` H8). The rule was written for a corpus that does not exist.

**Recommendation:** adopt the manuscript's `k=5`, hours, and learnable λ; make
the window a config knob (`temporal_window_hours: float | None = 48`) and
report both settings as an ablation row. That converts a data weakness into a
stated finding instead of a silent deviation.

**Either way, this must be said in the results:** with ~2/3 of rows presenting
an empty temporal channel, RQ2's temporal condition measures *data
availability*, not whether temporal context helps. Reporting the coverage table
above next to the ablation matrix is what keeps that conclusion honest.

---

## M3 · Retrieval — the pipeline is right, amend the manuscript

| | manuscript §3.4(3) | pipeline |
|---|---|---|
| k | 3 ("subject to hyperparameter tuning") | 5 |
| index | **XLM-R `[CLS]` embedding** | frozen `paraphrase-multilingual-MiniLM-L12-v2` |

**k is not a conflict** — the manuscript explicitly licenses tuning. Ablate
k ∈ {3, 5, 10} and report; no text change needed.

**The embedder is a real disagreement, and the pipeline wins on two counts.**

1. *Quality.* Un-fine-tuned `[CLS]` vectors from a masked-LM encoder are a
   poor similarity space — they are dominated by frequency and length effects
   and are well known to underperform for cosine retrieval. A sentence encoder
   trained with a contrastive objective is the right tool, and MiniLM's
   multilingual variant covers Tagalog and English in one space.
2. *Stationarity.* If the index is built from the model's own XLM-R encoder,
   the retrieval bank shifts as the encoder trains — the exemplars a target
   sees in epoch 1 are not the ones it sees in epoch 8. That makes the
   retrieval channel a moving target and forces a re-embed per epoch per fold.
   A frozen external encoder keeps the bank fixed and cheap.

**Recommendation:** amend §3.4(3) to "a frozen multilingual sentence encoder",
and keep XLM-R `[CLS]` as one ablation row so the swap is justified by a number
rather than by assertion.

---

## M4 · Fold construction — the pipeline is right, and it is load-bearing

**Manuscript §3.5:** "stratified 5-fold cross-validation, with stratification
applied jointly on the sarcasm label and language category… 80% train, 10%
validation, 10% test." No grouping.

**Pipeline:** `StratifiedGroupKFold(5)` stratified on `sarcastic × language`
and **grouped by `submission_fullname`**, with validation carved out of the
train side under the same grouping.

**Verdict: pipeline is clearly better — do not revert this.**

**96.3% of dataset rows sit in a thread with at least one other annotated
row** (679 threads over 4,416 rows, mean 6.5 rows/thread, largest 144).
Ungrouped splitting would therefore put a target and its own thread-mates on
opposite sides of the split for almost every row, and both context channels
would leak:

- the **conversational** channel feeds a test target text that is a *training
  target* in another row;
- the **retrieval** bank would return same-thread neighbours as "semantically
  similar exemplars", complete with their labels.

The direction of that leak is the problem: it inflates the context conditions
relative to the target-only baseline — i.e. it manufactures the thesis's
headline result. Ungrouped folds would produce a bigger RQ1 gap and a bigger
RQ2 effect, and neither would be real.

**Recommendation:** keep it, and state it in §3.5 as an explicit leakage
control. A deviation that *shrinks* your own claimed effect is the kind
examiners reward.

The pipeline additionally asserts this at runtime (`fold {i}: thread spans
train/test`) and freezes the fold file to `results/folds-v2.json` against the
dataset identity, so every ablation condition is scored on identical splits —
which §3.5 requires but does not say how to enforce.

---

## Is the pipeline methodology better or worse overall?

**Better, on everything the manuscript is silent about.** These are all
things an examiner asks for and Chapter III does not mention:

- thread-grouped folds with runtime leakage assertions (M4)
- retrieval banks rebuilt per fold, same-thread excluded, asserted at build time
- folds frozen to a file and identity-checked, so conditions are comparable
- mean ± std over folds × seeds instead of a single run — with ~80 test
  positives per fold, a single F1 swings wildly
- significance testing (paired bootstrap, approximate randomization, McNemar)
- natural-distribution vs all-rows metric hygiene
- per-instance gate logging, disaggregation by language / `record_type` /
  `resolved_by`, calibration + ECE
- a data-readiness gate that **refuses to run non-smoke training** while the
  data is not ready

**Worse, on two model specifics** — M1 (the gate loses target conditioning)
and the λ default in M2. Both contradict specific sentences in §3.4.1, and M1
undercuts the mechanism the thesis is built on. Fix both in code.

**Differently scoped** on retrieval (M3): better embedder, and the manuscript
already licenses the k change.

For the session recreating the methodology: **adopt the pipeline's protocol
wholesale, fix M1 and the λ default, keep the pipeline's retrieval embedder
and document the swap, and make the temporal window a reported knob rather
than a hard-coded 48 h.**

---

## Data-side findings that constrain the methodology

Not deviations — facts about the export that any methodology has to live with.
Full detail in `UYAM_HANDOFF.md`.

1. **Sarcasm-label agreement is weak.** Fleiss' κ = **0.386** across the three
   annotator models on the primary target label (vs 0.641 literal sentiment,
   0.568 intended sentiment). With no human gold subset (κ = n/a, 0 items),
   nothing currently validates the labels. This is the one gate check still
   failing.
2. **Every surviving sarcasm label is unanimous.** After dropping unresolved
   rows the votes are only `3-0` (3,549) or `2-0` (867) — uyam never
   adjudicated a split. Sample weighting by `resolved_by` (§9.1) is therefore
   degenerate; the real quality axis is *how many models answered*, which the
   adapter exposes as `reliability.n_annotators`.
3. **The positive class is thin and thread-concentrated.** 403 positives, 9.1%
   base rate, ~80 per test fold. Report mean ± std across folds × seeds and
   never a single-run F1.
4. **Sarcasm lives in comments.** Submissions are 286 rows with almost no
   positives; per-`record_type` disaggregation will be lopsided.
5. **Conversational coverage is thinner than the architecture assumes.** 93.5%
   of rows have a submission (all comments do), but only **31.9%** have a
   parent chain and **30.0%** have a captured reply — most annotated comments
   are depth-0 replies to the post. The conversational channel is largely
   "target + post title".
6. **RQ3 loses 233 rows** whose `intended_sentiment` vote never resolved; the
   pipeline now excludes them rather than scoring against a null.

---

## Changes already made

- `tools/build_uyam_export.py` — derives the dataset contract from the two
  uyam CSVs (see the provenance table in `UYAM_HANDOFF.md`).
- Contract validation now treats `labels.cues`, `aux.*` and `human_gold` as
  optional rather than asserting on fields uyam does not collect, and accepts
  `resolved_by = "unanimous_partial"`.
- Aux losses mask rows whose labels were never collected (`-1` sentinel)
  instead of training on fabricated zeros.
- RQ3 stage 1 rebuilt to the **manuscript's** §3.5 specification — a sentiment
  head trained on `literal_sentiment` over the **target-only** embedding —
  because MODEL_PLAN's proposed `aux.tx_sentiment` is not collected. RQ3 now
  filters to rows carrying sentiment ground truth.
- The loader prints, every run, which fields are absent and whether
  conversational context is the annotator snapshot or a corpus rebuild.

**Not yet changed:** M1–M4 above, pending a decision.
