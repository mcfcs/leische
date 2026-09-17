# Stage 4 v2 — a stronger gated fusion that keeps the thesis's mechanism (design draft, not yet run)

The thesis's central claim is *per-instance gating*: three context sources,
one scalar gate each, computed from the target and the source, normalised to
sum to one (§3.4.1 Stage 4). The 37-hour run showed the mechanism works
mechanically (gate std ≈ 0.1 per channel, target-conditioned) but buys no
accuracy (full model 0.355 vs baseline 0.368). This draft keeps the three
channels and the per-instance gate and changes **what the gate sees, what it
may choose, how it is trained, and what it feeds the classifier**. Every change
is motivated by a measured failure and every one is a config flag, so the
thesis architecture remains the reported baseline and v2 is an added row.

Numbers referenced below are in [docs/RESULTS_LOG.md](../docs/RESULTS_LOG.md)
and [IMPROVEMENTS.md](IMPROVEMENTS.md) §5.

---

## 0 · What the evidence says about the current gate

| measured | implication |
|---|---|
| gate weights: conv 0.16, temp 0.33, ret 0.50 (3 seeds); conv weight does **not** rise with the number of conversational items (0.155 with none, 0.149 with 3+) | the gate is not reading *availability* or *usefulness*; it has settled on a prior over channels |
| retrieval gets half the weight while every retrieval condition is at or below the baseline | the gate has no way to learn that a channel is a distractor: there is no "none of these" option, and the channel vectors carry no signal about their own reliability |
| under the annotator-majority label the same gate puts 0.34 on conv (vs 0.16) | the channel *is* informative; late fusion of a pooled vector under the stricter label wastes it |
| early fusion (context tokens and target tokens in one input) beats the baseline on 5/5 folds; late fusion never does | the target must condition the *encoding* of each item, not just the pooling of already-encoded items |
| one head per annotator + vote-share labels is the necessary ingredient in every variant that won | the gate and the classifier should be trained on the votes, not on the collapsed label |
| the cue labels `contextual_incongruity` (thread clashes with the literal reading) and `polarity_inversion` exist on every row | the gate has a *free supervision signal*: when the annotators flagged incongruity with the thread, the conversational gate should be open |

---

## 1 · The v2 design in one figure

```
target t ─────────────────────────────────────────────────────────────┐
                                                                      │
conv items  ──► [t </s> item_j] ──► shared encoder ──► e_j ─┐         │
                (target-aware item encoding, §2.1)          │ attn(t) ──► c_conv ─┐
temp items  ──► [t </s> post_k] ──► shared encoder ──► e_k ─┘  −λΔt   ──► c_temp ─┤
retrieval   ──► prototype contrast over label-aware banks (§2.6) ──► c_ret ──┤
                                                                     null ──► c_0 = 0
                                                                              │
   gate_i = σ( W_g [ t ; c_i ; t⊙c_i ; |t−c_i| ; s_i ] )   for i ∈ {conv, temp, ret, null}   (§2.2, §2.3)
   g = normalise(gate) with channel dropout in training (§2.4)
   c_fused = Σ_i g_i c_i
   feats = [ t ; c_fused ; t⊙c_fused ; |t−c_fused| ]                                     (§2.5)
   logits = MLP(feats);  annotator heads_k(feats);  cue heads(feats);  gate-cue loss    (§2.7)
```

---

## 2 · The changes

### 2.1 Target-aware item encoding (fixes the keyhole)

**Now:** each context item is encoded alone, mean-pooled, projected to 256-d;
the target meets it only as an attention query over pooled vectors.

**v2:** encode each item *with* the target in front of it — one pass of
`[<s> target </s></s> ROLE: item </s>]` through the shared encoder, mean-pool
the item's tokens only (mask the target span out of the pool), then project.
The item vector `e_j` now already encodes the target–item interaction at token
level (the incongruity between "great service as always" and "6 hours at the
LTO"), which is what early fusion delivered and late fusion cannot. The
Stage-2 attention over items and the Stage-4 gate are unchanged; they now
operate on target-aware vectors.

Cost: each item pass carries the target's tokens too (median 31, cap 192).
Conversational items are few (mean 0.9 turns + the post), temporal items ≤ 5,
so per-example tokens go from ~10 passes × 96 to roughly the same number of
passes at ≤ 288 tokens — about 1.5–2× the current full model *if retrieval is
kept*, and **less than the current full model** once retrieval exemplar texts
are replaced by prototypes (§2.6). Batch 16 × 2 still fits (the full model
peaked at 13.3 GB).

### 2.2 Matching features in the gate (lets the gate judge, not just look)

**Now:** `gate_i = σ(W [t ; c_i])`.

**v2:** `gate_i = σ(W [t ; c_i ; t ⊙ c_i ; |t − c_i| ; s_i])`. The product and
absolute-difference terms are the standard "matching" features from natural
language inference (ESIM, InferSent): they make agreement and contradiction
between the target and the context linearly readable, which a plain
concatenation does not. `s_i` is a small vector of **channel self-report**
scalars the gate can use to judge reliability:

- conv: number of items, whether a parent and a reply exist, log length of
  the post;
- temp: number of prior posts (48 h and unbounded), mean Δt in hours,
  smoothed author sarcasm rate from training-fold labels (leave-one-out);
- ret: kNN-25 / kNN-100 sarcasm rate over training rows, mean similarity of
  the top-5, gap between the sarcastic and non-sarcastic bank similarities.

These are the scalars the exploration measured to be weakly informative on
their own (AUPRC ~0.15); their job here is not to predict sarcasm but to tell
the gate *when a channel is worth trusting*.

### 2.3 A null channel (lets the gate say "none of these")

Add a fourth, fixed channel `c_0 = 0` with its own gate `σ(W_0 [t ; s])`. After
normalisation the mass on `c_0` is the model's per-instance estimate that no
context helps this comment, and `c_fused` shrinks toward zero accordingly — the
target-only baseline becomes recoverable *per instance*. Report the null-gate
distribution: on this corpus, 48% of rows have no thread turn at all, and the
gate should say so.

### 2.4 Gate regularisation (stops the collapse onto a prior)

- **Channel dropout** (ModDrop-style): during training, with probability 0.2
  per channel, replace `c_i` by zeros and set its self-report scalars to the
  "absent" values, so the classifier can never rely on one channel being
  present and the gate learns what each channel contributes alone.
- **Gate entropy target.** A weak penalty toward *lower* entropy
  (`+β·H(g)`, β ≈ 0.01) so the gate commits per instance instead of averaging;
  optionally sparsemax instead of normalised sigmoids so it can put exactly
  zero on a channel.
- **Temperature** on the gate logits, learned, so the sharpness is not fixed
  by initialisation.

### 2.5 An interaction-aware classifier input

**Now:** `MLP([t ; c_fused])`. **v2:** `MLP([t ; c_fused ; t ⊙ c_fused ;
|t − c_fused|])`. Same NLI reasoning as §2.2: sarcasm is a *mismatch* between
what the target says and what the context makes it mean; give the classifier
the mismatch directly.

### 2.6 Retrieval as prototype contrast, not exemplar texts

Every retrieval variant (k = 3 / 5 / 10, two encoders, two fusion styles, two
prior forms) was at or below the baseline, at 60% of the compute. Keep the
thesis's *idea* — expectation from sarcastic vs non-sarcastic examples — but
implement it as a contrast of **attention-pooled prototypes** over the frozen
retrieval space: `c_ret = W_r [ p_sarc − p_nonsarc ; p_sarc ⊙ p_nonsarc ]`,
where `p_sarc` is the similarity-weighted mean of the top-k sarcastic
training-row *embeddings* (already cached, no encoder pass) and likewise for
the non-sarcastic bank, projected to 256-d. Zero encoder passes; the channel
becomes a 384-d lookup. If it still adds nothing under the v2 gate, the null
channel will say so and the manuscript can report retrieval as a negative
result with a clean mechanism behind it.

### 2.7 Training signals

- **One head per annotator + vote-share soft labels** — exactly the recipe
  that produced the +0.037 in the exploration; heads sit on `feats`.
- **Cue-supervised gates**: auxiliary BCE `contextual_incongruity → g_conv`
  (the thread clashes with the literal reading ⇒ the conversational gate
  should be open) and `polarity_inversion ∧ ¬contextual_incongruity → g_null`
  (inversion that is readable from the target alone ⇒ no context needed).
  Weight ~0.1; masked where cues are missing. This is the first supervision
  the gate has ever had, and it comes free from the dataset.
- Keep the committed lr 2e-5 for base, 1e-5 for large; class weights per fold;
  early stopping on validation F1; R-Drop optional if seed variance stays ugly.

---

## 3 · Pseudo-code (drop-in for `ContextAwareSarcasmModel`)

```python
class GatedFusionV2(nn.Module):
    def __init__(self, cfg):
        d = cfg.d_model
        self.encoder = SharedEncoder(cfg)                       # unchanged
        self.conv_attn, self.temp_attn = TargetAttention(d), TargetAttention(d)
        self.ret_proj = nn.Linear(2 * cfg.retrieval_dim, d)      # prototypes, §2.6
        self.self_report = {"conv": 3, "temp": 4, "ret": 4, "null": 0}
        self.gate = nn.ModuleDict({c: nn.Linear(4 * d + n, 1) for c, n in self.self_report.items()})
        self.log_temp = nn.Parameter(torch.zeros(()))
        self.classifier = nn.Sequential(nn.Linear(4 * d, cfg.mlp_hidden), nn.GELU(),
                                        nn.Dropout(cfg.dropout), nn.Linear(cfg.mlp_hidden, 2))
        self.annot_heads = nn.ModuleList([nn.Linear(4 * d, 2) for _ in range(3)])
        self.cue_head = nn.Linear(4 * d, 4)

    def encode_items_with_target(self, part, t_ids, t_mask):
        # [t </s></s> item] per item; pool only the item span (mask the target out)
        ids, mask, item_span = pair_with_target(part, t_ids, t_mask)   # collator helper
        h = self.encoder.backbone(input_ids=ids, attention_mask=mask).last_hidden_state
        pooled = (h * item_span.unsqueeze(-1)).sum(1) / item_span.sum(1, keepdim=True).clamp(min=1)
        return scatter_items(self.encoder.proj(pooled), part["batch_idx"], t_ids.shape[0])

    def forward(self, b):
        t = self.encoder(b["target"]["input_ids"], b["target"]["attention_mask"])
        chans, reports = {}, {}
        items, mask = self.encode_items_with_target(b["conv"], *b["target"].values())
        chans["conv"] = self.conv_attn(t, items + role_emb, mask)
        items, mask = self.encode_items_with_target(b["temp"], *b["target"].values())
        chans["temp"] = self.temp_attn(t, items, mask, score_bias=-self.lam * b["temp"]["delta_hours"])
        chans["ret"] = self.ret_proj(torch.cat([b["ret"]["p_sarc"] - b["ret"]["p_nonsarc"],
                                               b["ret"]["p_sarc"] * b["ret"]["p_nonsarc"]], -1))
        chans["null"] = torch.zeros_like(t)
        if self.training:                                      # channel dropout, §2.4
            for c in ("conv", "temp", "ret"):
                drop = (torch.rand(t.shape[0], 1, device=t.device) < 0.2)
                chans[c] = torch.where(drop, torch.zeros_like(t), chans[c])
                b["report"][c] = torch.where(drop, ABSENT[c], b["report"][c])
        logits = torch.cat([self.gate[c](torch.cat([t, v, t * v, (t - v).abs(), b["report"][c]], -1))
                            for c, v in chans.items()], -1) / self.log_temp.exp()
        g = torch.sigmoid(logits); g = g / g.sum(-1, keepdim=True).clamp(min=1e-6)   # or sparsemax
        fused = sum(g[:, i:i + 1] * v for i, v in enumerate(chans.values()))
        feats = torch.cat([t, fused, t * fused, (t - fused).abs()], -1)
        return {"logits": self.classifier(feats), "gates": g, "features": feats, "target_emb": t,
                "annot_logits": torch.stack([h(feats) for h in self.annot_heads], 1),
                "cue_logits": self.cue_head(feats)}
```

Loss: `CE(main, y) + 0.5·Σ_k CE(annot_k, vote_k) + 0.25·BCE(cues)
+ 0.1·BCE(g_conv, contextual_incongruity) + 0.1·BCE(g_null, inversion_only)
+ 0.01·H(g)`, with the class weights the pipeline already computes per fold;
optionally the vote-share soft target on the main head.

---

## 4 · Cost

| model | encoder passes per example | tokens per pass | peak VRAM (measured / est.) | min per fold |
|---|---|---|---|---|
| committed full model | ~10 (target + conv + temp + 6 exemplars) | ≤ 96–192 | 13.3 GB | 23 |
| v2 with prototype retrieval | ~1 + conv (≈1.9) + temp (≈0.8) ≈ 4 | ≤ 192 + 96 | ~9–11 GB (est.) | ~12 (est.) |
| v2 on xlm-roberta-large | same | same | ~16 GB at batch 8 (est.) | ~35 (est.) |

---

## 5 · Experiment plan (what to run, in order, and what would falsify it)

| row | change | keep if | otherwise |
|---|---|---|---|
| v2-a | committed model + matching features + null channel + channel dropout (no re-encoding) | null gate varies with context availability; F1 ≥ baseline | the gate cannot judge from pooled vectors — go to v2-b |
| v2-b | v2-a + target-aware item encoding (§2.1), prototype retrieval (§2.6) | F1 clears the baseline on ≥ 4/5 folds | the channel structure itself is the limit; report early fusion as the model |
| v2-c | v2-b + annotator heads + vote-share labels + cue-supervised gates | ≥ exploration x10 (0.405) | the votes carry the gain, not the gate |
| v2-d | v2-c on xlm-roberta-large | ≥ x13 (0.438 fold 0) | capacity and mechanism are not additive |
| v2-e | seeds 42 / 7 on the best row; 5 folds; the §12 significance block | CI clears zero | it was seed noise |

Total ≈ 3–4 GPU-hours on the base encoder, +2 h for the large row.

---

## 6 · What to expect, honestly

- On the current adjudicated labels: **0.41–0.45 F1** if v2-c works as
  designed (the exploration's +0.037 plus a gate that stops spending half its
  weight on a distractor), 0.44–0.48 with the large encoder. Not 0.6: the
  three annotator LLMs score 0.49–0.58 against the verdict they produced, and
  their majority vote 0.606 (see [SUMMARY.md](SUMMARY.md) §5). A fine-tuned
  encoder will not out-agree the labelling process.
- What v2 *can* deliver that the current model cannot: a gate whose weights
  track context availability and the incongruity cue, a null channel that
  says "no context" on the 48% of rows without a thread turn, and an honest
  retrieval verdict with the mechanism isolated — i.e., the per-instance
  gating claim becomes supportable with a figure, not just a std.
- What it cannot fix: the contested 2-1 rows (44% of the positives, F1 ≈ 0.27
  for every model including the annotators' own majority at 0.345). Those
  move only when the label does.

---

## References specific to this draft

The 23 references for the diagnosis and the exploration are in
[IMPROVEMENTS.md](IMPROVEMENTS.md). The design elements above draw on:

- Arevalo, J., Solorio, T., Montes-y-Gómez, M., & González, F. A. (2017). Gated Multimodal Units for Information Fusion. *ICLR 2017 Workshop*. https://arxiv.org/abs/1702.01992 — the GMU the thesis cites for Stage 4.
- Chen, Q., Zhu, X., Ling, Z., Wei, S., Jiang, H., & Inkpen, D. (2017). Enhanced LSTM for Natural Language Inference. *Proceedings of ACL 2017*. https://aclanthology.org/P17-1152/ — the `[a ; b ; a⊙b ; |a−b|]` matching features (§2.2, §2.5).
- Conneau, A., Kiela, D., Schwenk, H., Barrault, L., & Bordes, A. (2017). Supervised Learning of Universal Sentence Representations from Natural Language Inference Data. *Proceedings of EMNLP 2017*. https://aclanthology.org/D17-1070/ — same matching features at sentence level.
- Neverova, N., Wolf, C., Taylor, G. W., & Nebout, F. (2016). ModDrop: Adaptive Multi-Modal Gesture Recognition. *IEEE TPAMI*, 38(8), 1692–1706. https://arxiv.org/abs/1501.00102 — channel (modality) dropout (§2.4).
- Martins, A. F. T., & Astudillo, R. F. (2016). From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification. *Proceedings of ICML 2016*. https://arxiv.org/abs/1602.02068 — sparse gates (§2.4).
- Alayrac, J.-B., et al. (2022). Flamingo: a Visual Language Model for Few-Shot Learning. *NeurIPS 2022*. https://arxiv.org/abs/2204.14198 — gated cross-attention with a learned, initially-closed gate (the "null by default" idea in §2.3).
- Liang, X., et al. (2021). R-Drop: Regularized Dropout for Neural Networks. *NeurIPS 2021*. https://arxiv.org/abs/2106.14448 — seed-variance control (§2.7).
- Davani, A. M., Díaz, M., & Prabhakaran, V. (2022). Dealing with Disagreements: Looking Beyond the Majority Vote in Subjective Annotations. *TACL*, 10, 92–110. https://aclanthology.org/2022.tacl-1.6/ — per-annotator heads (§2.7).

The first seven are standard references I have not re-verified against the
anthology in this session; check the exact venues before citing them in the
manuscript.
