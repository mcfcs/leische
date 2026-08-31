# %% [markdown]
# # 01 — Data EDA (guide §5)
#
# **Rerun on every dataset version.** Panels: label counts and stratification
# cells, sampling strategy, reliability, text lengths vs encoder budgets,
# conversational-context coverage, temporal coverage (viability check),
# agreement echo, manual read of sarcastic rows, readiness gate, frozen folds.

# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leische import data as D
from leische.nbsupport import bootstrap

cfg, df, corpus, card, gate = bootstrap(need_corpus=True)
SMOKE = cfg.tag()  # prefix for every printed number below

# %% [markdown]
# ## Label counts and stratification cells
#
# Any `language × sarcastic` cell < 10 is flagged — those cells make 5-fold
# stratification meaningless (§10 gate criterion).

# %%
sarc = D.sarcastic(df)
lang = D.language(df)
print(f"{SMOKE}sarcastic: {sarc.sum()} / {len(df)} ({sarc.mean():.1%})")
cells = pd.crosstab(lang, sarc)
print(f"\n{SMOKE}language × sarcastic cells:")
print(cells)
thin = [(l, s, int(cells.loc[l, s])) for l in cells.index for s in cells.columns
        if cells.loc[l, s] < 10]
for l, s, n in thin:
    print(f"  FLAG: cell ({l}, sarcastic={s}) has only {n} rows (<10)")
print(f"\n{SMOKE}{len(thin)} thin cell(s) — the pilot cannot support stratified 5-fold "
      "training; this is exactly why the §10 gate fails.")

# %%
fig, axes = plt.subplots(1, 3, figsize=(13, 3.2))
cells.plot.bar(ax=axes[0], title=f"{SMOKE}language × sarcastic")
df["sampling_strategy"].fillna("null (→natural)").value_counts().plot.bar(
    ax=axes[1], title=f"{SMOKE}sampling_strategy")
df["reliability"].map(lambda r: r["resolved_by"]).value_counts().plot.bar(
    ax=axes[2], title=f"{SMOKE}resolved_by")
plt.tight_layout()
plt.show()

# %% [markdown]
# ## Reliability: vote patterns × resolution path

# %%
rel = pd.DataFrame({
    "resolved_by": df["reliability"].map(lambda r: r["resolved_by"]),
    "sarcasm_votes": df["reliability"].map(lambda r: r["sarcasm_votes"]),
    "mean_confidence": df["reliability"].map(lambda r: r["mean_confidence"]),
})
print(f"{SMOKE}votes × resolution:")
print(pd.crosstab(rel["sarcasm_votes"], rel["resolved_by"]))
print(f"\n{SMOKE}mean self-reported confidence: {rel['mean_confidence'].mean():.3f} "
      "(weakly calibrated — vote pattern is the trusted signal, contract §8)")

# %% [markdown]
# ## Text lengths vs the Stage-1 truncation budgets
#
# Budgets (guide §4.1): target 192, context items 96, selftext 128 tokens.

# %%
from leische.contexts import build_conversational
from leische.encoders import Tokenize

tok = Tokenize(cfg.encoder_name, cfg.max_len_target, cfg.max_len_context, cfg.max_len_selftext)
target_tokens = df["text"].map(tok.token_count)
ctx_tokens = [tok.token_count(it.text) for _, r in df.iterrows()
              for it in build_conversational(r)]

fig, axes = plt.subplots(1, 2, figsize=(11, 3.2))
axes[0].hist(target_tokens, bins=30)
axes[0].axvline(cfg.max_len_target, color="r", ls="--", label=f"budget {cfg.max_len_target}")
axes[0].set_title(f"{SMOKE}target token counts"); axes[0].legend()
axes[1].hist(ctx_tokens, bins=30)
axes[1].axvline(cfg.max_len_context, color="r", ls="--", label=f"budget {cfg.max_len_context}")
axes[1].set_title(f"{SMOKE}conversational-item token counts"); axes[1].legend()
plt.tight_layout(); plt.show()

over_t = (target_tokens > cfg.max_len_target).mean()
over_c = np.mean([c > cfg.max_len_context for c in ctx_tokens]) if ctx_tokens else 0.0
print(f"{SMOKE}targets over budget: {over_t:.1%} | context items over budget: {over_c:.1%}")
print("note: corpus text carries mojibake (e.g. â€™ for ’) from collection-time "
      "encoding; labels were conditioned on the text AS-IS, so we never 'fix' it (§11.7).")

# %% [markdown]
# ## Conversational-context coverage

# %%
conv_stats = pd.DataFrame([
    {
        "has_submission": r["context"]["submission"] is not None,
        "n_parents": len(r["context"]["parent_chain"] or []),
        "n_replies": len(r["context"]["replies"] or []),
    }
    for _, r in df.iterrows()
])
print(f"{SMOKE}% with submission snapshot: {conv_stats['has_submission'].mean():.1%}")
print(f"{SMOKE}% with ≥1 parent: {(conv_stats['n_parents'] > 0).mean():.1%} "
      f"(mean chain length {conv_stats['n_parents'].mean():.2f})")
print(f"{SMOKE}% with ≥1 reply: {(conv_stats['n_replies'] > 0).mean():.1%} "
      f"(mean {conv_stats['n_replies'].mean():.2f})")

# %% [markdown]
# ## Temporal coverage — is author-history context viable at all?
#
# Prior-post counts per dataset row, reconstructed from `corpus-v1.jsonl`
# by `author_hash` (strictly-before rule). Guide §5 calls this the panel that
# "decides whether temporal context is viable".

# %%
from leische.contexts import TemporalIndex

tindex = TemporalIndex(corpus)
prior_counts = tindex.prior_counts(df)

fig, ax = plt.subplots(figsize=(7, 3.2))
ax.hist(prior_counts, bins=range(0, int(prior_counts.max()) + 2))
ax.set_title(f"{SMOKE}prior posts per target author (from corpus dump)")
ax.set_xlabel("prior posts before target"); ax.set_ylabel("rows")
plt.tight_layout(); plt.show()

for k in (1, 3, cfg.temporal_k):
    print(f"{SMOKE}rows with ≥{k} prior posts: {(prior_counts >= k).mean():.1%}")
print(f"{SMOKE}median prior posts: {prior_counts.median():.0f} — rows without history "
      "get the zeros/'missing' channel and the Stage-4 gate learns to suppress it (§4.2).")

# %% [markdown]
# ## Agreement statistics (echoed from dataset_card.json)
#
# These are the dataset's reliability evidence — cite them, don't recompute.

# %%
ag = card["agreement"]["labels"]
print(pd.DataFrame({
    label: {"fleiss_kappa": v["fleiss_kappa"], "krippendorff_alpha": v["krippendorff_alpha"]}
    for label, v in ag.items()
}).T)
gold = card["agreement"]["gold_vs_ensemble_cohen_kappa"]
print(f"\ngold subset: n={gold['n_gold_items']} → human-vs-ensemble κ not yet available "
      "(gate criterion #3)")
print(f"fastText-LID vs ensemble language agreement: {card['agreement']['lid_vs_ensemble_language']['agreement']:.2f}")

# %% [markdown]
# ## Manual read: every sarcastic row with all three annotator rationales
#
# Guide §5 asks for a 20-row read; the pilot has only 8 sarcastic rows, so all
# of them are shown (SMOKE).

# %%
for _, r in df[sarc].iterrows():
    labels = r["labels"]
    print("=" * 100)
    print(f"[{r['reddit_fullname']}] lang={labels['language']} "
          f"literal={labels['literal_sentiment']} intended={labels['intended_sentiment']} "
          f"votes={r['reliability']['sarcasm_votes']} resolved={r['reliability']['resolved_by']}")
    print(f"TEXT: {r['text'][:300]}")
    cues = [k for k, v in labels["cues"].items() if v]
    print(f"CUES: {', '.join(cues) or '—'}")
    for a in r["reliability"]["annotators"]:
        print(f"  {a['model_key']:<8} sarcastic={a['sarcastic']} conf={a['confidence']:.2f} | {a['rationale']}")

# %% [markdown]
# ## Freeze the fold assignment for this dataset version (§7.1)
#
# `StratifiedGroupKFold(5)` on `sarcastic×language`, grouped by
# `submission_fullname`, val carved from the train side with the same grouping.
# Saved once to `results/folds-v1.json`; every experiment reuses it.

# %%
import warnings

if cfg.folds_file().exists():
    folds = D.load_folds(cfg.folds_file(), card)
    print(f"frozen folds already exist and match this export: {cfg.folds_file()}")
else:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folds = D.make_folds(df, n_splits=cfg.n_folds, seed=13, smoke=True)
    for w in caught:
        print(f"WARNING: {w.message}")
    D.freeze_folds(folds, card, cfg.folds_file(), n_splits=cfg.n_folds, seed=13)
    print(f"froze {len(folds)} folds → {cfg.folds_file()}")

fold_pos = [
    {"fold": f["fold"],
     **{p: f"{len(f[p])} rows / "
           f"{int(sarc[df['reddit_fullname'].isin(f[p])].sum())} pos" for p in ("train", "val", "test")}}
    for f in folds
]
print(f"\n{SMOKE}fold composition (rows / sarcastic positives):")
print(pd.DataFrame(fold_pos).to_string(index=False))

# %%
rep = D.author_overlap_report(df, folds)
print(f"{SMOKE}author overlap audit (§7.1):")
print(rep.to_string(index=False))
print(f"authors appearing in >1 test fold: {rep.attrs['authors_in_multiple_test_folds']} "
      f"/ {rep.attrs['total_test_authors']} — if a few authors dominate later exports, "
      "add author_hash to the grouping key as a robustness run.")

# %% [markdown]
# ## Readiness gate verdict

# %%
print(gate.render())
