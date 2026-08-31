# %% [markdown]
# # 02 — Context assembly (guide §6)
#
# Builds and caches the three context channels keyed by `reddit_fullname`,
# prints an audit of exactly which texts entered each channel for 5 random
# rows, and proves the retrieval leakage filter.
#
# Channel rules (§11 pitfalls):
# - **conversational** — only the embedded snapshot the annotators saw (§11.7)
# - **temporal** — corpus posts by the same author strictly before the target
# - **retrieval** — per-fold, training-rows-only, same-thread excluded (§11.1);
#   this notebook caches the frozen embeddings and demonstrates one fold's banks

# %%
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leische import contexts as C
from leische import data as D
from leische.nbsupport import bootstrap
from leische.train import Assembly

cfg, df, corpus, card, gate = bootstrap(need_corpus=True)
SMOKE = cfg.tag()
folds = D.load_folds(cfg.folds_file(), card)

# %% [markdown]
# ## Build conversational + temporal channels (fold-independent)

# %%
assembly = Assembly(cfg, df, corpus)
n_conv = sum(len(v) for v in assembly.conv.values())
n_temp = sum(len(v) for v in assembly.temp.values())
print(f"{SMOKE}conversational items: {n_conv} across {len(assembly.conv)} rows "
      f"(mean {n_conv / len(df):.2f}/row)")
print(f"{SMOKE}temporal items (K={cfg.temporal_k}): {n_temp} "
      f"(mean {n_temp / len(df):.2f}/row)")

# %% [markdown]
# ## Cache the assembled channels keyed by `reddit_fullname`
#
# JSON cache under `data/cache/` — cheap to rebuild, but caching makes the
# training notebooks' inputs inspectable on disk.

# %%
cache = {
    f: {
        "conv": [{"text": it.text, "role": it.role, "is_submitter": it.is_submitter}
                 for it in assembly.conv[f]],
        "temp": [{"text": it.text, "delta_days": it.delta_days,
                  "reddit_fullname": it.reddit_fullname}
                 for it in assembly.temp[f]],
    }
    for f in df["reddit_fullname"]
}
cache_file = cfg.cache_path() / f"contexts-{cfg.dataset_version}.json"
cache_file.write_text(json.dumps(cache), encoding="utf-8")
print(f"cached → {cache_file} ({cache_file.stat().st_size:,} bytes)")

# %% [markdown]
# ## Δt sanity: temporal deltas are positive and in days

# %%
deltas = [it.delta_days for items in assembly.temp.values() for it in items]
assert all(d > 0 for d in deltas), "strictly-before violated"
if deltas:
    plt.figure(figsize=(7, 3))
    plt.hist(deltas, bins=30)
    plt.title(f"{SMOKE}Δt of temporal items (days before target)")
    plt.xlabel("days"); plt.tight_layout(); plt.show()
    lam = cfg.temporal_lambda_init
    print(f"{SMOKE}Δt range: {min(deltas):.2f}–{max(deltas):.2f} days | "
          f"decay exp(−λΔt) at λ={lam}: median weight "
          f"{np.exp(-lam * np.median(deltas)):.3f}")

# %% [markdown]
# ## Frozen retrieval embeddings (cached once per dataset version)

# %%
emb = assembly.retrieval_embeddings()
print(f"{SMOKE}embedded {len(emb.fullnames)} targets with {cfg.retrieval_model} "
      f"→ dim {emb.vectors.shape[1]} (cache: data/cache/)")

# %% [markdown]
# ## Demonstrate one fold's retrieval banks + leakage filter (§11.1)
#
# Banks are per-fold TRAINING artifacts — this cell builds fold 0's banks the
# same way `train.py` does and runs the hard leakage assertions. The same
# checks run on synthetic data in `tests/test_contexts.py`.

# %%
fold0 = folds[0]
queries = fold0["train"] + fold0["val"] + fold0["test"]
retrieval = assembly.fold_retrieval(fold0["train"], queries)  # asserts internally
print(f"leakage assertions PASSED for {len(retrieval)} queries "
      f"(banks ⊆ {len(fold0['train'])} train rows, same-thread excluded, k={cfg.retrieval_k})")

train_pos = int(D.sarcastic(df[df["reddit_fullname"].isin(fold0["train"])]).sum())
sizes = pd.DataFrame([{"sarc_bank_hits": len(r.sarc), "nonsarc_bank_hits": len(r.nonsarc)}
                      for r in retrieval.values()])
print(f"\n{SMOKE}fold-0 train bank: {train_pos} sarcastic / "
      f"{len(fold0['train']) - train_pos} non-sarcastic rows")
print(sizes.describe().loc[["mean", "min", "max"]])
print(f"{SMOKE}NOTE: with only {train_pos} sarcastic rows in the fold-0 bank, the "
      "sarcastic side retrieves fewer than k exemplars — expected on the pilot; "
      "the attention mask handles short banks.")

# %%
# show the filter doing real work: does any query's nearest neighbor share its thread?
thread_of = dict(zip(df["reddit_fullname"], df["submission_fullname"]))
multi = df.groupby("submission_fullname")["reddit_fullname"].apply(list)
multi = multi[multi.map(len) > 1]
demo = None
for thread, members in multi.items():
    in_train = [m for m in members if m in set(fold0["train"])]
    in_any = [m for m in members if m in retrieval]
    if in_train and in_any:
        q = in_any[0]
        mates = [m for m in members if m != q]
        retrieved = set(retrieval[q].sarc + retrieval[q].nonsarc)
        assert not (retrieved & set(mates)), "thread-mate leaked into retrieval!"
        demo = (q, mates)
        break
if demo:
    print(f"filter demo: query {demo[0]} has {len(demo[1])} thread-mate(s) in the dataset "
          f"({demo[1]}) — none were retrievable, none retrieved. PASS")
else:
    print("no thread with ≥2 dataset rows intersects fold-0 queries+train — filter "
          "exercised on synthetic threads in tests/test_contexts.py instead.")
print(f"({SMOKE.strip() or 'REAL'} data note: pilot threads rarely contribute >1 row)")

# %% [markdown]
# ## Audit: exactly which texts entered each channel (5 random rows)

# %%
rng = np.random.default_rng(13)
rows_dict = D.rows_by_fullname(df)
audited = [f for f in rng.choice(list(retrieval.keys()), size=5, replace=False)]
for f in audited:
    row = rows_dict[f]
    print(C.audit_context(row, assembly.conv[f], assembly.temp[f], retrieval[f], df))
    print()
