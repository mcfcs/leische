# %% [markdown]
# # 06 — RQ3: two-stage sentiment evaluation (guide §8) — SMOKE
#
# Ground truth: `labels.intended_sentiment`.
#
# - **Stage 1 (pre-sarcasm):** the shipped `aux.tx_sentiment` probabilities
#   (external, model-independent — the guide's primary candidate).
# - **Stage 2 (post-sarcasm):** rows the sarcasm model flags get their sentiment
#   re-predicted by a small head trained on `intended_sentiment` over the FROZEN
#   `[t ; c_fused]` features exported by notebook 04; unflagged rows keep stage 1.
#
# The interesting cell is *sarcastic ∧ literal≠intended* — where
# sarcasm-awareness must show its value. **Every number is SMOKE.**

# %%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from leische import data as D
from leische import evaluate as E
from leische.nbsupport import bootstrap

cfg, df, corpus, card, gate = bootstrap(config="configs/smoke.yaml", run_name="smoke-rq3")
SMOKE = cfg.tag()
folds = D.load_folds(cfg.folds_file(), card)

# %% [markdown]
# ## The polarity-shift landscape (why RQ3 exists)

# %%
literal = df["labels"].map(lambda l: l["literal_sentiment"])
intended = df["labels"].map(lambda l: l["intended_sentiment"])
sarc = D.sarcastic(df)
shift = literal != intended
print(f"{SMOKE}rows with literal≠intended: {shift.sum()} / {len(df)}")
print(f"{SMOKE}sarcastic ∧ literal≠intended: {(sarc & shift).sum()} rows "
      "(the cell where stage 2 must earn its keep)")
print(f"\n{SMOKE}literal → intended transitions among sarcastic rows:")
print(pd.crosstab(literal[sarc], intended[sarc]))

# %% [markdown]
# ## Stage 1 — external sentiment classifier vs intended_sentiment

# %%
stage1 = E.stage1_tx_predictions(df)
print(f"{SMOKE}stage-1 (tx_sentiment argmax) vs intended_sentiment, all 100 rows:")
print(E.rq3_report(df, stage1, tag=SMOKE + "stage1").to_string(index=False))

# %% [markdown]
# ## Stage 2 — sarcasm-aware reinterpretation (fold-0 protocol)
#
# Head trained on fold-0 TRAIN rows' frozen features only; evaluated on fold-0
# TEST rows. Sarcasm flags come from the notebook-04 smoke checkpoint.

# %%
z = np.load(cfg.cache_path() / "smoke_full_features.npz", allow_pickle=True)
feat_of = {f: i for i, f in enumerate(z["fullnames"])}
flags = dict(zip(z["fullnames"], z["pred"].astype(bool)))

fold0 = folds[0]
X = z["features"]
y_intended = dict(zip(df["reddit_fullname"], intended))

train_idx = [feat_of[f] for f in fold0["train"]]
head = LogisticRegression(max_iter=2000)
head.fit(X[train_idx], [y_intended[f] for f in fold0["train"]])
print(f"{SMOKE}stage-2 head trained on {len(train_idx)} fold-0 train rows "
      f"(classes: {list(head.classes_)})")

# %%
test_df = df[df["reddit_fullname"].isin(fold0["test"])].copy()
test_names = test_df["reddit_fullname"].tolist()
s1_test = E.stage1_tx_predictions(test_df)
flagged = pd.Series([flags[f] for f in test_names], index=test_df.index)
s2_test = pd.Series(head.predict(X[[feat_of[f] for f in test_names]]), index=test_df.index)
final = E.two_stage_predictions(test_df, s1_test, flagged, s2_test)
print(f"{SMOKE}fold-0 test: {len(test_df)} rows, {int(flagged.sum())} flagged sarcastic "
      f"by the smoke model")

# %%
print(f"{SMOKE}RQ3 comparison on fold-0 test rows (macro-F1 vs intended_sentiment):")
r1 = E.rq3_report(test_df, s1_test, tag=SMOKE + "stage1-only")
r2 = E.rq3_report(test_df, final, tag=SMOKE + "two-stage")
print(pd.concat([r1, r2], ignore_index=True).to_string(index=False))
print(f"\n{SMOKE}the pilot's fold-0 test slice holds ≤2 sarcastic rows — these tables "
      "prove the pipeline shape (per-language, per-gold-label, shift slice), nothing more.")

# %% [markdown]
# ## Calibration note (§9.8)
#
# RQ3 stage-2 quality depends on WHICH rows get flagged; a badly calibrated
# flag ruins the sentiment story. ECE + reliability curve for the smoke flags:

# %%
import matplotlib.pyplot as plt

y_true_sarc = test_df["labels"].map(lambda l: int(l["sarcastic"])).to_numpy()
probs = z["prob"][[feat_of[f] for f in test_names]]
print(f"{SMOKE}sarcasm-flag ECE on fold-0 test: {E.ece(y_true_sarc, probs):.4f}")
curve = E.reliability_curve(y_true_sarc, probs)
if len(curve):
    plt.figure(figsize=(4.5, 4))
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.plot(curve["mean_prob"], curve["frac_pos"], "o-")
    plt.xlabel("mean predicted p(sarcastic)"); plt.ylabel("empirical fraction")
    plt.title(f"{SMOKE}reliability (fold-0 test)")
    plt.tight_layout(); plt.show()
print("temperature scaling + F1-optimal thresholding exist behind config flags "
      "(temperature_scaling, tune_threshold_on_val) — evaluated post-gate.")

# %% [markdown]
# ## Verdict

# %%
print(f"{SMOKE}RQ3 pipeline exercised end-to-end: external stage-1, model-flagged "
      "stage-2 reinterpretation on frozen features, disaggregated reporting incl. "
      "the sarcastic∧shift slice. Re-run against dataset-v2 once the §10 gate passes.")
