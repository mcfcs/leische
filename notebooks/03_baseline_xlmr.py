# %% [markdown]
# # 03 — Context-agnostic XLM-R baseline (RQ1) + the two §10 smoke tests
#
# The baseline is `ContextAwareSarcasmModel` with all three channels OFF —
# same class as the full model, so the RQ1 comparison can never drift.
#
# Smoke tests required by guide §10 before the data gate is met:
# 1. **overfit-16** — drive training loss to ~0 on 16 rows (architecture sanity)
# 2. **tiny-settings 5-fold dry run** — prove the fold harness end-to-end
#
# **Every number below is SMOKE** — tiny settings, pilot data, meaningless folds.

# %%
import matplotlib.pyplot as plt
import torch

from leische import data as D
from leische import evaluate as E
from leische import train as T
from leische.nbsupport import bootstrap

cfg, df, corpus, card, gate = bootstrap(config="configs/smoke.yaml", run_name="smoke-baseline")
SMOKE = cfg.tag()
folds = D.load_folds(cfg.folds_file(), card)
device = torch.device("cuda")

# %% [markdown]
# ## Smoke test 1 — overfit 16 rows to ~zero loss
#
# Includes all 8 pilot positives + 8 negatives. A healthy architecture must be
# able to memorize 16 rows; failure here means wiring bugs, not data problems.

# %%
overfit = T.overfit_smoke(cfg, df, None, card, n=16, max_steps=150, target_loss=0.05)
plt.figure(figsize=(7, 3))
plt.plot(overfit["losses"])
plt.axhline(0.05, color="r", ls="--", label="target 0.05")
plt.title(f"{SMOKE}overfit-16 loss (baseline: target-only)")
plt.xlabel("step"); plt.ylabel("loss"); plt.legend(); plt.tight_layout(); plt.show()
assert overfit["passed"], "baseline failed to overfit 16 rows — architecture bug"

# %% [markdown]
# ## Mean vs CLS pooling (guide §4.1 asks to verify both)
#
# One tiny fold-0 run per pooling mode. At SMOKE scale this only proves both
# paths execute — the real comparison happens post-gate.

# %%
from leische.config import LeischeConfig, find_repo_root

root = find_repo_root()
pooling_results = {}
for pooling in ("mean", "cls"):
    cfg_p = LeischeConfig.from_yaml(root / "configs/smoke.yaml", pooling=pooling,
                                    run_name=f"smoke-pooling-{pooling}")
    res = T.run_cv(cfg_p, df, None, card, folds[:1], verbose=False)
    m = res["fold_metrics"].iloc[0]
    pooling_results[pooling] = m
    print(f"{SMOKE}pooling={pooling}: fold-0 test F1={m['f1']:.4f} "
          f"(n_pos={int(m['n_pos'])}), best val F1={m['best_val_f1']:.4f}")
print(f"{SMOKE}both pooling paths run; thesis default stays pooling=mean.")

# %% [markdown]
# ## Smoke test 2 — tiny-settings 5-fold dry run (baseline)
#
# Full harness: frozen folds → per-fold class weights → early stopping →
# per-fold metrics → natural-only hygiene split → disaggregation.

# %%
res = T.run_cv(cfg, df, None, card, folds, verbose=True)
fm, preds = res["fold_metrics"], res["predictions"]

# %%
print(f"{SMOKE}5-fold dry run, mean ± std across folds (1 seed, tiny settings):")
print(E.fold_summary(fm).to_string(index=False))
print(f"\n{SMOKE}§7.2 hygiene split (pilot has no keyword_oversampled rows, so "
      "natural-only == all — the code path is what's being exercised):")
print(E.natural_and_all(preds).to_string(index=False))

# %%
print(f"{SMOKE}F1 by language (several test folds hold 0 positives — that is the "
      "pilot's stratification failure, reported not hidden):")
print(E.slice_metrics(preds, "language").to_string(index=False))
print(f"\n{SMOKE}confusion matrix (rows=true, cols=pred):")
print(E.confusion(preds))
print(f"\n{SMOKE}calibration ECE (10 bins): {E.ece(preds['y_true'].to_numpy(), preds['prob'].to_numpy()):.4f}")

# %% [markdown]
# ## Verdict
#
# Both §10 smoke tests pass ⇒ the baseline harness is one config change away
# from real training (`dataset_version: v2`, `smoke: false`) once the gate is met.

# %%
print(f"{SMOKE}smoke test 1 (overfit-16): PASS")
print(f"{SMOKE}smoke test 2 (5-fold dry run): PASS — {len(fm)} fold-runs completed, "
      f"artifacts in results/{cfg.run_name}/")
