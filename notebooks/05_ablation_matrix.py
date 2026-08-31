# %% [markdown]
# # 05 — Ablation matrix (RQ2, guide §7.3) — SMOKE dry run
#
# The full 8-condition × 5-fold harness at ultra-tiny settings (1 seed, 1 epoch,
# 6 steps/epoch, short sequences). This proves the loop, the per-condition
# artifacts, and the significance tests end-to-end. **Every number is SMOKE** —
# with ~1–2 test positives per fold the metrics are noise by construction (§11.6).
#
# Disabled channels are REMOVED from the gate softmax (renormalized), not
# zero-filled — `model.py` builds the gate only over active channels.

# %%
import time

import pandas as pd

from leische import data as D
from leische import evaluate as E
from leische import train as T
from leische.config import LeischeConfig, find_repo_root
from leische.nbsupport import bootstrap

cfg0, df, corpus, card, gate = bootstrap(config="configs/ablation_smoke.yaml", need_corpus=True)
SMOKE = cfg0.tag()
folds = D.load_folds(cfg0.folds_file(), card)
root = find_repo_root()

# the §7.3 matrix
CONDITIONS = [
    ("1_baseline", False, False, False),
    ("2_conv", True, False, False),
    ("3_temp", False, True, False),
    ("4_ret", False, False, True),
    ("5_conv_temp", True, True, False),
    ("6_conv_ret", True, False, True),
    ("7_temp_ret", False, True, True),
    ("8_full", True, True, True),
]

# %% [markdown]
# ## Run all 8 conditions × 5 folds (× 1 seed, tiny settings)

# %%
results = {}
for name, use_conv, use_temp, use_ret in CONDITIONS:
    t0 = time.time()
    cfg = LeischeConfig.from_yaml(root / "configs/ablation_smoke.yaml",
                                  use_conv=use_conv, use_temp=use_temp, use_ret=use_ret,
                                  run_name=f"ablation-smoke-{name}")
    results[name] = T.run_cv(cfg, df, corpus, card, folds, verbose=False)
    fm = results[name]["fold_metrics"]
    print(f"{SMOKE}{name:<12} conv={int(use_conv)} temp={int(use_temp)} ret={int(use_ret)} "
          f"| mean F1 {fm['f1'].mean():.3f} | {time.time() - t0:5.1f}s "
          f"({len(fm)} fold-runs)")

# %% [markdown]
# ## Ablation table (mean ± std across folds)

# %%
table = E.ablation_table({n: r["fold_metrics"] for n, r in results.items()})
print(f"{SMOKE}RQ2 ablation matrix — HARNESS CHECK ONLY, pilot metrics are noise:")
print(table.to_string(index=False))
csv_path = cfg0.results_path() / "SMOKE-ablation-matrix.csv"
table.to_csv(csv_path, index=False)
print(f"\nsaved → {csv_path}")

# %% [markdown]
# ## Significance tests: condition 8 (full) vs condition 1 (baseline)
#
# Paired bootstrap + approximate randomization on shared test predictions,
# plus McNemar for the headline pair (guide §7.3). Wired now, meaningful
# only post-gate.

# %%
pred_base = results["1_baseline"]["predictions"]
pred_full = results["8_full"]["predictions"]

boot = E.paired_bootstrap(pred_base, pred_full, n_boot=1000, seed=13)
print(f"{SMOKE}paired bootstrap ΔF1 (full − baseline): {boot['observed_delta_f1']:+.4f} "
      f"CI95 [{boot['ci95'][0]:+.4f}, {boot['ci95'][1]:+.4f}] p={boot['p_two_sided']:.3f} "
      f"(n={boot['n_rows']})")

ar = E.approximate_randomization(pred_base, pred_full, n_iter=1000, seed=13)
print(f"{SMOKE}approximate randomization |ΔF1|={ar['observed_abs_delta_f1']:.4f} "
      f"p={ar['p_value']:.3f}")

mc = E.mcnemar_test(pred_base, pred_full)
print(f"{SMOKE}McNemar (correct/incorrect discordants {mc['table'][0][1]} vs "
      f"{mc['table'][1][0]}): p={mc['p_value']:.3f}")
print(f"\n{SMOKE}with ~8 positives these p-values are definitionally meaningless — "
      "the deliverable is that the machinery runs.")

# %% [markdown]
# ## §7.2 hygiene + diagnostics for the full condition

# %%
print(f"{SMOKE}natural-only vs all rows (condition 8):")
print(E.natural_and_all(pred_full).to_string(index=False))
print("\npilot has 0 keyword_oversampled rows → the two slices coincide; the "
      "filter path is what's being exercised.")
print(f"\n{SMOKE}condition-8 F1 by resolved_by (is the model only right on easy "
      "unanimous items? — §7.4):")
print(E.slice_metrics(pred_full, "resolved_by").to_string(index=False))
print(f"\n{SMOKE}condition-8 mean gates by language:")
print(E.gate_summary(pred_full, by="language").to_string(index=False))
