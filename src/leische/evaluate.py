"""Metrics, ablation tables, significance tests, RQ3 evaluation (guide §7–§8).

Everything here is positive-count-aware: pilot folds can hold 0 sarcastic
rows, so every metric reports n_pos and uses zero_division=0 instead of
crashing or silently emitting misleading 0/1 values.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)

SENT_ORDER = ("positive", "neutral", "negative")


# ------------------------------------------------------------------ core metrics


def safe_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)) if len(y_true) else float("nan"),
            "n": int(len(y_true)),
            "n_pos": int(np.sum(y_true)),
        }


def natural_and_all(pred_df: pd.DataFrame) -> pd.DataFrame:
    """§7.2 hygiene: main metrics on natural-only rows, all-rows separately."""
    rows = []
    for name, sub in (("natural_only", pred_df[pred_df["natural"]]), ("all_rows", pred_df)):
        rows.append({"slice": name,
                     **safe_metrics(sub["y_true"].to_numpy(), sub["pred"].to_numpy())})
    return pd.DataFrame(rows)


def slice_metrics(pred_df: pd.DataFrame, by: str) -> pd.DataFrame:
    """F1 etc. disaggregated by a metadata column (language/record_type/resolved_by)."""
    rows = []
    for value, sub in pred_df.groupby(by):
        rows.append({by: value,
                     **safe_metrics(sub["y_true"].to_numpy(), sub["pred"].to_numpy())})
    return pd.DataFrame(rows)


def fold_summary(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    """mean ± std across folds × seeds (§11.6: never report single-run F1)."""
    cols = ["f1", "precision", "recall", "accuracy"]
    agg = fold_metrics[cols].agg(["mean", "std"])
    out = pd.DataFrame({
        c: [f"{agg.loc['mean', c]:.4f} ± {0.0 if np.isnan(agg.loc['std', c]) else agg.loc['std', c]:.4f}"]
        for c in cols
    })
    out.insert(0, "runs", len(fold_metrics))
    return out


def confusion(pred_df: pd.DataFrame) -> np.ndarray:
    return confusion_matrix(pred_df["y_true"], pred_df["pred"], labels=[0, 1])


def gate_summary(pred_df: pd.DataFrame, by: str | None = None) -> pd.DataFrame:
    """Mean per-instance gate values (thesis-discussion material, guide §4.4)."""
    gate_cols = [c for c in pred_df.columns if c.startswith("gate_")]
    if not gate_cols:
        return pd.DataFrame()
    if by is None:
        return pred_df[gate_cols].describe().loc[["mean", "std"]]
    return pred_df.groupby(by)[gate_cols].mean().reset_index()


# ------------------------------------------------------------------ calibration (§9.8 / RQ3)


def ece(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total, err = len(prob), 0.0
    if total == 0:
        return float("nan")
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.sum() == 0:
            continue
        err += (mask.sum() / total) * abs(prob[mask].mean() - y_true[mask].mean())
    return float(err)


def reliability_curve(y_true: np.ndarray, prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (prob >= lo) & (prob < hi if hi < 1 else prob <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"bin_mid": (lo + hi) / 2, "mean_prob": float(prob[mask].mean()),
                     "frac_pos": float(y_true[mask].mean()), "n": int(mask.sum())})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ significance (§7.3)


def _aligned(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> pd.DataFrame:
    """Align two prediction frames on (reddit_fullname, seed) shared rows."""
    keys = ["reddit_fullname", "seed"] if "seed" in pred_a.columns else ["reddit_fullname"]
    m = pred_a[keys + ["y_true", "pred"]].merge(
        pred_b[keys + ["pred"]], on=keys, suffixes=("_a", "_b"))
    return m


def paired_bootstrap(pred_a: pd.DataFrame, pred_b: pd.DataFrame,
                     n_boot: int = 2000, seed: int = 13) -> dict:
    """Bootstrap the F1 difference (B − A) over shared test rows."""
    m = _aligned(pred_a, pred_b)
    rng = np.random.default_rng(seed)
    y, a, b = m["y_true"].to_numpy(), m["pred_a"].to_numpy(), m["pred_b"].to_numpy()
    n = len(y)
    diffs = np.empty(n_boot)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            diffs[i] = (f1_score(y[idx], b[idx], zero_division=0)
                        - f1_score(y[idx], a[idx], zero_division=0))
    observed = float(f1_score(y, b, zero_division=0) - f1_score(y, a, zero_division=0))
    return {
        "observed_delta_f1": observed,
        "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
        "p_two_sided": float(min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))),
        "n_rows": n,
    }


def approximate_randomization(pred_a: pd.DataFrame, pred_b: pd.DataFrame,
                              n_iter: int = 2000, seed: int = 13) -> dict:
    """Swap A/B predictions per row with p=0.5; p-value of |ΔF1| under the null."""
    m = _aligned(pred_a, pred_b)
    rng = np.random.default_rng(seed)
    y, a, b = m["y_true"].to_numpy(), m["pred_a"].to_numpy(), m["pred_b"].to_numpy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observed = abs(f1_score(y, b, zero_division=0) - f1_score(y, a, zero_division=0))
        count = 0
        for _ in range(n_iter):
            swap = rng.random(len(y)) < 0.5
            a2, b2 = np.where(swap, b, a), np.where(swap, a, b)
            d = abs(f1_score(y, b2, zero_division=0) - f1_score(y, a2, zero_division=0))
            if d >= observed - 1e-12:
                count += 1
    return {"observed_abs_delta_f1": float(observed),
            "p_value": float((count + 1) / (n_iter + 1)), "n_rows": len(y)}


def mcnemar_test(pred_a: pd.DataFrame, pred_b: pd.DataFrame) -> dict:
    """McNemar on the discordant-correctness table (headline pair, guide §7.3)."""
    from statsmodels.stats.contingency_tables import mcnemar

    m = _aligned(pred_a, pred_b)
    ok_a = (m["pred_a"] == m["y_true"]).to_numpy()
    ok_b = (m["pred_b"] == m["y_true"]).to_numpy()
    table = np.array([
        [int((ok_a & ok_b).sum()), int((ok_a & ~ok_b).sum())],
        [int((~ok_a & ok_b).sum()), int((~ok_a & ~ok_b).sum())],
    ])
    res = mcnemar(table, exact=True)
    return {"table": table.tolist(), "statistic": float(res.statistic),
            "p_value": float(res.pvalue)}


def ablation_table(per_condition: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """condition name → fold_metrics frame ⇒ mean±std table for the 8 conditions."""
    rows = []
    for name, fm in per_condition.items():
        row = {"condition": name, "runs": len(fm)}
        for c in ("f1", "precision", "recall", "accuracy"):
            std = fm[c].std()
            row[c] = f"{fm[c].mean():.4f} ± {0.0 if np.isnan(std) else std:.4f}"
        row["mean_test_pos"] = float(fm["n_pos"].mean())
        rows.append(row)
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ RQ3 (§8)


def sentiment_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict:
    if len(y_true) == 0:  # pilot slices (e.g. sarcastic∧shift on a tiny fold) can be empty
        return {"macro_f1": float("nan"), "accuracy": float("nan"), "n": 0}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "macro_f1": float(f1_score(y_true, y_pred, labels=list(SENT_ORDER),
                                       average="macro", zero_division=0)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "n": int(len(y_true)),
        }


def stage1_tx_predictions(df: pd.DataFrame) -> pd.Series:
    """Pre-sarcasm sentiment: argmax of the shipped aux.tx_sentiment probs.

    External and model-independent — the guide's primary stage-1 candidate.
    """
    def argmax(aux: dict) -> str:
        tx = aux["tx_sentiment"]
        probs = {"positive": tx["p_pos"], "neutral": tx["p_neu"], "negative": tx["p_neg"]}
        return max(probs, key=probs.get)

    return df["aux"].map(argmax)


def two_stage_predictions(df: pd.DataFrame, stage1: pd.Series,
                          flagged: pd.Series, stage2: pd.Series) -> pd.Series:
    """Final RQ3 predictions: sarcasm-flagged rows take stage-2, rest keep stage-1."""
    return stage2.where(flagged, stage1)


def rq3_report(df: pd.DataFrame, final_pred: pd.Series, tag: str = "") -> pd.DataFrame:
    """Macro-F1/accuracy vs intended_sentiment, disaggregated per guide §8."""
    y_true = df["labels"].map(lambda l: l["intended_sentiment"])
    lang = df["labels"].map(lambda l: l["language"])
    sarc = df["labels"].map(lambda l: l["sarcastic"])
    literal = df["labels"].map(lambda l: l["literal_sentiment"])

    rows = [{"slice": "overall", **sentiment_metrics(y_true, final_pred)}]
    for lg in sorted(lang.unique()):
        m = lang == lg
        rows.append({"slice": f"language={lg}", **sentiment_metrics(y_true[m], final_pred[m])})
    for s in (False, True):
        m = sarc == s
        rows.append({"slice": f"gold_sarcastic={s}", **sentiment_metrics(y_true[m], final_pred[m])})
    # the cell where sarcasm-awareness must show its value
    m = sarc & (literal != y_true)
    rows.append({"slice": "sarcastic ∧ literal≠intended", **sentiment_metrics(y_true[m], final_pred[m])})

    out = pd.DataFrame(rows)
    if tag:
        out.insert(0, "run", tag.strip())
    return out
