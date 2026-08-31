"""Metric safety on degenerate pilot slices."""

import math

import numpy as np
import pandas as pd

from leische import evaluate as E


def test_sentiment_metrics_empty_slice_is_nan_not_crash():
    # regression: fold-0 test has 0 sarcastic rows → gold_sarcastic=True slice empty
    m = E.sentiment_metrics(pd.Series([], dtype=object), pd.Series([], dtype=object))
    assert m["n"] == 0
    assert math.isnan(m["macro_f1"]) and math.isnan(m["accuracy"])


def test_safe_metrics_zero_positive_fold():
    y = np.zeros(10, dtype=int)
    m = E.safe_metrics(y, y)
    assert m["f1"] == 0.0 and m["n_pos"] == 0 and m["accuracy"] == 1.0


def test_ece_empty_is_nan():
    assert math.isnan(E.ece(np.array([]), np.array([])))


def test_rq3_report_handles_empty_shift_slice(pilot):
    sub = pilot[~pilot["labels"].map(lambda l: l["sarcastic"])].head(10)
    stage1 = E.stage1_tx_predictions(sub)
    rep = E.rq3_report(sub, stage1)
    row = rep[rep["slice"] == "sarcastic ∧ literal≠intended"].iloc[0]
    assert row["n"] == 0  # no sarcastic rows in this subset — must not raise
