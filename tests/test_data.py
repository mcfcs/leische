"""Contract validation, hygiene masks, and the §10 readiness gate."""

import pandas as pd
import pytest

from leische import data as D


def test_pilot_passes_contract_validation(pilot):
    assert len(pilot) == 100
    assert pilot["reddit_fullname"].is_unique


def test_null_sampling_strategy_counts_as_natural(pilot):
    # pilot export ships null sampling_strategy; §7.2 mask must keep those rows
    assert D.natural_mask(pilot).all()


def test_natural_mask_excludes_keyword_oversampled(pilot):
    df = pilot.copy()
    df.loc[df.index[:3], "sampling_strategy"] = "keyword_oversampled"
    mask = D.natural_mask(df)
    assert (~mask).sum() == 3


def test_readiness_gate_fails_on_pilot(card, pilot):
    gate = D.readiness_gate(card, pilot)
    assert not gate.passed  # v1 pilot: 8 positives, no gold subset
    failed = {c.name for c in gate.checks if not c.passed}
    assert "≥400 sarcastic positives" in failed
    assert "gold subset labeled + κ reported" in failed


def test_validation_rejects_broken_rows():
    import copy
    import json
    from pathlib import Path

    repo = Path(__file__).resolve().parents[1]
    raw = json.loads((repo / "data" / "dataset-v1.jsonl").read_text(encoding="utf-8").splitlines()[0])
    broken = copy.deepcopy(raw)
    broken["labels"]["language"] = "klingon"
    problems = D.validate_rows([broken])
    assert any("language" in p for p in problems)
    assert not D.validate_rows([raw])


def test_class_weights_inverse_frequency(pilot):
    w_neg, w_pos = D.class_weights(pilot)
    # 92 neg / 8 pos → positive weight is the large one
    assert w_pos > w_neg
    assert w_pos == pytest.approx(100 / (2 * 8))


def test_class_weights_degenerate_fold_falls_back():
    df = pd.DataFrame({
        "reddit_fullname": ["a", "b"],
        "labels": [{"sarcastic": False}, {"sarcastic": False}],
    })
    with pytest.warns(UserWarning):
        assert D.class_weights(df) == (1.0, 1.0)


def test_smoke_false_run_refused_on_pilot(cfg, card, pilot):
    """§10 hard constraint: non-smoke training must raise while the gate fails."""
    from leische import train as T

    bad_cfg = type(cfg)(**{**cfg.to_dict(), "smoke": False})
    with pytest.raises(RuntimeError, match="readiness gate FAILED"):
        T.enforce_gate(bad_cfg, card, pilot)
