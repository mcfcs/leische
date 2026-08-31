"""§11.2 thread leakage + §11.8 version mixing as executable properties."""

import copy

import pytest

from leische import data as D


@pytest.fixture(scope="module")
def folds(pilot):
    with pytest.warns(UserWarning):  # pilot folds legitimately lack positives
        return D.make_folds(pilot, n_splits=5, seed=13, smoke=True)


def test_no_thread_spans_splits(pilot, folds):
    thread_of = dict(zip(pilot["reddit_fullname"], pilot["submission_fullname"]))
    for fold in folds:
        parts = {p: {thread_of[f] for f in fold[p]} for p in ("train", "val", "test")}
        assert not parts["train"] & parts["test"]
        assert not parts["train"] & parts["val"]
        assert not parts["val"] & parts["test"]


def test_every_row_in_exactly_one_test_fold(pilot, folds):
    all_test = [f for fold in folds for f in fold["test"]]
    assert sorted(all_test) == sorted(pilot["reddit_fullname"])


def test_folds_are_deterministic(pilot, folds):
    with pytest.warns(UserWarning):
        again = D.make_folds(pilot, n_splits=5, seed=13, smoke=True)
    assert folds == again


def test_partitions_are_disjoint_and_complete(pilot, folds):
    for fold in folds:
        ids = fold["train"] + fold["val"] + fold["test"]
        assert len(ids) == len(set(ids)) == len(pilot)


def test_freeze_load_roundtrip_and_identity_check(pilot, card, folds, tmp_path):
    path = tmp_path / "folds-v1.json"
    D.freeze_folds(folds, card, path, n_splits=5, seed=13)
    assert D.load_folds(path, card) == folds

    other = copy.deepcopy(card)
    other["dataset_version"] = "v2"
    with pytest.raises(ValueError, match="regenerate folds"):
        D.load_folds(path, other)  # §11.8: never mix v1 folds with a v2 export


def test_nonsmoke_folds_reject_positive_free_splits(pilot):
    with pytest.raises(ValueError, match="0 sarcastic positives"):
        D.make_folds(pilot, n_splits=5, seed=13, smoke=False)
