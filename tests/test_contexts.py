"""§11.1 retrieval leakage, §11.7 context drift, temporal strictly-before."""

import numpy as np
import pandas as pd
import pytest

from leische import contexts as C
from leische import data as D


# ------------------------------------------------------------- conversational


def test_conversational_uses_embedded_snapshot_only(pilot):
    # the builder's signature admits no corpus — context drift is impossible
    import inspect
    params = inspect.signature(C.build_conversational).parameters
    assert list(params) == ["row"]


def test_comment_row_starts_with_submission_item(pilot):
    row = pilot.iloc[0]  # pilot rows are all comments
    items = C.build_conversational(row)
    assert items, "comment row with a submission snapshot must yield items"
    assert items[0].role == C.ROLE_SUBMISSION
    assert row["context"]["submission"]["title"] in items[0].text


def test_conversational_order_and_roles(pilot):
    for _, row in pilot.iterrows():
        items = C.build_conversational(row)
        roles = [it.role for it in items]
        # submission (if any) first, then ancestors, then replies — never interleaved
        assert roles == sorted(roles)


# ------------------------------------------------------------- temporal


@pytest.fixture(scope="module")
def tiny_corpus():
    times = pd.to_datetime([
        "2026-08-01T00:00:00Z", "2026-08-05T00:00:00Z",
        "2026-08-10T00:00:00Z", "2026-08-20T00:00:00Z",
    ])
    return pd.DataFrame({
        "author_hash": ["a", "a", "a", "b"],
        "created_dt": times,
        "reddit_fullname": ["t1_1", "t1_2", "t1_3", "t1_4"],
        "text": ["first", "second", "third", "other author"],
    })


def test_temporal_strictly_before_and_excludes_target(tiny_corpus):
    idx = C.TemporalIndex(tiny_corpus)
    target_time = pd.Timestamp("2026-08-10T00:00:00Z")
    # target IS t1_3 (same timestamp): must exclude itself and anything not strictly before
    items = idx.history("a", target_time, "t1_3", k=10)
    names = [it.reddit_fullname for it in items]
    assert names == ["t1_2", "t1_1"]  # most recent first, t1_3 excluded
    assert all(it.delta_days > 0 for it in items)
    assert items[0].delta_days == pytest.approx(5.0)


def test_temporal_k_cap_takes_most_recent(tiny_corpus):
    idx = C.TemporalIndex(tiny_corpus)
    items = idx.history("a", pd.Timestamp("2026-08-21T00:00:00Z"), "t1_x", k=2)
    assert [it.reddit_fullname for it in items] == ["t1_3", "t1_2"]


def test_temporal_unknown_author_is_empty(tiny_corpus):
    idx = C.TemporalIndex(tiny_corpus)
    assert idx.history("nobody", pd.Timestamp("2026-08-21T00:00:00Z"), "t1_x") == []


# ------------------------------------------------------------- retrieval


@pytest.fixture(scope="module")
def synth():
    """8 rows in 4 threads; embeddings arranged so nearest neighbors are known."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(8):
        rows.append({
            "reddit_fullname": f"t1_q{i}",
            "submission_fullname": f"t3_thread{i // 2}",  # 2 rows per thread
            "text": f"text {i}",
            "labels": {"sarcastic": i % 2 == 0},
        })
    df = pd.DataFrame(rows)
    vectors = rng.normal(size=(8, 16))
    emb = C.RetrievalEmbeddings(df["reddit_fullname"].tolist(), vectors)
    return df, emb


def test_retrieval_banks_are_train_only_and_thread_safe(synth):
    df, emb = synth
    train = df["reddit_fullname"].tolist()[:6]  # threads 0..2
    queries = df["reddit_fullname"].tolist()
    res = C.build_fold_retrieval(df, emb, train, queries, k=3)
    train_set = set(train)
    thread_of = dict(zip(df["reddit_fullname"], df["submission_fullname"]))
    for q, r in res.items():
        for f in r.sarc + r.nonsarc:
            assert f in train_set
            assert thread_of[f] != thread_of[q]
            assert f != q


def test_assert_no_leakage_catches_out_of_fold(synth):
    df, emb = synth
    bad = {"t1_q0": C.RetrievalResult(sarc=["t1_q6"], nonsarc=[])}  # q6 outside train
    with pytest.raises(AssertionError, match="outside the training fold"):
        C.assert_no_leakage(bad, df, train_set={"t1_q2", "t1_q4"})


def test_assert_no_leakage_catches_same_thread(synth):
    df, emb = synth
    bad = {"t1_q0": C.RetrievalResult(sarc=["t1_q1"], nonsarc=[])}  # same thread0
    with pytest.raises(AssertionError, match="shares thread"):
        C.assert_no_leakage(bad, df, train_set={"t1_q1"})


def test_retrieval_never_returns_more_than_k(synth):
    df, emb = synth
    train = df["reddit_fullname"].tolist()[:6]
    res = C.build_fold_retrieval(df, emb, train, ["t1_q7"], k=2)
    r = res["t1_q7"]
    assert len(r.sarc) <= 2 and len(r.nonsarc) <= 2
