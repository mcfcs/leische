"""Assembly of the three context channels (guide §4.2–§4.3, §6).

Hard rules enforced here:
  - Conversational context comes ONLY from the row's embedded `context`
    snapshot (§11.7) — these functions never see the corpus dump.
  - Temporal history: corpus posts by the same author_hash strictly BEFORE
    the target's timestamp, target itself excluded.
  - Retrieval banks: training-fold rows only, same-thread neighbors excluded
    (§11.1); assert_no_leakage() is called by both tests and notebook 02.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ROLE_SUBMISSION, ROLE_ANCESTOR, ROLE_REPLY = 0, 1, 2
ROLE_NAMES = {ROLE_SUBMISSION: "submission", ROLE_ANCESTOR: "ancestor", ROLE_REPLY: "reply"}


# --------------------------------------------------------------------- conversational


@dataclass
class ConvItem:
    text: str
    role: int  # 0 submission, 1 ancestor, 2 reply
    is_submitter: bool


def build_conversational(row: dict | pd.Series) -> list[ConvItem]:
    """Ordered [submission, parent_1..P (oldest→newest), reply_1..R].

    Uses only the embedded snapshot the annotators saw. For submission targets
    context.submission is null (the target IS the submission) and replies hold
    its first top-level comments.
    """
    ctx = row["context"]
    items: list[ConvItem] = []

    sub = ctx.get("submission")
    if sub is not None:
        text = "\n\n".join(t for t in (sub.get("title"), sub.get("selftext")) if t)
        if text.strip():
            items.append(ConvItem(text=text, role=ROLE_SUBMISSION, is_submitter=True))

    for p in ctx.get("parent_chain") or []:
        if (p.get("text") or "").strip():
            items.append(ConvItem(text=p["text"], role=ROLE_ANCESTOR,
                                  is_submitter=bool(p.get("is_submitter"))))

    for rep in ctx.get("replies") or []:
        if (rep.get("text") or "").strip():
            items.append(ConvItem(text=rep["text"], role=ROLE_REPLY,
                                  is_submitter=bool(rep.get("is_submitter"))))
    return items


# --------------------------------------------------------------------- temporal


@dataclass
class TemporalItem:
    text: str
    delta_days: float  # time before the target, in days (Δt of guide §4.2)
    reddit_fullname: str


class TemporalIndex:
    """Author-history lookup over the corpus dump (contract §6)."""

    def __init__(self, corpus: pd.DataFrame):
        cols = corpus[["author_hash", "created_dt", "reddit_fullname", "text"]]
        self._by_author: dict[str, list[tuple]] = {}
        for author, grp in cols.groupby("author_hash", sort=False):
            grp = grp.sort_values("created_dt")
            self._by_author[author] = list(
                zip(grp["created_dt"], grp["reddit_fullname"], grp["text"])
            )

    def history(self, author_hash: str, target_created_dt, target_fullname: str,
                k: int = 10) -> list[TemporalItem]:
        """Up to k most-recent posts strictly before the target (target excluded)."""
        posts = self._by_author.get(author_hash, [])
        prior = [
            (dt, fid, text)
            for dt, fid, text in posts
            if dt < target_created_dt and fid != target_fullname and (text or "").strip()
        ]
        prior = prior[-k:]  # k most recent
        items = [
            TemporalItem(
                text=text,
                delta_days=float((target_created_dt - dt).total_seconds()) / 86400.0,
                reddit_fullname=fid,
            )
            for dt, fid, text in reversed(prior)  # most recent first
        ]
        return items

    def prior_counts(self, df: pd.DataFrame) -> pd.Series:
        """Prior-post count per dataset row (EDA temporal-viability panel)."""
        return df.apply(
            lambda r: len(self.history(r["author_hash"], r["created_dt"],
                                       r["reddit_fullname"], k=10**6)),
            axis=1,
        )


# --------------------------------------------------------------------- retrieval


class RetrievalEmbeddings:
    """Frozen sentence embeddings for all dataset targets, cached on disk.

    Cache key = dataset_version + model name; embeddings never change during
    training (retrieval similarity is frozen; the shared encoder re-encodes the
    retrieved TEXTS trainably — guide §4.3 step 2).
    """

    def __init__(self, fullnames: list[str], vectors: np.ndarray):
        assert len(fullnames) == len(vectors)
        self.fullnames = list(fullnames)
        self.vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        self.index = {f: i for i, f in enumerate(self.fullnames)}

    @classmethod
    def build(cls, df: pd.DataFrame, model_name: str, cache_dir: Path,
              dataset_version: str) -> "RetrievalEmbeddings":
        safe_model = model_name.replace("/", "__")
        cache = cache_dir / f"retrieval-{dataset_version}-{safe_model}.npz"
        fullnames = df["reddit_fullname"].tolist()
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            cached_names = z["fullnames"].tolist()
            if cached_names == fullnames:
                return cls(cached_names, z["vectors"])
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        vectors = model.encode(df["text"].tolist(), batch_size=64,
                               show_progress_bar=False, convert_to_numpy=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, fullnames=np.array(fullnames, dtype=object), vectors=vectors)
        return cls(fullnames, vectors)


@dataclass
class RetrievalResult:
    sarc: list[str]     # reddit_fullnames of retrieved sarcastic exemplars
    nonsarc: list[str]  # reddit_fullnames of retrieved non-sarcastic exemplars


def build_fold_retrieval(
    df: pd.DataFrame,
    emb: RetrievalEmbeddings,
    train_fullnames: list[str],
    query_fullnames: list[str],
    k: int = 5,
) -> dict[str, RetrievalResult]:
    """Top-k sarcastic + top-k non-sarcastic exemplars per query row.

    Banks contain ONLY training-fold rows; neighbors sharing the query's
    submission_fullname are excluded (§11.1). Rebuilt per fold.
    """
    thread_of = dict(zip(df["reddit_fullname"], df["submission_fullname"]))
    label_of = dict(zip(df["reddit_fullname"], df["labels"].map(lambda l: bool(l["sarcastic"]))))

    train_set = set(train_fullnames)
    banks = {True: [], False: []}
    for f in train_fullnames:
        banks[label_of[f]].append(f)

    out: dict[str, RetrievalResult] = {}
    for q in query_fullnames:
        q_vec = emb.vectors[emb.index[q]]
        q_thread = thread_of[q]
        picked: dict[bool, list[str]] = {}
        for lab, bank in banks.items():
            cands = [f for f in bank if f != q and thread_of[f] != q_thread]
            if not cands:
                picked[lab] = []
                continue
            mat = emb.vectors[[emb.index[f] for f in cands]]
            sims = mat @ q_vec
            top = np.argsort(-sims)[:k]
            picked[lab] = [cands[i] for i in top]
        out[q] = RetrievalResult(sarc=picked[True], nonsarc=picked[False])

    assert_no_leakage(out, df, train_set)
    return out


def assert_no_leakage(
    retrieval: dict[str, RetrievalResult],
    df: pd.DataFrame,
    train_set: set[str],
) -> None:
    """§11.1 hard checks: bank ⊆ training fold, never the query's own thread."""
    thread_of = dict(zip(df["reddit_fullname"], df["submission_fullname"]))
    for q, res in retrieval.items():
        for f in res.sarc + res.nonsarc:
            if f not in train_set:
                raise AssertionError(f"retrieval leakage: {f} retrieved for {q} is outside the training fold")
            if thread_of[f] == thread_of[q]:
                raise AssertionError(f"retrieval leakage: {f} shares thread {thread_of[q]} with query {q}")
            if f == q:
                raise AssertionError(f"retrieval leakage: query {q} retrieved itself")


# --------------------------------------------------------------------- audit (notebook 02)


def audit_context(
    row: pd.Series,
    conv: list[ConvItem],
    temp: list[TemporalItem],
    ret: RetrievalResult | None,
    df: pd.DataFrame,
    max_chars: int = 110,
) -> str:
    """Human-readable dump of exactly which texts entered each channel."""

    def clip(t: str) -> str:
        t = " ".join(t.split())
        return t[:max_chars] + ("…" if len(t) > max_chars else "")

    text_of = dict(zip(df["reddit_fullname"], df["text"]))
    lines = [
        f"=== {row['reddit_fullname']} ({row['record_type']}, {row['labels']['language']}, "
        f"sarcastic={row['labels']['sarcastic']}) ===",
        f"TARGET: {clip(row['text'])}",
        f"[conversational] {len(conv)} item(s):",
    ]
    for i, it in enumerate(conv):
        lines.append(f"  {i}. role={ROLE_NAMES[it.role]:<10} is_submitter={it.is_submitter} | {clip(it.text)}")
    lines.append(f"[temporal] {len(temp)} prior post(s) by author:")
    for i, it in enumerate(temp):
        lines.append(f"  {i}. Δt={it.delta_days:8.2f} d | {clip(it.text)}")
    if ret is not None:
        lines.append(f"[retrieval] {len(ret.sarc)} sarcastic / {len(ret.nonsarc)} non-sarcastic exemplars:")
        for f in ret.sarc:
            lines.append(f"  S: {clip(text_of[f])}")
        for f in ret.nonsarc:
            lines.append(f"  N: {clip(text_of[f])}")
    return "\n".join(lines)
