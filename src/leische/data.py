"""Load and validate uyam exports against the dataset contract; build frozen folds.

Contract: uyam/docs/dataset-contract-leische.md. Pitfalls enforced here:
  - §11.2 thread leakage — folds group on submission_fullname, no exceptions
  - §11.3 oversampled rows — natural_mask() for any natural-distribution metric
  - §11.8 version mixing — frozen fold files carry the dataset identity and
    refuse to load against a different export
  - §10 readiness gate — readiness_gate() is what train.py consults before
    allowing a non-smoke run
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from .config import LeischeConfig

LANGUAGES = ("english", "tagalog", "taglish")
SENTIMENTS = ("positive", "neutral", "negative")
RESOLVED_BY = ("unanimous", "majority", "adjudicator", "human")
RECORD_TYPES = ("submission", "comment")
CUE_KEYS = ("polarity_inversion", "rhetorical_intent", "contextual_incongruity", "hyperbole")
# pilot exports carry null sampling_strategy; treat null as natural for hygiene
SAMPLING = ("natural", "keyword_oversampled", None)


# --------------------------------------------------------------------------- load


def load_dataset(path: str | Path, validate: bool = True) -> pd.DataFrame:
    """dataset-vN.jsonl → DataFrame (nested objects kept as dict columns)."""
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if validate:
        problems = validate_rows(rows)
        if problems:
            raise ValueError(
                f"{len(problems)} contract violations in {path}; first 5: {problems[:5]}"
            )
    df = pd.DataFrame(rows)
    # ISO8601: the exports mix second- and microsecond-precision timestamps
    df["created_dt"] = pd.to_datetime(df["created_utc"], utc=True, format="ISO8601")
    return df


def load_corpus(path: str | Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    df = pd.DataFrame(rows)
    df["created_dt"] = pd.to_datetime(df["created_utc"], utc=True, format="ISO8601")
    return df


def load_card(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def validate_rows(rows: list[dict]) -> list[str]:
    """Check every row against the contract. Returns a list of problem strings."""
    problems: list[str] = []
    seen_ids: set[str] = set()
    for i, r in enumerate(rows):
        where = f"row {i} ({r.get('reddit_fullname', '?')})"

        def bad(msg: str) -> None:
            problems.append(f"{where}: {msg}")

        for key in ("reddit_fullname", "record_type", "submission_fullname", "created_utc",
                    "author_hash", "text", "labels", "reliability", "aux", "context", "provenance"):
            if key not in r:
                bad(f"missing {key}")
        rid = r.get("reddit_fullname")
        if rid in seen_ids:
            bad("duplicate reddit_fullname")
        seen_ids.add(rid)
        if r.get("record_type") not in RECORD_TYPES:
            bad(f"record_type={r.get('record_type')!r}")
        if r.get("sampling_strategy") not in SAMPLING:
            bad(f"sampling_strategy={r.get('sampling_strategy')!r}")
        if not isinstance(r.get("text"), str) or not r.get("text", "").strip():
            bad("empty text")

        labels = r.get("labels") or {}
        if not isinstance(labels.get("sarcastic"), bool):
            bad("labels.sarcastic not bool")
        if labels.get("language") not in LANGUAGES:
            bad(f"labels.language={labels.get('language')!r}")
        for k in ("literal_sentiment", "intended_sentiment"):
            if labels.get(k) not in SENTIMENTS:
                bad(f"labels.{k}={labels.get(k)!r}")
        cues = labels.get("cues") or {}
        for k in CUE_KEYS:
            if not isinstance(cues.get(k), bool):
                bad(f"labels.cues.{k} not bool")

        rel = r.get("reliability") or {}
        if rel.get("resolved_by") not in RESOLVED_BY:
            bad(f"reliability.resolved_by={rel.get('resolved_by')!r}")

        ctx = r.get("context")
        if not isinstance(ctx, dict):
            bad("context missing/not dict")
        else:
            if r.get("record_type") == "comment" and ctx.get("submission") is None:
                bad("comment row with null context.submission")
            if r.get("record_type") == "submission" and ctx.get("submission") is not None:
                bad("submission row should have null context.submission (target IS the submission)")
            for chan in ("parent_chain", "replies"):
                if not isinstance(ctx.get(chan), list):
                    bad(f"context.{chan} not a list")
    return problems


# --------------------------------------------------------------------------- derived columns


def strat_key(df: pd.DataFrame) -> pd.Series:
    """Joint stratification key: sarcastic × language (guide §7.1)."""
    sarc = df["labels"].map(lambda l: l["sarcastic"])
    lang = df["labels"].map(lambda l: l["language"])
    return lang + "|sarc=" + sarc.astype(str)


def sarcastic(df: pd.DataFrame) -> pd.Series:
    return df["labels"].map(lambda l: bool(l["sarcastic"]))


def language(df: pd.DataFrame) -> pd.Series:
    return df["labels"].map(lambda l: l["language"])


def natural_mask(df: pd.DataFrame) -> pd.Series:
    """§7.2: rows allowed in natural-distribution metrics.

    Null sampling_strategy (pilot export) counts as natural; only rows
    explicitly tagged keyword_oversampled are excluded.
    """
    return df["sampling_strategy"].map(lambda s: s != "keyword_oversampled")


# --------------------------------------------------------------------------- readiness gate (§10)


@dataclass
class GateCheck:
    name: str
    passed: bool
    detail: str


@dataclass
class GateReport:
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks)

    def render(self) -> str:
        lines = ["§10 data-readiness gate:"]
        for c in self.checks:
            lines.append(f"  [{'PASS' if c.passed else 'FAIL'}] {c.name} — {c.detail}")
        lines.append(
            "  => GATE PASSED: real training is legitimate."
            if self.passed
            else "  => GATE FAILED: notebooks run in SMOKE mode only; no reported metrics."
        )
        return "\n".join(lines)


def _version_number(v: str | None) -> int:
    try:
        return int(str(v).lstrip("sarc-v").lstrip("v") or 0)
    except ValueError:
        return 0


def readiness_gate(card: dict, df: pd.DataFrame, folds_file: Path | None = None) -> GateReport:
    rep = GateReport()
    dv = card.get("dataset_version")
    pv = card.get("prompt_version")
    rep.checks.append(GateCheck(
        "dataset-v2 under sarc-v2",
        _version_number(dv) >= 2 and _version_number(pv) >= 2,
        f"dataset_version={dv}, prompt_version={pv}",
    ))

    n_pos = int(sarcastic(df).sum())
    rep.checks.append(GateCheck(
        "≥400 sarcastic positives", n_pos >= 400, f"{n_pos} positives in export"))

    gold = (card.get("agreement") or {}).get("gold_vs_ensemble_cohen_kappa") or {}
    n_gold = int(gold.get("n_gold_items") or 0)
    kappa = gold.get("sarcastic")
    rep.checks.append(GateCheck(
        "gold subset labeled + κ reported",
        n_gold >= 250 and kappa is not None,
        f"n_gold_items={n_gold}, sarcastic κ={kappa}",
    ))

    cells = strat_key(df).value_counts()
    thin = {k: int(v) for k, v in cells.items() if v < 10}
    rep.checks.append(GateCheck(
        "every language×sarcastic cell ≥10",
        len(thin) == 0,
        f"thin cells: {thin}" if thin else "all cells ≥10",
    ))

    if folds_file is not None:
        rep.checks.append(GateCheck(
            "fold file frozen", folds_file.exists(), str(folds_file)))
    return rep


# --------------------------------------------------------------------------- folds (§7.1)


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 13,
    val_frac_of_train: float = 1 / 9,  # 90% train-side × 1/9 ≈ 10% overall → 80/10/10
    smoke: bool = True,
) -> list[dict]:
    """StratifiedGroupKFold: stratify sarcastic×language, group by submission_fullname.

    Val is carved out of each fold's train side with the same thread grouping.
    Returns [{fold, train, val, test}] as reddit_fullname lists.
    """
    y = strat_key(df).to_numpy()
    groups = df["submission_fullname"].to_numpy()
    ids = df["reddit_fullname"].to_numpy()

    with warnings.catch_warnings():
        if smoke:
            warnings.filterwarnings("ignore", message="The least populated class")
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        folds = []
        for fold_i, (train_idx, test_idx) in enumerate(sgkf.split(df, y, groups)):
            gss = GroupShuffleSplit(n_splits=1, test_size=val_frac_of_train, random_state=seed + fold_i)
            tr_rel, val_rel = next(gss.split(train_idx, groups=groups[train_idx]))
            fold = {
                "fold": fold_i,
                "train": ids[train_idx[tr_rel]].tolist(),
                "val": ids[train_idx[val_rel]].tolist(),
                "test": ids[test_idx].tolist(),
            }
            folds.append(fold)

    # sanity: no thread spans splits within a fold
    thread_of = dict(zip(df["reddit_fullname"], df["submission_fullname"]))
    for fold in folds:
        parts = {name: {thread_of[i] for i in fold[name]} for name in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            overlap = parts[a] & parts[b]
            if overlap:
                raise AssertionError(f"fold {fold['fold']}: threads span {a}/{b}: {sorted(overlap)[:3]}")

    # degenerate-positive warning (expected on the pilot; fatal outside smoke)
    pos_ids = set(df.loc[sarcastic(df), "reddit_fullname"])
    for fold in folds:
        for part in ("train", "val", "test"):
            n_pos = len(pos_ids & set(fold[part]))
            if n_pos == 0:
                msg = f"fold {fold['fold']} {part} has 0 sarcastic positives"
                if smoke:
                    warnings.warn(f"SMOKE: {msg} (tolerated on pilot data)", stacklevel=2)
                else:
                    raise ValueError(msg + " — not valid for real training")
    return folds


def dataset_identity(card: dict) -> dict:
    return {
        "dataset_version": card.get("dataset_version"),
        "prompt_version": card.get("prompt_version"),
        "uyam_commit": card.get("uyam_commit"),
    }


def freeze_folds(folds: list[dict], card: dict, path: Path, n_splits: int, seed: int) -> None:
    """Write the frozen fold assignment; every experiment reuses this file."""
    payload = {
        "identity": dataset_identity(card),
        "n_splits": n_splits,
        "seed": seed,
        "folds": folds,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_folds(path: Path, card: dict) -> list[dict]:
    """Load frozen folds; refuse an identity mismatch (§11.8 version mixing)."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = dataset_identity(card)
    if payload["identity"] != expected:
        raise ValueError(
            f"frozen folds {path} were built for {payload['identity']}, "
            f"current export is {expected} — regenerate folds for this version"
        )
    return payload["folds"]


def author_overlap_report(df: pd.DataFrame, folds: list[dict]) -> pd.DataFrame:
    """§7.1 audit: how many authors appear in more than one fold's test split."""
    author_of = dict(zip(df["reddit_fullname"], df["author_hash"]))
    rows = []
    test_authors = [
        {author_of[i] for i in fold["test"]} for fold in folds
    ]
    all_authors = set().union(*test_authors) if test_authors else set()
    multi = sum(1 for a in all_authors if sum(a in s for s in test_authors) > 1)
    for fold, authors in zip(folds, test_authors):
        rows.append({"fold": fold["fold"], "n_test_rows": len(fold["test"]), "n_test_authors": len(authors)})
    rep = pd.DataFrame(rows)
    rep.attrs["authors_in_multiple_test_folds"] = multi
    rep.attrs["total_test_authors"] = len(all_authors)
    return rep


# --------------------------------------------------------------------------- misc helpers


def rows_by_fullname(df: pd.DataFrame) -> dict[str, pd.Series]:
    return {r["reddit_fullname"]: r for _, r in df.iterrows()}


def class_weights(train_df: pd.DataFrame) -> tuple[float, float]:
    """Inverse-frequency (neg_weight, pos_weight), computed on ONE training fold."""
    y = sarcastic(train_df).to_numpy()
    n, n_pos = len(y), int(y.sum())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        warnings.warn(f"degenerate training fold (n_pos={n_pos}) — falling back to equal weights", stacklevel=2)
        return 1.0, 1.0
    return n / (2.0 * n_neg), n / (2.0 * n_pos)


def parse_utc(ts: str) -> datetime:
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))
