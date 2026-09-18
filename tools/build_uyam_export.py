"""Adapt the uyam CSV export into the dataset contract the pipeline consumes.

uyam currently ships two flat CSVs:

    data/annotated-review.csv   the annotated targets (one row per target)
    data/uyam_export.csv        the full scraped corpus (submissions + comments)

The notebook expects the nested per-row contract described in
docs/MODEL_PLAN.md section 2 (``labels`` / ``reliability`` / ``aux`` /
``context``).  This script derives that contract from the two CSVs and writes

    data/dataset-{V}.jsonl      one row per annotated target
    data/corpus-{V}.jsonl       every scraped record (temporal-context source)
    data/dataset_card.json      identity + counts + inter-annotator agreement

Every derived field is marked in ``provenance`` so nothing authentic is
confused with something this script invented.  Fields uyam does not collect at
all (cue labels, tx_sentiment, LID ratios, human gold) are emitted as ``null``
rather than guessed - see docs/UYAM_HANDOFF.md.

    uv run python tools/build_uyam_export.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

# uyam pseudonymises at export time but still ships the reddit handle; the
# thesis commits to one-way hashed author identifiers, so hash here until uyam
# hashes at collection time (handoff item H7).
AUTHOR_SALT = "leische/uyam/author-v1"

SENTIMENTS = ("positive", "neutral", "negative")

# "[sarcastic; neutral->negative] <free text>" - emitted by every annotator
# model, 100% parseable across all three on the current export.
RATIONALE_RE = re.compile(
    r"^\[(?P<sarcastic>sarcastic|not sarcastic);\s*"
    r"(?P<literal>positive|neutral|negative)\s*(?:→|->)\s*"
    r"(?P<intended>positive|neutral|negative)\]\s*(?P<rationale>.*)$",
    re.DOTALL,
)

ANNOTATOR_MODELS = ("gemma3", "qwen3", "sealion")


def author_hash(author: object) -> str | None:
    if not isinstance(author, str) or not author.strip():
        return None
    return hashlib.sha256(f"{AUTHOR_SALT}:{author}".encode()).hexdigest()[:16]


def fullname(record_type: str, rid: object) -> str:
    return ("t3_" if record_type == "submission" else "t1_") + str(rid)


def clean(value: object) -> str | None:
    """CSV blanks arrive as NaN; keep text byte-identical otherwise (section 11.7)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value)
    return text if text.strip() else None


def corpus_text(row) -> str | None:
    if row["record_type"] == "submission":
        parts = [clean(row.get("title")), clean(row.get("selftext"))]
        joined = "\n\n".join(p for p in parts if p)
        return joined or None
    return clean(row.get("body"))


# ---------------------------------------------------------------- agreement


def fleiss_kappa(items: list[list[str]], categories: tuple[str, ...]) -> float | None:
    """Fleiss' kappa generalised to a variable number of raters per item."""
    usable = [i for i in items if len(i) >= 2]
    if not usable:
        return None
    p_bar, totals = 0.0, defaultdict(int)
    for votes in usable:
        n = len(votes)
        counts = defaultdict(int)
        for v in votes:
            counts[v] += 1
            totals[v] += 1
        p_bar += (sum(c * c for c in counts.values()) - n) / (n * (n - 1))
    p_bar /= len(usable)
    grand = sum(totals.values())
    p_e = sum((totals[c] / grand) ** 2 for c in categories)
    return None if abs(1 - p_e) < 1e-12 else (p_bar - p_e) / (1 - p_e)


def krippendorff_alpha(items: list[list[str]], categories: tuple[str, ...]) -> float | None:
    """Nominal-scale Krippendorff's alpha (natively tolerates missing raters)."""
    usable = [i for i in items if len(i) >= 2]
    if not usable:
        return None
    coincidence: dict[tuple[str, str], float] = defaultdict(float)
    for votes in usable:
        m = len(votes)
        for a in votes:
            for b in votes:
                coincidence[(a, b)] += 1.0 / (m - 1)
        for v in votes:  # remove the self-pairs added by the double loop
            coincidence[(v, v)] -= 1.0 / (m - 1)
    n_c = {c: sum(coincidence[(c, k)] for k in categories) for c in categories}
    n = sum(n_c.values())
    if n < 2:
        return None
    d_o = sum(coincidence[(c, k)] for c in categories for k in categories if c != k) / n
    d_e = sum(n_c[c] * n_c[k] for c in categories for k in categories if c != k) / (n * (n - 1))
    return None if d_e == 0 else 1.0 - d_o / d_e


def agreement_block(per_model: dict[str, pd.DataFrame]) -> dict:
    """Inter-annotator agreement over the three LLM annotators.

    Only the labels each model states in its rationale prefix can be scored;
    `language` is resolved by uyam without a per-model trace, so it is null
    here rather than silently omitted (handoff item H6).
    """
    out: dict[str, dict] = {}
    specs = {
        "sarcastic": ("sarcastic", ("sarcastic", "not sarcastic")),
        "literal_sentiment": ("literal", SENTIMENTS),
        "intended_sentiment": ("intended", SENTIMENTS),
    }
    for label, (column, categories) in specs.items():
        items = [
            [v for v in votes if isinstance(v, str)]
            for votes in zip(*(per_model[m][column] for m in ANNOTATOR_MODELS))
        ]
        out[label] = {
            "fleiss_kappa": fleiss_kappa(items, categories),
            "krippendorff_alpha": krippendorff_alpha(items, categories),
            "n_items": len(items),
        }
    out["language"] = {
        "fleiss_kappa": None,
        "krippendorff_alpha": None,
        "n_items": 0,
        "note": "uyam does not export a per-annotator language vote (handoff H6)",
    }
    return out


# ------------------------------------------------------------------ context


class ThreadIndex:
    """Conversational context rebuilt from the corpus dump.

    NOTE (MODEL_PLAN section 11.7): this is *not* the snapshot the annotators
    saw - uyam does not export one yet (handoff item H2).  Every row records
    ``context.source = "corpus_rebuild"`` so the channel can be swapped for the
    authentic snapshot without touching anything downstream.
    """

    MAX_REPLIES = 3  # thesis 3.4(1): "up to three direct replies"

    def __init__(self, corpus: pd.DataFrame):
        self.by_id = {str(r["id"]): r for _, r in corpus.iterrows()}
        self.submissions = {
            str(r["id"]): r
            for _, r in corpus[corpus.record_type == "submission"].iterrows()
        }
        children: dict[str, list] = defaultdict(list)
        comments = corpus[corpus.record_type == "comment"].sort_values(["created_utc", "id"])
        for _, r in comments.iterrows():
            parent = clean(r["parent_comment_id"])
            # depth-0 comments have no parent comment; they reply to the post
            children[parent if parent else f"post:{r['post_id']}"].append(r)
        self.children = children

    def _item(self, row) -> dict:
        return {
            "reddit_fullname": fullname(row["record_type"], row["id"]),
            "text": corpus_text(row),
            "depth": None if pd.isna(row["depth"]) else int(row["depth"]),
            "is_submitter": bool(row["is_submitter"]) if not pd.isna(row["is_submitter"]) else False,
            "created_utc": clean(row["created_utc"]),
            "author_hash": author_hash(clean(row["author"])),
        }

    def build(self, row) -> dict:
        is_submission = row["record_type"] == "submission"
        submission = None
        if not is_submission:
            sub = self.submissions.get(str(row["post_id"]))
            if sub is not None:
                submission = {
                    "reddit_fullname": fullname("submission", sub["id"]),
                    "title": clean(sub["title"]),
                    "selftext": clean(sub["selftext"]),
                }

        # ancestors, oldest first, walking parent_comment_id up to the post
        chain, cursor, guard = [], clean(row["parent_comment_id"]), 0
        while cursor and guard < 32:
            parent = self.by_id.get(cursor)
            if parent is None:
                break
            chain.append(self._item(parent))
            cursor, guard = clean(parent["parent_comment_id"]), guard + 1
        chain.reverse()

        # direct replies, chronological - deliberately NOT ranked by score,
        # which is post-hoc information the model must not see (section 11.5)
        key = f"post:{row['id']}" if is_submission else str(row["id"])
        replies = [self._item(c) for c in self.children.get(key, [])[: self.MAX_REPLIES]]

        return {
            "source": "corpus_rebuild",
            "submission": submission,
            "parent_chain": [c for c in chain if c["text"]],
            "replies": [r for r in replies if r["text"]],
        }


# --------------------------------------------------------------------- main


def parse_annotators(annotated: pd.DataFrame) -> dict[str, pd.DataFrame]:
    per_model = {}
    for model in ANNOTATOR_MODELS:
        parsed = annotated[f"reason: {model}"].fillna("").str.extract(RATIONALE_RE)
        per_model[model] = parsed
    return per_model


def build(version: str) -> None:
    annotated = pd.read_csv(DATA / "annotated-review.csv", encoding="utf-8-sig")
    corpus = pd.read_csv(DATA / "uyam_export.csv", low_memory=False)
    corpus["id"] = corpus["id"].astype(str)

    per_model = parse_annotators(annotated)
    for model, parsed in per_model.items():
        present = annotated[f"reason: {model}"].notna()
        unparsed = int((present & parsed["sarcastic"].isna()).sum())
        assert not unparsed, f"{unparsed} unparseable {model} rationales"

    corpus_by_id = corpus.set_index("id", drop=False)
    threads = ThreadIndex(corpus)

    annotated["bare_id"] = annotated["reddit_fullname"].str.replace(r"^t[13]_", "", regex=True)
    missing = ~annotated["bare_id"].isin(corpus_by_id.index)
    assert not missing.any(), f"{int(missing.sum())} annotated rows absent from the corpus"

    rows, skipped = [], defaultdict(int)
    for i, a in annotated.iterrows():
        # sarcasm is null exactly on split votes (2-1 / 1-1 / 1-0) that uyam
        # never adjudicated; language is null when its own vote did not resolve.
        # Both are required: sarcasm is the target, language is the
        # stratification key and the disaggregation axis for every RQ.
        if pd.isna(a["sarcastic"]):
            skipped["sarcasm_unresolved"] += 1
            continue
        if not isinstance(a["language"], str):
            skipped["language_unresolved"] += 1
            continue

        c = corpus_by_id.loc[a["bare_id"]]
        votes = str(a["votes"])
        n_annotators = sum(int(x) for x in votes.split("-"))

        annotators = []
        for model in ANNOTATOR_MODELS:
            p = per_model[model].loc[i]
            if not isinstance(p["sarcastic"], str):
                continue
            annotators.append({
                "model_key": model,
                "sarcastic": p["sarcastic"] == "sarcastic",
                "literal_sentiment": p["literal"],
                "intended_sentiment": p["intended"],
                "rationale": p["rationale"].strip(),
            })

        rows.append({
            "reddit_fullname": a["reddit_fullname"],
            "record_type": a["record_type"],
            "subreddit": a["subreddit"],
            "submission_fullname": fullname("submission", c["post_id"]),
            "created_utc": clean(c["created_utc"]),
            "author_hash": author_hash(clean(c["author"])),
            "text": str(a["text"]),
            "title": clean(c["title"]),
            "selftext": clean(c["selftext"]),
            "depth": None if pd.isna(c["depth"]) else int(c["depth"]),
            "is_submitter": bool(c["is_submitter"]) if not pd.isna(c["is_submitter"]) else False,
            "score": None if pd.isna(c["score"]) else int(c["score"]),
            # uyam leaves this blank on comments and never ran keyword
            # oversampling; null reads as natural downstream (section 7.2)
            "sampling_strategy": clean(c["sampling_strategy"]) or "natural",
            "labels": {
                "sarcastic": bool(a["sarcastic"]),
                "language": a["language"],
                "literal_sentiment": clean(a["literal"]),
                "intended_sentiment": clean(a["intended"]),
                "cues": None,  # not collected by uyam (handoff H5)
            },
            "reliability": {
                "sarcasm_votes": votes,
                # every surviving row is 3-0 or 2-0: uyam never adjudicated a
                # split, so a resolved sarcasm label is always unanimous among
                # the models that answered.  "unanimous_partial" = one model
                # (sealion) returned nothing for this row (handoff H3).
                "resolved_by": "unanimous" if n_annotators >= 3 else "unanimous_partial",
                "n_annotators": n_annotators,
                "mean_confidence": None if pd.isna(a["confidence"]) else float(a["confidence"]),
                "all_labels_resolved_by": a["resolved_by"],
                "annotators": annotators,
            },
            "aux": {"tx_sentiment": None, "lid": None},  # LID not collected (handoff H6)
            "human_gold": None,  # no gold subset labelled yet (handoff H1)
            "context": threads.build(c),
            "provenance": {
                "prompt_version": a["prompt_version"],
                "dataset_version": version,
                "context_source": "corpus_rebuild",
                "derived_by": "tools/build_uyam_export.py",
            },
        })

    dataset_path = DATA / f"dataset-{version}.jsonl"
    dataset_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n", encoding="utf-8"
    )

    corpus_rows = [
        {
            "reddit_fullname": fullname(r["record_type"], r["id"]),
            "record_type": r["record_type"],
            "subreddit": r["subreddit"],
            "author_hash": author_hash(clean(r["author"])),
            "created_utc": clean(r["created_utc"]),
            "text": corpus_text(r),
        }
        for _, r in corpus.iterrows()
    ]
    corpus_rows = [r for r in corpus_rows if r["text"] and r["author_hash"]]
    corpus_path = DATA / f"corpus-{version}.jsonl"
    corpus_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in corpus_rows) + "\n", encoding="utf-8"
    )

    frame = pd.DataFrame(rows)
    labels = pd.DataFrame(list(frame["labels"]))
    card = {
        "dataset_version": version,
        "prompt_version": str(annotated["prompt_version"].mode().iloc[0]),
        "uyam_commit": None,  # uyam does not stamp one on the CSV export (handoff H11)
        "built_by": "leische/tools/build_uyam_export.py",
        "counts": {
            "annotated_rows_in_export": int(len(annotated)),
            "dataset_rows": int(len(frame)),
            "sarcastic_positives": int(labels["sarcastic"].sum()),
            "corpus_rows": int(len(corpus_rows)),
            "threads": int(frame["submission_fullname"].nunique()),
            "skipped": dict(skipped),
            "language_x_sarcastic": {
                f"{lang}|sarc={sarc}": int(n)
                for (lang, sarc), n in labels.groupby(["language", "sarcastic"]).size().items()
            },
        },
        "agreement": {
            "annotators": list(ANNOTATOR_MODELS),
            "labels": agreement_block(per_model),
            "gold_vs_ensemble_cohen_kappa": {"n_gold_items": 0, "sarcastic": None},
        },
    }
    (DATA / "dataset_card.json").write_text(json.dumps(card, indent=2), encoding="utf-8")

    print(f"wrote {dataset_path.name}: {len(rows)} rows, "
          f"{int(labels['sarcastic'].sum())} sarcastic")
    print(f"wrote {corpus_path.name}: {len(corpus_rows)} rows")
    print(f"skipped: {dict(skipped)}")
    print(json.dumps(card["counts"]["language_x_sarcastic"], indent=2))
    print(json.dumps(card["agreement"]["labels"], indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default="v2")
    build(parser.parse_args().version)
