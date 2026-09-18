"""Everything the demo server needs, loaded once at startup.

Reads, never writes (the one exception is the retrieval embedding cache, which
is rebuilt in the notebook's own format if it is missing).  Nothing here copies
post text anywhere on disk: `data/` and `cache/` are gitignored and are opened
at runtime only (FABILE_BRIEF §5).
"""

from __future__ import annotations

import json
import math
import os
import pickle
import threading
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch

from .notebook_source import (ROLE_ANCESTOR, ROLE_REPLY, ROLE_SUBMISSION, ROOT,
                              ConvItem, TemporalItem, load_pipeline)

DATA = ROOT / "data"
CACHE = ROOT / "cache"
CHECKPOINTS = CACHE / "checkpoints"
DEFAULT_CHECKPOINT = CHECKPOINTS / "ablation-8_full-fold0-seed13.pt"
RQ3_HEADS = CHECKPOINTS / "rq3-heads-fold0.pkl"
FOLDS_FILE = ROOT / "results" / "folds-v1-a419a4bc95.json"

# input clipping (POST /api/predict)
MAX_TEXT = 2000
MAX_CONV = 8
MAX_HISTORY = 5
EXEMPLAR_CHARS = 400          # display truncation for retrieved exemplars

LANGUAGES = ("english", "tagalog", "taglish")

HONESTY = ("Labels are LLM-ensemble generated (gemma3, qwen3, sealion; qwen3-32b "
           "adjudicator). Human-validated on 31 items so far, Cohen's κ = −0.148. "
           "Probabilities measure agreement with the ensemble, not human judgement.")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _logit(p: float) -> float:
    p = min(max(float(p), 1e-6), 1 - 1e-6)
    return math.log(p / (1.0 - p))


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _dt(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def _clip(text, limit=MAX_TEXT) -> str:
    return (text or "").strip()[:limit]


@dataclass
class Bank:
    """One retrieval bank (sarcastic / not), restricted to training rows."""
    names: list
    vecs: np.ndarray          # L2-normalised [n, 384]
    threads: np.ndarray       # submission_fullname per row


class Engine:
    # ------------------------------------------------------------- startup
    def __init__(self, log=print):
        self.log = log
        self.lock = threading.Lock()
        ns = load_pipeline()
        self.Config = ns["Config"]
        self.to_device = ns["to_device"]

        self.device = torch.device(self._resolve_device())
        self.log(f"device: {self.device}")

        self.rows = _read_jsonl(DATA / "dataset-v1.jsonl")
        self.by_name = {r["reddit_fullname"]: r for r in self.rows}
        self.log(f"dataset: {len(self.rows)} rows")

        ckpt = self._load_checkpoint()
        self.trained = ckpt is not None
        if ckpt is not None:
            self.cfg = self.Config(**ckpt["config"])
            self.threshold = float(ckpt.get("threshold", 0.5))
            self.temperature = float(ckpt.get("temperature", 1.0)) or 1.0
            self.train_fullnames = list(ckpt["train_fullnames"])
            self.label_authority = ckpt.get("label_authority")
        else:
            self.cfg = self.Config(use_conv=True, use_temp=True, use_ret=True, smoke=False)
            self.threshold, self.temperature = 0.5, 1.0
            folds = json.loads(FOLDS_FILE.read_text(encoding="utf-8"))["folds"]
            self.train_fullnames = list(folds[0]["train"])
            self.label_authority = None
            self.log("NO CHECKPOINT — random weights, threshold 0.5, fold-0 train rows as the bank")
        self.threshold_cal = _sigmoid(_logit(self.threshold) / self.temperature)

        self.model = ns["ContextAwareSarcasmModel"](self.cfg)
        if ckpt is not None:
            self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval().to(self.device)
        self.tok = ns["Tokenize"](self.cfg)
        self.collate = ns["Collator"](self.cfg, self.tok)
        self.log(f"model ready — active channels {self.model.active}, "
                 f"trained={self.trained}")

        self._build_banks()
        self.heads = self._load_sentiment_heads()
        self.examples = self._build_examples()
        self.log(f"examples: {len(self.examples)}")

    def _resolve_device(self) -> str:
        want = os.environ.get("LEISCHE_DEMO_DEVICE", "").strip().lower()
        if want:
            if want.startswith("cuda") and not torch.cuda.is_available():
                self.log(f"LEISCHE_DEMO_DEVICE={want} but CUDA is unavailable — using cpu")
                return "cpu"
            return want
        return "cuda" if torch.cuda.is_available() else "cpu"

    def _load_checkpoint(self):
        path = Path(os.environ.get("LEISCHE_CHECKPOINT") or DEFAULT_CHECKPOINT)
        self.checkpoint_path = path
        if not path.exists():
            self.log(f"checkpoint not found: {path}")
            return None
        self.log(f"checkpoint: {path}")
        return torch.load(path, weights_only=False, map_location="cpu")

    # ------------------------------------------------------------ retrieval
    def _embeddings(self) -> tuple[list, np.ndarray]:
        key = self.cfg.retrieval_model.replace("/", "__")
        cache = CACHE / f"retrieval-{self.cfg.dataset_version}-{key}.npz"
        fullnames = [r["reddit_fullname"] for r in self.rows]
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            if z["fullnames"].tolist() == fullnames:
                return fullnames, z["vectors"]
            self.log(f"{cache.name} does not match the dataset — re-encoding")
        self.log(f"encoding {len(fullnames)} texts for retrieval (one-off, a minute or two)")
        vectors = self.sentence_encoder.encode(
            [r["text"] for r in self.rows], batch_size=64,
            show_progress_bar=False, convert_to_numpy=True)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, fullnames=np.array(fullnames, dtype=object),
                            vectors=vectors.astype(np.float32))
        return fullnames, vectors

    def _build_banks(self) -> None:
        from sentence_transformers import SentenceTransformer
        # queries are typed live, so the query encoder stays on CPU regardless
        # of where the classifier runs — it is a 12-layer MiniLM, milliseconds
        self.sentence_encoder = SentenceTransformer(self.cfg.retrieval_model, device="cpu")

        fullnames, vectors = self._embeddings()
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
        index = {f: i for i, f in enumerate(fullnames)}

        banks = {}
        for label, key in ((True, "sarcastic"), (False, "non_sarcastic")):
            names = [f for f in self.train_fullnames
                     if f in index and self._bank_label(self.by_name[f]) is label]
            banks[key] = Bank(
                names=names,
                vecs=vectors[[index[f] for f in names]].astype(np.float32),
                threads=np.array([self.by_name[f]["submission_fullname"] for f in names]))
        self.banks = banks
        self.log(f"retrieval banks — sarcastic {len(banks['sarcastic'].names)}, "
                 f"non-sarcastic {len(banks['non_sarcastic'].names)} "
                 f"(training rows only, k={self.cfg.retrieval_k}, "
                 f"label source {getattr(self.cfg, 'label_source', 'adjudicated')})")

    def _bank_label(self, row: dict) -> bool:
        """The label the checkpoint was trained on. `label_source` is a Config
        knob: the shipped adjudicated label, or the annotator majority (the
        §4 robustness row) — the banks must be split the same way training was."""
        if getattr(self.cfg, "label_source", "adjudicated") != "annotator_majority":
            return bool(row["labels"]["sarcastic"])
        votes = [a.get("sarcastic") for a in ((row.get("reliability") or {}).get("annotators") or [])
                 if isinstance(a.get("sarcastic"), bool)]
        if not votes or 2 * sum(votes) == len(votes):
            return bool(row["labels"]["sarcastic"])
        return 2 * sum(votes) > len(votes)

    def retrieve(self, text: str, exclude_thread: str | None) -> dict:
        q = self.sentence_encoder.encode([text], convert_to_numpy=True)[0]
        q = q / (np.linalg.norm(q) + 1e-12)
        out = {}
        for key, bank in self.banks.items():
            sims = bank.vecs @ q
            if exclude_thread:      # leakage rule: never an exemplar from the query's own thread
                sims = np.where(bank.threads == exclude_thread, -np.inf, sims)
            k = min(int(self.cfg.retrieval_k), len(sims))
            if k <= 0:
                out[key] = []
                continue
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top], kind="stable")]
            out[key] = [{"fullname": bank.names[i],
                         "text": self.by_name[bank.names[i]]["text"],
                         "similarity": round(float(sims[i]), 4),
                         "language": self.by_name[bank.names[i]]["labels"]["language"]}
                        for i in top if np.isfinite(sims[i])]
        return out

    # ------------------------------------------------------------ sentiment
    def _load_sentiment_heads(self):
        path = Path(os.environ.get("LEISCHE_RQ3_HEADS") or RQ3_HEADS)
        if not path.exists():
            self.log(f"sentiment heads not found: {path} — /api/predict returns sentiment=null")
            return None
        with path.open("rb") as fh:
            heads = pickle.load(fh)
        self.log(f"sentiment heads: {path.name}")
        return heads

    # ------------------------------------------------------------- examples
    def _author_history_index(self) -> dict:
        """corpus rows grouped by author, oldest first — mirrors TemporalIndex."""
        authors = {r["author_hash"] for r in self.rows}
        by_author: dict = {}
        for r in _read_jsonl(DATA / "corpus-v1.jsonl"):
            if r["author_hash"] in authors and (r.get("text") or "").strip():
                by_author.setdefault(r["author_hash"], []).append(
                    (_dt(r["created_utc"]), r["reddit_fullname"], r["text"]))
        for v in by_author.values():
            v.sort(key=lambda t: t[0])
        return by_author

    def _history(self, row: dict, by_author: dict) -> list[dict]:
        """Up to k posts by the same author strictly before the target, inside
        the configured window, most recent first (channels cell, TemporalIndex)."""
        target_dt = _dt(row["created_utc"])
        window = self.cfg.temporal_window_hours
        prior = [(dt, fid, tx) for dt, fid, tx in by_author.get(row["author_hash"], [])
                 if dt < target_dt and fid != row["reddit_fullname"]]
        if window is not None:
            cutoff = target_dt - timedelta(hours=float(window))
            prior = [p for p in prior if p[0] >= cutoff]
        return [{"text": tx, "hours_ago": round((target_dt - dt).total_seconds() / 3600.0, 2)}
                for dt, fid, tx in reversed(prior[-int(self.cfg.temporal_k):])]

    def _example(self, row: dict, group: str, note: str, by_author: dict) -> dict:
        ctx = row["context"] or {}
        sub = ctx.get("submission")
        rel = row.get("reliability") or {}
        text = row["text"]
        return {
            "id": row["reddit_fullname"],
            "title": (text[:60] + "…") if len(text) > 60 else text,
            "text": _clip(text),
            "language": row["labels"]["language"],
            "ensemble_label": "sarcastic" if row["labels"]["sarcastic"] else "not sarcastic",
            "resolved_by": rel.get("resolved_by"),
            "votes": rel.get("sarcasm_votes"),
            "submission": ({"title": sub.get("title") or "",
                            "selftext": _clip(sub.get("selftext") or "")} if sub else None),
            "parents": [{"text": _clip(p.get("text")), "is_submitter": bool(p.get("is_submitter"))}
                        for p in (ctx.get("parent_chain") or []) if (p.get("text") or "").strip()][:MAX_CONV],
            "replies": [{"text": _clip(p.get("text")), "is_submitter": bool(p.get("is_submitter"))}
                        for p in (ctx.get("replies") or []) if (p.get("text") or "").strip()][:MAX_CONV],
            "history": self._history(row, by_author),
            "submission_fullname": row["submission_fullname"],
            "group": group,
            "note": note,
        }

    def _build_examples(self) -> list[dict]:
        by_author = self._author_history_index()
        train = set(self.train_fullnames)
        held_out = sorted((r for r in self.rows if r["reddit_fullname"] not in train),
                          key=lambda r: r["reddit_fullname"])

        def n_ctx(r):
            ctx = r["context"] or {}
            return len(ctx.get("parent_chain") or []) + len(ctx.get("replies") or [])

        picked, seen = [], set()
        for lang in LANGUAGES:
            for sarc in (True, False):
                pool = [r for r in held_out
                        if r["labels"]["language"] == lang
                        and bool(r["labels"]["sarcastic"]) is sarc
                        and n_ctx(r) >= 1
                        and r["reddit_fullname"] not in seen]
                best = next((r for r in pool if self._history(r, by_author)), None) or \
                    next(iter(pool), None)
                if best is None:
                    continue
                seen.add(best["reddit_fullname"])
                rel = best.get("reliability") or {}
                note = (f"LLM ensemble label: "
                        f"{'sarcastic' if best['labels']['sarcastic'] else 'not sarcastic'} "
                        f"(resolved by {rel.get('resolved_by')}, sarcasm votes "
                        f"{rel.get('sarcasm_votes')}). Held out of the model's training fold.")
                picked.append(self._example(best, lang, note, by_author))

        # every gold disagreement — the honest hard cases (ANNOTATION_PROVENANCE §3)
        for row in sorted(self.rows, key=lambda r: r["reddit_fullname"]):
            gold = row.get("human_gold")
            if not gold or gold.get("sarcastic") is None:
                continue
            if bool(gold["sarcastic"]) == bool(row["labels"]["sarcastic"]):
                continue
            if row["reddit_fullname"] in seen:
                continue
            seen.add(row["reddit_fullname"])
            human = "sarcastic" if gold["sarcastic"] else "not sarcastic"
            ens = "sarcastic" if row["labels"]["sarcastic"] else "not sarcastic"
            note = f"hard case: human annotator said {human}, LLM ensemble said {ens}."
            if row["reddit_fullname"] in train:
                note += " This row is inside the model's training fold, so the model was fit on the ensemble's call."
            picked.append(self._example(row, "hard", note, by_author))
        return picked

    # -------------------------------------------------------------- predict
    def predict(self, req: dict) -> dict:
        notes: list[str] = []
        text = _clip(req.get("text"))
        if not text:
            raise ValueError("text is empty")
        if len((req.get("text") or "")) > MAX_TEXT:
            notes.append(f"text clipped to {MAX_TEXT} characters")

        use_conv = bool(req.get("use_conv", True))
        use_temp = bool(req.get("use_temp", True))
        use_ret = bool(req.get("use_ret", True))

        conv: list[ConvItem] = []
        if use_conv:
            sub = req.get("submission") or None
            if sub:
                joined = "\n\n".join(t for t in (_clip(sub.get("title")), _clip(sub.get("selftext"))) if t)
                if joined.strip():
                    conv.append(ConvItem(joined, ROLE_SUBMISSION, True))
            for p in (req.get("parents") or [])[:MAX_CONV]:
                if _clip(p.get("text")):
                    conv.append(ConvItem(_clip(p.get("text")), ROLE_ANCESTOR, bool(p.get("is_submitter"))))
            for p in (req.get("replies") or [])[:MAX_CONV]:
                if _clip(p.get("text")):
                    conv.append(ConvItem(_clip(p.get("text")), ROLE_REPLY, bool(p.get("is_submitter"))))

        temp: list[TemporalItem] = []
        if use_temp:
            window = self.cfg.temporal_window_hours
            for i, h in enumerate((req.get("history") or [])[:MAX_HISTORY]):
                body = _clip(h.get("text"))
                if not body:
                    continue
                try:
                    hours = float(h.get("hours_ago"))
                except (TypeError, ValueError):
                    notes.append("an earlier post had no usable 'hours ago' and was dropped")
                    continue
                if hours <= 0:
                    notes.append("earlier posts must be strictly before the target — one was dropped")
                    continue
                if window is not None and hours > float(window):
                    notes.append(f"an earlier post was older than the {window:g} h temporal window and was dropped")
                    continue
                temp.append(TemporalItem(body, hours, f"demo-{i}"))

        exemplars = {"sarcastic": [], "non_sarcastic": []}
        if use_ret:
            exemplars = self.retrieve(text, req.get("exclude_thread"))

        sample = {
            "fullname": "demo", "target_text": text,
            "conv": conv, "temp": temp,
            "ret_sarc": [e["text"] for e in exemplars["sarcastic"]],
            "ret_nonsarc": [e["text"] for e in exemplars["non_sarcastic"]],
            "label": 0, "soft_pos": 0.5, "cues": [-1.0] * 4, "shift": -1, "lang": 0,
            "resolved_by": "unanimous",
        }
        with self.lock, torch.no_grad():
            batch = self.to_device(self.collate([sample]), self.device)
            out = self.model(batch)
            logits = out["logits"][0].float().cpu()
            gate_vec = out["gates"][0].float().cpu().tolist() if out["gates"] is not None else []
            target_emb = out["target_emb"][0].float().cpu().numpy()
            features = out["features"][0].float().cpu().numpy()

        margin = float(logits[1] - logits[0])
        prob_raw = _sigmoid(margin)
        prob_cal = _sigmoid(margin / self.temperature)
        verdict = "sarcastic" if prob_raw >= self.threshold else "not sarcastic"

        gates = {name: 0.0 for name in ("conv", "temp", "ret")}
        for name, value in zip(self.model.active, gate_vec):
            gates[name] = round(float(value), 4)

        sentiment = None
        if self.heads:
            literal = str(self.heads["stage1_literal_on_target_emb"].predict(target_emb[None, :])[0])
            flagged = verdict == "sarcastic"
            intended = (str(self.heads["stage2_intended_on_features"].predict(features[None, :])[0])
                        if flagged else literal)
            sentiment = {"literal": literal, "intended": intended, "flagged": flagged}

        return {
            "prob_raw": round(prob_raw, 6),
            "prob": round(prob_cal, 6),
            "threshold": round(self.threshold, 6),
            "threshold_cal": round(self.threshold_cal, 6),
            "verdict": verdict,
            "margin": round(margin, 6),
            "gates": gates,
            "channels_supplied": {
                "conv": len(conv), "temp": len(temp),
                "ret": len(sample["ret_sarc"]) + len(sample["ret_nonsarc"]),
            },
            "exemplars": {
                key: [{"text": (e["text"][:EXEMPLAR_CHARS] + "…")
                       if len(e["text"]) > EXEMPLAR_CHARS else e["text"],
                       "similarity": e["similarity"], "language": e["language"]}
                      for e in items]
                for key, items in exemplars.items()
            },
            "sentiment": sentiment,
            "notes": notes,
            "model": {"trained": self.trained, "checkpoint": self.checkpoint_path.name},
        }

    # ----------------------------------------------------------------- meta
    def meta(self) -> dict:
        return {
            "trained": self.trained,
            "checkpoint": self.checkpoint_path.name,
            "threshold": round(self.threshold, 6),
            "threshold_cal": round(self.threshold_cal, 6),
            "temperature": round(self.temperature, 6),
            "label_authority": self.label_authority,
            "bank_sizes": {"sarcastic": len(self.banks["sarcastic"].names),
                           "non_sarcastic": len(self.banks["non_sarcastic"].names)},
            "retrieval_k": int(self.cfg.retrieval_k),
            "temporal_k": int(self.cfg.temporal_k),
            "temporal_window_hours": self.cfg.temporal_window_hours,
            "device": str(self.device),
            "sentiment_heads": self.heads is not None,
            "honesty": HONESTY,
            "limits": {"text_chars": MAX_TEXT, "conv_items": MAX_CONV, "history_items": MAX_HISTORY},
        }
