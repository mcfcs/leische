"""Dataset/collator, fold loop, loss, early stopping, seeds (guide §4 defaults).

Readiness-gate enforcement lives here: run_cv() raises on any smoke=False run
while the §10 gate fails, so "do not train for results yet" is code, not
convention. Every printed line and artifact from a smoke config carries the
SMOKE prefix.
"""

from __future__ import annotations

import copy
import json
import random
import subprocess
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from . import contexts as C
from . import data as D
from .config import LeischeConfig
from .encoders import Tokenize
from .model import ContextAwareSarcasmModel

LANG2ID = {"english": 0, "tagalog": 1, "taglish": 2}


# ------------------------------------------------------------------ determinism


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return "unknown"


# ------------------------------------------------------------------ context assembly


class Assembly:
    """Fold-independent context maps, built once per dataset version.

    Conversational + temporal never change across folds; retrieval is
    fold-dependent and built by fold_retrieval().
    """

    def __init__(self, cfg: LeischeConfig, df: pd.DataFrame, corpus: pd.DataFrame | None):
        self.cfg = cfg
        self.df = df
        self.conv = {r["reddit_fullname"]: C.build_conversational(r) for _, r in df.iterrows()}
        if cfg.use_temp or corpus is not None:
            index = C.TemporalIndex(corpus) if corpus is not None else None
            self.temp = {
                r["reddit_fullname"]: (
                    index.history(r["author_hash"], r["created_dt"],
                                  r["reddit_fullname"], k=cfg.temporal_k)
                    if index else []
                )
                for _, r in df.iterrows()
            }
        else:
            self.temp = {f: [] for f in df["reddit_fullname"]}
        self.emb: C.RetrievalEmbeddings | None = None

    def retrieval_embeddings(self) -> C.RetrievalEmbeddings:
        if self.emb is None:
            self.emb = C.RetrievalEmbeddings.build(
                self.df, self.cfg.retrieval_model, self.cfg.cache_path(),
                self.cfg.dataset_version)
        return self.emb

    def fold_retrieval(self, train_fullnames: list[str],
                       query_fullnames: list[str]) -> dict[str, C.RetrievalResult]:
        return C.build_fold_retrieval(self.df, self.retrieval_embeddings(),
                                      train_fullnames, query_fullnames,
                                      k=self.cfg.retrieval_k)


# ------------------------------------------------------------------ dataset / collator


class SarcasmDataset(Dataset):
    def __init__(self, df: pd.DataFrame, fullnames: list[str], assembly: Assembly,
                 retrieval: dict[str, C.RetrievalResult] | None):
        rows = D.rows_by_fullname(df)
        self.samples = [rows[f] for f in fullnames]
        self.assembly = assembly
        self.retrieval = retrieval or {}
        self.text_of = dict(zip(df["reddit_fullname"], df["text"]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        r = self.samples[i]
        f = r["reddit_fullname"]
        labels, rel = r["labels"], r["reliability"]
        ret = self.retrieval.get(f)
        votes = rel.get("sarcasm_votes") or ""
        # §9.1 soft targets from vote share: 3-0 → 0.95, else (escalated) → 0.75
        p_conf = 0.95 if votes == "3-0" else 0.75
        soft_pos = p_conf if labels["sarcastic"] else 1.0 - p_conf
        return {
            "fullname": f,
            "target_text": r["text"],
            "conv": self.assembly.conv.get(f, []),
            "temp": self.assembly.temp.get(f, []),
            "ret_sarc": [self.text_of[x] for x in ret.sarc] if ret else [],
            "ret_nonsarc": [self.text_of[x] for x in ret.nonsarc] if ret else [],
            "label": int(labels["sarcastic"]),
            "soft_pos": soft_pos,
            "cues": [float(labels["cues"][k]) for k in D.CUE_KEYS],
            "shift": int(labels["literal_sentiment"] != labels["intended_sentiment"]),
            "lang": LANG2ID[labels["language"]],
            "weight": 1.0,  # provenance weight filled by collator when flagged
            "resolved_by": rel["resolved_by"],
        }


class Collator:
    def __init__(self, cfg: LeischeConfig, tok: Tokenize):
        self.cfg = cfg
        self.tok = tok

    def _flatten(self, per_sample: list[list[str]], kind: str = "context",
                 extras: list[list] | None = None) -> dict:
        texts, batch_idx = [], []
        flat_extras: list = []
        for b, items in enumerate(per_sample):
            for j, t in enumerate(items):
                texts.append(t)
                batch_idx.append(b)
                if extras is not None:
                    flat_extras.append(extras[b][j])
        enc = self.tok(texts, kind=kind)
        out = {**enc, "batch_idx": torch.tensor(batch_idx, dtype=torch.long)}
        if extras is not None:
            out["extras"] = flat_extras
        return out

    def __call__(self, samples: list[dict]) -> dict:
        cfg = self.cfg
        batch: dict = {
            "fullnames": [s["fullname"] for s in samples],
            "target": self.tok([s["target_text"] for s in samples], kind="target"),
            "labels": torch.tensor([s["label"] for s in samples], dtype=torch.long),
            "soft_pos": torch.tensor([s["soft_pos"] for s in samples], dtype=torch.float),
            "cues": torch.tensor([s["cues"] for s in samples], dtype=torch.float),
            "shift": torch.tensor([s["shift"] for s in samples], dtype=torch.long),
            "lang": torch.tensor([s["lang"] for s in samples], dtype=torch.long),
        }
        weights = [
            cfg.sample_weights.get(s["resolved_by"], 1.0) if cfg.sample_weighting else 1.0
            for s in samples
        ]
        batch["sample_weight"] = torch.tensor(weights, dtype=torch.float)

        if cfg.use_conv:
            conv = self._flatten([[it.text for it in s["conv"]] for s in samples],
                                 extras=[[(it.role, it.is_submitter) for it in s["conv"]]
                                         for s in samples])
            extras = conv.pop("extras")
            conv["role"] = torch.tensor([e[0] for e in extras], dtype=torch.long)
            conv["is_submitter"] = torch.tensor([int(e[1]) for e in extras], dtype=torch.long)
            batch["conv"] = conv
        if cfg.use_temp:
            temp = self._flatten([[it.text for it in s["temp"]] for s in samples],
                                 extras=[[it.delta_days for it in s["temp"]] for s in samples])
            temp["delta_days"] = torch.tensor(temp.pop("extras"), dtype=torch.float)
            batch["temp"] = temp
        if cfg.use_ret:
            batch["ret_sarc"] = self._flatten([s["ret_sarc"] for s in samples])
            batch["ret_nonsarc"] = self._flatten([s["ret_nonsarc"] for s in samples])
        return batch


def to_device(batch: dict, device: torch.device) -> dict:
    out = {}
    for k, v in batch.items():
        if isinstance(v, dict):
            out[k] = {kk: (vv.to(device) if torch.is_tensor(vv) else vv) for kk, vv in v.items()}
        elif torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


# ------------------------------------------------------------------ loss (guide §4.5 + §9)


def compute_loss(outputs: dict, batch: dict, cfg: LeischeConfig,
                 class_w: torch.Tensor) -> torch.Tensor:
    logits = outputs["logits"]
    labels = batch["labels"]

    if cfg.soft_labels:
        soft = torch.stack([1.0 - batch["soft_pos"], batch["soft_pos"]], dim=-1)
        logp = F.log_softmax(logits, dim=-1)
        per = -(soft * logp).sum(-1) * class_w[labels]
    elif cfg.focal_loss:
        ce = F.cross_entropy(logits, labels, weight=class_w, reduction="none")
        pt = torch.exp(-ce)
        per = ((1 - pt) ** cfg.focal_gamma) * ce
    else:
        per = F.cross_entropy(logits, labels, weight=class_w, reduction="none")

    per = per * batch["sample_weight"]
    loss = per.mean()

    aux = 0.0
    if cfg.aux_cue_heads:
        aux = aux + F.binary_cross_entropy_with_logits(outputs["cue_logits"], batch["cues"])
    if cfg.aux_polarity_shift:
        aux = aux + F.cross_entropy(outputs["shift_logits"], batch["shift"])
    if cfg.aux_language_head:
        aux = aux + F.cross_entropy(outputs["lang_logits"], batch["lang"])
    if isinstance(aux, torch.Tensor):
        loss = loss + cfg.aux_loss_weight * aux
    return loss


# ------------------------------------------------------------------ optimizer


def build_optimizer(model: ContextAwareSarcasmModel, cfg: LeischeConfig) -> torch.optim.AdamW:
    encoder_ids = {id(p) for p in model.encoder.backbone.parameters()}
    enc_params = [p for p in model.parameters() if id(p) in encoder_ids and p.requires_grad]
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids and p.requires_grad]
    groups = [
        {"params": enc_params, "lr": cfg.lr_encoder},
        {"params": head_params, "lr": cfg.lr_heads},
    ]
    if cfg.layerwise_lr_decay is not None:
        # §9.4: replace the flat encoder group with per-layer decayed lrs
        layers = model.encoder.backbone.encoder.layer
        n = len(layers)
        groups = [{"params": head_params, "lr": cfg.lr_heads}]
        emb_params = [p for p in model.encoder.backbone.embeddings.parameters() if p.requires_grad]
        groups.append({"params": emb_params, "lr": cfg.lr_encoder * cfg.layerwise_lr_decay ** n})
        for i, layer in enumerate(layers):
            groups.append({
                "params": [p for p in layer.parameters() if p.requires_grad],
                "lr": cfg.lr_encoder * cfg.layerwise_lr_decay ** (n - 1 - i),
            })
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer, total_steps: int, warmup_ratio: float):
    warmup = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        return max(0.0, (total_steps - step) / max(1, total_steps - warmup))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ------------------------------------------------------------------ train / predict


def _safe_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    from sklearn.metrics import f1_score
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(f1_score(y_true, y_pred, zero_division=0))


def predict(model: ContextAwareSarcasmModel, loader: DataLoader, device: torch.device,
            fp16: bool = True, collect_features: bool = False) -> pd.DataFrame:
    model.eval()
    rows = []
    feats_all = []
    with torch.no_grad():
        for batch in loader:
            batch = to_device(batch, device)
            with torch.autocast("cuda", enabled=fp16 and device.type == "cuda"):
                out = model(batch)
            probs = F.softmax(out["logits"].float(), dim=-1)[:, 1].cpu().numpy()
            gates = out["gates"].float().cpu().numpy() if out["gates"] is not None else None
            if collect_features:
                feats_all.append(out["features"].float().cpu().numpy())
            for i, f in enumerate(batch["fullnames"]):
                row = {
                    "reddit_fullname": f,
                    "y_true": int(batch["labels"][i].item()),
                    "prob": float(probs[i]),
                    "pred": int(probs[i] >= 0.5),
                }
                if gates is not None:
                    for j, name in enumerate(model.gate_channels()):
                        row[f"gate_{name}"] = float(gates[i, j])
                rows.append(row)
    df = pd.DataFrame(rows)
    if collect_features:
        df.attrs["features"] = np.concatenate(feats_all) if feats_all else np.zeros((0,))
    return df


def train_model(
    model: ContextAwareSarcasmModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: LeischeConfig,
    device: torch.device,
    class_w: tuple[float, float] = (1.0, 1.0),
    log_every: int = 0,
) -> dict:
    """Early-stopped training; restores the best-val-F1 weights. Returns history."""
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    steps_per_epoch = len(train_loader) if cfg.max_steps_per_epoch is None else min(
        len(train_loader), cfg.max_steps_per_epoch)
    total_steps = max(1, steps_per_epoch * cfg.max_epochs // cfg.grad_accum)
    scheduler = build_scheduler(optimizer, total_steps, cfg.warmup_ratio)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device.type == "cuda")
    w = torch.tensor(class_w, dtype=torch.float, device=device)

    best_f1, best_state, patience_left = -1.0, None, cfg.patience
    history = {"train_loss": [], "val_f1": []}

    for epoch in range(cfg.max_epochs):
        model.train()
        losses = []
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            if cfg.max_steps_per_epoch is not None and step >= cfg.max_steps_per_epoch:
                break
            batch = to_device(batch, device)
            with torch.autocast("cuda", enabled=cfg.fp16 and device.type == "cuda"):
                out = model(batch)
                loss = compute_loss(out, batch, cfg, w) / cfg.grad_accum
            scaler.scale(loss).backward()
            if (step + 1) % cfg.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
            losses.append(float(loss.item()) * cfg.grad_accum)

        val = predict(model, val_loader, device, fp16=cfg.fp16)
        val_f1 = _safe_f1(val["y_true"].to_numpy(), val["pred"].to_numpy())
        history["train_loss"].append(float(np.mean(losses)) if losses else float("nan"))
        history["val_f1"].append(val_f1)
        if log_every:
            print(f"{cfg.tag()}epoch {epoch}: train_loss={history['train_loss'][-1]:.4f} "
                  f"val_f1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1, patience_left = val_f1, cfg.patience
            best_state = copy.deepcopy(model.state_dict())
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    history["best_val_f1"] = best_f1
    return history


# ------------------------------------------------------------------ fold loop


def make_loaders(cfg: LeischeConfig, df: pd.DataFrame, assembly: Assembly,
                 fold: dict, tok: Tokenize, seed: int) -> dict[str, DataLoader]:
    retrieval = None
    if cfg.use_ret:
        queries = fold["train"] + fold["val"] + fold["test"]
        retrieval = assembly.fold_retrieval(fold["train"], queries)

    collate = Collator(cfg, tok)
    loaders = {}
    for part in ("train", "val", "test"):
        ds = SarcasmDataset(df, fold[part], assembly, retrieval)
        if part == "train":
            if cfg.weighted_sampler:
                y = np.array([int(r["labels"]["sarcastic"]) for r in ds.samples])
                pos_frac = max(y.mean(), 1e-6)
                w = np.where(y == 1, cfg.target_pos_frac / pos_frac,
                             (1 - cfg.target_pos_frac) / max(1 - pos_frac, 1e-6))
                sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double),
                                                num_samples=len(ds), replacement=True)
                loaders[part] = DataLoader(ds, batch_size=cfg.batch_size, sampler=sampler,
                                           collate_fn=collate)
            else:
                g = torch.Generator().manual_seed(seed)
                loaders[part] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                                           generator=g, collate_fn=collate)
        else:
            loaders[part] = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                                       collate_fn=collate)
    return loaders


def enforce_gate(cfg: LeischeConfig, card: dict, df: pd.DataFrame) -> D.GateReport:
    gate = D.readiness_gate(card, df, cfg.folds_file())
    if not gate.passed and not cfg.smoke:
        raise RuntimeError(
            "§10 readiness gate FAILED — real (non-smoke) training is not "
            "legitimate on this export:\n" + gate.render()
        )
    return gate


def run_cv(cfg: LeischeConfig, df: pd.DataFrame, corpus: pd.DataFrame | None,
           card: dict, folds: list[dict], verbose: bool = True) -> dict:
    """Folds × seeds loop. Returns {'fold_metrics': df, 'predictions': df, 'histories': ...}.

    Writes results/<run_name>/ with metrics, predictions, config snapshot and
    dataset identity.
    """
    from .evaluate import safe_metrics

    gate = enforce_gate(cfg, card, df)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = Tokenize(cfg.encoder_name, cfg.max_len_target, cfg.max_len_context,
                   cfg.max_len_selftext)
    assembly = Assembly(cfg, df, corpus)
    meta = df[["reddit_fullname"]].copy()
    meta["language"] = D.language(df)
    meta["record_type"] = df["record_type"]
    meta["resolved_by"] = df["reliability"].map(lambda r: r["resolved_by"])
    meta["natural"] = D.natural_mask(df)

    fold_rows, pred_frames, histories = [], [], {}
    for fold in folds:
        for seed in cfg.seeds:
            set_seed(seed)
            loaders = make_loaders(cfg, df, assembly, fold, tok, seed)
            train_df = df[df["reddit_fullname"].isin(fold["train"])]
            w = D.class_weights(train_df) if cfg.class_weighting == "inverse_freq" else (1.0, 1.0)
            model = ContextAwareSarcasmModel(cfg)
            hist = train_model(model, loaders["train"], loaders["val"], cfg, device, w)
            test_pred = predict(model, loaders["test"], device, fp16=cfg.fp16)
            test_pred["fold"], test_pred["seed"] = fold["fold"], seed
            test_pred = test_pred.merge(meta, on="reddit_fullname", how="left")
            pred_frames.append(test_pred)
            m = safe_metrics(test_pred["y_true"].to_numpy(), test_pred["pred"].to_numpy())
            fold_rows.append({"fold": fold["fold"], "seed": seed, **m,
                              "best_val_f1": hist["best_val_f1"]})
            histories[(fold["fold"], seed)] = hist
            if verbose:
                print(f"{cfg.tag()}fold {fold['fold']} seed {seed}: "
                      f"test_f1={m['f1']:.4f} (n_pos={m['n_pos']}) "
                      f"val_f1={hist['best_val_f1']:.4f}")
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    fold_metrics = pd.DataFrame(fold_rows)
    predictions = pd.concat(pred_frames, ignore_index=True)

    out_dir = cfg.results_path() / cfg.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "SMOKE-" if cfg.smoke else ""
    fold_metrics.to_csv(out_dir / f"{prefix}fold_metrics.csv", index=False)
    predictions.to_csv(out_dir / f"{prefix}predictions.csv", index=False)
    snapshot = {
        "config": cfg.to_dict(),
        "dataset_identity": D.dataset_identity(card),
        "leische_commit": git_commit(),
        "gate_passed": gate.passed,
        "smoke": cfg.smoke,
    }
    (out_dir / f"{prefix}run.json").write_text(json.dumps(snapshot, indent=2), encoding="utf-8")
    return {"fold_metrics": fold_metrics, "predictions": predictions,
            "histories": histories, "gate": gate}


# ------------------------------------------------------------------ smoke test 1: overfit 16 rows


def overfit_smoke(cfg: LeischeConfig, df: pd.DataFrame, corpus: pd.DataFrame | None,
                  card: dict, n: int = 16, max_steps: int = 150,
                  target_loss: float = 0.05, seed: int = 13) -> dict:
    """Architecture sanity (§10): drive training loss to ~0 on n rows.

    Includes every sarcastic row first (the pilot has only 8), tops up with
    non-sarcastic rows. Always a SMOKE run by construction.
    """
    cfg = copy.deepcopy(cfg)
    cfg.smoke = True  # overfitting is never a real result
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pos = df[D.sarcastic(df)]["reddit_fullname"].tolist()
    neg = df[~D.sarcastic(df)]["reddit_fullname"].tolist()
    chosen = (pos + neg)[:n]

    tok = Tokenize(cfg.encoder_name, cfg.max_len_target, cfg.max_len_context,
                   cfg.max_len_selftext)
    assembly = Assembly(cfg, df, corpus)
    retrieval = assembly.fold_retrieval(chosen, chosen) if cfg.use_ret else None
    ds = SarcasmDataset(df, chosen, assembly, retrieval)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(seed),
                        collate_fn=Collator(cfg, tok))

    model = ContextAwareSarcasmModel(cfg).to(device)
    # overfitting wants a hotter head lr; encoder lr stays at the baseline value
    opt_cfg = copy.deepcopy(cfg)
    opt_cfg.lr_heads = 1e-3
    optimizer = build_optimizer(model, opt_cfg)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.fp16 and device.type == "cuda")
    w = torch.tensor([1.0, 1.0], device=device)

    losses: list[float] = []
    model.train()
    step = 0
    while step < max_steps:
        for batch in loader:
            if step >= max_steps:
                break
            batch = to_device(batch, device)
            with torch.autocast("cuda", enabled=cfg.fp16 and device.type == "cuda"):
                out = model(batch)
                loss = compute_loss(out, batch, cfg, w)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))
            step += 1
            if len(losses) >= 5 and np.mean(losses[-5:]) < target_loss:
                step = max_steps
                break

    final = float(np.mean(losses[-5:]))
    passed = final < target_loss
    print(f"{cfg.tag()}overfit-{len(chosen)}: final mean loss {final:.4f} over last 5 steps "
          f"({len(losses)} steps) → {'PASS' if passed else 'FAIL'} (target < {target_loss})")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return {"losses": losses, "final_loss": final, "passed": passed, "rows": chosen}
