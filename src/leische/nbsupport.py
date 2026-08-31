"""Shared notebook bootstrap: load export + card, evaluate the §10 gate,
print the SMOKE banner. Keeps every notebook's first cell identical and thin."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import data as D
from .config import LeischeConfig, find_repo_root


def bootstrap(config: str | Path | None = None, need_corpus: bool = False,
              **overrides) -> tuple[LeischeConfig, pd.DataFrame, pd.DataFrame | None, dict, D.GateReport]:
    root = find_repo_root()
    if config is not None:
        cfg = LeischeConfig.from_yaml(root / config, **overrides)
    else:
        cfg = LeischeConfig(**overrides)

    if not cfg.dataset_file().exists():
        raise FileNotFoundError(
            f"{cfg.dataset_file()} missing — run `python scripts/sync_data.py` first")

    df = D.load_dataset(cfg.dataset_file())
    corpus = D.load_corpus(cfg.corpus_file()) if need_corpus else None
    card = D.load_card(cfg.card_file())
    gate = D.readiness_gate(card, df, cfg.folds_file())

    if not gate.passed and not cfg.smoke:
        # the gate, not the config, decides when real numbers become legitimate
        print("readiness gate failed → forcing smoke=True")
        cfg.smoke = True

    ident = D.dataset_identity(card)
    if cfg.smoke:
        print("=" * 72)
        print(" SMOKE MODE — every number in this notebook is a harness or")
        print(" architecture check on the pilot export, NOT a reportable result.")
        print("=" * 72)
    print(f"dataset identity: {ident}")
    print(f"rows: {len(df)}  sarcastic: {int(D.sarcastic(df).sum())}  "
          f"corpus: {'loaded' if corpus is not None else 'not loaded'}")
    print()
    print(gate.render())
    return cfg, df, corpus, card, gate
