"""One dataclass holding every switch in docs/MODEL_PLAN.md.

Defaults are the thesis-committed baseline (guide §4). Every §9 upgrade sits
behind a flag that defaults to OFF so each manuscript claim stays reproducible
with upgrades disabled.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import yaml


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until pyproject.toml is found (works from notebooks/ or root)."""
    p = (start or Path.cwd()).resolve()
    for candidate in [p, *p.parents]:
        if (candidate / "pyproject.toml").exists():
            return candidate
    raise FileNotFoundError(f"no pyproject.toml above {p}")


@dataclass
class LeischeConfig:
    # ---- paths / identity -------------------------------------------------
    data_dir: str = "data"
    dataset_version: str = "v1"
    results_dir: str = "results"
    cache_dir: str = "data/cache"
    run_name: str = "dev"

    # ---- stage 1: shared encoder (guide §4.1) ----------------------------
    encoder_name: str = "xlm-roberta-base"
    pooling: str = "mean"  # "mean" | "cls" — verify both in notebook 03
    d_model: int = 256
    max_len_target: int = 192
    max_len_context: int = 96
    max_len_selftext: int = 128

    # ---- ablation flags (RQ2, guide §7.3) --------------------------------
    use_conv: bool = False
    use_temp: bool = False
    use_ret: bool = False

    # ---- stage 2: conversational + temporal (guide §4.2) -----------------
    conv_role_embeddings: bool = True  # submission / ancestor / reply + is_submitter
    temporal_k: int = 10
    temporal_lambda_init: float = 0.1
    temporal_lambda_learnable: bool = False  # thesis: fixed decay; learnable = flagged generalization
    missing_channel: str = "zeros"  # "zeros" | "learned"

    # ---- stage 3: retrieval (guide §4.3) ---------------------------------
    retrieval_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    retrieval_k: int = 5
    retrieval_prototypes: bool = False  # §9.5 centroid alternative

    # ---- stage 5: classifier (guide §4.5) --------------------------------
    mlp_hidden: int = 256
    dropout: float = 0.2

    # ---- training defaults (guide §4) ------------------------------------
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    warmup_ratio: float = 0.1
    max_epochs: int = 10
    patience: int = 3
    batch_size: int = 8
    grad_accum: int = 2
    fp16: bool = True
    grad_clip: float = 1.0
    seeds: list[int] = field(default_factory=lambda: [13, 42, 7])
    class_weighting: str = "inverse_freq"  # per training fold, never global
    max_steps_per_epoch: int | None = None  # smoke throttle; None = full epoch

    # ---- §9 upgrades — ALL OFF by default --------------------------------
    sample_weighting: bool = False  # §9.1 provenance weights
    sample_weights: dict = field(
        default_factory=lambda: {"unanimous": 1.0, "majority": 0.9, "adjudicator": 0.7, "human": 1.0}
    )
    soft_labels: bool = False  # §9.1 vote-share targets
    focal_loss: bool = False  # §9.2
    focal_gamma: float = 2.0
    weighted_sampler: bool = False  # §9.2
    target_pos_frac: float = 0.25
    aux_cue_heads: bool = False  # §9.3
    aux_polarity_shift: bool = False  # §9.3
    aux_language_head: bool = False  # §9.3
    aux_loss_weight: float = 0.25
    freeze_bottom_layers: int = 0  # §9.4 (0 = unfreeze all, the thesis default)
    layerwise_lr_decay: float | None = None  # §9.4 (e.g. 0.9)
    temperature_scaling: bool = False  # §9.8
    tune_threshold_on_val: bool = False  # §9.8

    # ---- evaluation (guide §7) -------------------------------------------
    n_folds: int = 5
    natural_only_metrics: bool = True  # §7.2 test-fold hygiene

    # ---- smoke mode -------------------------------------------------------
    # True → every emitted metric / table / artifact is prefixed SMOKE.
    # train.py refuses smoke=False while the §10 readiness gate fails.
    smoke: bool = True

    # -----------------------------------------------------------------------
    def tag(self) -> str:
        """Prefix for every printed metric line and artifact name."""
        return "SMOKE " if self.smoke else ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path, **overrides) -> "LeischeConfig":
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        raw.update(overrides)
        unknown = set(raw) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise KeyError(f"unknown config keys in {path}: {sorted(unknown)}")
        return cls(**raw)

    # resolved paths (relative to the repo root so notebooks/ and root both work)
    def root(self) -> Path:
        return find_repo_root()

    def data_path(self) -> Path:
        return self.root() / self.data_dir

    def results_path(self) -> Path:
        return self.root() / self.results_dir

    def cache_path(self) -> Path:
        p = self.root() / self.cache_dir
        p.mkdir(parents=True, exist_ok=True)
        return p

    def dataset_file(self) -> Path:
        return self.data_path() / f"dataset-{self.dataset_version}.jsonl"

    def corpus_file(self) -> Path:
        return self.data_path() / f"corpus-{self.dataset_version}.jsonl"

    def card_file(self) -> Path:
        return self.data_path() / "dataset_card.json"

    def folds_file(self) -> Path:
        return self.results_path() / f"folds-{self.dataset_version}.json"
