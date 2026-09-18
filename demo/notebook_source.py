"""Pull the model code straight out of `leische_pipeline.ipynb`.

The notebook is the single source of truth (FABILE_BRIEF §0), so the demo does
not re-implement Config, the model or the collator — it locates the cells by a
content marker and execs them, exactly the way `tools/check_model_contract.py`
does.  If the notebook changes, the demo changes with it.

Three pieces are needed:

  §2  the Config dataclass          (marker: "@dataclass\\nclass Config:")
  §7  the 5-stage model             (marker: "class ContextAwareSarcasmModel")
  §8  Collator + to_device          (a slice of the training-harness cell —
                                     the rest of that cell builds dataframes
                                     from globals a server does not have)

`apply_memory_guard` at the end of the Config cell is the one thing that is
deliberately neutered: it would call `torch.cuda.set_per_process_memory_fraction`
on a GPU the trainer owns.  The cell is exec'd against a torch shim whose
`cuda.is_available()` is False, so the guard returns early and the demo never
touches the trainer's VRAM budget.  Nothing else in that cell uses torch.
"""

from __future__ import annotations

import json
import math
import types
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "leische_pipeline.ipynb"

# roles, copied from the §6 channels cell so conv items match training exactly
ROLE_SUBMISSION, ROLE_ANCESTOR, ROLE_REPLY = 0, 1, 2


@dataclass
class ConvItem:
    text: str
    role: int
    is_submitter: bool


@dataclass
class TemporalItem:
    text: str
    delta_hours: float
    reddit_fullname: str


def _code_cells() -> list[str]:
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def _cell_with(marker: str, cells: list[str]) -> str:
    for src in cells:
        if marker in src:
            return src
    raise RuntimeError(f"no code cell in {NOTEBOOK.name} contains {marker!r}")


def load_pipeline() -> dict:
    """Exec the three cells into one namespace and return it.

    Returns a dict carrying at least: Config, Tokenize, ContextAwareSarcasmModel,
    Collator, to_device.
    """
    cells = _code_cells()

    # ---- §2 Config -------------------------------------------------------
    cfg_ns: dict = {
        "dataclass": dataclass, "field": field, "asdict": asdict,
        # see the module docstring: the demo must not re-cap the trainer's GPU
        "torch": types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False)),
        "VRAM_GB": 0.0,
        "print": lambda *a, **k: None,
    }
    exec(compile(_cell_with("@dataclass\nclass Config:", cells), "nb:config", "exec"), cfg_ns)

    # ---- §7 model --------------------------------------------------------
    ns: dict = {
        "Config": cfg_ns["Config"], "torch": torch, "nn": nn, "F": F, "math": math,
        "AutoModel": AutoModel, "AutoTokenizer": AutoTokenizer,
        "print": lambda *a, **k: None,
    }
    exec(compile(_cell_with("class ContextAwareSarcasmModel", cells), "nb:model", "exec"), ns)

    # ---- §8 Collator + to_device (slice only) ----------------------------
    harness = _cell_with("class Collator", cells)
    start = harness.index("class Collator")
    end = harness.index("def class_weights")
    ns["ROLE_SUBMISSION"] = ROLE_SUBMISSION
    exec(compile(harness[start:end], "nb:collator", "exec"), ns)

    for name in ("Config", "Tokenize", "ContextAwareSarcasmModel", "Collator", "to_device"):
        if name not in ns:
            raise RuntimeError(f"{name} did not come out of the notebook — cell markers moved?")
    return ns
