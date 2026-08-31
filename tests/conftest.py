from pathlib import Path

import pandas as pd
import pytest

from leische import data as D
from leische.config import LeischeConfig

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def cfg() -> LeischeConfig:
    return LeischeConfig()


@pytest.fixture(scope="session")
def pilot(cfg) -> pd.DataFrame:
    path = REPO / "data" / "dataset-v1.jsonl"
    if not path.exists():
        pytest.skip("pilot data not synced — run scripts/sync_data.py")
    return D.load_dataset(path)


@pytest.fixture(scope="session")
def card(cfg) -> dict:
    path = REPO / "data" / "dataset_card.json"
    if not path.exists():
        pytest.skip("pilot data not synced — run scripts/sync_data.py")
    return D.load_card(path)
