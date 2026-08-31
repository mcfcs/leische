"""Copy the uyam dataset exports into leische/data (data/ is gitignored).

Usage:
    python scripts/sync_data.py [--uyam PATH] [--version v1]

uyam is the single source of truth for the exports; this script only copies.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_UYAM = REPO_ROOT.parent / "uyam"


def sync(uyam_path: Path, version: str) -> None:
    src = uyam_path / "data" / "annotated"
    dst = REPO_ROOT / "data"
    dst.mkdir(exist_ok=True)

    files = [
        f"dataset-{version}.jsonl",
        f"dataset-{version}.parquet",
        f"corpus-{version}.jsonl",
        "dataset_card.json",
    ]
    for name in files:
        source = src / name
        if not source.exists():
            print(f"  MISSING in uyam export: {source}")
            continue
        shutil.copy2(source, dst / name)
        print(f"  copied {name} ({source.stat().st_size:,} bytes)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--uyam", type=Path, default=DEFAULT_UYAM, help="path to the uyam repo")
    ap.add_argument("--version", default="v1", help="dataset version to copy (v1, v2, ...)")
    args = ap.parse_args()
    if not (args.uyam / "data" / "annotated").exists():
        raise SystemExit(f"uyam export dir not found under {args.uyam}")
    sync(args.uyam, args.version)
    print("done.")
