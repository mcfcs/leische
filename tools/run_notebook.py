"""Execute `leische_pipeline.ipynb` section by section from the terminal.

The notebook is the single source of truth — every function lives inline in
it.  Long training stages, though, are better driven from a terminal than from
a browser tab: the log survives a closed editor, a crash leaves a traceback in
a file, and a stage can be started on its own without re-running the EDA.

This script extracts the notebook's code cells, labels each one with the
section it belongs to (from the `## N · ...` / `### Nb · ...` headings), and
executes the selected sections in order inside ONE shared namespace — exactly
what "Run All" does, minus the browser.  Nothing is duplicated: change the
notebook and the next run picks it up.

    uv run python tools/run_notebook.py --list
    uv run python tools/run_notebook.py --until 6                 # §1–§6 (data + gate + channels)
    uv run python tools/run_notebook.py --sections 1,2,3,5,6,7,8,8b,9   # skip the EDA
    uv run python tools/run_notebook.py --sections 1,2,3,5,6,7,8,8b,12 --stages A
    uv run python tools/run_notebook.py --until 8b --exec scratch/analysis.py

`--stages` sets LEISCHE_STAGES, which the §12 cell reads to decide which of
the staged real-training runs (A/B/C/D) to execute; unset means all of them.
`--exec` runs an extra script in the notebook namespace after the selected
cells, for ad-hoc analysis against the loaded state.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import time
import traceback
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "leische_pipeline.ipynb"

_HEADING = re.compile(r"^\s*#{2,3}\s+(?:§)?(\d+[a-z]?)\s*[·\-—:]", re.UNICODE)


def load_cells(notebook: Path | None = None) -> list[dict]:
    nb = json.loads((notebook or NOTEBOOK).read_text(encoding="utf-8"))
    cells, section, title = [], "0", "preamble"
    for i, c in enumerate(nb["cells"]):
        src = "".join(c["source"])
        if c["cell_type"] == "markdown":
            first = src.strip().splitlines()[0] if src.strip() else ""
            m = _HEADING.match(first)
            if m:
                section, title = m.group(1), first.lstrip("# ").strip()
            continue
        if c["cell_type"] == "code" and src.strip():
            cells.append({"index": i, "section": section, "title": title, "source": src})
    return cells


def _section_key(s: str) -> tuple[int, str]:
    m = re.match(r"(\d+)([a-z]?)", s)
    return (int(m.group(1)), m.group(2)) if m else (10**6, s)


def select(cells: list[dict], until: str | None, sections: list[str] | None) -> list[dict]:
    if sections:
        wanted = set(sections)
        return [c for c in cells if c["section"] in wanted]
    if until:
        limit = _section_key(until)
        return [c for c in cells if _section_key(c["section"]) <= limit]
    return cells


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the code cells and their sections")
    ap.add_argument("--until", help="run every section up to and including this one (e.g. 6, 8b)")
    ap.add_argument("--sections", help="comma-separated section ids to run, in notebook order")
    ap.add_argument("--stages", help="value for LEISCHE_STAGES (e.g. A or A,B) read by §12")
    ap.add_argument("--exec", dest="extra", help="script to exec in the notebook namespace afterwards")
    ap.add_argument("--set", action="append", default=[],
                    help="KEY=VALUE environment overrides applied before execution")
    ap.add_argument("--notebook", default=None,
                    help="another .ipynb to execute (default: leische_pipeline.ipynb); "
                         "e.g. the exploration notebook, which pulls the pipeline in itself")
    args = ap.parse_args()
    global NOTEBOOK
    if args.notebook:
        NOTEBOOK = (ROOT / args.notebook).resolve() if not Path(args.notebook).is_absolute() else Path(args.notebook)

    cells = load_cells(NOTEBOOK)
    if args.list:
        for c in cells:
            print(f"cell {c['index']:2d}  §{c['section']:<4} {c['title'][:70]}")
        return 0

    for kv in args.set:
        k, _, v = kv.partition("=")
        os.environ[k] = v
    if args.stages:
        os.environ["LEISCHE_STAGES"] = args.stages
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    os.environ["LEISCHE_HEADLESS"] = "1"

    # the notebook asserts it runs from the repo root, and figures are saved by
    # the plot helpers themselves, so a non-interactive backend loses nothing
    os.chdir(ROOT)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    warnings.filterwarnings("ignore", message=".*non-interactive.*")

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
        sys.stderr.reconfigure(encoding="utf-8", line_buffering=True)

    chosen = select(cells, args.until, args.sections.split(",") if args.sections else None)
    print(f"executing {len(chosen)} code cells from {NOTEBOOK.name}: "
          f"sections {sorted({c['section'] for c in chosen}, key=_section_key)}")
    ns: dict = {"__name__": "__main__", "__file__": str(NOTEBOOK)}
    t_start = time.time()
    for c in chosen:
        print(f"\n{'=' * 78}\n[cell {c['index']}] §{c['section']} — {c['title']}\n{'=' * 78}", flush=True)
        t0 = time.time()
        try:
            exec(compile(c["source"], f"<cell {c['index']}>", "exec"), ns)
        except Exception:
            traceback.print_exc()
            print(f"\nFAILED in cell {c['index']} (§{c['section']}) after {time.time() - t0:.1f}s", flush=True)
            return 1
        finally:
            plt.close("all")
        print(f"[cell {c['index']} done in {time.time() - t0:.1f}s]", flush=True)

    if args.extra:
        path = Path(args.extra)
        print(f"\n{'=' * 78}\n[exec] {path}\n{'=' * 78}", flush=True)
        try:
            exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            plt.close("all")
    print(f"\nall done in {(time.time() - t_start) / 60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
