"""Fetch a Hugging Face model into the local HF cache with parallel range requests, then verify.

On this network the `huggingface_hub` downloader stalls at 0 bytes (read
timeouts against the CDN) and a single-connection mirror download crawls, but
several parallel HTTP range requests each run at ~15 MB/s.  So: ask
huggingface.co (tiny, fast) for the repo's commit sha and file list, download
each large file as N byte ranges in parallel with curl, stitch them, check the
SHA-256 huggingface.co publishes for LFS files, and place everything in the
cache layout `from_pretrained` expects.  Nothing is written outside
~/.cache/huggingface/hub.

    uv run python tools/hf_fetch.py xlm-roberta-large microsoft/mdeberta-v3-base
    uv run python tools/hf_fetch.py jcblaise/roberta-tagalog-base --parts 8 --source mirror

Only the files a text classifier needs are fetched (config, tokenizer files and
one weight file — safetensors preferred, else pytorch_model.bin).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
WANTED = {"config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
          "vocab.json", "merges.txt", "vocab.txt", "sentencepiece.bpe.model", "spm.model",
          "unigram.json", "added_tokens.json"}
WEIGHTS = ("model.safetensors", "pytorch_model.bin")
SOURCES = {"hf": "https://huggingface.co", "mirror": "https://hf-mirror.com"}


def api(repo: str) -> dict:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dst: Path, size: int | None, parts: int) -> bool:
    """Parallel range download when the size is known and large; plain curl otherwise."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not size or size < 8 << 20 or parts <= 1:
        return subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(dst), url]).returncode == 0 and dst.exists()
    step = -(-size // parts)
    ranges = [(i * step, min(size, (i + 1) * step) - 1) for i in range(parts) if i * step < size]
    files = [dst.with_suffix(dst.suffix + f".part{i}") for i in range(len(ranges))]

    def have(i):
        return files[i].stat().st_size if files[i].exists() else 0

    def spawn(i):
        """Resume from what the part already holds; abort a connection that crawls
        below 300 KB/s for 20 s so it can be re-spawned (some CDN connections stall)."""
        lo, hi = ranges[i]
        more = files[i].with_suffix(files[i].suffix + ".more")
        more.unlink(missing_ok=True)
        return subprocess.Popen(["curl", "-sfL", "--speed-limit", "300000", "--speed-time", "20",
                                 "-r", f"{lo + have(i)}-{hi}", "-o", str(more), url]), more

    pending = [i for i in range(len(ranges)) if have(i) != ranges[i][1] - ranges[i][0] + 1]
    for attempt in range(8):
        procs = {i: spawn(i) for i in pending}
        for i, (proc, more) in procs.items():
            proc.wait()
            if more.exists() and more.stat().st_size:
                with files[i].open("ab") as out:
                    out.write(more.read_bytes())
            more.unlink(missing_ok=True)
        pending = [i for i in pending if have(i) != ranges[i][1] - ranges[i][0] + 1]
        if not pending:
            break
        print(f" (attempt {attempt + 1}: parts {pending} incomplete, resuming)", end="", flush=True)
        time.sleep(2)
    ok = not pending
    if ok:
        with dst.open("wb") as out:
            for f in files:
                out.write(f.read_bytes())
        for f in files:
            f.unlink(missing_ok=True)      # parts are kept on failure so a re-run resumes
    return ok and dst.stat().st_size == size


def fetch(repo: str, base: str, parts: int) -> bool:
    meta = api(repo)
    sha = meta["sha"]
    files = {s["rfilename"]: s for s in meta.get("siblings", [])}
    weight = next((w for w in WEIGHTS if w in files), None)
    if weight is None:
        print(f"{repo}: no single-file weights found ({sorted(files)[:12]}...)")
        return False
    d = HUB / f"models--{repo.replace('/', '--')}"
    snap = d / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    (d / "refs").mkdir(exist_ok=True)
    (d / "refs" / "main").write_text(sha, encoding="utf-8")
    ok = True
    for f in [x for x in files if x in WANTED] + [weight]:
        dst, lfs = snap / f, files[f].get("lfs") or {}
        expect, size = lfs.get("sha256"), lfs.get("size") or files[f].get("size")
        if dst.exists() and (expect is None or (dst.stat().st_size == size and sha256(dst) == expect)):
            print(f"  have  {repo}/{f}")
            continue
        t0 = time.time()
        print(f"  fetch {repo}/{f} ({(size or 0) / 1e6:.0f} MB, {parts if size and size >= 8 << 20 else 1} connection(s)) ...", end="", flush=True)
        if not download(f"{base}/{repo}/resolve/{sha}/{f}", dst, size, parts):
            print(" FAILED"); dst.unlink(missing_ok=True); ok = False; continue
        if expect and sha256(dst) != expect:
            print(" SHA-256 MISMATCH"); dst.unlink(); ok = False; continue
        rate = (dst.stat().st_size / 1e6) / max(time.time() - t0, 1e-3)
        print(f" ok [{rate:.0f} MB/s{', sha256 verified' if expect else ''}]")
    print(f"{repo}: {'ready' if ok else 'INCOMPLETE'} at {snap}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--source", choices=sorted(SOURCES), default="hf")
    ap.add_argument("--parts", type=int, default=8, help="parallel range requests per large file")
    args = ap.parse_args()
    return 0 if all(fetch(r, SOURCES[args.source], args.parts) for r in args.repos) else 1


if __name__ == "__main__":
    sys.exit(main())
