"""Fetch a Hugging Face model into the local HF cache through a mirror, then verify.

On this network the `huggingface_hub` downloader stalls at 0 bytes against the
Hugging Face CDN (read timeouts), while plain HTTPS to a mirror runs at tens of
MB/s.  Pointing `HF_ENDPOINT` at the mirror does not work either — the client
rejects the mirror's metadata.  So: ask huggingface.co (fast, tiny) for the
repo's commit sha and file list, download each file from the mirror with curl
straight into the cache layout `from_pretrained` expects, and check every
LFS file's SHA-256 against the value huggingface.co publishes.

    uv run python tools/hf_fetch.py xlm-roberta-large microsoft/mdeberta-v3-base
    uv run python tools/hf_fetch.py jcblaise/roberta-tagalog-base --mirror https://hf-mirror.com

Only the files a text classifier needs are fetched (config, tokenizer files,
one weight file — safetensors preferred, else pytorch_model.bin).  Nothing is
written outside ~/.cache/huggingface/hub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

HUB = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
WANTED = {"config.json", "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json",
          "vocab.json", "merges.txt", "vocab.txt", "sentencepiece.bpe.model", "spm.model",
          "unigram.json", "added_tokens.json", "generation_config.json"}
WEIGHTS = ("model.safetensors", "pytorch_model.bin")


def api(repo: str) -> dict:
    with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}?blobs=true", timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(repo: str, mirror: str) -> bool:
    meta = api(repo)
    sha = meta["sha"]
    files = {s["rfilename"]: s for s in meta.get("siblings", [])}
    weight = next((w for w in WEIGHTS if w in files), None)
    if weight is None:
        print(f"{repo}: no single-file weights found ({sorted(files)[:12]}...)")
        return False
    picked = [f for f in files if f in WANTED] + [weight]
    d = HUB / f"models--{repo.replace('/', '--')}"
    snap = d / "snapshots" / sha
    snap.mkdir(parents=True, exist_ok=True)
    (d / "refs").mkdir(exist_ok=True)
    (d / "refs" / "main").write_text(sha, encoding="utf-8")
    ok = True
    for f in picked:
        dst = snap / f
        expect = (files[f].get("lfs") or {}).get("sha256")
        if dst.exists() and (expect is None or sha256(dst) == expect):
            print(f"  have  {repo}/{f}")
            continue
        url = f"{mirror}/{repo}/resolve/{sha}/{f}"
        print(f"  fetch {repo}/{f} ...", end="", flush=True)
        r = subprocess.run(["curl", "-sfL", "--retry", "3", "-o", str(dst), url])
        if r.returncode != 0 or not dst.exists():
            print(" FAILED"); ok = False; continue
        if expect:
            got = sha256(dst)
            if got != expect:
                print(f" SHA-256 MISMATCH ({got[:12]} vs {expect[:12]})"); dst.unlink(); ok = False; continue
            print(f" ok ({dst.stat().st_size / 1e6:.0f} MB, sha256 verified)")
        else:
            print(f" ok ({dst.stat().st_size / 1e3:.0f} KB)")
    print(f"{repo}: {'ready' if ok else 'INCOMPLETE'} at {snap}")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("repos", nargs="+")
    ap.add_argument("--mirror", default="https://hf-mirror.com")
    args = ap.parse_args()
    return 0 if all(fetch(r, args.mirror.rstrip("/")) for r in args.repos) else 1


if __name__ == "__main__":
    sys.exit(main())
