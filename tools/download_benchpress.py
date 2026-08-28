"""Download Qiskit/benchpress qasm circuits via GitHub API (CN-friendly).

Writes repo subtree `benchpress/qasm/**/*.qasm` (656 files) into
`data/benchpress/qasm/<benchmark>/<file>.qasm` -- same layout the bench
scripts consume (`data/benchpress/qasm` rglob). Resumable: skips files
whose size already matches.

Usage: python tools/download_benchpress.py
"""
import base64
import json
import time
import urllib.request
from pathlib import Path

REPO = "Qiskit/benchpress"
BRANCH = "main"
SUBTREE = "benchpress/qasm"
REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "benchpress" / "qasm"


def api_get(path: str) -> dict:
    url = f"https://api.github.com/repos/{REPO}{path}"
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0",
                      "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tree = api_get(f"/git/trees/{BRANCH}?recursive=1")
    files = [t for t in tree["tree"]
             if t["type"] == "blob" and t["path"].startswith(SUBTREE + "/")
             and t["path"].endswith(".qasm")]
    print(f"qasm files under {SUBTREE}: {len(files)}")
    n_ok = 0
    for t in files:
        path = t["path"]
        rel = path[len(SUBTREE) + 1:]          # <benchmark>/<file>.qasm
        dest = OUT_DIR / rel
        if dest.exists() and dest.stat().st_size == t.get("size", -1):
            n_ok += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # raw first (no rate limit), base64 contents API as fallback
        for attempt in range(3):
            try:
                url = (f"https://raw.githubusercontent.com/{REPO}/"
                       f"{BRANCH}/{path}")
                req = urllib.request.Request(
                    url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                dest.write_bytes(data)
                n_ok += 1
                break
            except Exception:
                try:
                    blob = api_get(f"/contents/{path}")
                    data = base64.b64decode(
                        blob["content"].replace("\n", ""))
                    dest.write_bytes(data)
                    n_ok += 1
                    break
                except Exception:
                    time.sleep(2)
        if n_ok % 100 == 0:
            print(f"  downloaded {n_ok}/{len(files)}")
    print(f"done: {n_ok}/{len(files)} files -> {OUT_DIR}")


if __name__ == "__main__":
    main()
