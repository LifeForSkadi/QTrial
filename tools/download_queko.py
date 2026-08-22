"""Download QUEKO-benchmark repo via GitHub API (git protocol is blocked in CN network)."""
import base64
import json
import time
import urllib.request
from pathlib import Path

REPO = "qu-tan-um/QUEKO-benchmark"
BRANCH = "master"
OUT_DIR = Path(r"f:\Study\Quantum Computing\信安竞赛\QTrial\data\Queko")


def api_get(path: str) -> dict:
    url = f"https://api.github.com/repos/{REPO}{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # get the full tree
    tree = api_get(f"/git/trees/{BRANCH}?recursive=1")
    files = [t for t in tree["tree"] if t["type"] == "blob"]
    print(f"total files: {len(files)}")
    n_ok = 0
    for t in files:
        path = t["path"]
        dest = OUT_DIR / path
        if dest.exists() and dest.stat().st_size == t.get("size", -1):
            n_ok += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        # try raw first (no rate limit), fall back to base64 contents API
        for attempt in range(3):
            try:
                url = f"https://raw.githubusercontent.com/{REPO}/{BRANCH}/{path}"
                req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = r.read()
                dest.write_bytes(data)
                n_ok += 1
                break
            except Exception:
                try:
                    blob = api_get(f"/contents/{path}")
                    data = base64.b64decode(blob["content"].replace("\n", ""))
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
