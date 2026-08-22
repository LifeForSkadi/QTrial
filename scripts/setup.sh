#!/usr/bin/env bash
# ============================================================
#  QTrial environment setup (Linux / macOS)
#  usage:   bash setup.sh        -> core deps only (judge-ready)
#           bash setup.sh full   -> core + qiskit/pytket/etc
# ============================================================
set -e
cd "$(dirname "$0")/.."

if ! command -v python3 >/dev/null 2>&1; then
    echo "[x] python3 not found. Install Python 3.10-3.13 first."
    exit 1
fi

if [ ! -d .venv ]; then
    echo "[*] creating virtual environment .venv ..."
    python3 -m venv .venv
else
    echo "[*] .venv already exists, reusing it"
fi
source .venv/bin/activate

echo "[*] upgrading pip ..."
python -m pip install --upgrade pip -q

echo "[*] installing core deps: torch numpy networkx pyyaml numba ..."
if ! pip install torch numpy networkx pyyaml numba -i https://pypi.tuna.tsinghua.edu.cn/simple; then
    echo "[!] tsinghua mirror failed, retrying with default index ..."
    pip install torch numpy networkx pyyaml numba
fi

if [ "${1:-}" = "full" ]; then
    echo "[*] installing full deps (qiskit/pytket/mqt.bench/fastapi/cqlib) ..."
    if ! pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple; then
        echo "[!] tsinghua mirror failed, retrying with default index ..."
        pip install -r requirements.txt
    fi
else
    echo "[*] core-only mode. For benchmark baselines run: bash setup.sh full"
fi

echo "[*] smoke test: mapping examples/ghz10.qasm ..."
if python -m qtrail.pure_cli examples/ghz10.qasm >/dev/null 2>&1; then
    echo "[*] smoke test passed. outputs written to out/"
else
    echo "[!] smoke test failed - see error above, but deps may still be fine."
fi

echo
echo "[*] done. quick start:"
echo "    source .venv/bin/activate"
echo "    python -m qtrail.pure_cli examples/qft5.qasm"
echo
echo "    note: default pip wheel is CPU-only torch. For CUDA, install torch"
echo "    from https://pytorch.org/get-started (optional - the model is tiny)."
