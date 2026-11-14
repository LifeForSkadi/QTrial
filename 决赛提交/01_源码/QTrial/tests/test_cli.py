"""CLI end-to-end tests: map CLI on QUEKO file, invalid input handling, QCIS."""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
QUEKO_FILE = ROOT / "data" / "Queko" / "BIGD" / "20QBT_45CYC_.0D1_.1D2_0.qasm"


def _run_map(*extra, input_file=None, timeout=600):
    cmd = [sys.executable, "-m", "qtrail.map",
           str(input_file or QUEKO_FILE),
           "--output", str(ROOT / "out" / "test_cli"),
           "--baseline", "", "--dev", "cpu", *extra]
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True,
                          timeout=timeout)


def test_map_cli_end_to_end():
    r = _run_map()
    assert r.returncode == 0, r.stderr
    out = ROOT / "out" / "test_cli"
    assert (out / "20QBT_45CYC_.0D1_.1D2_0_mapped.qasm").exists()
    assert (out / "20QBT_45CYC_.0D1_.1D2_0.qcis").exists()
    m = json.loads((out / "20QBT_45CYC_.0D1_.1D2_0_metrics.json").read_text(encoding="utf-8"))
    assert m["swap_count"] >= 0
    assert "depth" in m and "est_fidelity" in m
    assert m["method"] in ("rl_multistart", "rl_multistart_routed", "rl_greedy",
                           "heuristic", "trivial", "hybrid_sabre_adopted",
                           "hybrid_o3_adopted", "hybrid_tket_adopted")


def test_map_cli_baseline_comparison():
    r = _run_map("--baseline", "qiskit-o1")
    assert r.returncode == 0, r.stderr
    m = json.loads((ROOT / "out" / "test_cli" /
                    "20QBT_45CYC_.0D1_.1D2_0_metrics.json").read_text(encoding="utf-8"))
    assert "baseline_qiskit-o1" in m
    assert m["baseline_qiskit-o1"]["swap_count"] >= 0


def test_map_cli_output_qasm_reparses():
    _run_map()
    from qiskit import qasm2
    p = ROOT / "out" / "test_cli" / "20QBT_45CYC_.0D1_.1D2_0_mapped.qasm"
    qc = qasm2.load(str(p))
    assert qc.num_qubits == 105  # physical circuit spans the full device
    ops = set(qc.count_ops())
    # sx is emitted as standard-qelib1 rx(pi/2) in the QASM file
    assert ops <= {"rz", "rx", "x", "cz", "measure"}


def test_map_cli_invalid_qasm_friendly_error(tmp_path):
    bad = tmp_path / "bad.qasm"
    bad.write_text('OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[300];\ncx q[0],q[1];\n',
                   encoding="utf-8")
    r = _run_map(input_file=bad)
    # too many qubits -> clean error message, non-zero exit, no traceback
    assert r.returncode != 0
    assert "qubits" in (r.stderr or "") or "qubits" in (r.stdout or "")
    assert "Traceback" not in (r.stderr or "")


def test_qcis_contains_expected_instructions():
    _run_map()
    qcis = (ROOT / "out" / "test_cli" / "20QBT_45CYC_.0D1_.1D2_0.qcis").read_text(encoding="utf-8")
    assert "CZ" in qcis
    assert any(tok in qcis for tok in ("X2P", "RZ", "X "))
