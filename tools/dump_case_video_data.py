# -*- coding: utf-8 -*-
"""Dump the logical QUEKO case circuit for the case-film video.

Output: promo_video/src/data/case_logic.json
The display schedule is a simple per-qubit ASAP schedule; it is only for
visualisation, not for execution or metrics.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.pure.qasm import load_qasm_file

QASM = "data/Queko/BIGD/20QBT_45CYC_.5D1_.1D2_0.qasm"
OUT = Path("promo_video/src/data/case_logic.json")


def display_name(inst) -> str:
    if inst.name == "cx":
        return "CX"
    if inst.name == "u3":
        th, ph, la = inst.params
        if abs(th - 3.141592653589793) < 1e-6 and abs(ph) < 1e-6 \
                and abs(la - 3.141592653589793) < 1e-6:
            return "X"
        if abs(th - 1.5707963267948966) < 1e-6 and abs(ph) < 1e-6 \
                and abs(la - 3.141592653589793) < 1e-6:
            return "H"
        return "U3"
    if inst.name == "u1":
        return "RZ"
    return inst.name.upper()


def main() -> None:
    circ = load_qasm_file(QASM)
    last = [0] * circ.n
    gates = []
    for k, inst in enumerate(circ.ops):
        col = max(last[q] for q in inst.qubits) + 1
        for q in inst.qubits:
            last[q] = col
        gates.append({
            "id": k,
            "name": display_name(inst),
            "qubits": [int(q) for q in inst.qubits],
            "col": col,
            "long": bool(inst.name == "cx"
                         and abs(int(inst.qubits[0]) - int(inst.qubits[1])) > 1),
        })
    data = {
        "name": "20QBT_45CYC_.5D1_.1D2_0",
        "source": "QUEKO BIGD",
        "n": circ.n,
        "n_cols": max(g["col"] for g in gates),
        "n1q": sum(1 for g in gates if len(g["qubits"]) == 1),
        "n2q": sum(1 for g in gates if len(g["qubits"]) == 2),
        "n_long": sum(1 for g in gates if g["long"]),
        "gates": gates,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"saved {OUT}  n={data['n']} cols={data['n_cols']} "
          f"2q={data['n2q']} long={data['n_long']}")


if __name__ == "__main__":
    main()
