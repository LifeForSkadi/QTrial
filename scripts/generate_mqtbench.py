"""Generate the MQTBench program-graph pool for training + eval subsets.

Usage: python scripts/generate_mqtbench.py [--sizes 10,15,20,25] [--out data/mqtbench/graph_pool.pkl]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from qtrail.utils.bench import (generate_mqtbench_graphs, save_graph_pool,
                                load_queko_program_graphs, default_pool_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="4,6,8,10,12,14,15,16,18,20,22,24,25,26,28,30")
    ap.add_argument("--out", default=str(default_pool_path()))
    ap.add_argument("--queko", action="store_true", help="also add QUEKO graphs")
    args = ap.parse_args()

    sizes = tuple(int(s) for s in args.sizes.split(","))
    print(f"generating MQTBench graphs for sizes {sizes} ...")
    graphs = generate_mqtbench_graphs(sizes=sizes)
    print(f"  mqtbench: {len(graphs)} graphs")

    if args.queko:
        for split in ("BIGD", "BNTF"):
            gs = load_queko_program_graphs(split=split)
            print(f"  queko/{split}: {len(gs)} graphs")
            graphs.extend(gs)

    save_graph_pool(graphs, args.out)
    print(f"saved {len(graphs)} graphs -> {args.out}")


if __name__ == "__main__":
    main()
