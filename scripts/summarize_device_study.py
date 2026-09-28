"""Summarize the paired fixed-wiring results saved by bridge_study."""
import argparse
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=Path("results/device-network-01"))
    args = parser.parse_args()
    design = json.loads((args.study / "design.json").read_text(encoding="utf-8"))
    rows = json.loads((args.study / "evaluation.json").read_text(encoding="utf-8"))
    indexed = {(row["family"], row["layout"], row["mode"]): row for row in rows}
    pairs = sorted({(row["family"], row["layout"]) for row in rows})
    expected = {(family, layout, mode)
                for family, layout in pairs for mode in design["modes"]}
    if len(indexed) != len(rows) or indexed.keys() != expected:
        raise ValueError("Evaluation data contain duplicate or missing wiring records")

    print(f"{len(pairs)} case-layout pairs; {len(rows)} evaluation records")
    print("Control          Improved pairs   Median residual increase   Positive layout medians")
    for control in ("uniform", "random", "raw_designed"):
        ratios = np.array([
            indexed[family, layout, "residual_designed"]["residual_distance"]
            / indexed[family, layout, control]["residual_distance"]
            for family, layout in pairs
        ])
        layout_medians = [np.median([ratio for (_, li), ratio in zip(pairs, ratios)
                                     if li == layout]) for layout in design["layouts"]]
        improved_layouts = sum(value > 1 for value in layout_medians)
        print(f"{control:<16} {np.count_nonzero(ratios > 1):>3}/{len(pairs):<10} "
              f"{100 * (np.median(ratios) - 1):>10.1f}% "
              f"{improved_layouts:>21}/{len(layout_medians)}")
    raw_ratios = [indexed[family, layout, "residual_designed"]["distance"]
                  / indexed[family, layout, "uniform"]["distance"]
                  for family, layout in pairs]
    print(f"Median unprojected-norm ratio against uniform: {np.median(raw_ratios):.3f}")


if __name__ == "__main__":
    main()
