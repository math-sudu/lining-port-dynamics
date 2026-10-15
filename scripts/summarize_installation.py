"""Generate manuscript tables and numerical facts from the fixed-installation study."""
import argparse
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "results/fixed-installation"
LABELS = {"uniform": "Uniform", "random": "Random", "raw_designed": "Raw-contrast design",
          "residual_designed": "Residual design", "layer_difference": "Layer difference",
          "strain_matched": "Strain match", "scan": "Four-channel scan", "mechanical_bridge": "Mechanical bridges"}


def read(name):
    return json.loads((STUDY / name).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-paper", action="store_true")
    args = parser.parse_args()
    summary, design = read("summary.json"), read("design.json")
    detection = read("detection.json")
    acquisition = read("acquisition.json")
    rows = read("evaluation.json")
    indexed = {(r["family"], r["layout"], r["mode"]): r for r in rows}
    pairs = sorted({(r["family"], r["layout"]) for r in rows})
    raw_ratio = np.median([indexed[f, l, "residual_designed"]["contrast_l2_v"] /
                           indexed[f, l, "uniform"]["contrast_l2_v"] for f, l in pairs])
    aggregate = []
    for mode in LABELS:
        source = detection if mode in design["modes"] else acquisition["detection"]
        for delta in (.1, .2, .3, .45, .6):
            selected = [r for r in source if r["mode"] == mode and r["reduction"] == delta]
            aggregate.append({"mode": mode, "reduction": delta,
                              **{key: float(np.mean([r[key] for r in selected]))
                                 for key in ("true_positive_rate", "false_positive_rate")}})
    summary.update(raw_ratio_vs_uniform=float(raw_ratio), detection=aggregate,
                   resources=acquisition["resources"], matched_false_alarm=acquisition["roc_at_rc_false_alarm"],
                   source="results/fixed-installation")
    if args.write_paper:
        (ROOT / "paper/numbers.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [r"\begin{tabular}{lrrr}", r"\toprule",
             r"Wiring & Raw ($\mu$V) & Residual ($\mu$V) & Design/control\\", r"\midrule"]
    for mode in design["modes"]:
        med = summary["medians"][mode]
        ratio = "---" if mode == "residual_designed" else f"{summary['comparisons'][mode]['median_ratio']:.3f}"
        lines.append(f"{LABELS[mode]} & {1e6 * med['contrast_l2_v']:.1f} & {1e6 * med['residual_l2_v']:.1f} & {ratio}\\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if args.write_paper:
        (ROOT / "paper/tables/installation_results.tex").write_text("\n".join(lines) + "\n")

    lines = [r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Acquisition & FPR (\%) & \multicolumn{3}{c}{TPR (\%)}\\",
             r"\cmidrule(lr){3-5}", r" & & $\delta=0.10$ & $\delta=0.20$ & $\delta=0.45$\\", r"\midrule"]
    for mode in LABELS:
        selected = {r["reduction"]: r for r in aggregate if r["mode"] == mode}
        rates = [100 * selected[.1]["false_positive_rate"]] + [100 * selected[d]["true_positive_rate"] for d in (.1, .2, .45)]
        lines.append(LABELS[mode] + " & " + " & ".join(f"{value:.1f}" for value in rates) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if args.write_paper:
        (ROOT / "paper/tables/detection.tex").write_text("\n".join(lines) + "\n")
    print(f"Raw paired ratio: {raw_ratio:.6f}")
    print(json.dumps({"comparisons": summary["comparisons"], "resources": acquisition["resources"]}, indent=2))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
