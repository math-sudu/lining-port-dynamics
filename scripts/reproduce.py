"""Generate the load cases and run the passive-bridge manuscript studies."""

import argparse
import json
from pathlib import Path
import runpy

from lining_port.installation_study import make_cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Generate ring load cases only")
    args = parser.parse_args()
    config = json.loads(Path("inputs/installation_task.json").read_text(encoding="utf-8"))
    cases = make_cases(config)
    output = Path("results/fixed-installation/cases.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, indent=2) + "\n", encoding="utf-8")
    print(f"Generated {len(cases)} ring load cases at {output}", flush=True)
    if not args.prepare_only:
        for name in ("random_bridge_study", "passive_bridge_study", "frame_bridge_study",
                     "passive_robustness_study"):
            print(f"Running {name}", flush=True)
            runpy.run_module(f"lining_port.{name}", run_name="__main__")


if __name__ == "__main__":
    main()
