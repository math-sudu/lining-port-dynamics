"""Generate the load cases and run the passive-bridge studies."""

import argparse
import json
from pathlib import Path
import subprocess
import sys

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
        studies = ["field_bridge_study", "random_bridge_study", "passive_bridge_study", "frame_bridge_study",
                   "passive_robustness_study", "passive_acquisition_study", "passive_design_study",
                   "passive_observables_study", "fixed_wiring_study",
                   "acquisition_architecture", "field_pressure_study"]
        for name in studies:
            print(f"Running {name}", flush=True)
            subprocess.run([sys.executable, "-m", f"lining_port.{name}"], check=True)


if __name__ == "__main__":
    main()
