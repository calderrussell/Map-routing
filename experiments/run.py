from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from data_processed.networks import diamond_cell_network
from experiments.audit import run_static_audit, write_audit
from experiments.matrix import load_and_validate_plan
from experiments.metadata import collect_metadata, write_metadata
from experiments.phase0 import run_phase0, write_phase0_result
from experiments.phase1 import run_phase1_smoke, write_phase1_result
from experiments.phase2 import run_phase2_smoke, write_phase2_result
from experiments.phase3 import run_phase3_smoke, write_phase3_result
from experiments.phase4 import run_phase4_smoke, write_phase4_result


def _print(result) -> None:
    print(json.dumps(asdict(result), indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gnn-dso")
    parser.add_argument(
        "command",
        choices=("phase0", "phase1", "phase2", "phase3", "phase4", "all-smoke", "matrix", "audit", "test"),
    )
    args = parser.parse_args(argv)
    runs = Path("artifacts/runs")
    if args.command == "phase0":
        result = run_phase0(); write_phase0_result(result, runs / "phase0.json"); _print(result)
    elif args.command == "phase1":
        result = run_phase1_smoke(); write_phase1_result(result, runs / "phase1_smoke.json"); _print(result)
    elif args.command == "phase2":
        result = run_phase2_smoke(); write_phase2_result(result, runs / "phase2_smoke.json"); _print(result)
    elif args.command == "phase3":
        result = run_phase3_smoke(); write_phase3_result(result, runs / "phase3_smoke.json"); _print(result)
    elif args.command == "phase4":
        result = run_phase4_smoke(); write_phase4_result(result, runs / "phase4_smoke.json"); _print(result)
    elif args.command == "all-smoke":
        results = [run_phase0(), run_phase1_smoke(), run_phase2_smoke(), run_phase3_smoke(), run_phase4_smoke()]
        writers = [write_phase0_result, write_phase1_result, write_phase2_result, write_phase3_result, write_phase4_result]
        for index, (result, writer) in enumerate(zip(results, writers)):
            writer(result, runs / f"phase{index}.json")
        metadata = collect_metadata(
            "configs/frozen_experiment_plan.yaml", diamond_cell_network().topology_hash(), tuple(range(30))
        )
        write_metadata(metadata, runs / "reproducibility.json")
        print(json.dumps({f"phase{i}": asdict(result) for i, result in enumerate(results)}, indent=2))
    elif args.command == "matrix":
        print(json.dumps(load_and_validate_plan("configs/frozen_experiment_plan.yaml"), indent=2))
    elif args.command == "audit":
        requirements = run_static_audit()
        print(json.dumps([asdict(item) for item in requirements], indent=2))
    elif args.command == "test":
        completed = subprocess.run([sys.executable, "-m", "pytest", "-q"])
        return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

