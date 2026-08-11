from __future__ import annotations

from pathlib import Path

import yaml


REQUIRED_EXPERIMENTS = {f"E{index}" for index in range(8)}
REQUIRED_PRIMARY_METRICS = {
    "total_system_travel_time",
    "oracle_relative_gap",
    "recovered_ue_so_gap",
    "conservation_residual",
    "capacity_violation",
    "end_to_end_latency",
}


def load_and_validate_plan(path: str | Path) -> dict:
    plan = yaml.safe_load(Path(path).read_text())
    if not plan.get("frozen"):
        raise ValueError("final experiment plan must be frozen")
    identifiers = {item["id"] for item in plan["experiments"]}
    if identifiers != REQUIRED_EXPERIMENTS:
        raise ValueError(f"experiment IDs must be exactly {sorted(REQUIRED_EXPERIMENTS)}")
    if plan.get("paired_seeds", 0) < 30:
        raise ValueError("central conditions require at least 30 paired seeds")
    metrics = set(plan.get("primary_metrics", []))
    missing = REQUIRED_PRIMARY_METRICS - metrics
    if missing:
        raise ValueError(f"missing primary metrics: {sorted(missing)}")
    train = set(plan["topology_splits"]["train"])
    validation = set(plan["topology_splits"]["validation"])
    test = set(plan["topology_splits"]["test"])
    if train & validation or train & test or validation & test:
        raise ValueError("topology splits overlap")
    return plan

