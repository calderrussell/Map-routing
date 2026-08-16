from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Requirement:
    identifier: str
    requirement: str
    implementation: str
    verification: str
    status: str = "implemented"


REQUIREMENTS = (
    Requirement("C1", "Receding-horizon CTM DSO oracle", "oracle/dso.py", "tests/test_oracle.py"),
    Requirement("C2", "Heterogeneous cell/intersection/OD/global spatio-temporal GNN", "models/heterogeneous.py", "tests/test_heterogeneous.py"),
    Requirement("C3", "Hard feasibility and differentiable projection", "projection/feasible.py", "tests/test_projection.py"),
    Requirement("C4", "Topology-held-out evaluation protocol", "configs/frozen_experiment_plan.yaml", "tests/test_metrics_matrix_metadata.py"),
    Requirement("P0-STATIC", "Static UE/SO, Braess, Sioux Falls, normalized gap", "oracle/static_assignment.py", "tests/test_static_assignment.py"),
    Requirement("P0-CTM", "Conservation, exits, source queues, capacity", "simulators/ctm.py", "tests/test_ctm.py"),
    Requirement("OUTPUT-1", "Local movement split logits and reachability mask", "models/homogeneous.py", "tests/test_features_models.py"),
    Requirement("OUTPUT-2", "Learned nonnegative marginal costs", "models/alternatives.py", "tests/test_features_models.py"),
    Requirement("OUTPUT-3", "Full multicommodity flow benchmark decoder", "models/alternatives.py", "tests/test_features_models.py"),
    Requirement("FEATURES", "Physical normalized feature schema without test leakage", "models/features.py", "tests/test_features_models.py"),
    Requirement("COMMODITY", "Sparse OD tokens and destination batching", "models/heterogeneous.py", "tests/test_heterogeneous.py"),
    Requirement("DATA", "Dynamic demand, five incident sets, topology-first split", "data_processed/scenarios.py", "tests/test_scenarios.py"),
    Requirement("DEMO", "Warm-up and sequential first-action oracle demonstrations", "data_processed/dataset.py", "tests/test_alternatives_dagger.py"),
    Requirement("TRAIN-1", "Normalized Huber imitation", "experiments/train.py", "tests/test_features_models.py"),
    Requirement("TRAIN-2", "Physics residuals and decision-focused rollout curriculum", "experiments/decision_training.py", "tests/test_decision_training.py"),
    Requirement("DAGGER", "On-policy prioritized oracle querying", "experiments/decision_training.py", "tests/test_alternatives_dagger.py"),
    Requirement("ROUTING", "Path decomposition, rounding, next-hop loop control", "routing/paths.py", "tests/test_routing.py"),
    Requirement("COMPLIANCE", "Partial control and compliance interface", "routing/compliance.py", "tests/test_routing.py"),
    Requirement("BASELINES", "Optimization, routing, MLP, GCN and external baseline adapters", "experiments/baselines.py", "tests/test_phase4.py"),
    Requirement("E0-E7", "Frozen complete experiment matrix", "configs/frozen_experiment_plan.yaml", "tests/test_metrics_matrix_metadata.py"),
    Requirement("METRICS", "System, optimality, feasibility, latency, fairness", "experiments/metrics.py", "tests/test_metrics_matrix_metadata.py"),
    Requirement("STATISTICS", "Paired seeds, bootstrap CI, effect sizes", "experiments/metrics.py", "tests/test_metrics_matrix_metadata.py"),
    Requirement("ROBUSTNESS", "Demand/capacity/forecast/noise/delay/control sweeps", "experiments/robustness.py", "tests/test_phase4.py"),
    Requirement("SCALING", "Network/commodity runtime and memory curves", "experiments/scaling.py", "tests/test_heterogeneous.py"),
    Requirement("REPRO", "Revision/config/network/seeds/solver/hardware/checkpoint metadata", "experiments/metadata.py", "tests/test_metrics_matrix_metadata.py"),
    Requirement("SUMO", "Frozen-controller TraCI transfer adapter", "simulators/sumo_adapter.py", "tests/test_sumo_adapter.py", "implemented; external run blocked by absent SUMO binary/network"),
    Requirement("REAL-EXAMPLE", "End-to-end Sioux Falls GNN walkthrough", "examples/sioux_falls_gnn.py", "tests/test_real_example.py"),
    Requirement("VISUALIZATION", "Coordinate-file import, traced GNN evaluation, and Folium result map", "visualization/cli.py", "tests/test_folium_visualization.py"),
    Requirement("CLAIMS", "Claim-to-evidence and failure-mode audits", "experiments/audit.py", "full pytest suite"),
)


def run_static_audit(root: str | Path = ".") -> tuple[Requirement, ...]:
    root = Path(root)
    missing = []
    for item in REQUIREMENTS:
        if not (root / item.implementation).exists():
            missing.append(f"{item.identifier}: {item.implementation}")
        if item.verification.endswith(".py") and not (root / item.verification).exists():
            missing.append(f"{item.identifier}: {item.verification}")
    if missing:
        raise FileNotFoundError("audit targets missing:\n" + "\n".join(missing))
    return REQUIREMENTS


def write_audit(path: str | Path, test_result: str) -> None:
    requirements = run_static_audit()
    lines = [
        "# Final requirement-by-requirement compliance audit",
        "",
        "Authoritative specification: `specification/gnn_dso_model.tex`.",
        "",
        f"Full verification result: **{test_result}**.",
        "",
        "| ID | Requirement | Implementation | Verification | Status |",
        "|---|---|---|---|---|",
    ]
    for item in requirements:
        lines.append(
            f"| {item.identifier} | {item.requirement} | `{item.implementation}` | `{item.verification}` | {item.status} |"
        )
    lines.extend(
        [
            "",
            "## Empirical claim boundary",
            "",
            "The checked-in smoke results validate software correctness and integration. They do not constitute the full 30-seed, held-out benchmark/SUMO study. Near-system-optimality, topology superiority, robustness, scalability, and microscopic-transfer claims must be made only after the frozen E0–E7 plan is executed on the named external data and SUMO environment.",
            "",
            "The local gradient DSO oracle is explicitly uncertified. Only `ExhaustiveTinyOracle` outputs are labeled certified. Projection, path decomposition, and end-to-end timings are separately exposed.",
            "",
        ]
    )
    Path(path).write_text("\n".join(lines))
