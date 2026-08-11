from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from data_processed.dataset import tiny_oracle_demonstrations
from data_processed.networks import diamond_cell_network
from experiments.ablations import ABLATIONS
from experiments.decision_training import fine_tune_decision_focused
from experiments.train import set_deterministic_seed, train_imitation
from models.homogeneous import DestinationGCNGRU
from projection.feasible import FeasibilityProjector
from simulators.ctm import CTMSimulator


@dataclass(frozen=True)
class Phase3Result:
    seed: int
    imitation_loss_initial: float
    imitation_loss_final: float
    decision_loss_initial: float
    decision_loss_final: float
    raw_violation_before: float
    raw_violation_after: float
    post_projection_violation_before: float
    post_projection_violation_after: float
    normalized_projection_correction_before: float
    normalized_projection_correction_after: float
    projection_status: str
    projection_solve_seconds: float
    curriculum_horizons: tuple[int, ...]
    registered_ablations: tuple[str, ...]


def _projection_probe(model, demonstration, simulator):
    with torch.no_grad():
        raw = torch.nn.functional.softplus(model(demonstration.features).logits).numpy()
    return FeasibilityProjector(simulator).project(demonstration.state, raw)


def run_phase3_smoke(seed: int = 13) -> Phase3Result:
    set_deterministic_seed(seed)
    network = diamond_cell_network()
    origins, destinations = np.asarray([0]), np.asarray([3])
    simulator = CTMSimulator(network, origins, destinations)
    demonstrations = tiny_oracle_demonstrations(
        network, origins, destinations, count=12, horizon=2, seed=seed
    )
    model = DestinationGCNGRU(hidden_dim=32)
    imitation = train_imitation(
        model, demonstrations, epochs=30, learning_rate=5e-3, seed=seed
    )
    before = _projection_probe(model, demonstrations[0], simulator)
    decision = fine_tune_decision_focused(
        model,
        demonstrations,
        simulator,
        epochs=20,
        maximum_horizon=2,
        learning_rate=5e-4,
    )
    after = _projection_probe(model, demonstrations[0], simulator)
    return Phase3Result(
        seed=seed,
        imitation_loss_initial=imitation.losses[0],
        imitation_loss_final=imitation.losses[-1],
        decision_loss_initial=decision.losses[0],
        decision_loss_final=decision.losses[-1],
        raw_violation_before=before.diagnostics.raw_violation.total,
        raw_violation_after=after.diagnostics.raw_violation.total,
        post_projection_violation_before=before.diagnostics.projected_violation.total,
        post_projection_violation_after=after.diagnostics.projected_violation.total,
        normalized_projection_correction_before=before.diagnostics.normalized_correction,
        normalized_projection_correction_after=after.diagnostics.normalized_correction,
        projection_status=after.diagnostics.status,
        projection_solve_seconds=after.diagnostics.solve_seconds,
        curriculum_horizons=decision.rollout_horizons,
        registered_ablations=tuple(item.name for item in ABLATIONS),
    )


def write_phase3_result(result: Phase3Result, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    result = run_phase3_smoke()
    write_phase3_result(result, "artifacts/runs/phase3_smoke.json")
    print(json.dumps(asdict(result), indent=2, sort_keys=True))

