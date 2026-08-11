from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import torch

from data_processed.dataset import gradient_oracle_demonstrations, tiny_oracle_demonstrations
from data_processed.networks import diamond_cell_network, grid_cell_network
from experiments.train import imitation_loss, set_deterministic_seed, train_imitation
from models.heterogeneous import HeteroSpatioTemporalGNN


@dataclass(frozen=True)
class Phase2Result:
    seed: int
    training_topologies: tuple[str, ...]
    validation_topology: str
    training_topology_hashes: tuple[str, ...]
    validation_topology_hash: str
    initial_training_loss: float
    final_training_loss: float
    validation_imitation_loss: float
    validation_illegal_flow: float
    validation_output_shape: tuple[int, int]
    oracle_labels_certified: int
    oracle_labels_uncertified: int


def _mean_loss(model, samples) -> float:
    values = []
    model.eval()
    with torch.no_grad():
        for item in samples:
            output = model(item.features)
            target = torch.as_tensor(item.target_action, dtype=output.splits.dtype)
            values.append(float(imitation_loss(output.splits, target, item.features.legal_mask)))
    return float(np.mean(values))


def run_phase2_smoke(seed: int = 11, epochs: int = 40) -> Phase2Result:
    """Train on two physical graphs and evaluate a completely held-out topology."""

    set_deterministic_seed(seed)
    diamond = diamond_cell_network("synthetic_diamond")
    train_grid = grid_cell_network(2, 2, seed=seed, name="synthetic_grid_2x2")
    validation_grid = grid_cell_network(2, 3, seed=seed + 1, name="synthetic_grid_2x3_validation")
    train = tiny_oracle_demonstrations(
        diamond, np.asarray([0]), np.asarray([3]), count=8, horizon=2, seed=seed
    )
    train += gradient_oracle_demonstrations(
        train_grid,
        np.asarray([0]),
        np.asarray([3]),
        count=4,
        horizon=2,
        seed=seed + 1,
        oracle_iterations=8,
    )
    validation = gradient_oracle_demonstrations(
        validation_grid,
        np.asarray([0]),
        np.asarray([5]),
        count=3,
        horizon=2,
        seed=seed + 2,
        oracle_iterations=8,
    )
    train_hashes = tuple(sorted({item.topology_hash for item in train}))
    if validation_grid.topology_hash() in train_hashes:
        raise RuntimeError("validation topology leaked into training")
    model = HeteroSpatioTemporalGNN(hidden_dim=32, layers=2)
    initial_loss = _mean_loss(model, train)
    history = train_imitation(model, train, epochs=epochs, learning_rate=3e-3, seed=seed)
    final_loss = _mean_loss(model, train)
    validation_loss = _mean_loss(model, validation)
    probe = validation[0]
    with torch.no_grad():
        output = model(probe.features)
    illegal = float(output.splits[~probe.features.legal_mask].sum())
    certified = sum(item.diagnostics.certified_gap == 0.0 for item in train)
    return Phase2Result(
        seed=seed,
        training_topologies=(diamond.name, train_grid.name),
        validation_topology=validation_grid.name,
        training_topology_hashes=train_hashes,
        validation_topology_hash=validation_grid.topology_hash(),
        initial_training_loss=initial_loss,
        final_training_loss=final_loss,
        validation_imitation_loss=validation_loss,
        validation_illegal_flow=illegal,
        validation_output_shape=tuple(output.splits.shape),
        oracle_labels_certified=certified,
        oracle_labels_uncertified=len(train) - certified,
    )


def write_phase2_result(result: Phase2Result, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    result = run_phase2_smoke()
    write_phase2_result(result, "artifacts/runs/phase2_smoke.json")
    print(json.dumps(asdict(result), indent=2, sort_keys=True))

