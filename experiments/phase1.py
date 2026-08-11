from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from data_processed.dataset import tiny_oracle_demonstrations
from data_processed.networks import diamond_cell_network
from experiments.baselines import dynamic_shortest_path
from experiments.train import set_deterministic_seed, train_imitation
from models.features import build_features
from models.homogeneous import DestinationGCNGRU
from oracle.dso import ExhaustiveTinyOracle
from simulators.ctm import CTMSimulator


@dataclass(frozen=True)
class Phase1Result:
    seed: int
    train_samples: int
    initial_imitation_loss: float
    final_imitation_loss: float
    heldout_demand: list[list[float]]
    model_tstt: float
    dynamic_shortest_path_tstt: float
    certified_discrete_oracle_tstt: float
    oracle_relative_gap: float
    improvement_over_dynamic_shortest_path: float
    model_inference_seconds: float
    dynamic_shortest_path_seconds: float
    oracle_seconds: float
    conservation_residual_max: float


def _model_actions(
    model: DestinationGCNGRU,
    simulator: CTMSimulator,
    demand: np.ndarray,
) -> tuple[np.ndarray, float]:
    state = simulator.empty_state()
    hidden = None
    actions = []
    started = time.perf_counter()
    model.eval()
    with torch.no_grad():
        for time_index in range(len(demand)):
            features = build_features(
                simulator.network,
                state,
                simulator.origins,
                simulator.destinations,
                demand[time_index:],
            )
            output = model(features, hidden)
            hidden = output.hidden
            action = output.splits.cpu().numpy()
            actions.append(action)
            state = simulator.step(state, demand[time_index], action).state
    return np.asarray(actions), time.perf_counter() - started


def _baseline_actions(simulator: CTMSimulator, demand: np.ndarray) -> tuple[np.ndarray, float]:
    state = simulator.empty_state()
    actions = []
    started = time.perf_counter()
    for time_index in range(len(demand)):
        action = dynamic_shortest_path(simulator.network, state, simulator.destinations)
        actions.append(action)
        state = simulator.step(state, demand[time_index], action).state
    return np.asarray(actions), time.perf_counter() - started


def run_phase1_smoke(
    *, seed: int = 7, train_samples: int = 32, epochs: int = 100
) -> Phase1Result:
    """Auditable same-topology experiment with a held-out demand profile."""

    set_deterministic_seed(seed)
    network = diamond_cell_network()
    origins, destinations = np.asarray([0]), np.asarray([3])
    simulator = CTMSimulator(network, origins, destinations)
    demonstrations = tiny_oracle_demonstrations(
        network, origins, destinations, count=train_samples, horizon=2, seed=seed
    )
    model = DestinationGCNGRU(hidden_dim=48)
    history = train_imitation(
        model, demonstrations, epochs=epochs, learning_rate=3e-3, seed=seed
    )
    # Fixed profile is outside the independent uniform draws used for demonstrations.
    heldout = np.asarray([[12.0], [12.0], [12.0], [8.0], [4.0], [0.0], [0.0], [0.0]])
    model_actions, model_seconds = _model_actions(model, simulator, heldout)
    baseline_actions, baseline_seconds = _baseline_actions(simulator, heldout)
    model_cost, model_steps = simulator.rollout(
        simulator.empty_state(), heldout, model_actions, terminal_weight=2.0
    )
    baseline_cost, _ = simulator.rollout(
        simulator.empty_state(), heldout, baseline_actions, terminal_weight=2.0
    )
    oracle = ExhaustiveTinyOracle(simulator)
    oracle_result = oracle.solve(simulator.empty_state(), heldout)
    denominator = max(abs(oracle_result.objective), 1e-9)
    return Phase1Result(
        seed=seed,
        train_samples=train_samples,
        initial_imitation_loss=history.losses[0],
        final_imitation_loss=history.losses[-1],
        heldout_demand=heldout.tolist(),
        model_tstt=model_cost,
        dynamic_shortest_path_tstt=baseline_cost,
        certified_discrete_oracle_tstt=oracle_result.objective,
        oracle_relative_gap=(model_cost - oracle_result.objective) / denominator,
        improvement_over_dynamic_shortest_path=(baseline_cost - model_cost) / baseline_cost,
        model_inference_seconds=model_seconds,
        dynamic_shortest_path_seconds=baseline_seconds,
        oracle_seconds=oracle_result.diagnostics.solve_seconds,
        conservation_residual_max=max(abs(step.conservation_residual) for step in model_steps),
    )


def write_phase1_result(result: Phase1Result, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    result = run_phase1_smoke()
    write_phase1_result(result, "artifacts/runs/phase1_smoke.json")
    print(json.dumps(asdict(result), indent=2, sort_keys=True))

