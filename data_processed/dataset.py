from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from models.features import FeatureBatch, build_features
from oracle.dso import ExhaustiveTinyOracle, OracleDiagnostics, RecedingHorizonOracle
from oracle.network import CellNetwork, CTMState
from simulators.ctm import CTMSimulator


@dataclass
class Demonstration:
    features: FeatureBatch
    target_action: np.ndarray
    oracle_objective: float
    diagnostics: OracleDiagnostics
    state: CTMState
    demand_forecast: np.ndarray
    topology_hash: str
    dual_variables: np.ndarray | None = None
    accepted: bool = True


def tiny_oracle_demonstrations(
    network: CellNetwork,
    origins: np.ndarray,
    destinations: np.ndarray,
    *,
    count: int,
    horizon: int,
    seed: int,
) -> list[Demonstration]:
    """Generate certified small demonstrations for Phase 1 correctness/smoke runs."""

    rng = np.random.default_rng(seed)
    simulator = CTMSimulator(network, origins, destinations)
    oracle = ExhaustiveTinyOracle(simulator)
    demonstrations = []
    for _ in range(count):
        state = simulator.empty_state()
        state.occupancy[:, 0] = rng.uniform(0.0, network.capacity[0])
        demand = rng.uniform(0.2, 1.4, size=(horizon, len(origins))) * network.capacity[0]
        result = oracle.solve(state, demand)
        demonstrations.append(
            Demonstration(
                features=build_features(network, state, origins, destinations, demand),
                target_action=result.first_action,
                oracle_objective=result.objective,
                diagnostics=result.diagnostics,
                state=state,
                demand_forecast=demand,
                topology_hash=network.topology_hash(),
            )
        )
    return demonstrations


def gradient_oracle_demonstrations(
    network: CellNetwork,
    origins: np.ndarray,
    destinations: np.ndarray,
    *,
    count: int,
    horizon: int,
    seed: int,
    oracle_iterations: int = 20,
) -> list[Demonstration]:
    """Generate topology-variable demonstrations while retaining solver confidence."""

    rng = np.random.default_rng(seed)
    simulator = CTMSimulator(network, origins, destinations)
    oracle = RecedingHorizonOracle(
        simulator,
        horizon=horizon,
        iterations=oracle_iterations,
        restarts=1,
        seed=seed,
    )
    demonstrations = []
    for _ in range(count):
        state = simulator.empty_state()
        for commodity, origin in enumerate(origins):
            state.occupancy[commodity, origin] = rng.uniform(0.0, network.capacity[origin])
        base = np.asarray([network.capacity[origin] for origin in origins])
        demand = rng.uniform(0.3, 1.2, size=(horizon, len(origins))) * base[None, :]
        result = oracle.solve(state, demand)
        demonstrations.append(
            Demonstration(
                features=build_features(network, state, origins, destinations, demand),
                target_action=result.first_action,
                oracle_objective=result.objective,
                diagnostics=result.diagnostics,
                state=state,
                demand_forecast=demand,
                topology_hash=network.topology_hash(),
            )
        )
    return demonstrations


def save_manifest(demonstrations: list[Demonstration], path: str | Path) -> None:
    """Store audit metadata without serializing executable Python objects."""

    records = [
        {
            "network": item.features.network_name,
            "topology_hash": item.topology_hash,
            "oracle_objective": item.oracle_objective,
            "oracle_status": item.diagnostics.status,
            "certified_gap": item.diagnostics.certified_gap,
            "lower_bound": item.diagnostics.lower_bound,
            "primal_residual": item.diagnostics.primal_residual,
            "solve_seconds": item.diagnostics.solve_seconds,
            "accepted": item.accepted,
            "dual_variables_available": item.dual_variables is not None,
        }
        for item in demonstrations
    ]
    Path(path).write_text(json.dumps(records, indent=2, sort_keys=True) + "\n")


@dataclass(frozen=True)
class TrajectoryGenerationResult:
    accepted: tuple[Demonstration, ...]
    rejected: tuple[Demonstration, ...]
    warmup_policy: str


def receding_horizon_demonstration_trajectory(
    network: CellNetwork,
    origins: np.ndarray,
    destinations: np.ndarray,
    realized_demand: np.ndarray,
    forecast_demand: np.ndarray,
    *,
    oracle: RecedingHorizonOracle,
    warmup_intervals: int = 0,
    warmup_policy: str = "free_flow_shortest_path",
    maximum_primal_residual: float = 1e-6,
) -> TrajectoryGenerationResult:
    """Full receding-horizon demonstration pipeline from the specification."""

    from experiments.baselines import free_flow_shortest_path

    simulator = CTMSimulator(network, origins, destinations)
    state = simulator.empty_state()
    if warmup_policy != "free_flow_shortest_path":
        raise ValueError("the implemented documented warm-up policy is free_flow_shortest_path")
    warmup_action = free_flow_shortest_path(network, destinations)
    for time_index in range(min(warmup_intervals, len(realized_demand))):
        state = simulator.step(state, realized_demand[time_index], warmup_action).state
    accepted, rejected = [], []
    for time_index in range(warmup_intervals, len(realized_demand)):
        visible_forecast = forecast_demand[time_index:]
        result = oracle.solve(state, visible_forecast)
        item = Demonstration(
            features=build_features(network, state, origins, destinations, visible_forecast),
            target_action=result.first_action,
            oracle_objective=result.objective,
            diagnostics=result.diagnostics,
            state=state.copy(),
            demand_forecast=visible_forecast.copy(),
            topology_hash=network.topology_hash(),
            dual_variables=None,
            accepted=result.diagnostics.primal_residual <= maximum_primal_residual,
        )
        (accepted if item.accepted else rejected).append(item)
        state = simulator.step(state, realized_demand[time_index], result.first_action).state
    return TrajectoryGenerationResult(tuple(accepted), tuple(rejected), warmup_policy)
