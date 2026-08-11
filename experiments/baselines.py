from __future__ import annotations

import numpy as np

from models.features import build_features
from oracle.dso import RecedingHorizonOracle
from oracle.network import CellNetwork, CTMState


def _path_policy(
    network: CellNetwork,
    destinations: np.ndarray,
    edge_cost: np.ndarray,
) -> np.ndarray:
    import networkx as nx

    result = np.zeros((len(destinations), network.n_movements), dtype=float)
    source = network.movement_sources
    target = network.movement_targets
    graph = network.graph()
    for movement, (u, v) in enumerate(network.movements):
        graph[u][v]["weight"] = float(edge_cost[movement])
    for commodity, destination in enumerate(destinations):
        reverse = graph.reverse(copy=True)
        lengths = nx.single_source_dijkstra_path_length(reverse, int(destination), weight="weight")
        for cell in range(network.n_cells):
            indices = np.flatnonzero(source == cell)
            feasible = [idx for idx in indices if int(target[idx]) in lengths]
            if not feasible or cell == destination:
                continue
            best = min(feasible, key=lambda idx: edge_cost[idx] + lengths[int(target[idx])])
            result[commodity, best] = 1.0
    return result


def free_flow_shortest_path(network: CellNetwork, destinations: np.ndarray) -> np.ndarray:
    return _path_policy(network, destinations, network.free_time[network.movement_targets])


def dynamic_shortest_path(
    network: CellNetwork, state: CTMState, destinations: np.ndarray
) -> np.ndarray:
    occupancy = np.sum(state.occupancy, axis=0) / network.storage
    cell_cost = network.free_time * (1.0 + 4.0 * occupancy**4)
    return _path_policy(network, destinations, cell_cost[network.movement_targets])


def marginal_cost_policy(
    network: CellNetwork, state: CTMState, destinations: np.ndarray
) -> np.ndarray:
    occupancy = np.sum(state.occupancy, axis=0) / network.storage
    cell_cost = network.free_time * (1.0 + 20.0 * occupancy**4)
    return _path_policy(network, destinations, cell_cost[network.movement_targets])


def backpressure_policy(
    network: CellNetwork, state: CTMState, destinations: np.ndarray, temperature: float = 0.2
) -> np.ndarray:
    total = np.sum(state.occupancy, axis=0) / network.storage
    source, target = network.movement_sources, network.movement_targets
    pressure = total[source] - total[target]
    legal = network.reachability_mask(destinations)
    logits = np.broadcast_to(pressure[None, :] / max(temperature, 1e-6), legal.shape).copy()
    logits[~legal] = -np.inf
    result = np.zeros_like(logits)
    for commodity in range(len(destinations)):
        for cell in range(network.n_cells):
            indices = np.flatnonzero((source == cell) & legal[commodity])
            if len(indices):
                values = logits[commodity, indices]
                values = np.exp(values - np.max(values))
                result[commodity, indices] = values / np.sum(values)
    return result


class NeuralPolicyAdapter:
    def __init__(self, model, simulator) -> None:
        self.model = model
        self.simulator = simulator
        self.hidden = None

    def __call__(self, state: CTMState, forecast: np.ndarray) -> np.ndarray:
        import torch

        features = build_features(
            self.simulator.network,
            state,
            self.simulator.origins,
            self.simulator.destinations,
            forecast,
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model(features, self.hidden)
        self.hidden = getattr(output, "hetero_hidden", output.hidden)
        return output.splits.cpu().numpy()


class OraclePolicyAdapter:
    """Receding-horizon DSO or shorter-budget MPC baseline."""

    def __init__(self, oracle: RecedingHorizonOracle) -> None:
        self.oracle = oracle
        self.diagnostics = []

    def __call__(self, state: CTMState, forecast: np.ndarray) -> np.ndarray:
        result = self.oracle.solve(state, forecast)
        self.diagnostics.append(result.diagnostics)
        return result.first_action


def short_budget_mpc(simulator, horizon: int = 3, seed: int = 0) -> OraclePolicyAdapter:
    return OraclePolicyAdapter(
        RecedingHorizonOracle(
            simulator,
            horizon=horizon,
            iterations=20,
            restarts=1,
            learning_rate=0.12,
            seed=seed,
        )
    )


class ExternalGraphModelAdapter:
    """Optional reproducible existing-model baseline with an explicit availability gate."""

    def __init__(self, implementation=None, citation: str = "liu2024hstgsn") -> None:
        self.implementation = implementation
        self.citation = citation

    @property
    def available(self) -> bool:
        return callable(self.implementation)

    def __call__(self, state: CTMState, forecast: np.ndarray) -> np.ndarray:
        if not self.available:
            raise RuntimeError(
                f"external baseline {self.citation} has no reproducible implementation configured"
            )
        return self.implementation(state, forecast)
