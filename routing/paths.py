from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import networkx as nx
import numpy as np
from scipy.optimize import minimize

from oracle.network import CellNetwork


@dataclass(frozen=True)
class RouteShareResult:
    paths: tuple[tuple[int, ...], ...]
    shares: np.ndarray
    reconstructed_flow: np.ndarray
    weighted_error: float
    objective: float
    status: str


def candidate_paths(
    network: CellNetwork,
    origin: int,
    destination: int,
    *,
    k_paths: int = 8,
    movement_cost: np.ndarray | None = None,
) -> tuple[tuple[int, ...], ...]:
    """Generate loop-free candidate cell paths in increasing generalized cost."""

    graph = network.graph()
    cost = (
        network.free_time[network.movement_targets]
        if movement_cost is None
        else np.asarray(movement_cost, dtype=float)
    )
    for movement, (source, target) in enumerate(network.movements):
        graph[source][target]["weight"] = float(cost[movement])
    try:
        generator = nx.shortest_simple_paths(graph, origin, destination, weight="weight")
        paths = []
        for path in generator:
            paths.append(tuple(path))
            if len(paths) >= k_paths:
                break
        return tuple(paths)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return ()


def path_movement_incidence(
    network: CellNetwork, paths: Iterable[tuple[int, ...]]
) -> np.ndarray:
    paths = tuple(paths)
    incidence = np.zeros((network.n_movements, len(paths)), dtype=float)
    lookup = {movement: index for index, movement in enumerate(network.movements)}
    for path_index, path in enumerate(paths):
        if len(set(path)) != len(path):
            raise ValueError("candidate paths must be loop free")
        for movement in zip(path[:-1], path[1:]):
            incidence[lookup[movement], path_index] = 1.0
    return incidence


def decompose_route_shares(
    network: CellNetwork,
    paths: tuple[tuple[int, ...], ...],
    target_flow: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    previous_shares: np.ndarray | None = None,
    switch_penalty: float = 0.01,
) -> RouteShareResult:
    """Constrained route-set decomposition from TeX equation (24)."""

    if not paths:
        raise ValueError("at least one reachable path is required")
    incidence = path_movement_incidence(network, paths)
    target = np.asarray(target_flow, dtype=float)
    weight = np.ones(network.n_movements) if weights is None else np.asarray(weights, dtype=float)
    previous = (
        np.full(len(paths), 1.0 / len(paths))
        if previous_shares is None
        else np.asarray(previous_shares, dtype=float)
    )

    def objective(shares):
        error = weight * (incidence @ shares - target)
        return float(error @ error + switch_penalty * np.sum((shares - previous) ** 2))

    result = minimize(
        objective,
        previous,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * len(paths),
        constraints={"type": "eq", "fun": lambda shares: np.sum(shares) - 1.0},
        options={"ftol": 1e-12, "maxiter": 1000},
    )
    if not result.success:
        raise RuntimeError(f"route decomposition failed: {result.message}")
    shares = np.asarray(result.x)
    reconstruction = incidence @ shares
    return RouteShareResult(
        paths=paths,
        shares=shares,
        reconstructed_flow=reconstruction,
        weighted_error=float(np.linalg.norm(weight * (reconstruction - target))),
        objective=float(result.fun),
        status="optimal",
    )


def balanced_rounding(shares: np.ndarray, drivers: int, seed: int = 0) -> np.ndarray:
    """Integer route counts with exact total and randomized tie breaking."""

    if drivers < 0:
        raise ValueError("drivers cannot be negative")
    shares = np.maximum(np.asarray(shares, dtype=float), 0.0)
    shares /= max(float(np.sum(shares)), 1e-12)
    expected = shares * drivers
    counts = np.floor(expected).astype(int)
    remainder = drivers - int(np.sum(counts))
    rng = np.random.default_rng(seed)
    order = np.lexsort((rng.random(len(shares)), -(expected - counts)))
    counts[order[:remainder]] += 1
    return counts


def sample_next_hop(
    network: CellNetwork,
    cell: int,
    probabilities: np.ndarray,
    *,
    visited: set[int],
    seed: int,
) -> int:
    """Sample an admissible unvisited next cell; refuse a routing cycle."""

    indices = np.flatnonzero(network.movement_sources == cell)
    indices = np.asarray([index for index in indices if network.movements[index][1] not in visited])
    if not len(indices):
        raise RuntimeError("no loop-free next hop is available")
    weights = np.maximum(np.asarray(probabilities)[indices], 0.0)
    weights = weights / np.sum(weights) if np.sum(weights) else np.full(len(indices), 1.0 / len(indices))
    chosen = int(np.random.default_rng(seed).choice(indices, p=weights))
    return network.movements[chosen][1]


@dataclass
class RouteCommitment:
    route: tuple[int, ...]
    committed_until: int
    expected_time: float


class RouteCommitmentManager:
    """Prevents unstable displayed routes unless improvement exceeds a threshold."""

    def __init__(self, minimum_intervals: int = 3, improvement_threshold: float = 0.1) -> None:
        self.minimum_intervals = minimum_intervals
        self.improvement_threshold = improvement_threshold
        self._routes: dict[str, RouteCommitment] = {}

    def recommend(
        self, driver_id: str, route: tuple[int, ...], expected_time: float, time: int
    ) -> tuple[int, ...]:
        current = self._routes.get(driver_id)
        improvement = (
            1.0
            if current is None
            else (current.expected_time - expected_time) / max(current.expected_time, 1e-9)
        )
        if current and time < current.committed_until and improvement < self.improvement_threshold:
            return current.route
        self._routes[driver_id] = RouteCommitment(
            route, time + self.minimum_intervals, float(expected_time)
        )
        return route

