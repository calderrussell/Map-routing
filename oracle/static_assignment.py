from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Mapping

import networkx as nx
import numpy as np
from scipy.optimize import minimize_scalar


Array = np.ndarray
Objective = Literal["ue", "so"]


@dataclass(frozen=True)
class StaticNetwork:
    name: str
    nodes: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    free_time: Array
    capacity: Array
    coefficient: Array
    power: Array

    def __post_init__(self) -> None:
        m = len(self.edges)
        for field in ("free_time", "capacity", "coefficient", "power"):
            value = np.asarray(getattr(self, field), dtype=float)
            if value.shape != (m,):
                raise ValueError(f"{field} must have shape ({m},)")
            object.__setattr__(self, field, value)
        if len(set(self.edges)) != m:
            raise ValueError("parallel links must be represented using distinct intermediate nodes")
        if np.any(self.capacity <= 0) or np.any(self.power < 1):
            raise ValueError("capacity must be positive and power at least one")

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def travel_time(self, flow: Array) -> Array:
        flow = np.maximum(np.asarray(flow, dtype=float), 0.0)
        return self.free_time + self.coefficient * np.power(flow / self.capacity, self.power)

    def marginal_time(self, flow: Array) -> Array:
        flow = np.maximum(np.asarray(flow, dtype=float), 0.0)
        return self.free_time + self.coefficient * (self.power + 1.0) * np.power(
            flow / self.capacity, self.power
        )

    def beckmann(self, flow: Array) -> float:
        flow = np.maximum(np.asarray(flow, dtype=float), 0.0)
        integral = self.free_time * flow + (
            self.coefficient
            * self.capacity
            / (self.power + 1.0)
            * np.power(flow / self.capacity, self.power + 1.0)
        )
        return float(np.sum(integral))

    def tstt(self, flow: Array) -> float:
        flow = np.asarray(flow, dtype=float)
        return float(np.dot(flow, self.travel_time(flow)))


@dataclass(frozen=True)
class StaticSolution:
    objective: Objective
    flow: Array
    travel_time: Array
    tstt: float
    iterations: int
    relative_gap: float


def _all_or_nothing(
    network: StaticNetwork,
    demand: Mapping[tuple[int, int], float],
    costs: Array,
) -> Array:
    graph = nx.DiGraph()
    edge_index = {edge: index for index, edge in enumerate(network.edges)}
    graph.add_nodes_from(network.nodes)
    for index, (source, target) in enumerate(network.edges):
        graph.add_edge(source, target, weight=float(costs[index]))
    flow = np.zeros(network.n_edges, dtype=float)
    for (origin, destination), volume in sorted(demand.items()):
        if volume <= 0:
            continue
        path = nx.shortest_path(graph, origin, destination, weight="weight")
        for edge in zip(path[:-1], path[1:]):
            flow[edge_index[edge]] += float(volume)
    return flow


def frank_wolfe(
    network: StaticNetwork,
    demand: Mapping[tuple[int, int], float],
    objective: Objective,
    *,
    max_iterations: int = 1000,
    tolerance: float = 1e-8,
) -> StaticSolution:
    """Solve separable UE or SO assignment with exact one-dimensional line search."""

    if objective not in ("ue", "so"):
        raise ValueError("objective must be 'ue' or 'so'")
    flow = _all_or_nothing(network, demand, network.free_time)
    relative_gap = float("inf")
    iterations = 0
    for iterations in range(1, max_iterations + 1):
        gradient = (
            network.travel_time(flow) if objective == "ue" else network.marginal_time(flow)
        )
        target = _all_or_nothing(network, demand, gradient)
        direction = target - flow
        fw_gap = max(0.0, -float(np.dot(gradient, direction)))
        scale = max(abs(float(np.dot(gradient, flow))), 1.0)
        relative_gap = fw_gap / scale
        if relative_gap <= tolerance:
            break
        scalar_objective = network.beckmann if objective == "ue" else network.tstt
        result = minimize_scalar(
            lambda step: scalar_objective(flow + step * direction),
            bounds=(0.0, 1.0),
            method="bounded",
            options={"xatol": 1e-12},
        )
        step = float(result.x) if result.success else 2.0 / (iterations + 2.0)
        flow = np.maximum(flow + step * direction, 0.0)
    return StaticSolution(
        objective=objective,
        flow=flow,
        travel_time=network.travel_time(flow),
        tstt=network.tstt(flow),
        iterations=iterations,
        relative_gap=relative_gap,
    )


def price_of_anarchy_gap(
    policy_cost: float,
    ue_cost: float,
    so_cost: float,
    epsilon: float = 1e-9,
) -> float:
    """Fraction of avoidable UE--SO cost remaining (TeX equation 9)."""

    return float((policy_cost - so_cost) / (ue_cost - so_cost + epsilon))

