from __future__ import annotations

import math

import networkx as nx
import numpy as np

from oracle.network import CellNetwork
from oracle.static_assignment import StaticNetwork


def braess_static(include_middle: bool = True) -> tuple[StaticNetwork, dict[tuple[int, int], float]]:
    # Classical 4,000-driver example: with the zero-cost middle road, UE TSTT is
    # 320,000 versus system-optimal 260,000 vehicle-minutes.
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    free = [0.0, 45.0, 45.0, 0.0]
    coefficient = [0.01, 0.0, 0.0, 0.01]
    if include_middle:
        edges.append((1, 2))
        free.append(0.0)
        coefficient.append(0.0)
    network = StaticNetwork(
        name="braess_with_middle" if include_middle else "braess_without_middle",
        nodes=(0, 1, 2, 3),
        edges=tuple(edges),
        free_time=np.asarray(free),
        capacity=np.ones(len(edges)),
        coefficient=np.asarray(coefficient),
        power=np.ones(len(edges)),
    )
    return network, {(0, 3): 4000.0}


def analytic_parallel_static() -> tuple[StaticNetwork, dict[tuple[int, int], float]]:
    # Two paths: t_upper(x)=10+x/100 and t_lower=25.  At UE x_upper=1500;
    # at SO x_upper=750 for demand 2000.
    network = StaticNetwork(
        name="analytic_parallel",
        nodes=(0, 1, 2, 3),
        edges=((0, 1), (1, 3), (0, 2), (2, 3)),
        free_time=np.asarray([10.0, 0.0, 25.0, 0.0]),
        capacity=np.ones(4),
        coefficient=np.asarray([0.01, 0.0, 0.0, 0.0]),
        power=np.ones(4),
    )
    return network, {(0, 3): 2000.0}


def sioux_falls_static() -> tuple[StaticNetwork, dict[tuple[int, int], float]]:
    """Canonical 24-node/76-link Sioux Falls topology.

    The built-in demand is a documented representative multi-OD pilot.  Full standard
    matrices are supported by :func:`data_raw.tntp.load_tntp_demand` and deliberately
    remain external immutable inputs.
    """

    undirected = [
        (1, 2, 25900.20064, 6), (1, 3, 23403.47319, 4),
        (2, 6, 4958.180928, 5), (3, 4, 17110.52372, 4),
        (3, 12, 23403.47319, 4), (4, 5, 17782.79410, 2),
        (4, 11, 4908.82673, 6), (5, 6, 4947.995469, 4),
        (5, 9, 10000.0, 5), (6, 7, 7841.81131, 3),
        (6, 8, 4898.587646, 2),
        (7, 8, 7841.81131, 3), (7, 18, 23403.47319, 2),
        (8, 9, 5050.193156, 10), (8, 16, 5045.822583, 5),
        (9, 10, 13915.78842, 3), (10, 11, 10000.0, 5),
        (10, 15, 13512.00155, 6), (10, 16, 4854.917717, 4),
        (11, 12, 4908.82673, 6), (11, 14, 4876.508287, 4),
        (12, 13, 25900.20064, 3), (13, 24, 5091.256152, 4),
        (14, 15, 5127.526119, 5), (14, 23, 4924.790605, 4),
        (15, 19, 14564.75315, 3), (15, 22, 9599.180565, 3),
        (16, 17, 5229.910063, 2), (16, 18, 19679.89671, 3),
        (17, 19, 4823.950831, 2), (18, 20, 23403.47319, 4),
        (19, 20, 5002.607563, 4), (20, 21, 5059.91234, 6),
        (20, 22, 5075.697193, 5), (21, 22, 5229.910063, 2),
        (21, 24, 4885.357564, 3), (22, 23, 5000.0, 4),
        (23, 24, 5078.508436, 2),
    ]
    # One pair (5,9) is already represented by the explicit reverse-like final link;
    # deduplicate while constructing both directions to retain 76 directed links.
    edge_data: dict[tuple[int, int], tuple[float, float]] = {}
    for u, v, capacity, free_time in undirected:
        edge_data[(u, v)] = (capacity, free_time)
        edge_data[(v, u)] = (capacity, free_time)
    edges = tuple(edge_data)
    cap = np.asarray([edge_data[e][0] for e in edges])
    fft = np.asarray([edge_data[e][1] for e in edges])
    network = StaticNetwork(
        name="sioux_falls",
        nodes=tuple(range(1, 25)),
        edges=edges,
        free_time=fft,
        capacity=cap,
        coefficient=0.15 * fft,
        power=np.full(len(edges), 4.0),
    )
    demand = {(1, 20): 6000.0, (2, 15): 5000.0, (3, 18): 4000.0, (7, 13): 3000.0, (24, 1): 5000.0}
    return network, demand


def diamond_cell_network(name: str = "diamond") -> CellNetwork:
    return CellNetwork(
        name=name,
        cell_ids=("origin", "upper", "lower", "destination"),
        movements=((0, 1), (0, 2), (1, 3), (2, 3)),
        storage=np.asarray([40.0, 20.0, 35.0, 50.0]),
        capacity=np.asarray([12.0, 5.0, 10.0, 15.0]),
        free_speed=np.ones(4),
        wave_speed=np.ones(4),
        free_time=np.asarray([1.0, 1.0, 2.0, 1.0]),
        movement_capacity=np.asarray([5.0, 10.0, 5.0, 10.0]),
        lanes=np.asarray([2.0, 1.0, 2.0, 2.0]),
        road_class=np.asarray([1.0, 2.0, 1.0, 1.0]),
        bearing=np.asarray([0.0, math.pi / 4, -math.pi / 4, 0.0]),
    )


def chain_cell_network(length: int = 4, name: str | None = None) -> CellNetwork:
    if length < 2:
        raise ValueError("chain needs at least two cells")
    movements = tuple((i, i + 1) for i in range(length - 1))
    return CellNetwork(
        name=name or f"chain_{length}",
        cell_ids=tuple(f"c{i}" for i in range(length)),
        movements=movements,
        storage=np.full(length, 20.0),
        capacity=np.full(length, 6.0),
        free_speed=np.ones(length),
        wave_speed=np.ones(length),
        free_time=np.ones(length),
        movement_capacity=np.full(length - 1, 6.0),
        lanes=np.ones(length),
        road_class=np.ones(length),
        bearing=np.zeros(length),
    )


def grid_cell_network(
    rows: int,
    columns: int,
    *,
    seed: int = 0,
    name: str | None = None,
) -> CellNetwork:
    """Bidirectional synthetic grid used only after a topology split is assigned."""

    if rows < 2 or columns < 2:
        raise ValueError("grid dimensions must both be at least two")
    rng = np.random.default_rng(seed)
    movements: list[tuple[int, int]] = []
    bearing = np.zeros(rows * columns)
    for row in range(rows):
        for column in range(columns):
            cell = row * columns + column
            bearing[cell] = math.atan2(row - (rows - 1) / 2, column - (columns - 1) / 2)
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = row + dr, column + dc
                if 0 <= rr < rows and 0 <= cc < columns:
                    movements.append((cell, rr * columns + cc))
    n = rows * columns
    capacity = rng.uniform(5.0, 10.0, n)
    storage = capacity * rng.uniform(2.5, 4.0, n)
    return CellNetwork(
        name=name or f"synthetic_grid_{rows}x{columns}_seed{seed}",
        cell_ids=tuple(f"r{row}c{column}" for row in range(rows) for column in range(columns)),
        movements=tuple(movements),
        storage=storage,
        capacity=capacity,
        free_speed=np.ones(n),
        wave_speed=np.ones(n),
        free_time=rng.uniform(0.8, 1.3, n),
        movement_capacity=np.asarray([min(capacity[u], capacity[v]) for u, v in movements]),
        lanes=np.maximum(1.0, np.round(capacity / 4.0)),
        road_class=rng.integers(1, 4, n).astype(float),
        bearing=bearing,
    )


def random_planar_cell_network(
    n_cells: int,
    *,
    seed: int,
    name: str | None = None,
) -> CellNetwork:
    """Connected bidirectional random-geometric graph with reproducible physics."""

    if n_cells < 4:
        raise ValueError("random planar network requires at least four cells")
    radius = 0.25
    while True:
        graph = nx.random_geometric_graph(n_cells, radius, seed=seed)
        if nx.is_connected(graph):
            break
        radius += 0.05
        if radius > 1.0:
            raise RuntimeError("could not generate a connected geometric graph")
    positions = nx.get_node_attributes(graph, "pos")
    movements = tuple(
        (u, v) for u, v in graph.edges for (u, v) in ((u, v), (v, u))
    )
    rng = np.random.default_rng(seed)
    capacity = rng.uniform(4.0, 12.0, n_cells)
    bearing = np.asarray([math.atan2(positions[i][1] - 0.5, positions[i][0] - 0.5) for i in range(n_cells)])
    return CellNetwork(
        name=name or f"planar_{n_cells}_seed{seed}",
        cell_ids=tuple(f"p{i}" for i in range(n_cells)),
        movements=movements,
        storage=capacity * rng.uniform(2.0, 4.0, n_cells),
        capacity=capacity,
        free_speed=np.ones(n_cells),
        wave_speed=np.ones(n_cells),
        free_time=rng.uniform(0.6, 1.8, n_cells),
        movement_capacity=np.asarray([min(capacity[u], capacity[v]) for u, v in movements]),
        lanes=np.maximum(1.0, np.round(capacity / 4.0)),
        road_class=rng.integers(1, 5, n_cells).astype(float),
        bearing=bearing,
    )


def static_links_to_cells(network: StaticNetwork, interval_capacity_scale: float = 0.001) -> CellNetwork:
    """Convert a link-based benchmark into a directed cell graph.

    Each directed road link becomes a cell; legal movements join consecutive links,
    excluding immediate U-turns.  TNTP hourly capacities are scaled to a control
    interval by ``interval_capacity_scale``.
    """

    movements = []
    for first, (u, v) in enumerate(network.edges):
        for second, (vv, w) in enumerate(network.edges):
            if v == vv and w != u:
                movements.append((first, second))
    capacity = np.maximum(network.capacity * interval_capacity_scale, 0.1)
    movement_capacity = np.asarray([min(capacity[u], capacity[v]) for u, v in movements])
    return CellNetwork(
        name=f"{network.name}_cells",
        cell_ids=tuple(f"{u}->{v}" for u, v in network.edges),
        movements=tuple(movements),
        storage=np.maximum(capacity * 4.0, 1.0),
        capacity=capacity,
        free_speed=np.ones(network.n_edges),
        wave_speed=np.ones(network.n_edges),
        free_time=np.maximum(network.free_time, 0.1),
        movement_capacity=movement_capacity,
        lanes=np.maximum(1.0, np.round(capacity / np.median(capacity) * 2.0)),
        road_class=np.ones(network.n_edges),
        bearing=np.zeros(network.n_edges),
    )
