from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import Iterable

import networkx as nx
import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class CellNetwork:
    """Directed CTM cell graph with physically scaled attributes.

    Cells are vertices and allowed intersection movements are directed edges.  The
    representation is deliberately array based so it can be shared by NumPy, CVXPY,
    and PyTorch implementations without graph-library state leaking into checkpoints.
    """

    name: str
    cell_ids: tuple[str, ...]
    movements: tuple[tuple[int, int], ...]
    storage: Array
    capacity: Array
    free_speed: Array
    wave_speed: Array
    free_time: Array
    movement_capacity: Array
    lanes: Array
    road_class: Array
    bearing: Array

    def __post_init__(self) -> None:
        n = len(self.cell_ids)
        m = len(self.movements)
        cell_fields = (
            "storage",
            "capacity",
            "free_speed",
            "wave_speed",
            "free_time",
            "lanes",
            "road_class",
            "bearing",
        )
        for field in cell_fields:
            value = np.asarray(getattr(self, field), dtype=float)
            if value.shape != (n,):
                raise ValueError(f"{field} must have shape ({n},), got {value.shape}")
            object.__setattr__(self, field, value)
        movement_capacity = np.asarray(self.movement_capacity, dtype=float)
        if movement_capacity.shape != (m,):
            raise ValueError(
                f"movement_capacity must have shape ({m},), got {movement_capacity.shape}"
            )
        object.__setattr__(self, "movement_capacity", movement_capacity)
        if len(set(self.cell_ids)) != n:
            raise ValueError("cell_ids must be unique")
        if np.any(self.storage <= 0) or np.any(self.capacity <= 0):
            raise ValueError("storage and capacity must be strictly positive")
        if np.any(self.free_speed <= 0) or np.any(self.wave_speed <= 0):
            raise ValueError("wave and free-speed coefficients must be positive")
        for source, target in self.movements:
            if source < 0 or source >= n or target < 0 or target >= n:
                raise ValueError(f"movement {(source, target)} references an invalid cell")
            if source == target:
                raise ValueError("self movements are not admissible CTM turns")

    @property
    def n_cells(self) -> int:
        return len(self.cell_ids)

    @property
    def n_movements(self) -> int:
        return len(self.movements)

    @property
    def movement_sources(self) -> Array:
        return np.fromiter((u for u, _ in self.movements), dtype=np.int64)

    @property
    def movement_targets(self) -> Array:
        return np.fromiter((v for _, v in self.movements), dtype=np.int64)

    def graph(self, disabled_movements: Iterable[int] = ()) -> nx.DiGraph:
        disabled = set(disabled_movements)
        graph = nx.DiGraph()
        graph.add_nodes_from(range(self.n_cells))
        graph.add_edges_from(
            movement for index, movement in enumerate(self.movements) if index not in disabled
        )
        return graph

    def reachability_mask(
        self, destinations: Array, disabled_movements: Iterable[int] = ()
    ) -> Array:
        """Return K x M legality mask including destination reachability."""

        destinations = np.asarray(destinations, dtype=np.int64)
        graph = self.graph(disabled_movements)
        disabled = set(disabled_movements)
        mask = np.zeros((len(destinations), self.n_movements), dtype=bool)
        for commodity, destination in enumerate(destinations):
            if destination < 0 or destination >= self.n_cells:
                raise ValueError(f"invalid destination cell {destination}")
            reverse_reachable = nx.ancestors(graph, int(destination)) | {int(destination)}
            for movement_index, (source, target) in enumerate(self.movements):
                mask[commodity, movement_index] = (
                    movement_index not in disabled
                    and source != destination
                    and target in reverse_reachable
                )
        return mask

    def topology_hash(self) -> str:
        payload = {
            "name": self.name,
            "cell_ids": self.cell_ids,
            "movements": self.movements,
            "storage": self.storage.tolist(),
            "capacity": self.capacity.tolist(),
            "free_time": self.free_time.tolist(),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def with_incident(
        self,
        affected_cells: Iterable[int],
        capacity_multiplier: float,
        speed_multiplier: float = 1.0,
    ) -> "CellNetwork":
        capacity = self.capacity.copy()
        free_speed = self.free_speed.copy()
        for cell in affected_cells:
            capacity[cell] *= capacity_multiplier
            free_speed[cell] *= speed_multiplier
        return replace(self, capacity=capacity, free_speed=free_speed)


@dataclass
class CTMState:
    occupancy: Array
    source_queue: Array
    time: int = 0
    previous_action: Array | None = None

    def copy(self) -> "CTMState":
        return CTMState(
            occupancy=np.asarray(self.occupancy, dtype=float).copy(),
            source_queue=np.asarray(self.source_queue, dtype=float).copy(),
            time=int(self.time),
            previous_action=None
            if self.previous_action is None
            else np.asarray(self.previous_action, dtype=float).copy(),
        )

    @property
    def vehicles(self) -> float:
        return float(np.sum(self.occupancy) + np.sum(self.source_queue))

