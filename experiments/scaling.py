from __future__ import annotations

from dataclasses import dataclass
import time
import tracemalloc

import numpy as np

from data_processed.networks import grid_cell_network
from models.features import build_features
from simulators.ctm import CTMSimulator


@dataclass(frozen=True)
class ScalingPoint:
    cells: int
    movements: int
    active_commodities: int
    preprocessing_seconds: float
    forward_seconds: float
    peak_memory_bytes: int


def scaling_benchmark(model_factory, sizes=((2, 2), (3, 3), (4, 4)), commodities=(1, 2, 4)):
    """E7 runtime/memory curves over network and active-commodity size."""

    points = []
    for (rows, columns), k in zip(sizes, commodities):
        network = grid_cell_network(rows, columns, seed=rows * 100 + columns)
        origins = np.arange(k) % network.n_cells
        destinations = (np.arange(k) + network.n_cells // 2) % network.n_cells
        simulator = CTMSimulator(network, origins, destinations)
        demand = np.ones((3, k))
        tracemalloc.start()
        started = time.perf_counter()
        features = build_features(
            network, simulator.empty_state(), origins, destinations, demand
        )
        preprocessing = time.perf_counter() - started
        model = model_factory()
        started = time.perf_counter()
        model(features)
        forward = time.perf_counter() - started
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        points.append(
            ScalingPoint(
                network.n_cells,
                network.n_movements,
                k,
                preprocessing,
                forward,
                peak,
            )
        )
    return tuple(points)
