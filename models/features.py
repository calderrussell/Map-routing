from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from oracle.network import CellNetwork, CTMState


CELL_FEATURES = (
    "occupancy_over_storage",
    "recent_inflow_over_capacity",
    "recent_outflow_over_capacity",
    "speed_over_free_speed",
    "travel_time_over_free_time",
    "log_storage",
    "capacity_over_network_median",
    "log_free_time",
    "lanes_over_four",
    "road_class_over_five",
    "bearing_sin",
    "bearing_cos",
    "incident_capacity_multiplier",
    "incident_speed_multiplier",
    "source_indicator",
    "destination_indicator",
)

INTERSECTION_FEATURES = (
    "fixed_phase_green_fraction",
    "conflicting_movement_degree",
    "movement_saturation",
    "normalized_pressure",
)

OD_FEATURES = (
    "source_queue_over_origin_capacity",
    "forecast_first_over_origin_capacity",
    "forecast_mean_over_origin_capacity",
    "forecast_peak_over_origin_capacity",
    "forecast_total_scaled",
    "free_flow_od_time_scaled",
    "origin_occupancy",
    "destination_occupancy",
)

MOVEMENT_FEATURES = (
    "capacity_over_source_capacity",
    "free_time_ratio",
    "turn_sin",
    "turn_cos",
    "closure_flag",
    "reachable_fraction",
)

GLOBAL_FEATURES = (
    "time_sin",
    "time_cos",
    "total_demand_scaled",
    "mean_occupancy",
    "disabled_capacity_fraction",
    "forecast_confidence",
)


@dataclass(frozen=True)
class FeatureBatch:
    cell: torch.Tensor
    intersection: torch.Tensor
    od: torch.Tensor
    movement: torch.Tensor
    global_context: torch.Tensor
    edge_index: torch.Tensor
    origins: torch.Tensor
    destinations: torch.Tensor
    legal_mask: torch.Tensor
    movement_sources: torch.Tensor
    movement_targets: torch.Tensor
    network_name: str

    def to(self, device: torch.device | str) -> "FeatureBatch":
        values = {
            field: getattr(self, field).to(device)
            for field in (
                "cell",
                "intersection",
                "od",
                "movement",
                "global_context",
                "edge_index",
                "origins",
                "destinations",
                "legal_mask",
                "movement_sources",
                "movement_targets",
            )
        }
        return FeatureBatch(network_name=self.network_name, **values)


def _shortest_free_time(network: CellNetwork, origin: int, destination: int) -> float:
    import networkx as nx

    graph = network.graph()
    for source, target in graph.edges:
        graph[source][target]["weight"] = float(network.free_time[target])
    try:
        return float(nx.shortest_path_length(graph, origin, destination, weight="weight"))
    except nx.NetworkXNoPath:
        return float(np.sum(network.free_time))


def build_features(
    network: CellNetwork,
    state: CTMState,
    origins: np.ndarray,
    destinations: np.ndarray,
    demand_forecast: np.ndarray,
    *,
    recent_inflow: np.ndarray | None = None,
    recent_outflow: np.ndarray | None = None,
    capacity_multiplier: np.ndarray | None = None,
    speed_multiplier: np.ndarray | None = None,
    disabled_movements: np.ndarray | None = None,
    forecast_confidence: float = 1.0,
    steps_per_day: int = 288,
) -> FeatureBatch:
    """Create physically normalized, deployment-safe model features."""

    origins = np.asarray(origins, dtype=np.int64)
    destinations = np.asarray(destinations, dtype=np.int64)
    forecast = np.asarray(demand_forecast, dtype=float)
    if forecast.ndim == 1:
        forecast = forecast[:, None]
    c = network.n_cells
    k = len(origins)
    total_occupancy = np.sum(state.occupancy, axis=0)
    occupancy_ratio = total_occupancy / network.storage
    recent_in = np.zeros(c) if recent_inflow is None else np.asarray(recent_inflow, dtype=float)
    recent_out = np.zeros(c) if recent_outflow is None else np.asarray(recent_outflow, dtype=float)
    cap_mult = np.ones(c) if capacity_multiplier is None else np.asarray(capacity_multiplier, dtype=float)
    speed_mult = np.ones(c) if speed_multiplier is None else np.asarray(speed_multiplier, dtype=float)
    source_indicator = np.zeros(c)
    destination_indicator = np.zeros(c)
    source_indicator[origins] = 1.0
    destination_indicator[destinations] = 1.0
    median_capacity = max(float(np.median(network.capacity)), 1e-9)
    cell = np.column_stack(
        [
            occupancy_ratio,
            recent_in / network.capacity,
            recent_out / network.capacity,
            speed_mult,
            1.0 + occupancy_ratio**2,
            np.log1p(network.storage),
            network.capacity / median_capacity,
            np.log1p(network.free_time),
            network.lanes / 4.0,
            network.road_class / 5.0,
            np.sin(network.bearing),
            np.cos(network.bearing),
            cap_mult,
            speed_mult,
            source_indicator,
            destination_indicator,
        ]
    )
    od_rows = []
    for commodity, (origin, destination) in enumerate(zip(origins, destinations)):
        scale = max(float(network.capacity[origin]), 1e-9)
        series = forecast[:, commodity]
        od_rows.append(
            [
                state.source_queue[commodity] / scale,
                series[0] / scale,
                float(np.mean(series)) / scale,
                float(np.max(series)) / scale,
                float(np.sum(series)) / (scale * max(len(series), 1)),
                _shortest_free_time(network, int(origin), int(destination))
                / max(float(np.sum(network.free_time)), 1e-9),
                occupancy_ratio[origin],
                occupancy_ratio[destination],
            ]
        )
    legal = network.reachability_mask(destinations)
    disabled = (
        np.zeros(network.n_movements, dtype=bool)
        if disabled_movements is None
        else np.asarray(disabled_movements, dtype=bool)
    )
    legal &= ~disabled[None, :]
    source = network.movement_sources
    target = network.movement_targets
    turn = network.bearing[target] - network.bearing[source]
    movement = np.column_stack(
        [
            network.movement_capacity / network.capacity[source],
            network.free_time[target] / np.maximum(network.free_time[source], 1e-9),
            np.sin(turn),
            np.cos(turn),
            disabled.astype(float),
            np.mean(legal, axis=0),
        ]
    )
    phase = 2.0 * math.pi * ((state.time % steps_per_day) / steps_per_day)
    global_context = np.asarray(
        [
            math.sin(phase),
            math.cos(phase),
            float(np.sum(forecast)) / max(float(np.sum(network.capacity)), 1e-9),
            float(np.mean(occupancy_ratio)),
            float(1.0 - np.sum(network.capacity * cap_mult) / np.sum(network.capacity)),
            float(forecast_confidence),
        ]
    )
    outgoing_degree = np.bincount(source, minlength=c).astype(float)
    incoming_degree = np.bincount(target, minlength=c).astype(float)
    intersection = np.column_stack(
        [
            np.ones(c),  # Routing-only paper: fixed known signal service.
            (outgoing_degree + incoming_degree) / max(float(np.max(outgoing_degree + incoming_degree)), 1.0),
            recent_out / network.capacity,
            np.clip(
                occupancy_ratio
                - np.asarray(
                    [np.mean(occupancy_ratio[target[source == cell]]) if np.any(source == cell) else 0.0 for cell in range(c)]
                ),
                -1.0,
                1.0,
            ),
        ]
    )
    return FeatureBatch(
        cell=torch.as_tensor(cell, dtype=torch.float32),
        intersection=torch.as_tensor(intersection, dtype=torch.float32),
        od=torch.as_tensor(np.asarray(od_rows), dtype=torch.float32),
        movement=torch.as_tensor(movement, dtype=torch.float32),
        global_context=torch.as_tensor(global_context, dtype=torch.float32),
        edge_index=torch.as_tensor(np.stack([source, target]), dtype=torch.long),
        origins=torch.as_tensor(origins, dtype=torch.long),
        destinations=torch.as_tensor(destinations, dtype=torch.long),
        legal_mask=torch.as_tensor(legal, dtype=torch.bool),
        movement_sources=torch.as_tensor(source, dtype=torch.long),
        movement_targets=torch.as_tensor(target, dtype=torch.long),
        network_name=network.name,
    )
