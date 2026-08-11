from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from data_processed.scenarios import Scenario
from experiments.baselines import dynamic_shortest_path
from experiments.metrics import SystemMetrics, latency_percentiles, system_metrics
from oracle.network import CTMState
from routing.compliance import combine_controlled_actions
from simulators.ctm import CTMSimulator


Policy = Callable[[CTMState, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EvaluationResult:
    controller: str
    scenario_id: str
    controlled_fraction: float
    metrics: SystemMetrics
    latency: dict[str, float]
    preprocessing_seconds: float
    policy_seconds: float
    projection_seconds: float
    route_decomposition_seconds: float
    end_to_end_seconds: float


def _incident_arrays(simulator: CTMSimulator, scenario: Scenario, time_index: int):
    network = simulator.network
    cap = np.ones(network.n_cells)
    speed = np.ones(network.n_cells)
    disabled = np.zeros(network.n_movements, dtype=bool)
    for incident in scenario.incidents:
        if incident.active(time_index):
            for cell in incident.affected_cells:
                cap[cell] *= incident.capacity_multiplier
                speed[cell] *= incident.speed_multiplier
                if incident.capacity_multiplier == 0:
                    disabled |= (network.movement_sources == cell) | (network.movement_targets == cell)
    return cap, speed, disabled


def evaluate_controller(
    name: str,
    simulator: CTMSimulator,
    scenario: Scenario,
    policy: Policy,
    *,
    controlled_fraction: float = 1.0,
    initial_state: CTMState | None = None,
    projection_seconds: float = 0.0,
    route_decomposition_seconds: float = 0.0,
) -> EvaluationResult:
    state = simulator.empty_state() if initial_state is None else initial_state.copy()
    initial_vehicles = state.vehicles
    results = []
    latencies, policy_total, preprocess_total = [], 0.0, 0.0
    started_all = time.perf_counter()
    for time_index, realized in enumerate(scenario.realized_demand):
        started = time.perf_counter()
        cap, speed, disabled = _incident_arrays(simulator, scenario, time_index)
        preprocess_total += time.perf_counter() - started
        forecast = scenario.forecast_demand[time_index:]
        policy_started = time.perf_counter()
        controlled = policy(state, forecast)
        policy_elapsed = time.perf_counter() - policy_started
        policy_total += policy_elapsed
        uncontrolled = dynamic_shortest_path(simulator.network, state, simulator.destinations)
        action = combine_controlled_actions(controlled, uncontrolled, controlled_fraction)
        step = simulator.step(
            state,
            realized,
            action,
            capacity_multiplier=cap,
            speed_multiplier=speed,
            disabled_movements=disabled,
        )
        results.append(step)
        state = step.state
        latencies.append(time.perf_counter() - started)
    end_to_end = time.perf_counter() - started_all + projection_seconds + route_decomposition_seconds
    return EvaluationResult(
        controller=name,
        scenario_id=scenario.scenario_id,
        controlled_fraction=controlled_fraction,
        metrics=system_metrics(
            results,
            initial_vehicles,
            float(np.sum(scenario.realized_demand)),
            simulator.delta_t,
        ),
        latency=latency_percentiles(latencies),
        preprocessing_seconds=preprocess_total,
        policy_seconds=policy_total,
        projection_seconds=projection_seconds,
        route_decomposition_seconds=route_decomposition_seconds,
        end_to_end_seconds=end_to_end,
    )

