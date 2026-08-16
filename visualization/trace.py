from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np

from data_processed.scenarios import Scenario
from experiments.baselines import dynamic_shortest_path
from experiments.evaluate import EvaluationResult, incident_arrays
from experiments.metrics import latency_percentiles, system_metrics
from oracle.network import CTMState
from routing.compliance import combine_controlled_actions
from simulators.ctm import CTMSimulator, CTMStepResult


Policy = Callable[[CTMState, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class TraceStep:
    time_index: int
    state_before: CTMState
    demand: np.ndarray
    controlled_action: np.ndarray
    applied_action: np.ndarray
    capacity_multiplier: np.ndarray
    speed_multiplier: np.ndarray
    disabled_movements: np.ndarray
    result: CTMStepResult


@dataclass(frozen=True)
class ControllerTrace:
    evaluation: EvaluationResult
    steps: tuple[TraceStep, ...]


def evaluate_controller_with_trace(
    name: str,
    simulator: CTMSimulator,
    scenario: Scenario,
    policy: Policy,
    *,
    controlled_fraction: float = 1.0,
    initial_state: CTMState | None = None,
) -> ControllerTrace:
    """Evaluate once while retaining the physical state/action/flow sequence for a map."""

    state = simulator.empty_state() if initial_state is None else initial_state.copy()
    initial_vehicles = state.vehicles
    results: list[CTMStepResult] = []
    steps: list[TraceStep] = []
    latencies: list[float] = []
    policy_total = 0.0
    preprocess_total = 0.0
    started_all = time.perf_counter()
    for time_index, realized in enumerate(scenario.realized_demand):
        started = time.perf_counter()
        cap, speed, disabled = incident_arrays(simulator, scenario, time_index)
        preprocess_total += time.perf_counter() - started
        forecast = scenario.forecast_demand[time_index:]
        policy_started = time.perf_counter()
        controlled = np.asarray(policy(state, forecast), dtype=float)
        policy_total += time.perf_counter() - policy_started
        uncontrolled = dynamic_shortest_path(simulator.network, state, simulator.destinations)
        applied = combine_controlled_actions(controlled, uncontrolled, controlled_fraction)
        state_before = state.copy()
        result = simulator.step(
            state,
            realized,
            applied,
            capacity_multiplier=cap,
            speed_multiplier=speed,
            disabled_movements=disabled,
        )
        steps.append(
            TraceStep(
                time_index=time_index,
                state_before=state_before,
                demand=np.asarray(realized, dtype=float).copy(),
                controlled_action=controlled.copy(),
                applied_action=np.asarray(applied, dtype=float).copy(),
                capacity_multiplier=cap.copy(),
                speed_multiplier=speed.copy(),
                disabled_movements=disabled.copy(),
                result=result,
            )
        )
        results.append(result)
        state = result.state
        latencies.append(time.perf_counter() - started)
    end_to_end = time.perf_counter() - started_all
    evaluation = EvaluationResult(
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
        projection_seconds=0.0,
        route_decomposition_seconds=0.0,
        end_to_end_seconds=end_to_end,
    )
    return ControllerTrace(evaluation=evaluation, steps=tuple(steps))
