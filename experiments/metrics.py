from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from simulators.ctm import CTMStepResult


@dataclass(frozen=True)
class SystemMetrics:
    total_system_travel_time: float
    throughput: float
    unfinished_vehicles: float
    source_queue_time: float
    maximum_accumulation: float
    conservation_residual: float
    capacity_violation: float
    invalid_turn_rate: float
    demand_completion: float


@dataclass(frozen=True)
class ProjectionAggregateMetrics:
    raw_violation: float
    post_projection_violation: float
    mean_normalized_correction: float
    projection_failures: int
    recirculating_flow: float


def system_metrics(
    results: list[CTMStepResult],
    initial_vehicles: float,
    total_demand: float,
    delta_t: float = 1.0,
) -> SystemMetrics:
    if not results:
        raise ValueError("at least one CTM step is required")
    accumulations = [step.state.vehicles for step in results]
    source_queue_time = delta_t * sum(float(np.sum(step.state.source_queue)) for step in results)
    throughput = sum(float(np.sum(step.exit_flow)) for step in results)
    available = initial_vehicles + total_demand
    invalid = sum(step.invalid_turn_flow for step in results)
    moved = sum(float(np.sum(step.movement_flow)) for step in results)
    return SystemMetrics(
        total_system_travel_time=delta_t * sum(accumulations),
        throughput=throughput,
        unfinished_vehicles=accumulations[-1],
        source_queue_time=source_queue_time,
        maximum_accumulation=max(accumulations),
        conservation_residual=max(abs(step.conservation_residual) for step in results),
        capacity_violation=sum(step.capacity_violation for step in results),
        invalid_turn_rate=invalid / max(moved, 1e-9),
        demand_completion=throughput / max(available, 1e-9),
    )


def oracle_relative_gap(policy_cost: float, oracle_cost: float) -> float:
    return float((policy_cost - oracle_cost) / max(abs(oracle_cost), 1e-9))


def solver_bound_aware_regret(policy_cost: float, lower_bound: float | None) -> float | None:
    if lower_bound is None:
        return None
    return float((policy_cost - lower_bound) / max(abs(lower_bound), 1e-9))


def recovered_ue_so_gap(policy_cost: float, ue_cost: float, so_cost: float) -> float:
    return float(1.0 - (policy_cost - so_cost) / max(ue_cost - so_cost, 1e-9))


def latency_percentiles(seconds: Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(seconds), dtype=float)
    return {key: float(np.percentile(values, value)) for key, value in (("p50", 50), ("p95", 95), ("p99", 99))}


def gini(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    if np.any(values < 0):
        raise ValueError("Gini inputs must be nonnegative")
    if not len(values) or np.sum(values) == 0:
        return 0.0
    difference = np.abs(values[:, None] - values[None, :]).sum()
    return float(difference / (2.0 * len(values) * np.sum(values)))


def fairness_metrics(recommended: np.ndarray, selfish: np.ndarray) -> dict[str, float]:
    ratios = np.asarray(recommended) / np.maximum(np.asarray(selfish), 1e-9)
    return {
        "median_detour_ratio": float(np.median(ratios)),
        "p95_detour_ratio": float(np.percentile(ratios, 95)),
        "maximum_detour_ratio": float(np.max(ratios)),
        "gini_recommended_time": gini(np.asarray(recommended)),
    }


def completed_trip_time_metrics(trip_times: np.ndarray) -> dict[str, float]:
    values = np.asarray(trip_times, dtype=float)
    if not len(values):
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan")}
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
    }


def aggregate_projection_metrics(results, recirculating_flow: float = 0.0) -> ProjectionAggregateMetrics:
    diagnostics = [item.diagnostics for item in results]
    failures = sum(item.status not in {"optimal", "optimal_inaccurate"} for item in diagnostics)
    return ProjectionAggregateMetrics(
        raw_violation=float(sum(item.raw_violation.total for item in diagnostics)),
        post_projection_violation=float(sum(item.projected_violation.total for item in diagnostics)),
        mean_normalized_correction=float(np.mean([item.normalized_correction for item in diagnostics])),
        projection_failures=failures,
        recirculating_flow=float(recirculating_flow),
    )


def paired_bootstrap_interval(
    left: np.ndarray,
    right: np.ndarray,
    *,
    seed: int,
    repetitions: int = 2000,
    confidence: float = 0.95,
) -> dict[str, float]:
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape:
        raise ValueError("paired samples must share a shape")
    differences = left - right
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [np.mean(rng.choice(differences, size=len(differences), replace=True)) for _ in range(repetitions)]
    )
    alpha = (1.0 - confidence) / 2.0
    std = np.std(differences, ddof=1) if len(differences) > 1 else 0.0
    return {
        "mean_difference": float(np.mean(differences)),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "paired_effect_size": float(np.mean(differences) / max(std, 1e-12)),
    }
