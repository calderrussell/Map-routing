from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


DemandRegime = Literal["low", "near_capacity", "oversaturated", "event"]
IncidentKind = Literal[
    "unfamiliar_location", "outside_severity", "multiple", "closure", "noisy_duration"
]


@dataclass(frozen=True)
class Incident:
    affected_cells: tuple[int, ...]
    start: int
    duration: int
    capacity_multiplier: float
    speed_multiplier: float
    observation_delay: int
    kind: IncidentKind

    def active(self, time: int) -> bool:
        return self.start <= time < self.start + self.duration


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    topology: str
    seed: int
    realized_demand: np.ndarray
    forecast_demand: np.ndarray
    incidents: tuple[Incident, ...]
    regime: DemandRegime


def dynamic_demand(
    base_od: np.ndarray,
    horizon: int,
    seed: int,
    regime: DemandRegime = "near_capacity",
    forecast_noise: float = 0.1,
) -> tuple[np.ndarray, np.ndarray]:
    """Equation (19): peaked demand with correlated multiplicative disturbance."""

    rng = np.random.default_rng(seed)
    base = np.asarray(base_od, dtype=float)
    time = np.linspace(0.0, 1.0, horizon)
    centers = [rng.uniform(0.25, 0.45)]
    if regime == "event":
        centers.append(rng.uniform(0.6, 0.8))
    width = rng.uniform(0.08, 0.22)
    profile = 0.25 + sum(np.exp(-0.5 * ((time - center) / width) ** 2) for center in centers)
    profile /= max(float(np.mean(profile)), 1e-9)
    scale = {"low": 0.55, "near_capacity": 1.0, "oversaturated": 1.45, "event": 1.2}[regime]
    common = np.zeros(horizon)
    for t in range(1, horizon):
        common[t] = 0.8 * common[t - 1] + rng.normal(0.0, 0.08)
    idiosyncratic = rng.normal(0.0, 0.06, size=(horizon, len(base)))
    disturbance = np.exp(common[:, None] + idiosyncratic)
    realized = np.maximum(scale * profile[:, None] * base[None, :] * disturbance, 0.0)
    forecast = np.maximum(
        realized * np.exp(rng.normal(0.0, forecast_noise, size=realized.shape)), 0.0
    )
    return realized, forecast


def sample_incidents(
    n_cells: int, horizon: int, seed: int, kind: IncidentKind
) -> tuple[Incident, ...]:
    rng = np.random.default_rng(seed)
    count = 2 if kind == "multiple" else 1
    incidents = []
    for _ in range(count):
        severity = rng.uniform(0.25, 0.75)
        if kind == "outside_severity":
            severity = rng.uniform(0.05, 0.2)
        if kind == "closure":
            severity = 0.0
        incidents.append(
            Incident(
                affected_cells=(int(rng.integers(0, n_cells)),),
                start=int(rng.integers(0, max(1, horizon // 2))),
                duration=max(1, int(rng.integers(1, max(2, horizon // 2)))),
                capacity_multiplier=float(severity),
                speed_multiplier=float(max(severity, 0.1)),
                observation_delay=int(rng.integers(1, 4)) if kind == "noisy_duration" else 0,
                kind=kind,
            )
        )
    return tuple(incidents)


def assert_topology_disjoint(splits: dict[str, list[str]]) -> None:
    names = list(splits)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = set(splits[left]) & set(splits[right])
            if overlap:
                raise ValueError(f"topology leakage between {left} and {right}: {sorted(overlap)}")

