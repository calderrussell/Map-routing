from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from data_processed.networks import diamond_cell_network
from data_processed.scenarios import Incident, Scenario, dynamic_demand
from experiments.baselines import backpressure_policy, dynamic_shortest_path
from experiments.evaluate import evaluate_controller
from experiments.matrix import load_and_validate_plan
from experiments.robustness import robustness_grid
from routing.paths import balanced_rounding, candidate_paths, decompose_route_shares
from simulators.ctm import CTMSimulator
from simulators.sumo_adapter import SUMOAdapter, SUMOConfig


@dataclass(frozen=True)
class Phase4Result:
    seed: int
    experiment_ids: tuple[str, ...]
    paired_seeds: int
    robustness_conditions: int
    controlled_fraction_costs: dict[str, float]
    incident_capacity_violation: float
    incident_conservation_residual: float
    route_count: int
    route_decomposition_error: float
    rounded_drivers: int
    sumo_available: bool
    sumo_status: str


def run_phase4_smoke(seed: int = 17) -> Phase4Result:
    network = diamond_cell_network()
    origins, destinations = np.asarray([0]), np.asarray([3])
    simulator = CTMSimulator(network, origins, destinations)
    realized, forecast = dynamic_demand(np.asarray([8.0]), 12, seed, "near_capacity")
    incident = Incident(
        affected_cells=(1,),
        start=3,
        duration=4,
        capacity_multiplier=0.2,
        speed_multiplier=0.4,
        observation_delay=1,
        kind="unfamiliar_location",
    )
    scenario = Scenario(
        scenario_id="phase4_incident",
        topology=network.name,
        seed=seed,
        realized_demand=realized,
        forecast_demand=forecast,
        incidents=(incident,),
        regime="near_capacity",
    )

    def controlled(state, visible_forecast):
        return backpressure_policy(network, state, destinations)

    costs = {}
    probe = None
    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        probe = evaluate_controller(
            "backpressure_mixed",
            simulator,
            scenario,
            controlled,
            controlled_fraction=fraction,
        )
        costs[f"{fraction:.2f}"] = probe.metrics.total_system_travel_time
    assert probe is not None
    paths = candidate_paths(network, 0, 3, k_paths=4)
    decomposition = decompose_route_shares(
        network,
        paths,
        np.asarray([0.25, 0.75, 0.25, 0.75]),
        switch_penalty=0.0,
    )
    counts = balanced_rounding(decomposition.shares, 101, seed=seed)
    plan = load_and_validate_plan("configs/frozen_experiment_plan.yaml")
    sumo = SUMOAdapter(SUMOConfig(Path("data_raw/sumo/network.sumocfg")))
    sumo_available, sumo_status = sumo.availability()
    return Phase4Result(
        seed=seed,
        experiment_ids=tuple(item["id"] for item in plan["experiments"]),
        paired_seeds=int(plan["paired_seeds"]),
        robustness_conditions=len(robustness_grid()),
        controlled_fraction_costs=costs,
        incident_capacity_violation=probe.metrics.capacity_violation,
        incident_conservation_residual=probe.metrics.conservation_residual,
        route_count=len(paths),
        route_decomposition_error=decomposition.weighted_error,
        rounded_drivers=int(np.sum(counts)),
        sumo_available=sumo_available,
        sumo_status=sumo_status,
    )


def write_phase4_result(result: Phase4Result, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    result = run_phase4_smoke()
    write_phase4_result(result, "artifacts/runs/phase4_smoke.json")
    print(json.dumps(asdict(result), indent=2, sort_keys=True))

