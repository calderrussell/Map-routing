from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time

import numpy as np
import torch

from data_processed.dataset import gradient_oracle_demonstrations
from data_processed.networks import sioux_falls_static, static_links_to_cells
from data_processed.scenarios import Incident, Scenario, dynamic_demand
from experiments.baselines import dynamic_shortest_path
from experiments.evaluate import evaluate_controller
from experiments.train import set_deterministic_seed, train_imitation
from models.features import build_features
from models.heterogeneous import HeteroSpatioTemporalGNN
from simulators.ctm import CTMSimulator


SEED = 31
OD_ZONE_PAIRS = ((1, 20), (7, 13), (24, 1))
# A zone is attached to one reproducible outbound/inbound road cell for this example.
OD_CELL_LINKS = (((1, 2), (18, 20)), ((7, 8), (12, 13)), ((24, 13), (2, 1)))
INCIDENT_LINK = (10, 15)


@dataclass(frozen=True)
class SiouxFallsProblem:
    static_network: object
    cell_network: object
    simulator: CTMSimulator
    origins: np.ndarray
    destinations: np.ndarray
    link_index: dict[tuple[int, int], int]


@dataclass(frozen=True)
class ControllerSummary:
    tstt: float
    throughput: float
    unfinished_vehicles: float
    source_queue_time: float
    capacity_violation: float
    conservation_residual: float
    latency_p95_seconds: float


@dataclass(frozen=True)
class SiouxFallsExampleResult:
    seed: int
    benchmark: str
    intersections: int
    directed_road_links: int
    cell_movements: int
    od_zone_pairs: tuple[tuple[int, int], ...]
    attached_od_cells: tuple[tuple[str, str], ...]
    incident_link: tuple[int, int]
    train_demonstrations: int
    oracle_statuses: tuple[str, ...]
    oracle_certified: bool
    model_parameters: int
    training_seconds: float
    imitation_loss_initial: float
    imitation_loss_final: float
    gnn: ControllerSummary
    dynamic_shortest_path: ControllerSummary
    gnn_minus_baseline_tstt: float


def build_problem() -> SiouxFallsProblem:
    """Create the canonical 24-node/76-link Sioux Falls benchmark as a CTM graph."""

    static_network, _ = sioux_falls_static()
    cell_network = static_links_to_cells(static_network)
    link_index = {edge: index for index, edge in enumerate(static_network.edges)}
    origins = np.asarray([link_index[origin_link] for origin_link, _ in OD_CELL_LINKS])
    destinations = np.asarray(
        [link_index[destination_link] for _, destination_link in OD_CELL_LINKS]
    )
    simulator = CTMSimulator(cell_network, origins, destinations)
    return SiouxFallsProblem(
        static_network, cell_network, simulator, origins, destinations, link_index
    )


class IncidentAwareGNNPolicy:
    """Stateful GRU policy with the same incident observation used by the simulator."""

    def __init__(
        self,
        model: HeteroSpatioTemporalGNN,
        problem: SiouxFallsProblem,
        incident: Incident,
    ) -> None:
        self.model = model
        self.problem = problem
        self.incident = incident
        self.hidden = None

    def __call__(self, state, forecast):
        network = self.problem.cell_network
        capacity_multiplier = np.ones(network.n_cells)
        speed_multiplier = np.ones(network.n_cells)
        if self.incident.active(state.time):
            for cell in self.incident.affected_cells:
                capacity_multiplier[cell] *= self.incident.capacity_multiplier
                speed_multiplier[cell] *= self.incident.speed_multiplier
        features = build_features(
            network,
            state,
            self.problem.origins,
            self.problem.destinations,
            forecast,
            capacity_multiplier=capacity_multiplier,
            speed_multiplier=speed_multiplier,
            forecast_confidence=0.85,
        )
        self.model.eval()
        with torch.no_grad():
            output = self.model(features, self.hidden)
        self.hidden = output.hetero_hidden
        return output.splits.cpu().numpy()


def _summary(evaluation) -> ControllerSummary:
    metrics = evaluation.metrics
    return ControllerSummary(
        tstt=metrics.total_system_travel_time,
        throughput=metrics.throughput,
        unfinished_vehicles=metrics.unfinished_vehicles,
        source_queue_time=metrics.source_queue_time,
        capacity_violation=metrics.capacity_violation,
        conservation_residual=metrics.conservation_residual,
        latency_p95_seconds=evaluation.latency["p95"],
    )


def run_example(*, quick: bool = True, seed: int = SEED) -> tuple[SiouxFallsExampleResult, HeteroSpatioTemporalGNN]:
    """Train on oracle demonstrations and evaluate one held-out incident scenario."""

    set_deterministic_seed(seed)
    problem = build_problem()
    demonstrations = 6 if quick else 30
    oracle_iterations = 12 if quick else 80
    training_epochs = 60 if quick else 250
    started = time.perf_counter()
    training_data = gradient_oracle_demonstrations(
        problem.cell_network,
        problem.origins,
        problem.destinations,
        count=demonstrations,
        horizon=10,
        seed=seed,
        oracle_iterations=oracle_iterations,
    )
    model = HeteroSpatioTemporalGNN(hidden_dim=48 if quick else 96, layers=2 if quick else 3)
    history = train_imitation(
        model,
        training_data,
        epochs=training_epochs,
        learning_rate=2e-3,
        seed=seed,
    )
    training_seconds = time.perf_counter() - started

    realized, forecast = dynamic_demand(
        np.asarray([6.0, 4.0, 4.0]),
        horizon=16,
        seed=seed + 100,
        regime="event",
        forecast_noise=0.15,
    )
    incident_cell = problem.link_index[INCIDENT_LINK]
    incident = Incident(
        affected_cells=(incident_cell,),
        start=5,
        duration=5,
        capacity_multiplier=0.25,
        speed_multiplier=0.5,
        observation_delay=1,
        kind="unfamiliar_location",
    )
    scenario = Scenario(
        scenario_id="sioux_falls_heldout_event_incident",
        topology="sioux_falls",
        seed=seed + 100,
        realized_demand=realized,
        forecast_demand=forecast,
        incidents=(incident,),
        regime="event",
    )
    gnn_evaluation = evaluate_controller(
        "heterogeneous_gnn",
        problem.simulator,
        scenario,
        IncidentAwareGNNPolicy(model, problem, incident),
    )
    baseline_evaluation = evaluate_controller(
        "dynamic_shortest_path",
        problem.simulator,
        scenario,
        lambda state, visible_forecast: dynamic_shortest_path(
            problem.cell_network, state, problem.destinations
        ),
    )
    gnn = _summary(gnn_evaluation)
    baseline = _summary(baseline_evaluation)
    result = SiouxFallsExampleResult(
        seed=seed,
        benchmark="Sioux Falls transportation network (stylized real-city benchmark)",
        intersections=len(problem.static_network.nodes),
        directed_road_links=problem.static_network.n_edges,
        cell_movements=problem.cell_network.n_movements,
        od_zone_pairs=OD_ZONE_PAIRS,
        attached_od_cells=tuple(
            (
                problem.cell_network.cell_ids[origin],
                problem.cell_network.cell_ids[destination],
            )
            for origin, destination in zip(problem.origins, problem.destinations)
        ),
        incident_link=INCIDENT_LINK,
        train_demonstrations=demonstrations,
        oracle_statuses=tuple(sorted({item.diagnostics.status for item in training_data})),
        oracle_certified=all(item.diagnostics.certified_gap == 0.0 for item in training_data),
        model_parameters=sum(parameter.numel() for parameter in model.parameters()),
        training_seconds=training_seconds,
        imitation_loss_initial=history.losses[0],
        imitation_loss_final=history.losses[-1],
        gnn=gnn,
        dynamic_shortest_path=baseline,
        gnn_minus_baseline_tstt=gnn.tstt - baseline.tstt,
    )
    return result, model


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the slower research-sized settings")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/runs/sioux_falls_example.json")
    )
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("artifacts/runs/sioux_falls_gnn.pt")
    )
    args = parser.parse_args()
    result, model = run_example(quick=not args.full, seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(asdict(result), indent=2, sort_keys=True) + "\n")
    torch.save(
        {
            "state_dict": model.state_dict(),
            "model_config": {
                "hidden_dim": model.hidden_dim,
                "layers": len(model.relations),
            },
            "example": "sioux_falls",
            "seed": args.seed,
            "result": asdict(result),
        },
        args.checkpoint,
    )
    print(json.dumps(asdict(result), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
