from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from data_processed.dataset import Demonstration
from experiments.train import imitation_loss
from oracle.dso import RecedingHorizonOracle
from oracle.network import CTMState
from projection.feasible import DifferentiableFeasibilityLayer
from simulators.ctm import CTMSimulator, DifferentiableCTM


@dataclass(frozen=True)
class DecisionWeights:
    imitation: float = 1.0
    decision: float = 0.2
    conservation: float = 0.1
    capacity: float = 0.1
    loop: float = 0.02
    projection: float = 0.05


@dataclass(frozen=True)
class DecisionTrainingHistory:
    losses: tuple[float, ...]
    rollout_horizons: tuple[int, ...]


def physics_residuals(
    simulator: CTMSimulator, state: CTMState, raw_flow: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Raw conservation/capacity/loop residuals from TeX equation (21)."""

    import networkx as nx

    network = simulator.network
    device = raw_flow.device
    source = torch.as_tensor(network.movement_sources, dtype=torch.long, device=device)
    target = torch.as_tensor(network.movement_targets, dtype=torch.long, device=device)
    occupancy = torch.as_tensor(state.occupancy, dtype=torch.float32, device=device)
    total = occupancy.sum(dim=0)
    capacity = torch.as_tensor(network.capacity, dtype=torch.float32, device=device)
    storage = torch.as_tensor(network.storage, dtype=torch.float32, device=device)
    free_speed = torch.as_tensor(network.free_speed, dtype=torch.float32, device=device)
    wave_speed = torch.as_tensor(network.wave_speed, dtype=torch.float32, device=device)
    sending = torch.minimum(free_speed * total, capacity)
    receiving = torch.minimum(wave_speed * torch.relu(storage - total), capacity)
    commodity_sending = occupancy / total.unsqueeze(0).clamp_min(1e-12) * sending.unsqueeze(0)
    outgoing = torch.zeros_like(occupancy).index_add(1, source, raw_flow)
    incoming = torch.zeros_like(total).index_add(0, target, raw_flow.sum(dim=0))
    conservation = torch.relu(outgoing - commodity_sending).sum()
    movement_capacity = torch.as_tensor(
        network.movement_capacity, dtype=torch.float32, device=device
    )
    capacity_residual = torch.relu(raw_flow.sum(dim=0) - movement_capacity).sum()
    capacity_residual = capacity_residual + torch.relu(incoming - receiving).sum()
    graph = network.graph()
    loop_mask = torch.zeros_like(raw_flow, dtype=torch.bool)
    for commodity, destination in enumerate(simulator.destinations):
        reverse = graph.reverse(copy=False)
        distances = nx.single_source_shortest_path_length(reverse, int(destination))
        for movement, (u, v) in enumerate(network.movements):
            if distances.get(v, 10**9) >= distances.get(u, 10**9):
                loop_mask[commodity, movement] = True
    loop = (raw_flow * loop_mask).sum()
    return conservation, capacity_residual, loop


def decision_focused_loss(
    model,
    demonstration: Demonstration,
    simulator: CTMSimulator,
    horizon: int,
    weights: DecisionWeights,
) -> tuple[torch.Tensor, dict[str, float]]:
    output = model(demonstration.features)
    target = torch.as_tensor(demonstration.target_action, dtype=output.splits.dtype)
    imitation = imitation_loss(output.splits, target, demonstration.features.legal_mask)
    demand = torch.as_tensor(demonstration.demand_forecast[:horizon], dtype=torch.float32)
    occupancy = torch.as_tensor(demonstration.state.occupancy, dtype=torch.float32)
    source_queue = torch.as_tensor(demonstration.state.source_queue, dtype=torch.float32)
    ctm = DifferentiableCTM(simulator)
    repeated_logits = output.logits.unsqueeze(0).expand(len(demand), -1, -1)
    rollout_objective, _, _ = ctm.rollout(
        occupancy, source_queue, demand, repeated_logits, terminal_weight=2.0
    )
    oracle_scale = max(abs(demonstration.oracle_objective), 1e-6)
    decision = (rollout_objective - demonstration.oracle_objective) / oracle_scale
    # Penalize reliance on projection using deliberately unnormalized positive proposals.
    raw = torch.nn.functional.softplus(output.logits)
    projected = DifferentiableFeasibilityLayer(simulator)(occupancy, raw)
    projection = torch.linalg.vector_norm(projected - raw) / torch.linalg.vector_norm(
        projected
    ).clamp_min(1e-6)
    invalid = (raw * ~demonstration.features.legal_mask).sum()
    conservation, capacity, loop = physics_residuals(simulator, demonstration.state, raw)
    total = (
        weights.imitation * imitation
        + weights.decision * decision
        + weights.projection * projection
        + weights.conservation * conservation
        + weights.capacity * (capacity + invalid)
        + weights.loop * loop
    )
    return total, {
        "imitation": float(imitation.detach()),
        "decision": float(decision.detach()),
        "projection": float(projection.detach()),
        "invalid": float(invalid.detach()),
        "conservation": float(conservation.detach()),
        "capacity": float(capacity.detach()),
        "loop": float(loop.detach()),
    }


def fine_tune_decision_focused(
    model,
    demonstrations: list[Demonstration],
    simulator: CTMSimulator,
    *,
    epochs: int = 30,
    maximum_horizon: int = 4,
    learning_rate: float = 5e-4,
    weights: DecisionWeights = DecisionWeights(),
) -> DecisionTrainingHistory:
    """Short-to-long rollout curriculum from TeX Stage 2."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    losses, horizons = [], []
    for epoch in range(epochs):
        horizon = min(maximum_horizon, 1 + epoch * maximum_horizon // max(epochs, 1))
        total_value = 0.0
        for demonstration in demonstrations:
            optimizer.zero_grad()
            loss, _ = decision_focused_loss(model, demonstration, simulator, horizon, weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_value += float(loss.detach())
        losses.append(total_value / len(demonstrations))
        horizons.append(horizon)
    return DecisionTrainingHistory(tuple(losses), tuple(horizons))


@dataclass(frozen=True)
class DaggerCandidate:
    state: CTMState
    forecast: np.ndarray
    uncertainty: float
    projection_correction: float
    maximum_occupancy_ratio: float

    @property
    def priority(self) -> float:
        return self.uncertainty + self.projection_correction + self.maximum_occupancy_ratio


def select_dagger_queries(candidates: list[DaggerCandidate], budget: int) -> list[DaggerCandidate]:
    """Prioritize uncertainty, projection corrections, and near-storage states."""

    return sorted(candidates, key=lambda item: item.priority, reverse=True)[:budget]


def query_dagger_oracle(
    oracle: RecedingHorizonOracle, candidates: list[DaggerCandidate], budget: int
):
    return [
        (candidate, oracle.solve(candidate.state, candidate.forecast))
        for candidate in select_dagger_queries(candidates, budget)
    ]


def collect_dagger_demonstrations(
    model,
    simulator: CTMSimulator,
    oracle: RecedingHorizonOracle,
    demand_scenarios: list[np.ndarray],
    *,
    query_budget: int,
) -> list[Demonstration]:
    """Roll out the current model, select shifted states, and query the oracle."""

    from models.features import build_features
    from projection.feasible import FeasibilityProjector

    candidates: list[DaggerCandidate] = []
    for demand in demand_scenarios:
        state = simulator.empty_state()
        hidden = None
        for time_index in range(len(demand)):
            features = build_features(
                simulator.network,
                state,
                simulator.origins,
                simulator.destinations,
                demand[time_index:],
            )
            output = model(features, hidden)
            hidden = getattr(output, "hetero_hidden", output.hidden)
            probability = output.splits.detach().cpu().numpy()
            entropy = -float(np.sum(probability * np.log(np.maximum(probability, 1e-12))))
            raw = torch.nn.functional.softplus(output.logits).detach().cpu().numpy()
            correction = FeasibilityProjector(simulator).project(
                state, raw
            ).diagnostics.normalized_correction
            occupancy_ratio = float(
                np.max(np.sum(state.occupancy, axis=0) / simulator.network.storage)
            )
            candidates.append(
                DaggerCandidate(
                    state.copy(), demand[time_index:].copy(), entropy, correction, occupancy_ratio
                )
            )
            state = simulator.step(state, demand[time_index], probability).state
    demonstrations = []
    for candidate, result in query_dagger_oracle(oracle, candidates, query_budget):
        demonstrations.append(
            Demonstration(
                features=build_features(
                    simulator.network,
                    candidate.state,
                    simulator.origins,
                    simulator.destinations,
                    candidate.forecast,
                ),
                target_action=result.first_action,
                oracle_objective=result.objective,
                diagnostics=result.diagnostics,
                state=candidate.state,
                demand_forecast=candidate.forecast,
                topology_hash=simulator.network.topology_hash(),
                accepted=result.diagnostics.primal_residual <= 1e-6,
            )
        )
    return demonstrations


def run_dagger(
    model,
    initial_demonstrations: list[Demonstration],
    simulator: CTMSimulator,
    oracle: RecedingHorizonOracle,
    demand_scenarios: list[np.ndarray],
    *,
    rounds: int,
    query_budget: int,
    training_epochs_per_round: int = 10,
) -> list[Demonstration]:
    """Iterative behavioral cloning/data aggregation until the requested audit rounds."""

    from experiments.train import train_imitation

    dataset = list(initial_demonstrations)
    for _ in range(rounds):
        train_imitation(model, dataset, epochs=training_epochs_per_round, learning_rate=1e-3)
        additions = collect_dagger_demonstrations(
            model, simulator, oracle, demand_scenarios, query_budget=query_budget
        )
        dataset.extend(item for item in additions if item.accepted)
    return dataset
