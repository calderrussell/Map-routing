from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from oracle.network import CellNetwork, CTMState


Array = np.ndarray


@dataclass(frozen=True)
class CTMStepResult:
    state: CTMState
    movement_flow: Array
    admission: Array
    exit_flow: Array
    sending: Array
    receiving: Array
    conservation_residual: float
    capacity_violation: float
    invalid_turn_flow: float


def normalize_splits(raw: Array, network: CellNetwork, legal_mask: Array) -> Array:
    raw = np.maximum(np.asarray(raw, dtype=float), 0.0) * legal_mask
    result = np.zeros_like(raw)
    sources = network.movement_sources
    for commodity in range(raw.shape[0]):
        for cell in range(network.n_cells):
            indices = np.flatnonzero((sources == cell) & legal_mask[commodity])
            if not len(indices):
                continue
            total = float(np.sum(raw[commodity, indices]))
            result[commodity, indices] = (
                raw[commodity, indices] / total if total > 0 else 1.0 / len(indices)
            )
    return result


class CTMSimulator:
    """Finite-storage multicommodity CTM with explicit source queues."""

    def __init__(
        self,
        network: CellNetwork,
        origins: Array,
        destinations: Array,
        delta_t: float = 1.0,
    ) -> None:
        self.network = network
        self.origins = np.asarray(origins, dtype=np.int64)
        self.destinations = np.asarray(destinations, dtype=np.int64)
        if self.origins.shape != self.destinations.shape:
            raise ValueError("origins and destinations must have the same shape")
        self.k = len(self.origins)
        self.delta_t = float(delta_t)
        self.legal_mask = network.reachability_mask(self.destinations)

    def empty_state(self) -> CTMState:
        return CTMState(
            occupancy=np.zeros((self.k, self.network.n_cells), dtype=float),
            source_queue=np.zeros(self.k, dtype=float),
        )

    def step(
        self,
        state: CTMState,
        demand: Array,
        splits: Array,
        *,
        capacity_multiplier: Array | None = None,
        speed_multiplier: Array | None = None,
        disabled_movements: Array | None = None,
    ) -> CTMStepResult:
        network = self.network
        occupancy = np.asarray(state.occupancy, dtype=float)
        queue = np.asarray(state.source_queue, dtype=float) + np.asarray(demand, dtype=float)
        if occupancy.shape != (self.k, network.n_cells):
            raise ValueError("occupancy has incompatible shape")
        cap_mult = np.ones(network.n_cells) if capacity_multiplier is None else np.asarray(capacity_multiplier, dtype=float)
        speed_mult = np.ones(network.n_cells) if speed_multiplier is None else np.asarray(speed_multiplier, dtype=float)
        disabled = np.zeros(network.n_movements, dtype=bool) if disabled_movements is None else np.asarray(disabled_movements, dtype=bool)
        legal = self.legal_mask & ~disabled[None, :]
        normalized = normalize_splits(splits, network, legal)
        source = network.movement_sources
        target = network.movement_targets
        total_occupancy = np.sum(occupancy, axis=0)
        sending = np.minimum(
            network.free_speed * speed_mult * total_occupancy,
            network.capacity * cap_mult,
        )
        receiving = np.minimum(
            network.wave_speed * np.maximum(network.storage - total_occupancy, 0.0),
            network.capacity * cap_mult,
        )
        commodity_share = np.divide(
            occupancy,
            total_occupancy[None, :],
            out=np.zeros_like(occupancy),
            where=total_occupancy[None, :] > 1e-12,
        )
        commodity_sending = commodity_share * sending[None, :]
        exits = np.zeros(self.k, dtype=float)
        for commodity, destination in enumerate(self.destinations):
            exits[commodity] = min(
                occupancy[commodity, destination], commodity_sending[commodity, destination]
            )
        movement_flow = commodity_sending[:, source] * normalized
        # Movement capacity, then receiving capacity.  Scaling preserves commodity mix.
        aggregate = np.sum(movement_flow, axis=0)
        movement_limits = network.movement_capacity * np.minimum(cap_mult[source], cap_mult[target])
        movement_limits = np.where(disabled, 0.0, movement_limits)
        movement_scale = np.minimum(1.0, movement_limits / np.maximum(aggregate, 1e-12))
        movement_flow *= movement_scale[None, :]
        incoming = np.zeros(network.n_cells, dtype=float)
        np.add.at(incoming, target, np.sum(movement_flow, axis=0))
        receiving_scale = np.minimum(1.0, receiving / np.maximum(incoming, 1e-12))
        movement_flow *= receiving_scale[target][None, :]
        incoming[:] = 0.0
        np.add.at(incoming, target, np.sum(movement_flow, axis=0))
        # Source admission uses receiving space left after endogenous movement flow.
        admission = np.zeros(self.k, dtype=float)
        remaining_receiving = np.maximum(receiving - incoming, 0.0)
        for cell in np.unique(self.origins):
            commodities = np.flatnonzero(self.origins == cell)
            requests = queue[commodities]
            total_request = float(np.sum(requests))
            scale = min(1.0, remaining_receiving[cell] / max(total_request, 1e-12))
            admission[commodities] = requests * scale
        queue -= admission
        inflow_by_commodity = np.zeros_like(occupancy)
        outflow_by_commodity = np.zeros_like(occupancy)
        for movement, (u, v) in enumerate(network.movements):
            outflow_by_commodity[:, u] += movement_flow[:, movement]
            inflow_by_commodity[:, v] += movement_flow[:, movement]
        for commodity, origin in enumerate(self.origins):
            inflow_by_commodity[commodity, origin] += admission[commodity]
        for commodity, destination in enumerate(self.destinations):
            outflow_by_commodity[commodity, destination] += exits[commodity]
        next_occupancy = occupancy + inflow_by_commodity - outflow_by_commodity
        next_occupancy[np.abs(next_occupancy) < 1e-12] = 0.0
        capacity_violation = float(
            np.sum(np.maximum(np.sum(next_occupancy, axis=0) - network.storage, 0.0))
            + np.sum(np.maximum(-next_occupancy, 0.0))
        )
        invalid_turn_flow = float(np.sum(movement_flow * ~legal))
        next_state = CTMState(
            occupancy=np.maximum(next_occupancy, 0.0),
            source_queue=np.maximum(queue, 0.0),
            time=state.time + 1,
            previous_action=normalized,
        )
        before_plus_demand = state.vehicles + float(np.sum(demand))
        after_plus_exits = next_state.vehicles + float(np.sum(exits))
        residual = before_plus_demand - after_plus_exits
        return CTMStepResult(
            state=next_state,
            movement_flow=movement_flow,
            admission=admission,
            exit_flow=exits,
            sending=sending,
            receiving=receiving,
            conservation_residual=float(residual),
            capacity_violation=capacity_violation,
            invalid_turn_flow=invalid_turn_flow,
        )

    def rollout(
        self,
        state: CTMState,
        demand: Array,
        actions: Array,
        *,
        terminal_weight: float = 1.0,
    ) -> tuple[float, list[CTMStepResult]]:
        results: list[CTMStepResult] = []
        current = state.copy()
        objective = 0.0
        for time in range(len(demand)):
            objective += self.delta_t * current.vehicles
            result = self.step(current, demand[time], actions[time])
            results.append(result)
            current = result.state
        objective += terminal_weight * current.vehicles
        return float(objective), results


class DifferentiableCTM:
    """PyTorch CTM used by the DSO oracle and decision-focused training."""

    def __init__(self, simulator: CTMSimulator, device: torch.device | str = "cpu") -> None:
        self.simulator = simulator
        self.device = torch.device(device)
        network = simulator.network
        self.source = torch.as_tensor(network.movement_sources, dtype=torch.long, device=self.device)
        self.target = torch.as_tensor(network.movement_targets, dtype=torch.long, device=self.device)
        self.storage = torch.as_tensor(network.storage, dtype=torch.float32, device=self.device)
        self.capacity = torch.as_tensor(network.capacity, dtype=torch.float32, device=self.device)
        self.free_speed = torch.as_tensor(network.free_speed, dtype=torch.float32, device=self.device)
        self.wave_speed = torch.as_tensor(network.wave_speed, dtype=torch.float32, device=self.device)
        self.movement_capacity = torch.as_tensor(network.movement_capacity, dtype=torch.float32, device=self.device)
        self.origins = torch.as_tensor(simulator.origins, dtype=torch.long, device=self.device)
        self.destinations = torch.as_tensor(simulator.destinations, dtype=torch.long, device=self.device)
        self.legal = torch.as_tensor(simulator.legal_mask, dtype=torch.bool, device=self.device)

    def splits_from_logits(self, logits: torch.Tensor) -> torch.Tensor:
        masked = logits.masked_fill(~self.legal, -1e9)
        result = torch.zeros_like(masked)
        for cell in range(self.simulator.network.n_cells):
            indices = torch.nonzero(self.source == cell, as_tuple=False).flatten()
            if indices.numel() == 0:
                continue
            local_legal = self.legal[:, indices]
            local = torch.softmax(masked[:, indices], dim=-1) * local_legal
            local = local / local.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            result[:, indices] = local
        return result

    def step(
        self,
        occupancy: torch.Tensor,
        source_queue: torch.Tensor,
        demand: torch.Tensor,
        logits: torch.Tensor,
        capacity_multiplier: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        cap_mult = torch.ones_like(self.capacity) if capacity_multiplier is None else capacity_multiplier
        queue = source_queue + demand
        total = occupancy.sum(dim=0)
        sending = torch.minimum(self.free_speed * total, self.capacity * cap_mult)
        receiving = torch.minimum(
            self.wave_speed * torch.relu(self.storage - total), self.capacity * cap_mult
        )
        share = occupancy / total.unsqueeze(0).clamp_min(1e-12)
        commodity_sending = share * sending.unsqueeze(0)
        splits = self.splits_from_logits(logits)
        flow = commodity_sending[:, self.source] * splits
        aggregate = flow.sum(dim=0)
        limits = self.movement_capacity * torch.minimum(cap_mult[self.source], cap_mult[self.target])
        flow = flow * torch.minimum(torch.ones_like(aggregate), limits / aggregate.clamp_min(1e-12)).unsqueeze(0)
        incoming = torch.zeros_like(total).index_add(0, self.target, flow.sum(dim=0))
        receive_scale = torch.minimum(torch.ones_like(receiving), receiving / incoming.clamp_min(1e-12))
        flow = flow * receive_scale[self.target].unsqueeze(0)
        incoming = torch.zeros_like(total).index_add(0, self.target, flow.sum(dim=0))
        admission = torch.zeros_like(queue)
        remaining = torch.relu(receiving - incoming)
        for cell in torch.unique(self.origins).tolist():
            indices = torch.nonzero(self.origins == cell, as_tuple=False).flatten()
            requests = queue[indices]
            scale = torch.minimum(
                torch.ones((), device=self.device), remaining[cell] / requests.sum().clamp_min(1e-12)
            )
            admission = admission.index_copy(0, indices, requests * scale)
        queue = torch.relu(queue - admission)
        inflow = torch.zeros_like(occupancy).index_add(1, self.target, flow)
        outflow = torch.zeros_like(occupancy).index_add(1, self.source, flow)
        exit_flow = torch.zeros_like(queue)
        for commodity, destination in enumerate(self.destinations.tolist()):
            exit_flow[commodity] = torch.minimum(
                occupancy[commodity, destination], commodity_sending[commodity, destination]
            )
        origin_injection = torch.zeros_like(occupancy)
        origin_injection[torch.arange(len(self.origins)), self.origins] = admission
        destination_exit = torch.zeros_like(occupancy)
        destination_exit[torch.arange(len(self.destinations)), self.destinations] = exit_flow
        next_occupancy = torch.relu(occupancy + inflow + origin_injection - outflow - destination_exit)
        return next_occupancy, queue, flow, exit_flow

    def rollout(
        self,
        occupancy: torch.Tensor,
        source_queue: torch.Tensor,
        demand: torch.Tensor,
        logits: torch.Tensor,
        *,
        terminal_weight: float = 1.0,
        switch_weight: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        objective = torch.zeros((), device=self.device)
        previous_splits: torch.Tensor | None = None
        for time in range(demand.shape[0]):
            objective = objective + occupancy.sum() + source_queue.sum()
            if switch_weight and previous_splits is not None:
                objective = objective + switch_weight * torch.mean(
                    torch.abs(self.splits_from_logits(logits[time]) - previous_splits)
                )
            occupancy, source_queue, _, _ = self.step(
                occupancy, source_queue, demand[time], logits[time]
            )
            previous_splits = self.splits_from_logits(logits[time])
        objective = objective + terminal_weight * (occupancy.sum() + source_queue.sum())
        return objective, occupancy, source_queue

