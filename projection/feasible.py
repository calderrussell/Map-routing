from __future__ import annotations

from dataclasses import dataclass
import time

import cvxpy as cp
import numpy as np
import torch
from torch import nn

from oracle.network import CTMState
from simulators.ctm import CTMSimulator


@dataclass(frozen=True)
class ViolationMetrics:
    negativity: float
    invalid_turn: float
    sending: float
    receiving: float
    movement_capacity: float

    @property
    def total(self) -> float:
        return self.negativity + self.invalid_turn + self.sending + self.receiving + self.movement_capacity


@dataclass(frozen=True)
class ProjectionDiagnostics:
    status: str
    solve_seconds: float
    objective: float
    normalized_correction: float
    raw_violation: ViolationMetrics
    projected_violation: ViolationMetrics
    holding_slack: float


@dataclass(frozen=True)
class ProjectionResult:
    flow: np.ndarray
    holding: np.ndarray
    diagnostics: ProjectionDiagnostics


def _envelopes(simulator: CTMSimulator, state: CTMState):
    network = simulator.network
    total = np.sum(state.occupancy, axis=0)
    sending = np.minimum(network.free_speed * total, network.capacity)
    receiving = np.minimum(network.wave_speed * np.maximum(network.storage - total, 0.0), network.capacity)
    share = np.divide(
        state.occupancy,
        total[None, :],
        out=np.zeros_like(state.occupancy, dtype=float),
        where=total[None, :] > 1e-12,
    )
    return sending, receiving, share * sending[None, :]


def violation_metrics(
    simulator: CTMSimulator,
    state: CTMState,
    flow: np.ndarray,
) -> ViolationMetrics:
    network = simulator.network
    flow = np.asarray(flow, dtype=float)
    source, target = network.movement_sources, network.movement_targets
    sending, receiving, commodity_sending = _envelopes(simulator, state)
    outgoing = np.zeros((simulator.k, network.n_cells))
    incoming = np.zeros(network.n_cells)
    np.add.at(incoming, target, np.sum(flow, axis=0))
    for movement, cell in enumerate(source):
        outgoing[:, cell] += flow[:, movement]
    aggregate_movement = np.sum(flow, axis=0)
    return ViolationMetrics(
        negativity=float(np.sum(np.maximum(-flow, 0.0))),
        invalid_turn=float(np.sum(np.abs(flow[~simulator.legal_mask]))),
        sending=float(np.sum(np.maximum(outgoing - commodity_sending, 0.0))),
        receiving=float(np.sum(np.maximum(incoming - receiving, 0.0))),
        movement_capacity=float(
            np.sum(np.maximum(aggregate_movement - network.movement_capacity, 0.0))
        ),
    )


class FeasibilityProjector:
    """Weighted nearest-flow QP corresponding to TeX equation (18)."""

    def __init__(self, simulator: CTMSimulator, holding_penalty: float = 1e-3) -> None:
        self.simulator = simulator
        self.holding_penalty = float(holding_penalty)

    def project(self, state: CTMState, proposed: np.ndarray) -> ProjectionResult:
        started = time.perf_counter()
        simulator = self.simulator
        network = simulator.network
        proposed = np.asarray(proposed, dtype=float)
        if proposed.shape != (simulator.k, network.n_movements):
            raise ValueError("proposed flow must have K x M shape")
        _, receiving, commodity_sending = _envelopes(simulator, state)
        source, target = network.movement_sources, network.movement_targets
        flow = cp.Variable(proposed.shape)
        holding = cp.Variable((simulator.k, network.n_cells))
        constraints = [flow >= 0, holding >= 0]
        for commodity in range(simulator.k):
            for movement in range(network.n_movements):
                if not simulator.legal_mask[commodity, movement]:
                    constraints.append(flow[commodity, movement] == 0)
            for cell in range(network.n_cells):
                indices = np.flatnonzero(source == cell)
                if len(indices):
                    constraints.append(
                        cp.sum(flow[commodity, indices]) + holding[commodity, cell]
                        == commodity_sending[commodity, cell]
                    )
                else:
                    constraints.append(holding[commodity, cell] == commodity_sending[commodity, cell])
        constraints.extend(
            cp.sum(flow[:, movement]) <= network.movement_capacity[movement]
            for movement in range(network.n_movements)
        )
        constraints.extend(
            cp.sum(flow[:, np.flatnonzero(target == cell)]) <= receiving[cell]
            for cell in range(network.n_cells)
            if np.any(target == cell)
        )
        weights = 1.0 / np.maximum(network.movement_capacity, 1e-6)
        objective = cp.Minimize(
            0.5 * cp.sum_squares(cp.multiply(weights[None, :], flow - proposed))
            + self.holding_penalty * cp.sum(holding)
        )
        problem = cp.Problem(objective, constraints)
        try:
            value = problem.solve(
                solver=cp.OSQP,
                eps_abs=1e-8,
                eps_rel=1e-8,
                max_iter=100_000,
                warm_start=True,
                verbose=False,
            )
        except cp.SolverError:
            value = problem.solve(solver=cp.CLARABEL, verbose=False)
        if flow.value is None or holding.value is None:
            raise RuntimeError(f"projection failed with status {problem.status}")
        projected = np.asarray(flow.value)
        held = np.asarray(holding.value)
        correction = np.linalg.norm(projected - proposed) / max(np.linalg.norm(projected), 1e-9)
        diagnostics = ProjectionDiagnostics(
            status=str(problem.status),
            solve_seconds=time.perf_counter() - started,
            objective=float(value),
            normalized_correction=float(correction),
            raw_violation=violation_metrics(simulator, state, proposed),
            projected_violation=violation_metrics(simulator, state, projected),
            holding_slack=float(np.sum(held)),
        )
        return ProjectionResult(projected, held, diagnostics)


class DifferentiableFeasibilityLayer(nn.Module):
    """Fast masked-and-scaled CTM allocator used inside training rollouts."""

    def __init__(self, simulator: CTMSimulator) -> None:
        super().__init__()
        network = simulator.network
        self.n_cells = network.n_cells
        self.register_buffer("source", torch.as_tensor(network.movement_sources, dtype=torch.long))
        self.register_buffer("target", torch.as_tensor(network.movement_targets, dtype=torch.long))
        self.register_buffer("storage", torch.as_tensor(network.storage, dtype=torch.float32))
        self.register_buffer("capacity", torch.as_tensor(network.capacity, dtype=torch.float32))
        self.register_buffer("free_speed", torch.as_tensor(network.free_speed, dtype=torch.float32))
        self.register_buffer("wave_speed", torch.as_tensor(network.wave_speed, dtype=torch.float32))
        self.register_buffer(
            "movement_capacity", torch.as_tensor(network.movement_capacity, dtype=torch.float32)
        )
        self.register_buffer("legal", torch.as_tensor(simulator.legal_mask, dtype=torch.bool))

    def forward(self, occupancy: torch.Tensor, proposed: torch.Tensor) -> torch.Tensor:
        total = occupancy.sum(dim=0)
        sending = torch.minimum(self.free_speed * total, self.capacity)
        receiving = torch.minimum(self.wave_speed * torch.relu(self.storage - total), self.capacity)
        share = occupancy / total.unsqueeze(0).clamp_min(1e-12)
        commodity_sending = share * sending.unsqueeze(0)
        flow = torch.relu(proposed) * self.legal
        for cell in range(self.n_cells):
            indices = torch.nonzero(self.source == cell, as_tuple=False).flatten()
            if not indices.numel():
                continue
            outgoing = flow[:, indices].sum(dim=-1)
            scale = torch.minimum(
                torch.ones_like(outgoing), commodity_sending[:, cell] / outgoing.clamp_min(1e-12)
            )
            replacement = flow[:, indices] * scale.unsqueeze(-1)
            flow = flow.clone()
            flow[:, indices] = replacement
        aggregate = flow.sum(dim=0)
        flow = flow * torch.minimum(
            torch.ones_like(aggregate), self.movement_capacity / aggregate.clamp_min(1e-12)
        ).unsqueeze(0)
        incoming = torch.zeros_like(total).index_add(0, self.target, flow.sum(dim=0))
        flow = flow * torch.minimum(
            torch.ones_like(receiving), receiving / incoming.clamp_min(1e-12)
        )[self.target].unsqueeze(0)
        return flow

