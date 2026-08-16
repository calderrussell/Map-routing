from __future__ import annotations

from dataclasses import dataclass
import itertools
import time

import numpy as np
import torch

from oracle.network import CTMState
from simulators.ctm import CTMSimulator, DifferentiableCTM


@dataclass(frozen=True)
class OracleDiagnostics:
    status: str
    iterations: int
    solve_seconds: float
    objective: float
    lower_bound: float | None
    certified_gap: float | None
    primal_residual: float


@dataclass(frozen=True)
class OracleResult:
    first_action: np.ndarray
    action_sequence: np.ndarray
    objective: float
    diagnostics: OracleDiagnostics


class RecedingHorizonOracle:
    """Gradient-based CTM model-predictive DSO oracle.

    This solver reports no certified lower bound.  Callers must preserve that fact in
    demonstration metadata instead of describing a timed solution as exactly optimal.
    Tiny instances can be checked with :class:`ExhaustiveTinyOracle` below.
    """

    def __init__(
        self,
        simulator: CTMSimulator,
        horizon: int = 8,
        iterations: int = 120,
        learning_rate: float = 0.15,
        restarts: int = 3,
        terminal_weight: float = 2.0,
        switch_weight: float = 0.01,
        seed: int = 0,
    ) -> None:
        self.simulator = simulator
        self.horizon = int(horizon)
        self.iterations = int(iterations)
        self.learning_rate = float(learning_rate)
        self.restarts = int(restarts)
        self.terminal_weight = float(terminal_weight)
        self.switch_weight = float(switch_weight)
        self.seed = int(seed)
        self.ctm = DifferentiableCTM(simulator)

    def solve(self, state: CTMState, demand_forecast: np.ndarray) -> OracleResult:
        started = time.perf_counter()
        demand_forecast = np.asarray(demand_forecast, dtype=np.float32)
        horizon = min(self.horizon, len(demand_forecast))
        demand = torch.as_tensor(demand_forecast[:horizon], dtype=torch.float32)
        occupancy = torch.as_tensor(state.occupancy, dtype=torch.float32)
        queue = torch.as_tensor(state.source_queue, dtype=torch.float32)
        best_objective = float("inf")
        best_logits: torch.Tensor | None = None
        generator = torch.Generator().manual_seed(self.seed + state.time)
        total_iterations = 0
        for restart in range(self.restarts):
            initialization = torch.zeros(
                (horizon, self.simulator.k, self.simulator.network.n_movements),
                dtype=torch.float32,
            )
            if restart:
                initialization += 0.1 * torch.randn(initialization.shape, generator=generator)
            logits = torch.nn.Parameter(initialization)
            optimizer = torch.optim.Adam([logits], lr=self.learning_rate)
            last = float("inf")
            for _ in range(self.iterations):
                total_iterations += 1
                optimizer.zero_grad()
                objective, _, _ = self.ctm.rollout(
                    occupancy,
                    queue,
                    demand,
                    logits,
                    terminal_weight=self.terminal_weight,
                    switch_weight=self.switch_weight,
                )
                objective.backward()
                torch.nn.utils.clip_grad_norm_([logits], 10.0)
                optimizer.step()
                value = float(objective.detach())
                if np.isfinite(last) and abs(last - value) <= 1e-7 * max(abs(last), 1.0):
                    break
                last = value
            with torch.no_grad():
                objective, _, _ = self.ctm.rollout(
                    occupancy, queue, demand, logits, terminal_weight=self.terminal_weight
                )
                value = float(objective)
                if value < best_objective:
                    best_objective = value
                    best_logits = logits.detach().clone()
        if best_logits is None:
            raise RuntimeError("DSO oracle failed to produce an action")
        actions = np.stack(
            [self.ctm.splits_from_logits(best_logits[t]).cpu().numpy() for t in range(horizon)]
        )
        # Independently replay in NumPy and expose the maximum conservation residual.
        _, replay = self.simulator.rollout(state, demand_forecast[:horizon], actions, terminal_weight=self.terminal_weight)
        primal = max((abs(step.conservation_residual) for step in replay), default=0.0)
        diagnostics = OracleDiagnostics(
            status="locally_solved_uncertified",
            iterations=total_iterations,
            solve_seconds=time.perf_counter() - started,
            objective=best_objective,
            lower_bound=None,
            certified_gap=None,
            primal_residual=primal,
        )
        return OracleResult(actions[0], actions, best_objective, diagnostics)


class ExhaustiveTinyOracle:
    """Certified discrete action enumeration for small correctness fixtures."""

    def __init__(self, simulator: CTMSimulator, split_grid: tuple[float, ...] = (0.0, 0.5, 1.0)) -> None:
        if simulator.k != 1:
            raise ValueError("exhaustive fixture currently supports one commodity")
        self.simulator = simulator
        self.split_grid = split_grid

    def solve(self, state: CTMState, demand_forecast: np.ndarray) -> OracleResult:
        started = time.perf_counter()
        network = self.simulator.network
        divergent_cells = [
            cell
            for cell in range(network.n_cells)
            if np.sum(network.movement_sources == cell) == 2
        ]
        horizon = len(demand_forecast)
        choices = list(itertools.product(self.split_grid, repeat=horizon * len(divergent_cells)))
        best_cost = float("inf")
        best_actions: np.ndarray | None = None
        for choice in choices:
            actions = np.zeros((horizon, 1, network.n_movements), dtype=float)
            cursor = 0
            for t in range(horizon):
                for cell in range(network.n_cells):
                    indices = np.flatnonzero(network.movement_sources == cell)
                    if len(indices) == 1:
                        actions[t, 0, indices[0]] = 1.0
                    elif len(indices) == 2:
                        share = choice[cursor]
                        cursor += 1
                        actions[t, 0, indices] = (share, 1.0 - share)
            cost, _ = self.simulator.rollout(state, demand_forecast, actions, terminal_weight=2.0)
            if cost < best_cost:
                best_cost, best_actions = cost, actions
        if best_actions is None:
            raise RuntimeError("empty exhaustive action space")
        diagnostics = OracleDiagnostics(
            status="certified_discrete_optimum",
            iterations=len(choices),
            solve_seconds=time.perf_counter() - started,
            objective=best_cost,
            lower_bound=best_cost,
            certified_gap=0.0,
            primal_residual=0.0,
        )
        return OracleResult(best_actions[0], best_actions, best_cost, diagnostics)
