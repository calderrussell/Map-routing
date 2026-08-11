import numpy as np
import torch

from data_processed.networks import diamond_cell_network
from oracle.network import CTMState
from projection.feasible import DifferentiableFeasibilityLayer, FeasibilityProjector
from simulators.ctm import CTMSimulator


def test_qp_projection_eliminates_hard_violations_and_reports_correction() -> None:
    network = diamond_cell_network()
    simulator = CTMSimulator(network, np.array([0]), np.array([3]))
    state = CTMState(np.array([[12.0, 8.0, 10.0, 0.0]]), np.zeros(1))
    proposed = np.full((1, network.n_movements), 100.0)
    result = FeasibilityProjector(simulator).project(state, proposed)
    assert result.diagnostics.raw_violation.total > 0
    assert result.diagnostics.projected_violation.total < 1e-5
    assert result.diagnostics.normalized_correction > 0
    assert result.diagnostics.status in {"optimal", "optimal_inaccurate"}
    assert np.all(result.flow >= -1e-7)


def test_differentiable_projection_has_gradients_and_respects_limits() -> None:
    network = diamond_cell_network()
    simulator = CTMSimulator(network, np.array([0]), np.array([3]))
    occupancy = torch.tensor([[12.0, 8.0, 10.0, 0.0]])
    proposed = torch.full((1, network.n_movements), 100.0, requires_grad=True)
    flow = DifferentiableFeasibilityLayer(simulator)(occupancy, proposed)
    flow.sum().backward()
    assert proposed.grad is not None
    assert torch.all(flow.sum(dim=0) <= torch.tensor(network.movement_capacity) + 1e-5)

