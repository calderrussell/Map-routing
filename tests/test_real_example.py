import numpy as np
import torch

from examples.sioux_falls_gnn import OD_ZONE_PAIRS, build_problem
from models.features import build_features
from models.heterogeneous import HeteroSpatioTemporalGNN


def test_real_sioux_falls_problem_and_gnn_forward() -> None:
    problem = build_problem()
    assert len(problem.static_network.nodes) == 24
    assert problem.static_network.n_edges == 76
    assert problem.cell_network.n_cells == 76
    assert len(OD_ZONE_PAIRS) == 3
    forecast = np.ones((4, 3))
    features = build_features(
        problem.cell_network,
        problem.simulator.empty_state(),
        problem.origins,
        problem.destinations,
        forecast,
    )
    with torch.no_grad():
        output = HeteroSpatioTemporalGNN(hidden_dim=24, layers=1)(features)
    assert output.splits.shape == (3, problem.cell_network.n_movements)
    assert torch.all(output.splits[~features.legal_mask] == 0)

