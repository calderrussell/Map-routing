import numpy as np
import pytest
import torch

from data_processed.networks import diamond_cell_network, grid_cell_network, sioux_falls_static, static_links_to_cells
from data_processed.splits import TopologySplits
from models.features import build_features
from models.heterogeneous import HeteroSpatioTemporalGNN, active_od_indices, destination_batches
from simulators.ctm import CTMSimulator


def _forward(network, origins, destinations):
    simulator = CTMSimulator(network, np.asarray(origins), np.asarray(destinations))
    forecast = np.ones((3, len(origins)))
    features = build_features(
        network, simulator.empty_state(), np.asarray(origins), np.asarray(destinations), forecast
    )
    model = HeteroSpatioTemporalGNN(hidden_dim=24, layers=2)
    output = model(features)
    output.splits.sum().backward()
    return output, features, model


def test_heterograph_runs_on_variable_topologies_and_commodities() -> None:
    diamond = diamond_cell_network()
    grid = grid_cell_network(2, 3, seed=3)
    output_a, features_a, _ = _forward(diamond, [0], [3])
    output_b, features_b, model_b = _forward(grid, [0, 2], [5, 3])
    assert output_a.splits.shape == (1, diamond.n_movements)
    assert output_b.splits.shape == (2, grid.n_movements)
    assert torch.all(output_b.splits[~features_b.legal_mask] == 0)
    assert any(parameter.grad is not None for parameter in model_b.parameters())


def test_sioux_falls_cell_conversion_and_scaling_helpers() -> None:
    static, _ = sioux_falls_static()
    cells = static_links_to_cells(static)
    assert cells.n_cells == 76
    demand = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 1.0]])
    assert active_od_indices(demand).tolist() == [0, 2]
    batches = destination_batches(torch.tensor([4, 4, 5, 6]), maximum_unique=2)
    assert [batch.tolist() for batch in batches] == [[0, 1, 2], [3]]


def test_topology_split_guard_blocks_test_leakage() -> None:
    splits = TopologySplits.load("configs/topology_splits.yaml")
    splits.authorize("sioux_falls", "fit")
    splits.authorize("eastern_massachusetts", "tune")
    splits.authorize("anaheim", "final_test")
    with pytest.raises(PermissionError):
        splits.authorize("anaheim", "fit")
    with pytest.raises(PermissionError):
        splits.authorize("winnipeg", "tune")

