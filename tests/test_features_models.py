import numpy as np
import torch

from data_processed.dataset import tiny_oracle_demonstrations
from data_processed.networks import diamond_cell_network
from experiments.train import train_imitation
from models.features import CELL_FEATURES, OD_FEATURES, build_features
from models.alternatives import FullMulticommodityFlowDecoder, LearnedMarginalCostDecoder
from models.homogeneous import DestinationGCNGRU
from simulators.ctm import CTMSimulator


def test_physical_features_and_masked_model_output() -> None:
    network = diamond_cell_network()
    origins, destinations = np.array([0]), np.array([3])
    simulator = CTMSimulator(network, origins, destinations)
    features = build_features(
        network, simulator.empty_state(), origins, destinations, np.ones((3, 1)) * 8.0
    )
    assert features.cell.shape == (network.n_cells, len(CELL_FEATURES))
    assert features.intersection.shape[0] == network.n_cells
    assert features.od.shape == (1, len(OD_FEATURES))
    model = DestinationGCNGRU(hidden_dim=24)
    output = model(features)
    assert output.splits.shape == (1, network.n_movements)
    for cell in range(network.n_cells):
        indices = np.flatnonzero(network.movement_sources == cell)
        legal = indices[features.legal_mask[0, indices].numpy()]
        if len(legal):
            assert torch.isclose(output.splits[0, legal].sum(), torch.tensor(1.0))
    assert torch.all(output.splits[~features.legal_mask] == 0)
    externality, cost = LearnedMarginalCostDecoder(hidden_dim=16)(features)
    full_flow = FullMulticommodityFlowDecoder(hidden_dim=16, horizon=3)(features)
    assert torch.all(externality >= 0) and torch.all(cost >= externality)
    assert full_flow.shape == (3, 1, network.n_movements)
    assert torch.all(full_flow[:, ~features.legal_mask] == 0)


def test_phase1_imitation_loss_decreases() -> None:
    network = diamond_cell_network()
    demonstrations = tiny_oracle_demonstrations(
        network, np.array([0]), np.array([3]), count=8, horizon=2, seed=4
    )
    model = DestinationGCNGRU(hidden_dim=24)
    history = train_imitation(model, demonstrations, epochs=20, learning_rate=0.01, seed=3)
    assert np.isfinite(history.best_loss)
    assert min(history.losses[-5:]) < history.losses[0]
