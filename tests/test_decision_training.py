import numpy as np

from data_processed.dataset import tiny_oracle_demonstrations
from data_processed.networks import diamond_cell_network
from experiments.decision_training import (
    DaggerCandidate,
    fine_tune_decision_focused,
    select_dagger_queries,
)
from experiments.train import train_imitation
from models.homogeneous import DestinationGCNGRU
from simulators.ctm import CTMSimulator


def test_decision_focused_curriculum_and_dagger_priority() -> None:
    network = diamond_cell_network()
    origins, destinations = np.array([0]), np.array([3])
    simulator = CTMSimulator(network, origins, destinations)
    demos = tiny_oracle_demonstrations(
        network, origins, destinations, count=6, horizon=2, seed=9
    )
    model = DestinationGCNGRU(hidden_dim=24)
    train_imitation(model, demos, epochs=5, learning_rate=0.01, seed=9)
    history = fine_tune_decision_focused(
        model, demos, simulator, epochs=4, maximum_horizon=2, learning_rate=1e-4
    )
    assert len(history.losses) == 4
    assert history.rollout_horizons[0] == 1
    assert history.rollout_horizons[-1] == 2
    assert np.all(np.isfinite(history.losses))
    state = simulator.empty_state()
    candidates = [
        DaggerCandidate(state, np.ones((2, 1)), 0.1, 0.1, 0.1),
        DaggerCandidate(state, np.ones((2, 1)), 0.5, 0.4, 0.9),
    ]
    assert select_dagger_queries(candidates, 1)[0].priority > 1.0

