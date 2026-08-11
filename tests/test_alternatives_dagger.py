import numpy as np

from data_processed.dataset import receding_horizon_demonstration_trajectory
from data_processed.networks import diamond_cell_network
from experiments.decision_training import collect_dagger_demonstrations
from experiments.train import train_imitation
from models.homogeneous import DestinationGCNGRU
from oracle.dso import RecedingHorizonOracle
from simulators.ctm import CTMSimulator


def test_full_trajectory_generation_and_dagger_collection() -> None:
    network = diamond_cell_network()
    origins, destinations = np.array([0]), np.array([3])
    simulator = CTMSimulator(network, origins, destinations)
    oracle = RecedingHorizonOracle(simulator, horizon=2, iterations=4, restarts=1, seed=2)
    demand = np.array([[5.0], [8.0], [0.0]])
    generated = receding_horizon_demonstration_trajectory(
        network, origins, destinations, demand, demand, oracle=oracle, warmup_intervals=1
    )
    assert generated.warmup_policy == "free_flow_shortest_path"
    assert len(generated.accepted) == 2
    model = DestinationGCNGRU(hidden_dim=16)
    train_imitation(model, list(generated.accepted), epochs=2, learning_rate=0.005, seed=2)
    additions = collect_dagger_demonstrations(
        model, simulator, oracle, [demand[:2]], query_budget=1
    )
    assert len(additions) == 1
    assert additions[0].topology_hash == network.topology_hash()

