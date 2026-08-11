import numpy as np

from data_processed.networks import diamond_cell_network
from oracle.dso import ExhaustiveTinyOracle, RecedingHorizonOracle
from simulators.ctm import CTMSimulator


def test_exhaustive_oracle_is_certified_and_prefers_capacity() -> None:
    network = diamond_cell_network()
    simulator = CTMSimulator(network, np.array([0]), np.array([3]))
    demand = np.array([[12.0], [12.0], [0.0], [0.0]])
    result = ExhaustiveTinyOracle(simulator).solve(simulator.empty_state(), demand)
    assert result.diagnostics.certified_gap == 0.0
    assert result.first_action.shape == (1, network.n_movements)
    # Movement 1 enters the higher-capacity lower branch.
    assert result.first_action[0, 1] >= result.first_action[0, 0]


def test_gradient_oracle_replays_conservatively() -> None:
    network = diamond_cell_network()
    simulator = CTMSimulator(network, np.array([0]), np.array([3]))
    demand = np.array([[8.0], [8.0], [0.0]], dtype=float)
    oracle = RecedingHorizonOracle(
        simulator, horizon=3, iterations=15, restarts=1, learning_rate=0.1
    )
    result = oracle.solve(simulator.empty_state(), demand)
    assert np.isfinite(result.objective)
    assert result.diagnostics.lower_bound is None
    assert result.diagnostics.status == "locally_solved_uncertified"
    assert result.diagnostics.primal_residual < 1e-5

