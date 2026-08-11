import numpy as np

from data_processed.networks import chain_cell_network, diamond_cell_network
from oracle.network import CTMState
from simulators.ctm import CTMSimulator


def test_vehicle_conservation_without_exit() -> None:
    network = chain_cell_network(3)
    simulator = CTMSimulator(network, origins=np.array([0]), destinations=np.array([2]))
    state = CTMState(np.array([[3.0, 0.0, 0.0]]), np.zeros(1))
    splits = np.ones((1, network.n_movements))
    result = simulator.step(state, np.zeros(1), splits)
    assert abs(result.conservation_residual) < 1e-10
    assert np.isclose(result.state.vehicles + result.exit_flow.sum(), state.vehicles)


def test_known_exit_and_oversaturated_source_queue() -> None:
    network = chain_cell_network(2)
    simulator = CTMSimulator(network, origins=np.array([0]), destinations=np.array([1]))
    state = CTMState(np.array([[0.0, 4.0]]), np.zeros(1))
    result = simulator.step(state, np.array([50.0]), np.ones((1, 1)))
    assert np.isclose(result.exit_flow[0], 4.0)
    assert result.state.source_queue[0] > 0.0
    assert abs(result.conservation_residual) < 1e-10
    assert result.capacity_violation == 0.0


def test_receiving_capacity_and_invalid_turn_mask() -> None:
    network = diamond_cell_network()
    simulator = CTMSimulator(network, origins=np.array([0]), destinations=np.array([3]))
    state = CTMState(np.array([[20.0, 19.0, 0.0, 0.0]]), np.zeros(1))
    splits = np.array([[1.0, 0.0, 1.0, 1.0]])
    result = simulator.step(state, np.zeros(1), splits)
    assert np.sum(result.movement_flow[:, 0]) <= result.receiving[1] + 1e-10
    assert result.invalid_turn_flow == 0.0
    assert abs(result.conservation_residual) < 1e-10

