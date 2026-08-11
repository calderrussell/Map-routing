import numpy as np
import pytest

from data_processed.networks import diamond_cell_network
from routing.compliance import combine_controlled_actions, compliance_probability
from routing.paths import balanced_rounding, candidate_paths, decompose_route_shares, sample_next_hop


def test_route_decomposition_rounding_and_loop_prevention() -> None:
    network = diamond_cell_network()
    paths = candidate_paths(network, 0, 3, k_paths=4)
    assert len(paths) == 2
    target = np.array([0.25, 0.75, 0.25, 0.75])
    result = decompose_route_shares(network, paths, target, switch_penalty=0.0)
    assert result.weighted_error < 1e-5
    counts = balanced_rounding(result.shares, 101, seed=4)
    assert counts.sum() == 101
    assert abs(counts[1] - 76) <= 1
    hop = sample_next_hop(network, 0, target, visited={0, 1}, seed=1)
    assert hop == 2
    with pytest.raises(RuntimeError):
        sample_next_hop(network, 0, target, visited={0, 1, 2}, seed=1)


def test_partial_control_and_compliance_interface() -> None:
    controlled = np.array([[0.0, 1.0, 1.0, 1.0]])
    selfish = np.array([[1.0, 0.0, 1.0, 1.0]])
    mixed = combine_controlled_actions(controlled, selfish, 0.25)
    assert np.allclose(mixed[0, :2], [0.75, 0.25])
    probabilities = compliance_probability(np.array([1.0, 1.5]), np.array([1.0, 1.0]))
    assert probabilities[0] > probabilities[1]

