import numpy as np

from data_processed.networks import analytic_parallel_static, braess_static, sioux_falls_static
from oracle.static_assignment import frank_wolfe, price_of_anarchy_gap


def test_analytic_parallel_ue_and_so() -> None:
    network, demand = analytic_parallel_static()
    ue = frank_wolfe(network, demand, "ue")
    so = frank_wolfe(network, demand, "so")
    assert np.isclose(ue.flow[0], 1500.0, atol=1.0)
    assert np.isclose(so.flow[0], 750.0, atol=1.0)
    assert so.tstt < ue.tstt
    assert abs(price_of_anarchy_gap(ue.tstt, ue.tstt, so.tstt) - 1.0) < 1e-8
    assert abs(price_of_anarchy_gap(so.tstt, ue.tstt, so.tstt)) < 1e-8


def test_braess_price_of_anarchy_and_road_removal() -> None:
    with_middle, demand = braess_static(True)
    without_middle, _ = braess_static(False)
    ue = frank_wolfe(with_middle, demand, "ue")
    so = frank_wolfe(with_middle, demand, "so")
    removed = frank_wolfe(without_middle, demand, "ue")
    assert ue.tstt > so.tstt
    assert removed.tstt < ue.tstt
    assert np.isclose(ue.tstt, 320000.0, rtol=2e-3)
    # The continuous SO uses the middle link for 500 drivers and is slightly better
    # than simply reproducing the 260,000 road-removal solution.
    assert np.isclose(so.tstt, 258750.0, rtol=2e-3)


def test_sioux_falls_static_ue_so_are_finite() -> None:
    network, demand = sioux_falls_static()
    assert len(network.nodes) == 24
    assert network.n_edges == 76
    ue = frank_wolfe(network, demand, "ue", max_iterations=300, tolerance=1e-6)
    so = frank_wolfe(network, demand, "so", max_iterations=300, tolerance=1e-6)
    assert np.isfinite(ue.tstt) and np.isfinite(so.tstt)
    assert so.tstt <= ue.tstt * 1.0001
