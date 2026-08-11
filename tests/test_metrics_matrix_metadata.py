import numpy as np

from data_processed.networks import diamond_cell_network
from experiments.matrix import load_and_validate_plan
from experiments.metadata import collect_metadata
from experiments.metrics import fairness_metrics, gini, paired_bootstrap_interval


def test_frozen_experiment_matrix_and_reproducibility_metadata() -> None:
    plan = load_and_validate_plan("configs/frozen_experiment_plan.yaml")
    assert len(plan["experiments"]) == 8
    metadata = collect_metadata(
        "configs/frozen_experiment_plan.yaml",
        diamond_cell_network().topology_hash(),
        tuple(range(30)),
    )
    assert metadata.configuration_sha256
    assert metadata.network_sha256
    assert metadata.solver_versions["torch"]


def test_fairness_and_paired_statistics() -> None:
    assert np.isclose(gini(np.ones(5)), 0.0)
    fairness = fairness_metrics(np.array([10.0, 20.0]), np.array([10.0, 10.0]))
    assert fairness["maximum_detour_ratio"] == 2.0
    interval = paired_bootstrap_interval(
        np.array([8.0, 9.0, 10.0]), np.array([10.0, 10.0, 10.0]), seed=2, repetitions=100
    )
    assert interval["mean_difference"] < 0

