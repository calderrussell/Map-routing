import numpy as np
import pytest

from data_processed.scenarios import assert_topology_disjoint, dynamic_demand, sample_incidents


def test_dynamic_demand_is_reproducible_and_forecast_is_distinct() -> None:
    realized_a, forecast_a = dynamic_demand(np.array([5.0, 3.0]), 12, 42, "event")
    realized_b, forecast_b = dynamic_demand(np.array([5.0, 3.0]), 12, 42, "event")
    assert np.allclose(realized_a, realized_b)
    assert np.allclose(forecast_a, forecast_b)
    assert not np.allclose(realized_a, forecast_a)


def test_incident_families_and_topology_leakage_guard() -> None:
    for kind in ("unfamiliar_location", "outside_severity", "multiple", "closure", "noisy_duration"):
        incidents = sample_incidents(10, 12, 1, kind)
        assert len(incidents) == (2 if kind == "multiple" else 1)
    assert_topology_disjoint({"train": ["a", "b"], "validation": ["c"], "test": ["d"]})
    with pytest.raises(ValueError, match="topology leakage"):
        assert_topology_disjoint({"train": ["a"], "test": ["a"]})

