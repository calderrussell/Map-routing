from experiments.phase4 import run_phase4_smoke


def test_phase4_robustness_routing_and_sumo_gate() -> None:
    result = run_phase4_smoke()
    assert result.experiment_ids == tuple(f"E{i}" for i in range(8))
    assert result.paired_seeds >= 30
    assert result.robustness_conditions >= 20
    assert set(result.controlled_fraction_costs) == {"0.00", "0.25", "0.50", "0.75", "1.00"}
    assert result.incident_capacity_violation < 1e-6
    assert result.incident_conservation_residual < 1e-6
    assert result.route_count == 2
    assert result.route_decomposition_error < 1e-5
    assert result.rounded_drivers == 101
    assert result.sumo_status

