from experiments.phase1 import run_phase1_smoke


def test_phase1_heldout_demand_gate() -> None:
    result = run_phase1_smoke(train_samples=24, epochs=60)
    assert result.final_imitation_loss < result.initial_imitation_loss
    assert result.model_tstt < result.dynamic_shortest_path_tstt
    assert result.improvement_over_dynamic_shortest_path > 0.1
    assert result.oracle_relative_gap < 0.1
    assert result.conservation_residual_max < 1e-6

