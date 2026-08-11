from experiments.phase3 import run_phase3_smoke


def test_phase3_projection_and_decision_gate() -> None:
    result = run_phase3_smoke()
    assert result.decision_loss_final < result.decision_loss_initial
    assert result.raw_violation_after < result.raw_violation_before
    assert result.normalized_projection_correction_after < result.normalized_projection_correction_before
    assert result.post_projection_violation_after < 1e-5
    assert result.projection_status in {"optimal", "optimal_inaccurate"}
    assert result.curriculum_horizons[0] == 1
    assert result.curriculum_horizons[-1] == 2
    assert "full" in result.registered_ablations

