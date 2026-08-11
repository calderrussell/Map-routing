from experiments.phase2 import run_phase2_smoke


def test_phase2_topology_heldout_gate() -> None:
    result = run_phase2_smoke(epochs=20)
    assert result.validation_topology_hash not in result.training_topology_hashes
    assert result.final_training_loss < result.initial_training_loss
    assert result.validation_illegal_flow == 0.0
    assert result.validation_output_shape[0] == 1
    assert result.oracle_labels_certified > 0
    assert result.oracle_labels_uncertified > 0

