from experiments.audit import run_static_audit


def test_every_audit_target_exists() -> None:
    requirements = run_static_audit()
    assert len(requirements) >= 25
    assert any(item.identifier == "SUMO" for item in requirements)

