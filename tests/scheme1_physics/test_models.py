import torch

from roughness.scheme1_physics.models import (
    PhysicsResidualRegressor,
    assert_frozen_state_unchanged,
    freeze_residual_for_gate,
    load_ordinary_into_gated,
)


def _batch():
    return {
        "signal": torch.randn(2, 3, 3, 256),
        "window_mask": torch.ones(2, 3, dtype=torch.bool),
        "process": torch.randn(2, 3),
        "physics": torch.randn(2, 1),
        "base_ra": torch.tensor([0.2, 0.4]),
    }


def test_ordinary_prediction_is_physics_plus_residual():
    batch = _batch()
    output = PhysicsResidualRegressor("ordinary", dropout=0.0)(**batch)
    assert torch.allclose(
        output.prediction, batch["base_ra"] + output.residual
    )


def test_gated_prediction_uses_one_minus_physical_confidence():
    batch = _batch()
    output = PhysicsResidualRegressor("gated", dropout=0.0)(**batch)
    assert torch.all((0 <= output.gate) & (output.gate <= 1))
    assert torch.allclose(
        output.prediction,
        batch["base_ra"] + (1.0 - output.gate) * output.residual,
    )


def test_gate_training_freezes_residual_path_and_disables_dropout():
    ordinary = PhysicsResidualRegressor("ordinary", dropout=0.3)
    gated = PhysicsResidualRegressor("gated", dropout=0.3)
    load_ordinary_into_gated(gated, ordinary.state_dict())
    snapshot = freeze_residual_for_gate(gated)
    gated.train()

    assert gated.gate_layer.weight.requires_grad
    assert all(
        not parameter.requires_grad
        for name, parameter in gated.named_parameters()
        if not name.startswith("gate_layer.")
    )
    assert not gated.encoder.training
    assert not gated.residual_mlp.training
    assert gated.gate_layer.training
    assert_frozen_state_unchanged(gated, snapshot)


def test_frozen_state_check_detects_a_change():
    model = PhysicsResidualRegressor("gated", dropout=0.0)
    snapshot = freeze_residual_for_gate(model)
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1.0)
    try:
        assert_frozen_state_unchanged(model, snapshot)
    except AssertionError:
        pass
    else:
        raise AssertionError("changed frozen weights were not detected")
