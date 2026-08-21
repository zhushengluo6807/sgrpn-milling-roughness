from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from roughness.sgrpn.models import (
    DirectFusionModel,
    ModelOutput,
    OrderSpectrumEncoder,
    ProcessMLP,
    ResidualExpert,
    ResidualFusionModel,
    SelectiveGatedModel,
    VibrationOnlyModel,
    average_swap_predictions,
    combine_prediction,
    sgrpn_loss,
    weighted_huber,
)


def _inputs(batch_size: int = 2, windows: int = 3) -> tuple[torch.Tensor, ...]:
    return (
        torch.zeros(batch_size, windows, 3, 361),
        torch.ones(batch_size, windows, dtype=torch.bool),
        torch.zeros(batch_size, 9),
        torch.zeros(batch_size, 7),
    )


def _module_types(module: nn.Sequential) -> list[type[nn.Module]]:
    return [type(layer) for layer in module]


def test_process_mlp_has_registered_architecture_and_returns_one_mean_per_row():
    model = ProcessMLP()
    assert _module_types(model.network) == [
        nn.Linear,
        nn.ReLU,
        nn.Dropout,
        nn.Linear,
        nn.ReLU,
        nn.Linear,
    ]
    assert model.network[0].in_features == 9
    assert model.network[0].out_features == 32
    assert model.network[2].p == 0.10
    assert model.network[3].in_features == 32
    assert model.network[3].out_features == 16
    assert model.network[5].in_features == 16
    assert model.network[5].out_features == 1
    assert model(torch.zeros(4, 9)).shape == (4,)


def test_encoder_has_registered_cnn_and_returns_64_dimensional_embedding():
    encoder = OrderSpectrumEncoder()
    assert _module_types(encoder.window_encoder) == [
        nn.Conv1d,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Conv1d,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.Conv1d,
        nn.BatchNorm1d,
        nn.ReLU,
        nn.AdaptiveAvgPool1d,
    ]
    convolutions = [layer for layer in encoder.window_encoder if isinstance(layer, nn.Conv1d)]
    assert [(layer.in_channels, layer.out_channels, layer.kernel_size, layer.stride) for layer in convolutions] == [
        (3, 16, (7,), (2,)),
        (16, 32, (5,), (2,)),
        (32, 64, (3,), (2,)),
    ]
    out = encoder(torch.zeros(2, 3, 3, 361), torch.ones(2, 3, dtype=torch.bool))
    assert out.shape == (2, 64)


def test_registered_models_return_model_output_contract():
    spectrum, mask, process, quality = _inputs()
    encoder = OrderSpectrumEncoder()

    vibration = VibrationOnlyModel(encoder)(spectrum, mask)
    direct = DirectFusionModel(OrderSpectrumEncoder())(spectrum, mask, process)
    residual = ResidualFusionModel(
        ProcessMLP(), ResidualExpert(OrderSpectrumEncoder())
    )(spectrum, mask, process)
    gated = SelectiveGatedModel(
        ProcessMLP(), ResidualExpert(OrderSpectrumEncoder())
    )(spectrum, mask, process, quality)

    assert vibration.prediction.shape == (2,)
    assert vibration.embedding is not None and vibration.embedding.shape == (2, 64)
    assert direct.prediction.shape == (2,)
    assert direct.embedding is not None and direct.embedding.shape == (2, 64)
    assert residual.process_mean is not None and residual.process_mean.shape == (2,)
    assert residual.residual is not None and residual.residual.shape == (2,)
    assert gated.gate is not None and gated.gate.shape == (2,)
    assert bool(((gated.gate >= 0) & (gated.gate <= 1)).all())


def test_vibration_and_residual_heads_have_registered_architecture():
    for head in (
        VibrationOnlyModel().head,
        ResidualExpert().head,
    ):
        assert _module_types(head) == [nn.Linear, nn.ReLU, nn.Dropout, nn.Linear]
        assert (head[0].in_features, head[0].out_features) == (64, 32)
        assert head[2].p == 0.10
        assert (head[3].in_features, head[3].out_features) == (32, 1)

    direct_head = DirectFusionModel().head
    assert (direct_head[0].in_features, direct_head[0].out_features) == (73, 32)


def test_gated_model_uses_exact_80_to_16_to_1_gate():
    model = SelectiveGatedModel(ProcessMLP(), ResidualExpert(OrderSpectrumEncoder()))
    assert _module_types(model.gate) == [nn.Linear, nn.ReLU, nn.Linear, nn.Sigmoid]
    assert (model.gate[0].in_features, model.gate[0].out_features) == (80, 16)
    assert (model.gate[2].in_features, model.gate[2].out_features) == (16, 1)


def test_gated_model_has_exact_fallback_boundaries():
    base = torch.tensor([1.0, 2.0])
    delta = torch.tensor([0.2, -0.3])
    torch.testing.assert_close(combine_prediction(base, delta, torch.zeros(2)), base)
    torch.testing.assert_close(combine_prediction(base, delta, torch.ones(2)), base + delta)


def test_gate_training_freezes_both_experts_and_keeps_gate_trainable():
    model = SelectiveGatedModel(ProcessMLP(), ResidualExpert(OrderSpectrumEncoder()))
    model.freeze_experts()
    assert not any(parameter.requires_grad for parameter in model.process_expert.parameters())
    assert not any(parameter.requires_grad for parameter in model.residual_expert.parameters())
    assert any(parameter.requires_grad for parameter in model.gate.parameters())


def test_frozen_expert_state_is_equal_after_one_gate_optimizer_step():
    torch.manual_seed(7)
    model = SelectiveGatedModel(ProcessMLP(), ResidualExpert(OrderSpectrumEncoder()))
    model.freeze_experts()
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
        if name.startswith(("process_expert.", "residual_expert."))
    }
    optimizer = torch.optim.AdamW(model.gate.parameters(), lr=1e-3)
    spectrum, mask, process, quality = _inputs()
    model.train()
    output = model(spectrum, mask, process, quality)
    loss = sgrpn_loss(output, torch.tensor([0.2, 0.4]), torch.ones(2), delta=0.1)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    after = model.state_dict()
    assert all(torch.equal(expected, after[name]) for name, expected in before.items())


def test_weighted_huber_is_normalized_by_sum_of_weights():
    prediction = torch.tensor([0.0, 1.0])
    target = torch.tensor([0.0, 0.0])
    weight = torch.tensor([3.0, 1.0])
    # Per-row Huber losses at delta=0.5 are [0.0, 0.375].
    torch.testing.assert_close(
        weighted_huber(prediction, target, weight, delta=0.5),
        torch.tensor(0.375 / 4.0),
    )


def test_sgrpn_loss_matches_registered_formula():
    output = ModelOutput(
        prediction=torch.tensor([1.2]),
        process_mean=torch.tensor([1.0]),
        residual=torch.tensor([0.4]),
        gate=torch.tensor([0.5]),
        embedding=torch.zeros(1, 64),
    )
    loss = sgrpn_loss(output, torch.tensor([1.0]), torch.tensor([1.0]), delta=0.1)
    expected = weighted_huber(output.prediction, torch.tensor([1.0]), torch.tensor([1.0]), 0.1)
    expected = expected + 1e-3 * 0.5**2 + 1e-2 * (0.5 * 0.4) ** 2
    torch.testing.assert_close(loss, expected)


@pytest.mark.parametrize(
    ("prediction", "target", "weight", "delta", "message"),
    [
        (torch.zeros(2, 1), torch.zeros(2), torch.ones(2), 0.1, "shape"),
        (torch.tensor([float("nan")]), torch.zeros(1), torch.ones(1), 0.1, "finite"),
        (torch.zeros(1), torch.zeros(1), torch.tensor([-1.0]), 0.1, "non-negative"),
        (torch.zeros(1), torch.zeros(1), torch.tensor([0.0]), 0.1, "sum"),
        (torch.zeros(1), torch.zeros(1), torch.ones(1), 0.0, "delta"),
    ],
)
def test_weighted_huber_rejects_invalid_inputs(prediction, target, weight, delta, message):
    with pytest.raises(ValueError, match=message):
        weighted_huber(prediction, target, weight, delta)


@pytest.mark.parametrize(
    "process",
    [torch.zeros(2, 8), torch.full((2, 9), float("nan")), torch.zeros(2, 9, 1)],
)
def test_process_mlp_rejects_wrong_width_rank_or_non_finite_values(process):
    with pytest.raises(ValueError, match="process"):
        ProcessMLP()(process)


def test_encoder_rejects_non_three_channel_spectra():
    with pytest.raises(ValueError, match="spectrum"):
        OrderSpectrumEncoder()(torch.zeros(2, 3, 2, 361), torch.ones(2, 3, dtype=torch.bool))


def test_encoder_rejects_zero_window_masks():
    with pytest.raises(ValueError, match="at least one|valid window"):
        OrderSpectrumEncoder()(torch.zeros(2, 3, 3, 361), torch.zeros(2, 3, dtype=torch.bool))


@pytest.mark.parametrize(
    ("spectrum", "mask", "message"),
    [
        (torch.zeros(2, 3, 3, 360), torch.ones(2, 3, dtype=torch.bool), "361"),
        (torch.full((2, 3, 3, 361), float("inf")), torch.ones(2, 3, dtype=torch.bool), "finite"),
        (torch.zeros(2, 3, 3, 361), torch.ones(2, 2, dtype=torch.bool), "mask"),
        (torch.zeros(2, 3, 3, 361), torch.ones(2, 3), "bool"),
    ],
)
def test_encoder_rejects_invalid_shape_finiteness_and_mask(spectrum, mask, message):
    with pytest.raises(ValueError, match=message):
        OrderSpectrumEncoder()(spectrum, mask)


def test_direct_fusion_rejects_wrong_process_width():
    spectrum, mask, _, _ = _inputs()
    with pytest.raises(ValueError, match="process"):
        DirectFusionModel()(spectrum, mask, torch.zeros(2, 8))


def test_gated_model_rejects_wrong_quality_width():
    spectrum, mask, process, _ = _inputs()
    model = SelectiveGatedModel(ProcessMLP(), ResidualExpert(OrderSpectrumEncoder()))
    with pytest.raises(ValueError, match="quality"):
        model(spectrum, mask, process, torch.zeros(2, 6))


class _SwapAwareModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[torch.Tensor] = []

    def forward(self, spectrum, window_mask, process, quality):
        del window_mask, process, quality
        self.calls.append(spectrum.detach().clone())
        channel_value = spectrum[:, :, 0].mean(dim=(1, 2))
        embedding = channel_value[:, None].expand(-1, 64)
        return ModelOutput(
            prediction=channel_value,
            process_mean=channel_value + 1,
            residual=channel_value + 2,
            gate=channel_value + 3,
            embedding=embedding,
        )


def test_average_swap_predictions_is_deep_non_mutating_and_averages_all_fields():
    spectrum, mask, process, quality = _inputs(batch_size=1, windows=2)
    spectrum[:, :, 0] = 2.0
    spectrum[:, :, 1] = 6.0
    batch = {
        "spectrum": spectrum,
        "window_mask": mask,
        "process": process,
        "quality": quality,
        "target": torch.tensor([0.5]),
        "sample_weight": torch.tensor([1.0]),
        "sample_id": ["s1"],
        "group_id": ["g1"],
    }
    original = deepcopy(batch)
    model = _SwapAwareModel()

    output = average_swap_predictions(model, batch)

    assert len(model.calls) == 2
    assert torch.equal(model.calls[0], original["spectrum"])
    assert torch.equal(model.calls[1][:, :, 0], original["spectrum"][:, :, 1])
    torch.testing.assert_close(output.prediction, torch.tensor([4.0]))
    torch.testing.assert_close(output.process_mean, torch.tensor([5.0]))
    torch.testing.assert_close(output.residual, torch.tensor([6.0]))
    torch.testing.assert_close(output.gate, torch.tensor([7.0]))
    torch.testing.assert_close(output.embedding, torch.full((1, 64), 4.0))
    assert torch.equal(batch["spectrum"], original["spectrum"])
    assert batch["sample_id"] == original["sample_id"]


def test_average_swap_predictions_preserves_none_fields():
    class PredictionOnly(nn.Module):
        def forward(self, spectrum, window_mask, process, quality):
            del window_mask, process, quality
            return ModelOutput(prediction=spectrum[:, :, 0].mean(dim=(1, 2)))

    spectrum, mask, process, quality = _inputs(batch_size=1, windows=1)
    batch = {
        "spectrum": spectrum,
        "window_mask": mask,
        "process": process,
        "quality": quality,
        "target": torch.tensor([0.5]),
        "sample_weight": torch.tensor([1.0]),
        "sample_id": ["s1"],
        "group_id": ["g1"],
    }
    output = average_swap_predictions(PredictionOnly(), batch)
    assert output.process_mean is None
    assert output.residual is None
    assert output.gate is None
    assert output.embedding is None
