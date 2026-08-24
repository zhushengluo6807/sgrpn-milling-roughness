"""Fixed Phase A model architectures and losses for SGRPN."""

from __future__ import annotations

from dataclasses import dataclass, fields
import inspect
import math
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from . import dataset


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    process_mean: torch.Tensor | None = None
    residual: torch.Tensor | None = None
    gate: torch.Tensor | None = None
    embedding: torch.Tensor | None = None


def _require_finite_matrix(value: Tensor, *, name: str, width: int) -> None:
    if not isinstance(value, Tensor) or value.ndim != 2 or value.shape[1] != width:
        raise ValueError(f"{name} must have shape [B, {width}]")
    if value.shape[0] == 0:
        raise ValueError(f"{name} must contain at least one row")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be floating point and finite")


def _require_same_batch(*values: tuple[str, Tensor]) -> None:
    batch_sizes = {int(value.shape[0]) for _, value in values}
    if len(batch_sizes) != 1:
        names = ", ".join(name for name, _ in values)
        raise ValueError(f"{names} must have the same batch size")


class ProcessMLP(nn.Module):
    """P1: the fixed nine-feature process expert."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(9, 32),
            nn.ReLU(),
            nn.Dropout(0.10),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, process: Tensor) -> Tensor:
        _require_finite_matrix(process, name="process", width=9)
        return self.network(process).squeeze(1)


class OrderSpectrumEncoder(nn.Module):
    """Encode valid windows and masked-mean pool them into segment embeddings."""

    embedding_dim = 64

    def __init__(self) -> None:
        super().__init__()
        self.window_encoder = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=7, stride=2),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )

    @staticmethod
    def _validate_inputs(spectrum: Tensor, window_mask: Tensor) -> None:
        if (
            not isinstance(spectrum, Tensor)
            or spectrum.ndim != 4
            or tuple(spectrum.shape[2:]) != (3, 361)
        ):
            raise ValueError("spectrum must have shape [B, W, 3, 361]")
        if spectrum.shape[0] == 0 or spectrum.shape[1] == 0:
            raise ValueError("spectrum must contain at least one row and window")
        if not spectrum.is_floating_point() or not bool(torch.isfinite(spectrum).all()):
            raise ValueError("spectrum must be floating point and finite")
        if not isinstance(window_mask, Tensor) or window_mask.ndim != 2:
            raise ValueError("window mask must have shape [B, W]")
        if tuple(window_mask.shape) != tuple(spectrum.shape[:2]):
            raise ValueError("window mask must match spectrum batch and window dimensions")
        if window_mask.dtype != torch.bool:
            raise ValueError("window mask must have bool dtype")
        if window_mask.device != spectrum.device:
            raise ValueError("spectrum and window mask must be on the same device")
        if bool((window_mask.sum(dim=1) == 0).any()):
            raise ValueError("every segment must contain at least one valid window")

    def forward(self, spectrum: Tensor, window_mask: Tensor) -> Tensor:
        self._validate_inputs(spectrum, window_mask)
        batch_size, window_count = spectrum.shape[:2]
        flat_spectrum = spectrum.reshape(batch_size * window_count, 3, 361)
        flat_mask = window_mask.reshape(-1)
        valid_embedding = self.window_encoder(flat_spectrum[flat_mask]).squeeze(-1)
        flat_embedding = valid_embedding.new_zeros(
            (batch_size * window_count, self.embedding_dim)
        )
        flat_embedding = flat_embedding.index_copy(0, flat_mask.nonzero().squeeze(1), valid_embedding)
        embedding = flat_embedding.reshape(batch_size, window_count, self.embedding_dim)
        counts = window_mask.sum(dim=1, keepdim=True).to(embedding.dtype)
        return embedding.sum(dim=1) / counts


def _regression_head(input_width: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(input_width, 32),
        nn.ReLU(),
        nn.Dropout(0.10),
        nn.Linear(32, 1),
    )


class VibrationOnlyModel(nn.Module):
    """V1: direct Ra prediction from the order-spectrum embedding."""

    def __init__(self, encoder: OrderSpectrumEncoder | None = None) -> None:
        super().__init__()
        self.encoder = OrderSpectrumEncoder() if encoder is None else encoder
        self.head = _regression_head(64)

    def forward(self, spectrum: Tensor, window_mask: Tensor) -> ModelOutput:
        embedding = self.encoder(spectrum, window_mask)
        prediction = self.head(embedding).squeeze(1)
        return ModelOutput(prediction=prediction, embedding=embedding)


class DirectFusionModel(nn.Module):
    """F1: direct Ra prediction from concatenated process and vibration inputs."""

    def __init__(self, encoder: OrderSpectrumEncoder | None = None) -> None:
        super().__init__()
        self.encoder = OrderSpectrumEncoder() if encoder is None else encoder
        self.head = _regression_head(9 + 64)

    def forward(
        self,
        spectrum: Tensor,
        window_mask: Tensor,
        process: Tensor,
    ) -> ModelOutput:
        _require_finite_matrix(process, name="process", width=9)
        embedding = self.encoder(spectrum, window_mask)
        _require_same_batch(("process", process), ("spectrum", embedding))
        prediction = self.head(torch.cat([process, embedding], dim=1)).squeeze(1)
        return ModelOutput(prediction=prediction, embedding=embedding)


class ResidualExpert(nn.Module):
    """Residual-CNN expert trained on leakage-safe P1 residual targets."""

    def __init__(self, encoder: OrderSpectrumEncoder | None = None) -> None:
        super().__init__()
        self.encoder = OrderSpectrumEncoder() if encoder is None else encoder
        self.head = _regression_head(64)

    def forward(self, spectrum: Tensor, window_mask: Tensor) -> ModelOutput:
        embedding = self.encoder(spectrum, window_mask)
        residual = self.head(embedding).squeeze(1)
        return ModelOutput(
            prediction=residual,
            residual=residual,
            embedding=embedding,
        )


def combine_prediction(base: Tensor, delta: Tensor, gate: Tensor) -> Tensor:
    """Apply the exact safe-fallback residual combination."""
    if not all(isinstance(value, Tensor) for value in (base, delta, gate)):
        raise ValueError("base, delta, and gate must be tensors")
    if base.ndim != 1 or delta.shape != base.shape or gate.shape != base.shape:
        raise ValueError("base, delta, and gate must have identical shape [B]")
    if not all(value.is_floating_point() for value in (base, delta, gate)):
        raise ValueError("base, delta, and gate must be floating point")
    if not all(bool(torch.isfinite(value).all()) for value in (base, delta, gate)):
        raise ValueError("base, delta, and gate must be finite")
    if bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError("gate must lie in [0, 1]")
    return base + gate * delta


class ResidualFusionModel(nn.Module):
    """R1: frozen or trainable P1 plus the complete learned residual."""

    def __init__(
        self,
        process_expert: ProcessMLP | None = None,
        residual_expert: ResidualExpert | None = None,
    ) -> None:
        super().__init__()
        self.process_expert = ProcessMLP() if process_expert is None else process_expert
        self.residual_expert = ResidualExpert() if residual_expert is None else residual_expert

    def forward(
        self,
        spectrum: Tensor,
        window_mask: Tensor,
        process: Tensor,
    ) -> ModelOutput:
        process_mean = self.process_expert(process)
        residual_output = self.residual_expert(spectrum, window_mask)
        residual = residual_output.residual
        if residual is None or residual_output.embedding is None:
            raise ValueError("residual expert must return residual and embedding tensors")
        _require_same_batch(("process", process_mean), ("spectrum", residual))
        prediction = process_mean + residual
        return ModelOutput(
            prediction=prediction,
            process_mean=process_mean,
            residual=residual,
            embedding=residual_output.embedding,
        )


class SelectiveGatedModel(nn.Module):
    """G1: selectively gate the vibration residual onto the process expert."""

    def __init__(
        self,
        process_expert: ProcessMLP | None = None,
        residual_expert: ResidualExpert | None = None,
    ) -> None:
        super().__init__()
        self.process_expert = ProcessMLP() if process_expert is None else process_expert
        self.residual_expert = ResidualExpert() if residual_expert is None else residual_expert
        self.gate = nn.Sequential(
            nn.Linear(80, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )
        self._experts_frozen = False

    def freeze_experts(self) -> None:
        """Freeze both experts, including their training-time module state."""
        for parameter in self.process_expert.parameters():
            parameter.requires_grad_(False)
        for parameter in self.residual_expert.parameters():
            parameter.requires_grad_(False)
        for parameter in self.gate.parameters():
            parameter.requires_grad_(True)
        self._experts_frozen = True
        self.process_expert.eval()
        self.residual_expert.eval()

    def train(self, mode: bool = True) -> SelectiveGatedModel:
        super().train(mode)
        if self._experts_frozen:
            self.process_expert.eval()
            self.residual_expert.eval()
            self.gate.train(mode)
        return self

    def forward(
        self,
        spectrum: Tensor,
        window_mask: Tensor,
        process: Tensor,
        quality: Tensor,
    ) -> ModelOutput:
        _require_finite_matrix(process, name="process", width=9)
        _require_finite_matrix(quality, name="quality", width=7)
        process_mean = self.process_expert(process)
        residual_output = self.residual_expert(spectrum, window_mask)
        residual = residual_output.residual
        embedding = residual_output.embedding
        if residual is None or embedding is None:
            raise ValueError("residual expert must return residual and embedding tensors")
        _require_same_batch(
            ("process", process),
            ("embedding", embedding),
            ("quality", quality),
        )
        gate_input = torch.cat([process, embedding, quality], dim=1)
        gate = self.gate(gate_input).squeeze(1)
        prediction = combine_prediction(process_mean, residual, gate)
        return ModelOutput(
            prediction=prediction,
            process_mean=process_mean,
            residual=residual,
            gate=gate,
            embedding=embedding,
        )


class VarianceHead(nn.Module):
    """Heteroscedastic Gaussian scale head for frozen G1 means."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(82, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, features: Tensor) -> Tensor:
        _require_finite_matrix(features, name="scale_features", width=82)
        return F.softplus(self.network(features).squeeze(1)) + 1e-4


class GlobalScale(nn.Module):
    """Homoscedastic ablation with exactly one learned positive scale."""

    def __init__(self) -> None:
        super().__init__()
        self.raw_scale = nn.Parameter(torch.zeros(()))

    def forward(self, features: Tensor) -> Tensor:
        _require_finite_matrix(features, name="scale_features", width=82)
        return (F.softplus(self.raw_scale) + 1e-4).expand(features.shape[0])


def _require_finite_vector(value: Tensor, *, name: str) -> None:
    if not isinstance(value, Tensor) or value.ndim != 1 or value.numel() == 0:
        raise ValueError(f"{name} must have shape [B]")
    if not value.is_floating_point() or not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be floating point and finite")


def build_scale_features(output: ModelOutput, process: Tensor, quality: Tensor) -> Tensor:
    """Assemble the registered 64+9+7+1+1 scale-regression features."""
    if not isinstance(output, ModelOutput):
        raise ValueError("output must be a ModelOutput")
    required = ("prediction", "process_mean", "embedding", "gate")
    if any(getattr(output, name) is None for name in required):
        raise ValueError("output must contain prediction, process_mean, embedding, and gate")
    prediction = output.prediction
    process_mean = output.process_mean
    embedding = output.embedding
    gate = output.gate
    assert process_mean is not None and embedding is not None and gate is not None
    _require_finite_vector(prediction, name="prediction")
    _require_finite_vector(process_mean, name="process_mean")
    _require_finite_vector(gate, name="gate")
    _require_finite_matrix(embedding, name="embedding", width=64)
    _require_finite_matrix(process, name="process", width=9)
    _require_finite_matrix(quality, name="quality", width=7)
    _require_same_batch(
        ("prediction", prediction),
        ("process_mean", process_mean),
        ("embedding", embedding),
        ("gate", gate),
        ("process", process),
        ("quality", quality),
    )
    if bool(((gate < 0) | (gate > 1)).any()):
        raise ValueError("gate must lie in [0, 1]")
    correction_abs = (prediction - process_mean).abs()
    return torch.cat([embedding, process, quality, gate[:, None], correction_abs[:, None]], dim=1)


def repeated_gaussian_nll(mu: Tensor, sigma: Tensor, readings: Tensor, weight: Tensor) -> Tensor:
    """Weight region-averaged Gaussian NLL over exactly three Ra readings."""
    _require_finite_vector(mu, name="mu")
    _require_finite_vector(sigma, name="sigma")
    _require_finite_vector(weight, name="weight")
    if not isinstance(readings, Tensor) or readings.ndim != 2 or readings.shape[1] != 3:
        raise ValueError("readings must have shape [B, 3]")
    if not readings.is_floating_point() or not bool(torch.isfinite(readings).all()):
        raise ValueError("readings must be floating point and finite")
    _require_same_batch(("mu", mu), ("sigma", sigma), ("readings", readings), ("weight", weight))
    if bool((sigma <= 0).any()):
        raise ValueError("sigma must be positive")
    if bool((weight <= 0).any()):
        raise ValueError("weight must be positive")
    per_read = 0.5 * torch.log(2.0 * math.pi * sigma[:, None].square())
    per_read = per_read + 0.5 * ((readings - mu[:, None]) / sigma[:, None]).square()
    return (weight * per_read.mean(dim=1)).sum() / weight.sum()


def freeze_mean_model(model: SelectiveGatedModel) -> dict[str, Tensor]:
    """Freeze all G1 mean parameters and snapshot parameters plus buffers."""
    if not isinstance(model, SelectiveGatedModel):
        raise ValueError("model must be a SelectiveGatedModel")
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


def assert_mean_model_unchanged(model: SelectiveGatedModel, snapshot: Mapping[str, Tensor]) -> None:
    """Assert that every frozen-mean parameter and buffer matches its snapshot."""
    if not isinstance(model, SelectiveGatedModel):
        raise ValueError("model must be a SelectiveGatedModel")
    current = model.state_dict()
    if set(current) != set(snapshot):
        raise AssertionError("mean model state keys changed")
    for name, expected in snapshot.items():
        actual = current[name]
        if actual.dtype != expected.dtype or actual.shape != expected.shape:
            raise AssertionError(f"mean model state tensor {name} changed shape or dtype")
        if not torch.equal(actual.detach().cpu(), expected):
            raise AssertionError(f"mean model state tensor {name} changed")


def weighted_huber(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
    delta: float,
) -> Tensor:
    """Return weighted Huber loss normalized by the sum of sample weights."""
    if not all(isinstance(value, Tensor) for value in (prediction, target, weight)):
        raise ValueError("prediction, target, and weight must be tensors")
    if prediction.ndim != 1 or target.shape != prediction.shape or weight.shape != prediction.shape:
        raise ValueError("prediction, target, and weight must have identical shape [B]")
    if prediction.numel() == 0:
        raise ValueError("prediction, target, and weight must not be empty")
    if not all(value.is_floating_point() for value in (prediction, target, weight)):
        raise ValueError("prediction, target, and weight must be floating point")
    if not all(bool(torch.isfinite(value).all()) for value in (prediction, target, weight)):
        raise ValueError("prediction, target, and weight must be finite")
    if bool((weight < 0).any()):
        raise ValueError("weight must be non-negative")
    weight_sum = weight.sum()
    if not bool(weight_sum > 0):
        raise ValueError("weight sum must be positive")
    try:
        numeric_delta = float(delta)
    except (TypeError, ValueError) as error:
        raise ValueError("delta must be finite and positive") from error
    if not math.isfinite(numeric_delta) or numeric_delta <= 0:
        raise ValueError("delta must be finite and positive")
    per_row = F.huber_loss(prediction, target, reduction="none", delta=numeric_delta)
    return (weight * per_row).sum() / weight_sum


def sgrpn_loss(
    output: ModelOutput,
    target: Tensor,
    weight: Tensor,
    delta: float,
) -> Tensor:
    """Registered G1 loss: weighted Huber plus both fixed shrinkage terms."""
    if not isinstance(output, ModelOutput):
        raise ValueError("output must be a ModelOutput")
    if output.gate is None or output.residual is None:
        raise ValueError("SGRPN output must contain gate and residual tensors")
    if output.gate.shape != output.prediction.shape or output.residual.shape != output.prediction.shape:
        raise ValueError("prediction, gate, and residual must have identical shape [B]")
    if not bool(torch.isfinite(output.gate).all()) or not bool(torch.isfinite(output.residual).all()):
        raise ValueError("gate and residual must be finite")
    if bool(((output.gate < 0) | (output.gate > 1)).any()):
        raise ValueError("gate must lie in [0, 1]")
    return (
        weighted_huber(output.prediction, target, weight, delta)
        + 1e-3 * output.gate.square().mean()
        + 1e-2 * (output.gate * output.residual).square().mean()
    )


def _call_with_batch(model: nn.Module, batch: Mapping[str, Any]) -> ModelOutput:
    signature = inspect.signature(model.forward)
    parameters = list(signature.parameters.values())
    positional = [
        parameter
        for parameter in parameters
        if parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) == 1 and positional[0].name == "batch":
        output = model(batch)
    else:
        allowed = {"spectrum", "window_mask", "process", "quality"}
        names = {parameter.name for parameter in parameters}
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters):
            names |= allowed
        kwargs = {name: batch[name] for name in allowed & names if name in batch}
        output = model(**kwargs)
    if not isinstance(output, ModelOutput):
        raise ValueError("model must return ModelOutput")
    return output


def average_swap_predictions(model: nn.Module, batch: dict[str, Any]) -> ModelOutput:
    """Average original/swapped inference for every populated output field."""
    if not isinstance(model, nn.Module):
        raise ValueError("model must be an nn.Module")
    if not isinstance(batch, dict):
        raise ValueError("batch must be a dict")
    original_output = _call_with_batch(model, batch)
    swapped_output = _call_with_batch(model, dataset.swap_horizontal(batch))
    averaged: dict[str, Tensor | None] = {}
    for field in fields(ModelOutput):
        original_value = getattr(original_output, field.name)
        swapped_value = getattr(swapped_output, field.name)
        if original_value is None and swapped_value is None:
            averaged[field.name] = None
            continue
        if not isinstance(original_value, Tensor) or not isinstance(swapped_value, Tensor):
            raise ValueError(f"model output field {field.name} must have matching tensor values")
        if original_value.shape != swapped_value.shape:
            raise ValueError(f"model output field {field.name} shapes do not match")
        if not bool(torch.isfinite(original_value).all()) or not bool(torch.isfinite(swapped_value).all()):
            raise ValueError(f"model output field {field.name} must be finite")
        averaged[field.name] = (original_value + swapped_value) / 2
    return ModelOutput(**averaged)


__all__ = [
    "DirectFusionModel",
    "GlobalScale",
    "ModelOutput",
    "OrderSpectrumEncoder",
    "ProcessMLP",
    "ResidualExpert",
    "ResidualFusionModel",
    "SelectiveGatedModel",
    "VibrationOnlyModel",
    "VarianceHead",
    "assert_mean_model_unchanged",
    "average_swap_predictions",
    "build_scale_features",
    "combine_prediction",
    "sgrpn_loss",
    "freeze_mean_model",
    "repeated_gaussian_nll",
    "weighted_huber",
]
