from typing import Literal

import torch
from torch import nn

from roughness.scheme1.models import CNNEncoder, ModelOutput, masked_mean_pool


class PhysicsResidualRegressor(nn.Module):
    def __init__(
        self,
        mode: Literal["ordinary", "gated"],
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if mode not in {"ordinary", "gated"}:
            raise ValueError("mode must be ordinary or gated")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must satisfy 0 <= value < 1")
        self.mode = mode
        self.encoder = CNNEncoder()
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.encoder.embedding_dim + 4, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.gate_layer = nn.Linear(4, 1) if mode == "gated" else None
        self._gate_only_training = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self._gate_only_training:
            self.encoder.eval()
            self.residual_mlp.eval()
            if self.gate_layer is not None:
                self.gate_layer.train(mode)
        return self

    def encode_segment(
        self, signal: torch.Tensor, window_mask: torch.Tensor
    ) -> torch.Tensor:
        if signal.ndim != 4:
            raise ValueError("signal must have shape [B,W,3,L]")
        batch, windows, channels, samples = signal.shape
        embedded = self.encoder(
            signal.reshape(batch * windows, channels, samples)
        ).reshape(batch, windows, -1)
        return masked_mean_pool(embedded, window_mask)

    def forward(
        self,
        signal: torch.Tensor,
        window_mask: torch.Tensor,
        process: torch.Tensor,
        physics: torch.Tensor,
        base_ra: torch.Tensor,
    ) -> ModelOutput:
        if process.ndim != 2 or process.shape[-1] != 3:
            raise ValueError("process must have shape [B,3]")
        if physics.ndim != 2 or physics.shape[-1] != 1:
            raise ValueError("physics must have shape [B,1]")
        if base_ra.ndim != 1 or base_ra.shape[0] != process.shape[0]:
            raise ValueError("base_ra must have shape [B]")
        context = torch.cat([process, physics], dim=-1)
        embedding = self.encode_segment(signal, window_mask)
        residual = self.residual_mlp(
            torch.cat([embedding, context], dim=-1)
        ).squeeze(-1)
        if self.mode == "ordinary":
            return ModelOutput(
                prediction=base_ra + residual,
                residual=residual,
            )
        gate = torch.sigmoid(self.gate_layer(context).squeeze(-1))
        return ModelOutput(
            prediction=base_ra + (1.0 - gate) * residual,
            residual=residual,
            gate=gate,
        )


def load_ordinary_into_gated(
    gated: PhysicsResidualRegressor,
    ordinary_state: dict[str, torch.Tensor],
) -> None:
    if gated.mode != "gated":
        raise ValueError("target model must be gated")
    missing, unexpected = gated.load_state_dict(
        ordinary_state, strict=False
    )
    if set(missing) != {"gate_layer.weight", "gate_layer.bias"} or unexpected:
        raise ValueError("ordinary checkpoint is incompatible with gated model")


def freeze_residual_for_gate(
    model: PhysicsResidualRegressor,
) -> dict[str, torch.Tensor]:
    if model.mode != "gated" or model.gate_layer is None:
        raise ValueError("gate-only training requires a gated model")
    snapshot: dict[str, torch.Tensor] = {}
    for name, parameter in model.named_parameters():
        is_gate = name.startswith("gate_layer.")
        parameter.requires_grad_(is_gate)
        if not is_gate:
            snapshot[name] = parameter.detach().cpu().clone()
    model._gate_only_training = True
    model.train()
    return snapshot


def assert_frozen_state_unchanged(
    model: PhysicsResidualRegressor,
    snapshot: dict[str, torch.Tensor],
) -> None:
    current = dict(model.named_parameters())
    if set(snapshot) - set(current):
        raise AssertionError("Frozen parameter set no longer matches model")
    for name, expected in snapshot.items():
        actual = current[name].detach().cpu()
        if not torch.equal(actual, expected):
            raise AssertionError(f"Frozen parameter changed: {name}")
