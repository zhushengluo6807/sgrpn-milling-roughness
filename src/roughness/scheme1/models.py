from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class ModelOutput:
    prediction: torch.Tensor
    residual: torch.Tensor | None = None
    gate: torch.Tensor | None = None


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


class ConvBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(3, 16, kernel_size=31, stride=4),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv1d(16, 32, kernel_size=15, stride=4),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=9, stride=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
        )

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        if signal.ndim != 3 or signal.shape[1] != 3:
            raise ValueError("signal must have shape [windows, 3, samples]")
        return self.layers(signal)


class _EmbeddingProjection(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(128, 32),
            nn.GELU(),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        average = sequence.mean(dim=-1)
        maximum = sequence.amax(dim=-1)
        return self.projection(torch.cat([average, maximum], dim=1))


class CNNEncoder(nn.Module):
    embedding_dim = 32

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ConvBackbone()
        self.embedding = _EmbeddingProjection()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        return self.embedding(self.backbone(signal))


class TCNBlock(nn.Module):
    def __init__(self, dilation: int, dropout: float = 0.2) -> None:
        super().__init__()
        self.dilation = int(dilation)
        self.block = nn.Sequential(
            nn.Conv1d(
                64,
                64,
                kernel_size=3,
                dilation=self.dilation,
                padding=self.dilation,
            ),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence + self.block(sequence)


class CNNTCNEncoder(nn.Module):
    embedding_dim = 32

    def __init__(self) -> None:
        super().__init__()
        self.backbone = ConvBackbone()
        self.tcn_blocks = nn.ModuleList(
            [TCNBlock(dilation) for dilation in (1, 2, 4, 8)]
        )
        self.embedding = _EmbeddingProjection()

    def forward(self, signal: torch.Tensor) -> torch.Tensor:
        sequence = self.backbone(signal)
        for block in self.tcn_blocks:
            sequence = block(sequence)
        return self.embedding(sequence)


def masked_mean_pool(
    embeddings: torch.Tensor,
    window_mask: torch.Tensor,
) -> torch.Tensor:
    if embeddings.ndim != 3 or window_mask.ndim != 2:
        raise ValueError("Expected embeddings [B,W,D] and mask [B,W]")
    if embeddings.shape[:2] != window_mask.shape:
        raise ValueError("Embedding and mask dimensions do not match")
    counts = window_mask.sum(dim=1, keepdim=True)
    if torch.any(counts == 0):
        raise ValueError("Every segment must contain at least one window")
    weighted = embeddings * window_mask.unsqueeze(-1).to(embeddings.dtype)
    return weighted.sum(dim=1) / counts.to(embeddings.dtype)


class SegmentSignalRegressor(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.embedding_dim, 1)

    def encode_segment(
        self,
        signal: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        if signal.ndim != 4:
            raise ValueError("signal must have shape [B,W,3,L]")
        batch, windows, channels, samples = signal.shape
        encoded = self.encoder(
            signal.reshape(batch * windows, channels, samples)
        ).reshape(batch, windows, -1)
        return masked_mean_pool(encoded, window_mask)

    def forward(
        self,
        signal: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> ModelOutput:
        segment_embedding = self.encode_segment(signal, window_mask)
        prediction = self.head(segment_embedding).squeeze(-1)
        return ModelOutput(prediction=prediction)


class SegmentFusionRegressor(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        process_dim: int,
        physics_dim: int,
        mode: str,
    ) -> None:
        super().__init__()
        if mode not in {"direct", "residual", "gated"}:
            raise ValueError("mode must be direct, residual or gated")
        self.encoder = encoder
        self.process_dim = int(process_dim)
        self.physics_dim = int(physics_dim)
        self.mode = mode
        self.process_branch = (
            nn.Sequential(
                nn.Linear(self.process_dim, 16),
                nn.GELU(),
                nn.Linear(16, 16),
                nn.GELU(),
            )
            if self.process_dim
            else None
        )
        self.physics_branch = (
            nn.Sequential(
                nn.Linear(self.physics_dim, 16),
                nn.GELU(),
            )
            if self.physics_dim
            else None
        )
        fused_dim = encoder.embedding_dim
        if self.process_branch is not None:
            fused_dim += 16
        if self.physics_branch is not None:
            fused_dim += 16
        self.residual_head = nn.Linear(fused_dim, 1)
        self.gate_head = nn.Linear(fused_dim, 1) if mode == "gated" else None

    def _encode(
        self,
        signal: torch.Tensor,
        window_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, windows, channels, samples = signal.shape
        embeddings = self.encoder(
            signal.reshape(batch * windows, channels, samples)
        ).reshape(batch, windows, -1)
        return masked_mean_pool(embeddings, window_mask)

    def forward(
        self,
        signal: torch.Tensor,
        window_mask: torch.Tensor,
        process: torch.Tensor | None = None,
        physics: torch.Tensor | None = None,
        base_ra: torch.Tensor | None = None,
    ) -> ModelOutput:
        parts = [self._encode(signal, window_mask)]
        if self.process_branch is not None:
            if process is None or process.shape[-1] != self.process_dim:
                raise ValueError("process input is missing or has wrong shape")
            parts.append(self.process_branch(process))
        if self.physics_branch is not None:
            if physics is None or physics.shape[-1] != self.physics_dim:
                raise ValueError("physics input is missing or has wrong shape")
            parts.append(self.physics_branch(physics))
        fused = torch.cat(parts, dim=-1)
        learned = self.residual_head(fused).squeeze(-1)
        if self.mode == "direct":
            return ModelOutput(prediction=learned)
        if base_ra is None or base_ra.shape != learned.shape:
            raise ValueError("base_ra is required for residual models")
        if self.mode == "residual":
            return ModelOutput(
                prediction=base_ra + learned,
                residual=learned,
            )
        gate = torch.sigmoid(self.gate_head(fused).squeeze(-1))
        return ModelOutput(
            prediction=base_ra + gate * learned,
            residual=learned,
            gate=gate,
        )
