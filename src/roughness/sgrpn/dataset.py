"""Variable-window order-spectrum datasets for SGRPN Phase A."""

from __future__ import annotations

from copy import copy
from typing import Any, Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .order_spectrum import OrderSpectrumCache, QualityScaler, SpectrumScaler


_ITEM_KEYS = frozenset(
    {
        "spectrum",
        "process",
        "quality",
        "target",
        "sample_weight",
        "sample_id",
        "group_id",
    }
)
_BATCH_KEYS = frozenset((*_ITEM_KEYS, "window_mask"))


def _finite_array(value: Any, *, name: str, dtype: np.dtype | type = np.float32) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be numeric and finite") from error
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must be finite")
    return array


def _scaler_values(scaler: SpectrumScaler | QualityScaler, *, shape: tuple[int, ...], name: str) -> tuple[np.ndarray, np.ndarray]:
    mean = _finite_array(scaler.mean, name=f"{name}.mean")
    scale = _finite_array(scaler.scale, name=f"{name}.scale")
    if mean.shape != shape or scale.shape != shape:
        raise ValueError(f"{name} mean and scale must have shape {shape}")
    if np.any(scale == 0):
        raise ValueError(f"{name}.scale must be non-zero")
    return mean, scale


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.clone()
    if isinstance(value, np.ndarray):
        return value.copy()
    if isinstance(value, list):
        return list(value)
    if isinstance(value, tuple):
        return tuple(value)
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    return copy(value)


class OrderBagDataset(Dataset):
    """Expose one scaled, variable-length spectrum bag per manifest row."""

    def __init__(
        self,
        frame: pd.DataFrame,
        cache: OrderSpectrumCache,
        spectrum_scaler: SpectrumScaler,
        quality_scaler: QualityScaler,
        process: np.ndarray,
        targets: np.ndarray,
        augment_horizontal_swap: bool,
    ) -> None:
        if not isinstance(frame, pd.DataFrame):
            raise TypeError("frame must be a pandas DataFrame")
        required = {"sample_id", "group_id", "sample_weight"}
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"frame missing columns: {missing}")
        if frame["sample_id"].isna().any() or frame["group_id"].isna().any():
            raise ValueError("frame sample_id and group_id must not be missing")
        sample_ids = tuple(frame["sample_id"].astype(str))
        if len(set(sample_ids)) != len(sample_ids):
            raise ValueError("frame sample_id values must be unique")
        weights = _finite_array(frame["sample_weight"].to_numpy(), name="sample_weight")
        if weights.ndim != 1 or np.any(weights <= 0):
            raise ValueError("sample_weight must be finite and positive")

        process_values = _finite_array(process, name="process")
        if process_values.shape != (len(frame), 9):
            raise ValueError("process must have shape [n_samples, 9]")
        target_values = _finite_array(targets, name="targets")
        if target_values.ndim != 1 or target_values.shape[0] != len(frame):
            raise ValueError("targets must have shape [n_samples]")

        self._frame = frame.reset_index(drop=True).copy()
        self._sample_ids = sample_ids
        self._group_ids = tuple(self._frame["group_id"].astype(str))
        self._weights = weights
        self._process = process_values
        self._targets = target_values
        self._cache = cache
        self._spectrum_mean, self._spectrum_scale = _scaler_values(
            spectrum_scaler, shape=(3, 361), name="spectrum_scaler"
        )
        self._quality_mean, self._quality_scale = _scaler_values(
            quality_scaler, shape=(7,), name="quality_scaler"
        )
        self.augment_horizontal_swap = bool(augment_horizontal_swap)

        for sample_id in self._sample_ids:
            bag = np.asarray(cache.bag(sample_id), dtype=np.float32)
            if bag.ndim != 3 or bag.shape[1:] != (3, 361) or bag.shape[0] == 0:
                raise ValueError(f"sample {sample_id} has an empty or invalid spectrum bag")
            if not np.isfinite(bag).all():
                raise ValueError(f"sample {sample_id} spectrum bag must be finite")
            quality = np.asarray(cache.quality_row(sample_id), dtype=np.float32)
            if quality.shape != (7,) or not np.isfinite(quality).all():
                raise ValueError(f"sample {sample_id} quality must be finite with shape [7]")

    def __len__(self) -> int:
        return len(self._sample_ids)

    def __getitem__(self, index: int) -> dict[str, Any]:
        bag = np.asarray(self._cache.bag(self._sample_ids[index]), dtype=np.float32)
        quality = np.asarray(self._cache.quality_row(self._sample_ids[index]), dtype=np.float32)
        scaled_spectrum = (bag - self._spectrum_mean) / self._spectrum_scale
        scaled_quality = (quality - self._quality_mean) / self._quality_scale
        if not np.isfinite(scaled_spectrum).all() or not np.isfinite(scaled_quality).all():
            raise ValueError("scaler transform produced non-finite values")

        item: dict[str, Any] = {
            "spectrum": torch.from_numpy(scaled_spectrum.copy()),
            "process": torch.from_numpy(self._process[index].copy()),
            "quality": torch.from_numpy(scaled_quality.copy()),
            "target": torch.tensor(self._targets[index]),
            "sample_weight": torch.tensor(self._weights[index]),
            "sample_id": self._sample_ids[index],
            "group_id": self._group_ids[index],
        }
        if self.augment_horizontal_swap and bool(torch.rand(()) < 0.5):
            item = swap_horizontal_item(item)
        return item


def swap_horizontal_item(item: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copied item with Ch9/Ch10 exchanged and Z unchanged."""
    if frozenset(item) != _ITEM_KEYS:
        raise ValueError(f"item keys must be {sorted(_ITEM_KEYS)}")
    output = {key: _clone_value(value) for key, value in item.items()}
    spectrum = output["spectrum"]
    if not isinstance(spectrum, torch.Tensor) or spectrum.ndim != 3 or tuple(spectrum.shape[1:]) != (3, 361):
        raise ValueError("item spectrum must have shape [W, 3, 361]")
    if spectrum.shape[0] == 0:
        raise ValueError("item spectrum bag must not be empty")
    output["spectrum"][:, [0, 1]] = output["spectrum"][:, [1, 0]]
    return output


def collate_order_bags(items: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length bags to ``[B, W, 3, 361]`` and return a mask."""
    if not items:
        raise ValueError("cannot collate an empty batch")
    for item in items:
        if frozenset(item) != _ITEM_KEYS:
            raise ValueError(f"item keys must be {sorted(_ITEM_KEYS)}")
    spectra = [torch.as_tensor(item["spectrum"]) for item in items]
    if any(spectrum.ndim != 3 or tuple(spectrum.shape[1:]) != (3, 361) for spectrum in spectra):
        raise ValueError("each spectrum must have shape [W, 3, 361]")
    if any(spectrum.shape[0] == 0 for spectrum in spectra):
        raise ValueError("spectrum bags must not be empty")
    if any(not bool(torch.isfinite(spectrum).all()) for spectrum in spectra):
        raise ValueError("spectrum values must be finite")

    batch_size = len(items)
    max_windows = max(int(spectrum.shape[0]) for spectrum in spectra)
    padded = torch.zeros(
        (batch_size, max_windows, 3, 361), dtype=spectra[0].dtype, device=spectra[0].device
    )
    mask = torch.zeros((batch_size, max_windows), dtype=torch.bool, device=spectra[0].device)
    for row, spectrum in enumerate(spectra):
        if spectrum.device != padded.device:
            raise ValueError("all spectra must be on the same device")
        width = spectrum.shape[0]
        padded[row, :width] = spectrum
        mask[row, :width] = True

    process = torch.stack([torch.as_tensor(item["process"]) for item in items])
    quality = torch.stack([torch.as_tensor(item["quality"]) for item in items])
    target = torch.stack([torch.as_tensor(item["target"]) for item in items])
    sample_weight = torch.stack([torch.as_tensor(item["sample_weight"]) for item in items])
    if process.ndim != 2 or process.shape[1] != 9 or not bool(torch.isfinite(process).all()):
        raise ValueError("process values must have shape [B, 9] and be finite")
    if quality.ndim != 2 or quality.shape[1] != 7 or not bool(torch.isfinite(quality).all()):
        raise ValueError("quality values must have shape [B, 7] and be finite")
    if target.ndim != 1 or not bool(torch.isfinite(target).all()):
        raise ValueError("target values must be finite scalars")
    if sample_weight.ndim != 1 or not bool(torch.isfinite(sample_weight).all()):
        raise ValueError("sample_weight values must be finite scalars")
    return {
        "spectrum": padded,
        "window_mask": mask,
        "process": process,
        "quality": quality,
        "target": target,
        "sample_weight": sample_weight,
        "sample_id": [str(item["sample_id"]) for item in items],
        "group_id": [str(item["group_id"]) for item in items],
    }


def swap_horizontal(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Return an independent batch with its first two spectrum channels exchanged."""
    if frozenset(batch) != _BATCH_KEYS:
        raise ValueError(f"batch keys must be {sorted(_BATCH_KEYS)}")
    output = {key: _clone_value(value) for key, value in batch.items()}
    spectrum = output["spectrum"]
    if not isinstance(spectrum, torch.Tensor) or spectrum.ndim != 4 or tuple(spectrum.shape[2:]) != (3, 361):
        raise ValueError("batch spectrum must have shape [B, W, 3, 361]")
    output["spectrum"][:, :, [0, 1]] = output["spectrum"][:, :, [1, 0]]
    return output
