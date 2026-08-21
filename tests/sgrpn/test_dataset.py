from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from roughness.sgrpn.dataset import OrderBagDataset, collate_order_bags, swap_horizontal
from roughness.sgrpn.order_spectrum import (
    OrderSpectrumCache,
    QualityScaler,
    SpectrumScaler,
)


def item_with_windows(count: int, weight: float) -> dict:
    return {
        "spectrum": torch.zeros(count, 3, 361),
        "process": torch.zeros(9),
        "quality": torch.zeros(7),
        "target": torch.tensor(0.5),
        "sample_weight": torch.tensor(weight),
        "sample_id": f"s{count}",
        "group_id": f"g{count}",
    }


def test_collate_preserves_weights_and_masks_variable_bags():
    batch = collate_order_bags([item_with_windows(1, weight=1.0), item_with_windows(3, weight=0.5)])
    assert batch["spectrum"].shape == (2, 3, 3, 361)
    assert batch["window_mask"].tolist() == [[True, False, False], [True, True, True]]
    assert batch["sample_weight"].tolist() == [1.0, 0.5]


def test_horizontal_swap_moves_only_first_two_channels():
    batch = collate_order_bags([item_with_windows(2, weight=1.0)])
    batch["spectrum"][:, :, 0] = 1.0
    batch["spectrum"][:, :, 1] = 2.0
    batch["spectrum"][:, :, 2] = 3.0
    original = batch["spectrum"].clone()
    swapped = swap_horizontal(batch)
    torch.testing.assert_close(swapped["spectrum"][:, :, 0], batch["spectrum"][:, :, 1])
    torch.testing.assert_close(swapped["spectrum"][:, :, 1], batch["spectrum"][:, :, 0])
    torch.testing.assert_close(swapped["spectrum"][:, :, 2], batch["spectrum"][:, :, 2])
    torch.testing.assert_close(batch["spectrum"], original)
    assert swapped is not batch
    assert swapped["sample_id"] is not batch["sample_id"]


def _dataset(*, augment: bool = False, spectra: np.ndarray | None = None) -> OrderBagDataset:
    spectra = np.zeros((2, 3, 361), dtype=np.float32) if spectra is None else spectra
    cache = OrderSpectrumCache(
        segment_ids=("s1",),
        spectra=spectra,
        offsets=np.array([0, len(spectra)], dtype=np.int64),
        quality=np.ones((1, 7), dtype=np.float32),
        durations_s=np.array([1.0]),
    )
    frame = pd.DataFrame(
        [{"sample_id": "s1", "group_id": "g1", "sample_weight": 2.0}]
    )
    return OrderBagDataset(
        frame=frame,
        cache=cache,
        spectrum_scaler=SpectrumScaler(np.zeros((3, 361), dtype=np.float32), np.ones((3, 361), dtype=np.float32)),
        quality_scaler=QualityScaler(np.zeros(7, dtype=np.float32), np.ones(7, dtype=np.float32)),
        process=np.ones((1, 9), dtype=np.float32),
        targets=np.array([0.25], dtype=np.float32),
        augment_horizontal_swap=augment,
    )


def test_dataset_applies_scalers_and_returns_contract():
    spectra = np.zeros((2, 3, 361), dtype=np.float32)
    spectra[:, 0] = 1.0
    dataset = _dataset(spectra=spectra)
    item = dataset[0]
    assert set(item) == {
        "spectrum", "process", "quality", "target", "sample_weight", "sample_id", "group_id"
    }
    assert item["spectrum"].shape == (2, 3, 361)
    assert item["process"].shape == (9,)
    assert item["quality"].shape == (7,)
    assert torch.isfinite(item["spectrum"]).all()
    assert item["sample_id"] == "s1"
    assert item["group_id"] == "g1"


def test_dataset_augments_only_when_enabled_and_random_below_half(monkeypatch):
    spectra = np.zeros((2, 3, 361), dtype=np.float32)
    spectra[:, 0] = 1.0
    spectra[:, 1] = 2.0
    dataset = _dataset(augment=True, spectra=spectra)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.25))
    item = dataset[0]
    torch.testing.assert_close(item["spectrum"][:, 0], torch.full((2, 361), 2.0))
    torch.testing.assert_close(item["spectrum"][:, 1], torch.full((2, 361), 1.0))
    torch.testing.assert_close(item["spectrum"][:, 2], torch.zeros((2, 361)))


def test_dataset_does_not_augment_at_or_above_half(monkeypatch):
    spectra = np.zeros((2, 3, 361), dtype=np.float32)
    spectra[:, 0] = 1.0
    spectra[:, 1] = 2.0
    dataset = _dataset(augment=True, spectra=spectra)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: torch.tensor(0.75))
    item = dataset[0]
    torch.testing.assert_close(item["spectrum"][:, 0], torch.full((2, 361), 1.0))
    torch.testing.assert_close(item["spectrum"][:, 1], torch.full((2, 361), 2.0))


def test_dataset_does_not_draw_random_when_augmentation_disabled(monkeypatch):
    dataset = _dataset(augment=False)
    monkeypatch.setattr(torch, "rand", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected draw")))
    dataset[0]


def test_dataset_rejects_empty_bag():
    with pytest.raises(ValueError, match="empty|zero|windows"):
        collate_order_bags([item_with_windows(0, weight=1.0)])
