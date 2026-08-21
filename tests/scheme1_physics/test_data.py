from pathlib import Path

import numpy as np
import pandas as pd

from roughness.scheme1_physics.config import load_scheme1_physics_config
from roughness.scheme1_physics.data import (
    build_physics_loaders,
    fit_input_scaler,
    prepare_physics_fold,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "scheme1_physics.yaml"


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a", "b", "c"],
            "group_id": ["g1", "g2", "g3"],
            "n_rpm": [1000.0, 2000.0, 3000.0],
            "fz_mm_per_tooth": [0.01, 0.02, 0.03],
            "ap_mm": [0.1, 0.2, 0.3],
            "base_ra": [0.2, 0.4, 0.6],
        }
    )


def test_scaler_uses_only_fit_rows_and_preserves_raw_base_ra():
    train = _frame().iloc[:2].copy()
    held_out = _frame().iloc[2:].copy()
    scaler = fit_input_scaler(train)
    transformed = scaler.transform(held_out)

    assert scaler.source_sample_ids == ("a", "b")
    assert np.array_equal(transformed["base_ra"], held_out["base_ra"])
    assert np.isfinite(
        transformed[
            ["n_rpm", "fz_mm_per_tooth", "ap_mm", "base_ra_scaled"]
        ].to_numpy()
    ).all()


def test_prepare_fold_selects_and_scales_from_outer_train_only():
    config = load_scheme1_physics_config(CONFIG_PATH)
    prepared = prepare_physics_fold(
        config, formula="word", outer_fold=0, seed=20260723
    )
    train_ids = set(prepared.train_raw["sample_id"].astype(str))
    test_ids = set(prepared.test_raw["sample_id"].astype(str))

    assert train_ids.isdisjoint(test_ids)
    assert set(prepared.scaler.source_sample_ids) == train_ids
    assert set(prepared.selection.source_sample_ids) == train_ids
    assert "base_ra_scaled" in prepared.train_scaled
    assert np.array_equal(
        prepared.test_raw["base_ra"], prepared.test_scaled["base_ra"]
    )


def test_loader_groups_are_disjoint():
    config = load_scheme1_physics_config(CONFIG_PATH)
    prepared = prepare_physics_fold(
        config, formula="word", outer_fold=0, seed=20260723
    )
    loaders = build_physics_loaders(
        prepared,
        config,
        batch_size=2,
        validation_fraction=0.25,
        seed=20260723,
        device="cpu",
    )

    assert loaders.train_groups.isdisjoint(loaders.validation_groups)
    assert loaders.train_groups.isdisjoint(loaders.test_groups)
    assert loaders.validation_groups.isdisjoint(loaders.test_groups)
    batch = next(iter(loaders.train_loader))
    assert batch["physics"].shape[1] == 1
    assert batch["base_ra"].ndim == 1
