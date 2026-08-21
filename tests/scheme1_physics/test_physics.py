import numpy as np
import pandas as pd
import pytest

from roughness.scheme1_physics.physics import (
    build_physics_outer_frames,
    exact_ra_um,
    physical_ra_um,
    select_effective_scale,
    word_ra_um,
)


def _frame() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    folds = []
    for index in range(15):
        rows.append(
            {
                "sample_id": f"s{index}",
                "group_id": f"g{index}",
                "fz_mm_per_tooth": 0.02 + 0.01 * (index % 4),
                "ra_mean": 0.1 + 0.02 * index,
                "sample_weight": 1.0,
            }
        )
        folds.append({"sample_id": f"s{index}", "fold": index % 5})
    return pd.DataFrame(rows), pd.DataFrame(folds)


def test_word_formula_converts_mm_to_um():
    assert word_ra_um(0.04, 0.2) == pytest.approx(
        1000.0 * 0.04**2 / (32.0 * 0.2)
    )


def test_exact_formula_rejects_invalid_domain():
    with pytest.raises(ValueError, match="fz_mm must satisfy"):
        exact_ra_um(0.21, 0.10)


def test_exact_converges_to_word_for_small_ratio():
    exact = exact_ra_um(1e-4, 0.8)
    approximate = word_ra_um(1e-4, 0.8)
    assert exact == pytest.approx(approximate, rel=1e-8)


def test_formula_dispatch_and_array_outputs():
    fz = np.array([0.02, 0.04, 0.08])
    assert np.isfinite(physical_ra_um("exact", fz, 0.2)).all()
    assert np.isfinite(physical_ra_um("word", fz, 0.2)).all()
    with pytest.raises(ValueError, match="formula"):
        physical_ra_um("unknown", fz, 0.2)


def test_scale_selection_is_auditable_and_deterministic():
    manifest, folds = _frame()
    train = manifest.merge(folds, on="sample_id")
    train = train[train["fold"] != 0].reset_index(drop=True)
    selection, audit = select_effective_scale(
        train,
        formula="exact",
        candidates_mm=(0.075, 0.1, 0.2),
        inner_splits=3,
        outer_fold=0,
        seed=20260723,
    )
    assert selection.radius_mm in {0.075, 0.1, 0.2}
    assert selection.source_sample_ids == tuple(
        sorted(train["sample_id"].astype(str))
    )
    assert set(audit["candidate_radius_mm"]) == {0.075, 0.1, 0.2}
    assert audit["source_sample_ids"].nunique() == 1


def test_outer_test_rows_never_enter_scale_selection():
    manifest, folds = _frame()
    train, test, selection, audit = build_physics_outer_frames(
        manifest,
        folds,
        outer_fold=0,
        seed=20260723,
        formula="word",
        candidates_mm=(0.1, 0.2),
        inner_splits=3,
    )
    train_ids = set(train["sample_id"].astype(str))
    test_ids = set(test["sample_id"].astype(str))
    source_ids = set(audit.iloc[0]["source_sample_ids"].split("|"))
    assert train_ids.isdisjoint(test_ids)
    assert source_ids == train_ids
    assert source_ids.isdisjoint(test_ids)
    assert train["base_ra"].notna().all()
    assert test["base_ra"].notna().all()
    assert selection.outer_fold == 0
