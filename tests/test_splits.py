import pandas as pd
import pytest

from roughness.splits import make_group_folds


def test_group_folds_are_deterministic_and_leak_free():
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(12)],
            "group_id": [f"g{i // 2}" for i in range(12)],
        }
    )

    first = make_group_folds(manifest, n_splits=3, seed=7)
    second = make_group_folds(manifest, n_splits=3, seed=7)

    pd.testing.assert_frame_equal(first, second)
    assert first.groupby("group_id")["fold"].nunique().max() == 1
    assert first["sample_id"].nunique() == len(manifest)
    assert set(first["fold"]) == {0, 1, 2}


def test_group_folds_reject_too_many_splits():
    manifest = pd.DataFrame(
        {"sample_id": ["s0", "s1"], "group_id": ["g0", "g1"]}
    )

    with pytest.raises(ValueError, match="smaller than n_splits"):
        make_group_folds(manifest, n_splits=3, seed=7)
