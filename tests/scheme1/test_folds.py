import pandas as pd
import pytest

from roughness.scheme1.folds import validate_outer_folds


def _manifest() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": ["a_1", "a_2", "b_1", "c_1", "d_1"],
            "group_id": ["a", "a", "b", "c", "d"],
        }
    )


def test_validate_outer_folds_accepts_complete_grouped_assignment():
    folds = pd.DataFrame(
        {
            "sample_id": ["a_1", "a_2", "b_1", "c_1", "d_1"],
            "group_id": ["a", "a", "b", "c", "d"],
            "fold": [0, 0, 1, 2, 3],
        }
    )

    audit = validate_outer_folds(_manifest(), folds, expected_n_folds=4)

    assert audit == {
        "n_samples": 5,
        "n_groups": 4,
        "n_folds": 4,
        "group_overlap_count": 0,
    }


def test_validate_outer_folds_rejects_group_leakage():
    folds = pd.DataFrame(
        {
            "sample_id": ["a_1", "a_2", "b_1", "c_1", "d_1"],
            "group_id": ["a", "a", "b", "c", "d"],
            "fold": [0, 1, 1, 2, 3],
        }
    )

    with pytest.raises(ValueError, match="Group leakage"):
        validate_outer_folds(_manifest(), folds, expected_n_folds=4)


@pytest.mark.parametrize(
    ("folds", "message"),
    [
        (
            pd.DataFrame(
                {
                    "sample_id": ["a_1", "a_2", "b_1", "c_1"],
                    "group_id": ["a", "a", "b", "c"],
                    "fold": [0, 0, 1, 2],
                }
            ),
            "Missing fold assignments",
        ),
        (
            pd.DataFrame(
                {
                    "sample_id": ["a_1", "a_1", "b_1", "c_1", "d_1"],
                    "group_id": ["a", "a", "b", "c", "d"],
                    "fold": [0, 0, 1, 2, 3],
                }
            ),
            "Duplicate fold assignments",
        ),
    ],
)
def test_validate_outer_folds_rejects_incomplete_or_duplicate_samples(
    folds, message
):
    with pytest.raises(ValueError, match=message):
        validate_outer_folds(_manifest(), folds, expected_n_folds=4)
