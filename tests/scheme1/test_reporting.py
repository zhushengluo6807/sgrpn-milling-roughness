import pandas as pd

from roughness.scheme1.reporting import (
    write_error_breakdowns,
    write_prediction_figures,
)


def test_reporting_writes_scatter_and_residual_figures(tmp_path):
    predictions = pd.DataFrame(
        {
            "model": ["N1"] * 4,
            "y_true": [0.5, 0.8, 1.0, 1.2],
            "y_pred": [0.55, 0.75, 1.05, 1.1],
        }
    )

    paths = write_prediction_figures(predictions, tmp_path)

    assert paths["scatter"].is_file()
    assert paths["residual"].is_file()


def test_reporting_writes_process_and_version_error_breakdowns(tmp_path):
    predictions = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "y_true": [0.5, 0.8, 1.0, 1.2],
            "y_pred": [0.55, 0.75, 1.05, 1.1],
            "sample_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    manifest = pd.DataFrame(
        {
            "sample_id": ["s1", "s2", "s3", "s4"],
            "n_rpm": [4000, 4000, 5000, 5000],
            "fz_mm_per_tooth": [0.03, 0.06, 0.03, 0.06],
            "ap_mm": [0.5, 0.5, 1.0, 1.0],
            "version": ["v3", "v3", "v4", "v4"],
            "region_index": [1, 2, 1, 2],
        }
    )

    paths = write_error_breakdowns(predictions, manifest, tmp_path)

    assert set(paths) == {
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
        "version",
        "region_index",
    }
    assert all(item["csv"].is_file() for item in paths.values())
    assert all(item["figure"].is_file() for item in paths.values())
