import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roughness.scheme1.features import (
    aggregate_window_features,
    band_limited_displacement_proxy,
    bottom_geometry_ra_um,
    build_segment_feature_table,
    extract_segment_feature_row,
    extract_channel_features,
    fit_feature_scaler,
    horizontal_feature_views,
    write_feature_artifacts,
)


def test_bottom_geometry_uses_effective_edge_radius_and_exact_arc():
    value = bottom_geometry_ra_um(fz_mm=0.09, re_mm=0.20)

    assert value == pytest.approx(1.282061825237, rel=1e-12)
    assert value > 1.265625  # registered small-feed approximation


@pytest.mark.parametrize(
    ("fz_mm", "re_mm"),
    [(0.20, 0.075), (0.01, 0.0), (0.01, -0.1)],
)
def test_bottom_geometry_rejects_invalid_generation_geometry(fz_mm, re_mm):
    with pytest.raises(ValueError):
        bottom_geometry_ra_um(fz_mm=fz_mm, re_mm=re_mm)


def test_bottom_geometry_increases_with_feed_and_decreases_with_radius():
    low_feed = bottom_geometry_ra_um(0.03, 0.20)
    high_feed = bottom_geometry_ra_um(0.09, 0.20)
    large_radius = bottom_geometry_ra_um(0.09, 0.80)

    assert high_feed > low_feed
    assert large_radius < high_feed


def test_displacement_proxy_matches_single_frequency_in_relative_units():
    sample_rate = 1000
    time = np.arange(2000) / sample_rate
    acceleration = np.sin(2 * np.pi * 50 * time)

    proxy = band_limited_displacement_proxy(
        acceleration,
        sample_rate_hz=sample_rate,
        low_hz=20.0,
        high_hz=200.0,
    )

    expected = (1 / math.sqrt(2)) / (2 * math.pi * 50) ** 2
    assert proxy == pytest.approx(expected, rel=0.01)
    assert np.isfinite(proxy)


def test_channel_features_find_nominal_rotation_and_tpf_energy():
    sample_rate = 2000
    time = np.arange(4000) / sample_rate
    signal = (
        np.sin(2 * np.pi * 100 * time)
        + 0.5 * np.sin(2 * np.pi * 300 * time)
    )

    features = extract_channel_features(
        signal,
        sample_rate_hz=sample_rate,
        n_rpm=6000,
        teeth=3,
        welch_nperseg=1024,
        welch_noverlap=512,
        hf_band_hz=(700.0, 900.0),
    )

    assert features["rms"] == pytest.approx(math.sqrt(0.625), rel=0.01)
    assert features["rotation_energy"] > features["hf_energy"]
    assert features["tpf_energy"] > features["hf_energy"]
    assert features["tpf_h2_energy"] < features["tpf_energy"]
    assert 0.0 <= features["spectral_entropy"] <= 1.0
    assert all(np.isfinite(value) for value in features.values())


def test_horizontal_symmetric_features_ignore_channel_order():
    sample_rate = 1000
    time = np.arange(2000) / sample_rate
    ch9 = np.sin(2 * np.pi * 50 * time)
    ch10 = 2 * np.sin(2 * np.pi * 80 * time)

    forward = horizontal_feature_views(
        ch9,
        ch10,
        sample_rate_hz=sample_rate,
        n_rpm=3000,
        welch_nperseg=512,
        welch_noverlap=256,
        hf_band_hz=(300.0, 450.0),
    )
    swapped = horizontal_feature_views(
        ch10,
        ch9,
        sample_rate_hz=sample_rate,
        n_rpm=3000,
        welch_nperseg=512,
        welch_noverlap=256,
        hf_band_hz=(300.0, 450.0),
    )

    assert forward["A_X_rms"] == pytest.approx(swapped["B_X_rms"])
    assert forward["A_Y_rms"] == pytest.approx(swapped["B_Y_rms"])
    symmetric_keys = [key for key in forward if key.startswith("H_sym_")]
    assert symmetric_keys
    for key in symmetric_keys:
        assert forward[key] == pytest.approx(swapped[key])


def test_aggregate_window_features_emits_mean_std_and_max():
    rows = [
        {"Z_rms": 1.0, "Z_kurtosis": 2.0},
        {"Z_rms": 3.0, "Z_kurtosis": 6.0},
    ]

    aggregated = aggregate_window_features(rows)

    assert aggregated == {
        "Z_kurtosis_mean": 4.0,
        "Z_kurtosis_std": 2.0,
        "Z_kurtosis_max": 6.0,
        "Z_rms_mean": 2.0,
        "Z_rms_std": 1.0,
        "Z_rms_max": 3.0,
    }


def test_segment_features_include_z_horizontal_and_registered_geometry():
    sample_rate = 1000
    time = np.arange(2000) / sample_rate
    signal = np.column_stack(
        [
            np.sin(2 * np.pi * 50 * time),
            np.sin(2 * np.pi * 80 * time),
            np.sin(2 * np.pi * 100 * time),
        ]
    )

    row = extract_segment_feature_row(
        signal,
        windows=[(0, 1000), (1000, 2000)],
        n_rpm=6000,
        fz_mm=0.09,
        re_candidates_mm=(0.10, 0.20),
        sample_rate_hz=sample_rate,
        welch_nperseg=512,
        welch_noverlap=256,
        hf_band_hz=(300.0, 450.0),
    )

    assert row["Z_rms_mean"] == pytest.approx(1 / math.sqrt(2), rel=0.01)
    assert "H_sym_mean_rms_mean" in row
    assert row["ra_geo_bottom_re_0p20_um"] == pytest.approx(1.282061825237)
    assert all(np.isfinite(value) for value in row.values())


def test_feature_scaler_fits_only_named_training_segments():
    frame = pd.DataFrame(
        {
            "sample_id": ["a", "b", "held_out"],
            "f1": [0.0, 2.0, 1000.0],
            "f2": [10.0, 14.0, 2000.0],
        }
    )

    scaler = fit_feature_scaler(frame, ["f1", "f2"], {"a", "b"})
    transformed = scaler.transform(frame.loc[:1, ["f1", "f2"]])

    assert scaler.source_segment_ids == ("a", "b")
    np.testing.assert_allclose(scaler.mean, [1.0, 12.0])
    np.testing.assert_allclose(transformed, [[-1.0, -1.0], [1.0, 1.0]])


def test_build_and_write_segment_feature_artifacts(tmp_path):
    sample_rate = 1000
    time = np.arange(2000) / sample_rate
    signal_path = tmp_path / "signal.csv"
    pd.DataFrame(
        {
            "Time_s": time,
            "Ch9_g": np.sin(2 * np.pi * 50 * time),
            "Ch10_g": np.sin(2 * np.pi * 80 * time),
            "Ch11_g": np.sin(2 * np.pi * 100 * time),
        }
    ).to_csv(signal_path, index=False)
    manifest = pd.DataFrame(
        {
            "sample_id": ["s1"],
            "group_id": ["g1"],
            "signal_path": [signal_path],
            "n_rpm": [6000.0],
            "fz_mm_per_tooth": [0.09],
            "ap_mm": [1.0],
            "ra_mean": [1.2],
            "sample_weight": [1.0],
        }
    )
    window_index = pd.DataFrame(
        {
            "segment_id": ["s1", "s1"],
            "start_sample": [0, 1000],
            "end_sample": [1000, 2000],
        }
    )

    table = build_segment_feature_table(
        manifest,
        window_index,
        re_candidates_mm=(0.20,),
        sample_rate_hz=sample_rate,
        welch_nperseg=512,
        welch_noverlap=256,
        hf_band_hz=(300.0, 450.0),
    )
    written = write_feature_artifacts(table, tmp_path / "features")

    assert table.loc[0, "sample_id"] == "s1"
    assert table.loc[0, "ra_geo_bottom_re_0p20_um"] == pytest.approx(
        1.282061825237
    )
    assert written["features"].is_file()
    schema = json.loads(written["schema"].read_text(encoding="utf-8"))
    assert schema["ra_geo_bottom_re_0p20_um"]["unit"] == "um"
    displacement_keys = [
        key for key in schema if "displacement_proxy_relative" in key
    ]
    assert displacement_keys
    assert all(schema[key]["unit"] == "relative_unit" for key in displacement_keys)
