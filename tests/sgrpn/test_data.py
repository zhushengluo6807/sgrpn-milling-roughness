from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roughness.sgrpn.config import SGRPNConfig
from roughness.sgrpn.data import (
    build_process_features,
    load_data_bundle,
    outer_indices,
    recompute_duration,
    validate_group_split,
    write_data_audit,
)


def _config(tmp_path: Path, manifest: Path, folds: Path, windows: Path) -> SGRPNConfig:
    m0_oof = tmp_path / "m0_oof.csv"
    m0_oof.write_text("sample_id,prediction\n", encoding="utf-8")
    return SGRPNConfig(
        manifest_path=manifest,
        folds_path=folds,
        window_index_path=windows,
        m0_oof_path=m0_oof,
        output_dir=tmp_path / "outputs",
        sample_rate_hz=25600,
        window_samples=25600,
        order_min=0.0,
        order_max=90.0,
        order_step=0.25,
        seeds=(20260723,),
        inner_splits=4,
        max_epochs=200,
        patience=20,
        process_learning_rate=0.001,
        signal_learning_rate=0.0003,
        gate_learning_rate=0.001,
        weight_decay=0.0001,
        huber_delta_um=0.1,
        gate_penalty=0.001,
        correction_penalty=0.01,
        bootstrap_repetitions=10,
    )


def _write_signal(path: Path) -> None:
    values = np.arange(25600, dtype=float)
    pd.DataFrame(
        {"Ch9_g": values, "Ch10_g": values + 1.0, "Ch11_g": values + 2.0}
    ).to_csv(path, index=False)


def _fixture_paths(tmp_path: Path) -> tuple[SGRPNConfig, Path]:
    signal_a = tmp_path / "a.csv"
    signal_b = tmp_path / "b.csv"
    _write_signal(signal_a)
    _write_signal(signal_b)
    manifest = tmp_path / "manifest.csv"
    rows = []
    for fold in range(5):
        rows.append(
            {
                "sample_id": f"s{fold}",
                "group_id": f"g{fold}",
                "signal_path": str(signal_a if fold % 2 == 0 else signal_b),
                "n_rpm": 4000.0,
                "fz_mm_per_tooth": 0.05,
                "ap_mm": 1.0,
                "ra_1": 0.8,
                "ra_2": 0.9,
                "ra_3": 1.0,
                "ra_mean": 0.9,
                "sample_weight": 1.0,
                "split_count": 1,
                "version": "v3",
                "duration_s": 1.1 if fold == 0 else 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(manifest, index=False)
    folds = tmp_path / "folds.csv"
    pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(5)], "group_id": [f"g{i}" for i in range(5)], "fold": range(5)}
    ).to_csv(folds, index=False)
    windows = tmp_path / "windows.csv"
    pd.DataFrame(
        {
            "segment_id": [f"s{i}" for i in range(5)],
            "group_id": [f"g{i}" for i in range(5)],
            "csv_path": [str(signal_a if i % 2 == 0 else signal_b) for i in range(5)],
            "window_id": [0] * 5,
            "start_sample": [0] * 5,
            "end_sample": [25600] * 5,
            "is_tail_aligned": [False] * 5,
        }
    ).to_csv(windows, index=False)
    return _config(tmp_path, manifest, folds, windows), manifest


def test_process_features_are_exactly_nine_columns():
    frame = pd.DataFrame({"n_rpm": [4000.0], "fz_mm_per_tooth": [0.05], "ap_mm": [1.0]})

    features = build_process_features(frame)

    np.testing.assert_allclose(features[0], [4000.0, 0.05, 1.0, 16_000_000.0, 0.0025, 1.0, 200.0, 4000.0, 0.05])
    assert features.shape == (1, 9)


def test_duration_uses_row_count_not_metadata():
    assert recompute_duration(row_count=25600, sample_rate_hz=25600) == 1.0


def test_group_overlap_fails_fast():
    with pytest.raises(ValueError, match="group leakage"):
        validate_group_split(np.array(["g1", "g2"]), np.array(["g2", "g3"]))


def test_load_records_duration_mismatch_and_preserves_manifest(tmp_path: Path):
    config, manifest = _fixture_paths(tmp_path)
    original_bytes = manifest.read_bytes()

    bundle = load_data_bundle(config)

    audit = bundle.duration_audit.set_index("sample_id")
    assert audit.at["s0", "duration_recomputed_s"] == 1.0
    assert audit.at["s0", "mismatch_over_1ms"]
    assert manifest.read_bytes() == original_bytes
    train, test = outer_indices(bundle, fold=0)
    assert set(train) == {1, 2, 3, 4}
    assert test.tolist() == [0]
    audit_path = write_data_audit(bundle, tmp_path / "audit" / "duration.csv")
    assert pd.read_csv(audit_path).shape == (5, 6)


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [("ra_2", np.nan, "ra_2"), ("n_rpm", np.inf, "finite")],
)
def test_load_rejects_missing_repeat_and_non_finite_process(
    tmp_path: Path, column: str, value: float, message: str
):
    config, _ = _fixture_paths(tmp_path)
    frame = pd.read_csv(config.manifest_path)
    frame.loc[0, column] = value
    frame.to_csv(config.manifest_path, index=False)

    with pytest.raises(ValueError, match=message):
        load_data_bundle(config)
