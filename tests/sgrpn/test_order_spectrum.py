import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roughness.sgrpn.config import SGRPNConfig
from roughness.sgrpn.data import DataBundle
from roughness.sgrpn.order_spectrum import (
    OrderSpectrumCache,
    build_order_cache,
    fit_quality_scaler,
    fit_spectrum_scaler,
    load_order_cache,
    order_grid,
    save_order_cache,
    signal_quality_features,
    window_to_order_spectrum,
)


def test_sixth_order_sine_peaks_near_six():
    fs = 25600
    n_rpm = 4000.0
    shaft_hz = n_rpm / 60.0
    t = np.arange(fs) / fs
    one = np.sin(2 * np.pi * 6.0 * shaft_hz * t)
    signal = np.column_stack([one, 0.5 * one, 0.25 * one])

    spectrum = window_to_order_spectrum(signal, n_rpm, fs, 0.0, 90.0, 0.25)
    grid = order_grid(0.0, 90.0, 0.25)

    assert spectrum.shape == (3, 361)
    assert abs(grid[np.argmax(spectrum[0])] - 6.0) <= 0.25


def test_order_spectrum_rejects_non_fixed_window_and_non_positive_rpm():
    window = np.ones((25599, 3))

    with pytest.raises(ValueError, match=r"\[25600, 3\]"):
        window_to_order_spectrum(window, 4000.0, 25600, 0.0, 90.0, 0.25)
    with pytest.raises(ValueError, match="positive"):
        window_to_order_spectrum(np.ones((25600, 3)), 0.0, 25600, 0.0, 90.0, 0.25)


def test_quality_vector_is_horizontal_swap_invariant():
    signal = np.column_stack([
        np.linspace(-1.0, 1.0, 25600),
        np.linspace(1.0, -1.0, 25600),
        np.sin(np.linspace(0.0, 20.0, 25600)),
    ])

    q_ab = signal_quality_features(signal)
    q_ba = signal_quality_features(signal[:, [1, 0, 2]])

    np.testing.assert_allclose(q_ab, q_ba)
    assert q_ab.shape == (7,)


def test_scalers_use_only_requested_segments():
    cache = OrderSpectrumCache(
        segment_ids=("s1", "s2"),
        spectra=np.stack([np.ones((3, 361)), np.full((3, 361), 100.0)]).astype(np.float32),
        offsets=np.array([0, 1, 2], dtype=np.int64),
        quality=np.array([np.zeros(7), np.full(7, 100.0)], dtype=np.float32),
        durations_s=np.ones(2),
    )

    spectrum_scaler = fit_spectrum_scaler(cache, ["s1"])
    quality_scaler = fit_quality_scaler(cache, ["s1"])

    np.testing.assert_allclose(spectrum_scaler.mean, 1.0)
    np.testing.assert_allclose(quality_scaler.mean, 0.0)


def _config(tmp_path: Path, manifest_path: Path, windows_path: Path) -> SGRPNConfig:
    folds_path = tmp_path / "folds.csv"
    folds_path.write_text("sample_id,group_id,fold\ns1,g1,0\n", encoding="utf-8")
    m0_oof_path = tmp_path / "m0.csv"
    m0_oof_path.write_text("sample_id,prediction\n", encoding="utf-8")
    return SGRPNConfig(
        manifest_path=manifest_path,
        folds_path=folds_path,
        window_index_path=windows_path,
        m0_oof_path=m0_oof_path,
        output_dir=tmp_path / "outputs" / "sgrpn" / "phase_a",
        sample_rate_hz=25600,
        window_samples=25600,
        order_min=0.0,
        order_max=90.0,
        order_step=0.25,
        seeds=(20260723,),
        inner_splits=4,
        max_epochs=1,
        patience=1,
        process_learning_rate=0.001,
        signal_learning_rate=0.001,
        gate_learning_rate=0.001,
        weight_decay=0.0,
        huber_delta_um=0.1,
        gate_penalty=0.0,
        correction_penalty=0.0,
        bootstrap_repetitions=1,
    )


def test_build_save_and_load_cache_follow_window_index_and_fingerprint(tmp_path: Path):
    signal_path = tmp_path / "signal.csv"
    t = np.arange(25600, dtype=np.float64) / 25600.0
    sixth_order = np.sin(2 * np.pi * 6.0 * (4000.0 / 60.0) * t)
    samples = np.concatenate([np.zeros(25600), sixth_order])
    pd.DataFrame({"Ch9_g": samples, "Ch10_g": samples * 2, "Ch11_g": samples * 3}).to_csv(signal_path, index=False)
    manifest_path = tmp_path / "manifest.csv"
    manifest = pd.DataFrame({
        "sample_id": ["s1"], "signal_path": [str(signal_path)], "n_rpm": [4000.0], "duration_s": [2.0],
    })
    manifest.to_csv(manifest_path, index=False)
    windows_path = tmp_path / "windows.csv"
    windows = pd.DataFrame({
        "segment_id": ["s1"], "start_sample": [25600], "end_sample": [51200],
    })
    windows.to_csv(windows_path, index=False)
    bundle = DataBundle(
        manifest=manifest,
        folds=pd.DataFrame(),
        windows=windows,
        fold_audit={},
        duration_audit=pd.DataFrame({"mismatch_over_1ms": [True]}),
    )
    config = _config(tmp_path, manifest_path, windows_path)

    cache = build_order_cache(bundle, config)
    assert order_grid(0.0, 90.0, 0.25)[np.argmax(cache.bag("s1")[0, 0])] == 6.0
    np.testing.assert_allclose(cache.durations_s, [2.0])
    npz_path, json_path = save_order_cache(cache, bundle, config)

    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    assert npz_path.parent == config.output_dir / "features"
    assert metadata["window_count"] == 1
    assert metadata["mismatch_count"] == 1
    loaded = load_order_cache(bundle, config)
    np.testing.assert_allclose(loaded.spectra, cache.spectra)

    windows_path.write_text(windows_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        load_order_cache(bundle, config)
