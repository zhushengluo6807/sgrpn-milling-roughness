from pathlib import Path
from types import SimpleNamespace

import pytest

import json

from roughness.scheme1.config import (
    load_scheme1_config,
    write_protocol_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "scheme1.yaml"


def test_scheme1_config_resolves_paths_from_config_directory(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)

    config = load_scheme1_config(CONFIG_PATH)

    assert config.segments_dir.is_dir()
    assert config.manifest_path.is_file()
    assert config.folds_path.is_file()
    assert config.output_dir.samefile(PROJECT_ROOT / "outputs" / "scheme1")


def test_scheme1_config_exposes_frozen_protocol():
    config = load_scheme1_config(CONFIG_PATH)

    assert config.sample_rate_hz == 25_600
    assert config.window_samples == 25_600
    assert config.stride_samples == 25_600
    assert config.re_candidates_mm == (0.075, 0.10, 0.20, 0.40, 0.80)
    assert config.seeds == (20260723, 20260724, 20260725)
    assert config.inner_splits == 4
    assert config.bootstrap_repetitions == 10_000


def test_scheme1_config_rejects_invalid_welch_overlap(tmp_path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "\n".join(
            [
                f"segments_dir: {PROJECT_ROOT / '切削实验'}",
                f"manifest_path: {PROJECT_ROOT / 'outputs/first_round_baseline/manifest.csv'}",
                f"folds_path: {PROJECT_ROOT / 'outputs/first_round_baseline/folds.csv'}",
                f"output_dir: {tmp_path / 'out'}",
                "sample_rate_hz: 25600",
                "window_samples: 25600",
                "stride_samples: 25600",
                "re_candidates_mm: [0.075]",
                "seeds: [20260723]",
                "inner_splits: 4",
                "welch_nperseg: 8192",
                "welch_noverlap: 8192",
                "nominal_band_min_halfwidth_hz: 5.0",
                "nominal_band_relative_halfwidth: 0.05",
                "hf_band_hz: [1000.0, 10000.0]",
                "max_epochs: 200",
                "patience: 20",
                "learning_rate: 0.001",
                "weight_decay: 0.0001",
                "bootstrap_repetitions: 10000",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="welch_noverlap"):
        load_scheme1_config(path)


def test_write_protocol_checkpoint_records_fold_audit_and_config(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "roughness.scheme1.config.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(returncode=0),
    )
    config = load_scheme1_config(CONFIG_PATH)

    checkpoint = write_protocol_checkpoint(
        config,
        {"n_samples": 586, "n_groups": 212, "n_folds": 5},
        output_path=tmp_path / "run_manifest.json",
    )

    saved = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert saved["protocol"]["sample_rate_hz"] == 25_600
    assert saved["protocol"]["seeds"] == [20260723, 20260724, 20260725]
    assert saved["outer_folds"]["n_groups"] == 212
    assert saved["git_available"] is True
