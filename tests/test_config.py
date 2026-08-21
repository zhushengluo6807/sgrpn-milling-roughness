from pathlib import Path

import pytest

from roughness.config import load_config


def test_load_config_resolves_existing_paths(tmp_path: Path):
    for name in ["labels.xlsx", "v3.xlsx", "v4.xlsx"]:
        (tmp_path / name).touch()
    (tmp_path / "segments").mkdir()
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "labels: labels.xlsx\n"
        "v3_records: v3.xlsx\n"
        "v4_records: v4.xlsx\n"
        "segments_root: segments\n"
        "output_dir: out\n"
        "seed: 20260723\n"
        "n_splits: 5\n",
        encoding="utf-8",
    )

    cfg = load_config(cfg_path)

    assert cfg.labels == tmp_path / "labels.xlsx"
    assert cfg.output_dir == tmp_path / "out"
    assert cfg.seed == 20260723
    assert cfg.n_splits == 5


def test_load_config_rejects_missing_required_keys(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text("labels: missing.xlsx\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Missing required config keys"):
        load_config(cfg_path)


def test_load_config_rejects_missing_input_path(tmp_path: Path):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        "labels: missing.xlsx\n"
        "v3_records: missing-v3.xlsx\n"
        "v4_records: missing-v4.xlsx\n"
        "segments_root: missing-segments\n"
        "output_dir: out\n"
        "seed: 20260723\n"
        "n_splits: 5\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Input path does not exist"):
        load_config(cfg_path)
