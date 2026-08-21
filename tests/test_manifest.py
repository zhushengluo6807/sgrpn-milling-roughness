from pathlib import Path

import numpy as np
import pytest

from roughness.config import load_config
from roughness.manifest import build_manifest, geometry_ra_um, relative_after_segments


def test_relative_after_segments_preserves_nested_suffix():
    source = r"切削实验\预处理并切分后实验数据\segments\500\3_seg1.csv"

    assert relative_after_segments(source) == Path("500") / "3_seg1.csv"


def test_relative_after_segments_rejects_path_without_marker():
    with pytest.raises(ValueError, match="does not contain segments"):
        relative_after_segments(r"切削实验\500\3_seg1.csv")


def test_geometry_ra_converts_mm_to_um():
    assert geometry_ra_um(0.15, radius_mm=5.0) == pytest.approx(0.140625)


def test_real_manifest_has_expected_shape_and_ranges():
    cfg = load_config(Path("configs/first_round.yaml"))

    manifest, audit = build_manifest(cfg)

    assert len(manifest) == 586
    assert manifest["group_id"].nunique() == 212
    assert manifest["sample_id"].is_unique
    assert audit["missing_signal_files"] == 0
    assert audit["n_range"] == [4000.0, 8500.0]
    assert audit["fz_range"] == [0.03, 0.15]
    assert audit["ap_range"] == [0.5, 2.0]
    assert np.allclose(manifest.groupby("group_id")["sample_weight"].sum(), 1.0)
    assert manifest["signal_path"].map(Path.is_file).all()
