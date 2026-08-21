from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch

from roughness.scheme1_physics.config import load_scheme1_physics_config
from roughness.scheme1_physics.physics import ScaleSelection
from roughness.scheme1_physics.training import (
    completed_physics_run_matches,
    ordinary_parent_model,
    physics_model_formula,
    physics_model_mode,
    train_gated_physics_fold,
    train_physics_fold,
    write_physics_baseline_oof,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG = load_scheme1_physics_config(
    PROJECT_ROOT / "configs" / "scheme1_physics.yaml"
)


@pytest.mark.parametrize(
    ("model", "formula", "mode"),
    [
        ("PW0", "word", "baseline"),
        ("PW1", "word", "ordinary"),
        ("PW2", "word", "gated"),
        ("PE0", "exact", "baseline"),
        ("PE1", "exact", "ordinary"),
        ("PE2", "exact", "gated"),
    ],
)
def test_model_mapping_is_fixed(model, formula, mode):
    assert physics_model_formula(model) == formula
    assert physics_model_mode(model) == mode


def test_unknown_model_mapping_is_rejected():
    with pytest.raises(ValueError, match="Unknown physics model"):
        physics_model_formula("M0")


def test_baseline_oof_has_all_rows_once_per_model_seed(tmp_path, monkeypatch):
    config = replace(CONFIG, output_dir=tmp_path / "physics")
    predictions, selections, audits = write_physics_baseline_oof(
        config, force=True
    )

    manifest = pd.read_csv(config.source.manifest_path)
    expected = len(manifest) * len(config.source.seeds) * 2
    assert len(predictions) == expected
    assert set(predictions["model"]) == {"PW0", "PE0"}
    assert not predictions.duplicated(
        ["sample_id", "model", "seed"]
    ).any()
    assert set(selections["formula"]) == {"word", "exact"}
    assert set(audits["candidate_radius_mm"]) == set(
        config.source.re_candidates_mm
    )
    assert (
        tmp_path / "physics" / "physics" / "oof_predictions.csv"
    ).is_file()


def test_train_ordinary_fold_writes_complete_oof(tmp_path, monkeypatch):
    import roughness.scheme1_physics.training as training

    selection = ScaleSelection(
        formula="word",
        outer_fold=0,
        seed=20260723,
        radius_mm=0.2,
        inner_weighted_mae=0.1,
        source_sample_ids=("a", "b"),
    )
    prepared = SimpleNamespace(selection=selection)
    batch = {
        "signal": torch.randn(2, 1, 3, 256),
        "window_mask": torch.ones(2, 1, dtype=torch.bool),
        "process": torch.randn(2, 3),
        "physics": torch.randn(2, 1),
        "base_ra": torch.tensor([0.2, 0.4]),
        "target": torch.tensor([0.3, 0.5]),
        "sample_weight": torch.ones(2),
        "segment_id": ["a", "b"],
        "group_id": ["g1", "g2"],
    }
    loaders = SimpleNamespace(
        train_loader=[batch],
        validation_loader=[batch],
        test_loader=[batch],
    )
    monkeypatch.setattr(training, "prepare_physics_fold", lambda *a, **k: prepared)
    monkeypatch.setattr(training, "build_physics_loaders", lambda *a, **k: loaders)
    config = replace(CONFIG, output_dir=tmp_path / "physics")

    result = train_physics_fold(
        "PW1",
        outer_fold=0,
        seed=20260723,
        config=config,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
    )
    oof = pd.read_csv(result.oof_path)
    checkpoint = torch.load(
        result.checkpoint_path, map_location="cpu", weights_only=False
    )

    assert {"base_ra", "residual", "formula", "radius_mm"} <= set(oof)
    assert oof["formula"].eq("word").all()
    assert checkpoint["model"] == "PW1"
    assert checkpoint["formula"] == "word"
    assert checkpoint["run_fingerprint"]


def test_gate_parent_mapping_is_strict():
    assert ordinary_parent_model("PW2") == "PW1"
    assert ordinary_parent_model("PE2") == "PE1"
    with pytest.raises(ValueError, match="gated model"):
        ordinary_parent_model("PW1")


def test_gated_fold_preserves_parent_and_writes_gate_identity(
    tmp_path, monkeypatch
):
    import numpy as np
    import roughness.scheme1_physics.training as training

    selection = ScaleSelection(
        formula="word",
        outer_fold=0,
        seed=20260723,
        radius_mm=0.2,
        inner_weighted_mae=0.1,
        source_sample_ids=("a", "b"),
    )
    prepared = SimpleNamespace(selection=selection)
    batch = {
        "signal": torch.randn(2, 1, 3, 256),
        "window_mask": torch.ones(2, 1, dtype=torch.bool),
        "process": torch.randn(2, 3),
        "physics": torch.randn(2, 1),
        "base_ra": torch.tensor([0.2, 0.4]),
        "target": torch.tensor([0.3, 0.5]),
        "sample_weight": torch.ones(2),
        "segment_id": ["a", "b"],
        "group_id": ["g1", "g2"],
    }
    loaders = SimpleNamespace(
        train_loader=[batch],
        validation_loader=[batch],
        test_loader=[batch],
    )
    monkeypatch.setattr(training, "prepare_physics_fold", lambda *a, **k: prepared)
    monkeypatch.setattr(training, "build_physics_loaders", lambda *a, **k: loaders)
    config = replace(CONFIG, output_dir=tmp_path / "physics")
    parent = train_physics_fold(
        "PW1",
        0,
        20260723,
        config,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
    )
    result = train_gated_physics_fold(
        "PW2",
        0,
        20260723,
        config,
        parent_checkpoint=parent.checkpoint_path,
        device=torch.device("cpu"),
        max_epochs=1,
        batch_size=2,
    )
    oof = pd.read_csv(result.oof_path)
    reconstructed = oof["base_ra"] + (1.0 - oof["gate"]) * oof["residual"]

    assert oof["gate"].between(0, 1).all()
    assert np.allclose(oof["y_pred"], reconstructed, atol=1e-6)
    assert (result.checkpoint_path.parent / "frozen_parameter_audit.json").is_file()
    checkpoint = torch.load(
        result.checkpoint_path, map_location="cpu", weights_only=False
    )
    assert checkpoint["parent_checkpoint_fingerprint"]
    assert completed_physics_run_matches(
        result.checkpoint_path.parent, checkpoint["run_fingerprint"]
    )
