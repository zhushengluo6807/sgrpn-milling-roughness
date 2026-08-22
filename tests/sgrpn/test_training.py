from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from roughness.sgrpn.config import SGRPNConfig
from roughness.sgrpn.data import DataBundle
from roughness.sgrpn.order_spectrum import OrderSpectrumCache
from roughness.sgrpn.training import (
    EpochSelection,
    MODEL_SEQUENCE,
    TrainingResult,
    completed_fold_matches,
    run_phase_a_fold,
)


EXPECTED_COLUMNS = {
    "sample_id",
    "group_id",
    "version",
    "fold",
    "seed",
    "model",
    "target",
    "prediction",
    "sample_weight",
    "process_mean",
    "residual",
    "gate",
}


def _fixture(tmp_path: Path, *, max_epochs: int = 2) -> tuple[SGRPNConfig, DataBundle, OrderSpectrumCache]:
    sample_ids = [f"s{index}" for index in range(10)]
    group_ids = [f"g{index}" for index in range(10)]
    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_id": group_ids,
            "signal_path": ["unused.csv"] * 10,
            "n_rpm": np.linspace(4000.0, 8000.0, 10),
            "fz_mm_per_tooth": np.linspace(0.03, 0.12, 10),
            "ap_mm": np.linspace(0.5, 2.0, 10),
            "ra_1": np.linspace(0.2, 1.1, 10),
            "ra_2": np.linspace(0.2, 1.1, 10),
            "ra_3": np.linspace(0.2, 1.1, 10),
            "ra_mean": np.linspace(0.2, 1.1, 10),
            "sample_weight": np.ones(10),
            "split_count": np.ones(10, dtype=int),
            "version": ["v3"] * 5 + ["v4"] * 5,
            "duration_s": np.ones(10),
        }
    )
    folds = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_id": group_ids,
            "fold": np.arange(10) % 5,
        }
    )
    windows = pd.DataFrame(
        {
            "segment_id": sample_ids,
            "group_id": group_ids,
            "csv_path": ["unused.csv"] * 10,
            "window_id": np.zeros(10, dtype=int),
            "start_sample": np.zeros(10, dtype=int),
            "end_sample": np.full(10, 25600, dtype=int),
            "is_tail_aligned": np.zeros(10, dtype=bool),
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    folds_path = tmp_path / "folds.csv"
    windows_path = tmp_path / "windows.csv"
    m0_path = tmp_path / "m0.csv"
    manifest.to_csv(manifest_path, index=False)
    folds.to_csv(folds_path, index=False)
    windows.to_csv(windows_path, index=False)
    m0_path.write_text("sample_id,prediction\n", encoding="utf-8")
    config = SGRPNConfig(
        manifest_path=manifest_path,
        folds_path=folds_path,
        window_index_path=windows_path,
        m0_oof_path=m0_path,
        output_dir=tmp_path / "outputs" / "sgrpn" / "phase_a",
        sample_rate_hz=25600,
        window_samples=25600,
        order_min=0.0,
        order_max=90.0,
        order_step=0.25,
        seeds=(20260723,),
        inner_splits=4,
        max_epochs=max_epochs,
        patience=1,
        process_learning_rate=1e-3,
        signal_learning_rate=3e-4,
        gate_learning_rate=1e-3,
        weight_decay=1e-4,
        huber_delta_um=0.10,
        gate_penalty=1e-3,
        correction_penalty=1e-2,
        bootstrap_repetitions=10_000,
    )
    bundle = DataBundle(
        manifest=manifest,
        folds=folds,
        windows=windows,
        fold_audit={},
        duration_audit=pd.DataFrame(),
    )
    rng = np.random.default_rng(73)
    cache = OrderSpectrumCache(
        segment_ids=tuple(sample_ids),
        spectra=rng.normal(size=(10, 3, 361)).astype(np.float32),
        offsets=np.arange(11, dtype=np.int64),
        quality=rng.normal(size=(10, 7)).astype(np.float32),
        durations_s=np.ones(10),
    )
    return config, bundle, cache


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def fit(
        self,
        *,
        stage,
        model,
        train_loader,
        validation_loader,
        max_epochs,
        patience,
        learning_rate,
        weight_decay,
        loss_name,
        device,
        seed,
    ) -> TrainingResult:
        del patience, learning_rate, weight_decay, loss_name, seed
        train_batches = list(train_loader)
        valid_batches = [] if validation_loader is None else list(validation_loader)
        train_ids = [item for batch in train_batches for item in batch["sample_id"]]
        valid_ids = [item for batch in valid_batches for item in batch["sample_id"]]
        train_targets = torch.cat([batch["target"] for batch in train_batches]).numpy()
        valid_targets = (
            np.empty(0)
            if not valid_batches
            else torch.cat([batch["target"] for batch in valid_batches]).numpy()
        )
        if stage == "G1":
            assert not any(parameter.requires_grad for parameter in model.process_expert.parameters())
            assert not any(parameter.requires_grad for parameter in model.residual_expert.parameters())
            assert not model.process_expert.training
            assert not model.residual_expert.training
            assert any(parameter.requires_grad for parameter in model.gate.parameters())
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.zero_()
        self.calls.append(
            {
                "stage": stage,
                "train_ids": train_ids,
                "valid_ids": valid_ids,
                "train_targets": train_targets,
                "valid_targets": valid_targets,
                "device": str(device),
            }
        )
        return TrainingResult(
            model=model,
            best_epoch=max(1, min(int(max_epochs), 2)),
            history=pd.DataFrame(
                [{"epoch": 1, "train_loss": 0.0, "validation_loss": 0.0}]
            ),
        )


def test_epoch_selection_uses_rounded_median_inner_best_epoch():
    assert EpochSelection(best_epochs=(3, 9, 5, 7)).refit_epochs == 6


def test_resume_requires_exact_fingerprint_fold_seed_and_model_set(tmp_path: Path):
    marker = tmp_path / "complete.json"
    marker.write_text(
        json.dumps(
            {
                "status": "complete",
                "fingerprint": "current",
                "fold": 2,
                "seed": 20260723,
                "models": list(MODEL_SEQUENCE),
                "completed_stages": list(MODEL_SEQUENCE),
            }
        ),
        encoding="utf-8",
    )
    assert completed_fold_matches(marker, "current", fold=2, seed=20260723)
    assert not completed_fold_matches(marker, "old", fold=2, seed=20260723)
    assert not completed_fold_matches(marker, "current", fold=1, seed=20260723)
    assert not completed_fold_matches(marker, "current", fold=2, seed=1)


def test_fold_orchestration_is_leakage_safe_and_uses_fixed_stage_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    config, bundle, cache = _fixture(tmp_path)
    backend = RecordingBackend()
    test_ids = set(bundle.folds.loc[bundle.folds["fold"] == 0, "sample_id"])
    train = bundle.manifest.loc[~bundle.manifest["sample_id"].isin(test_ids)]
    expected_residual = np.sort(train["ra_mean"].to_numpy())

    import roughness.sgrpn.training as training

    original_reader = training._read_outer_targets

    def guarded_reader(frame):
        stages = [call["stage"] for call in backend.calls]
        assert stages
        assert stages[-1] == "G1"
        assert set(stages) == set(MODEL_SEQUENCE)
        return original_reader(frame)

    monkeypatch.setattr(training, "_read_outer_targets", guarded_reader)
    artifacts = run_phase_a_fold(
        config,
        bundle,
        cache,
        fold=0,
        seed=20260723,
        device="cpu",
        backend=backend,
        batch_size=4,
    )

    collapsed = [
        stage
        for index, stage in enumerate(call["stage"] for call in backend.calls)
        if index == 0 or stage != backend.calls[index - 1]["stage"]
    ]
    assert collapsed == list(MODEL_SEQUENCE)
    assert all(test_ids.isdisjoint(call["train_ids"]) for call in backend.calls)
    assert all(test_ids.isdisjoint(call["valid_ids"]) for call in backend.calls)
    r1_validation_targets = np.concatenate(
        [call["valid_targets"] for call in backend.calls if call["stage"] == "R1" and len(call["valid_targets"])]
    )
    np.testing.assert_allclose(np.sort(r1_validation_targets), expected_residual)
    assert set(artifacts.predictions.columns) == EXPECTED_COLUMNS
    assert set(artifacts.predictions["model"]) == set(MODEL_SEQUENCE)
    assert len(artifacts.predictions) == len(test_ids) * len(MODEL_SEQUENCE)
    assert not artifacts.predictions.duplicated(["sample_id", "model"]).any()
    assert np.isfinite(artifacts.predictions[["target", "prediction", "sample_weight"]]).all().all()
    gates = artifacts.predictions.loc[artifacts.predictions["model"] == "G1", "gate"]
    assert gates.between(0.0, 1.0).all()
    assert all(call["device"] == "cpu" for call in backend.calls)


def test_fold_checkpoint_is_atomic_and_exact_completed_fold_is_reused(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path)
    backend = RecordingBackend()
    first = run_phase_a_fold(
        config, bundle, cache, fold=0, seed=20260723, device="cpu", backend=backend
    )
    fold_dir = config.output_dir / "folds" / "fold_0" / "seed_20260723"
    marker = fold_dir / "complete.json"
    assert completed_fold_matches(marker, first.fingerprint.value, fold=0, seed=20260723)
    assert not list(config.output_dir.rglob("*.tmp-*"))

    class FailIfCalled:
        def fit(self, **kwargs):
            raise AssertionError(f"resume retrained {kwargs['stage']}")

    resumed = run_phase_a_fold(
        config, bundle, cache, fold=0, seed=20260723, device="cpu", backend=FailIfCalled()
    )
    pd.testing.assert_frame_equal(resumed.predictions, first.predictions)

    with pytest.raises(ValueError, match="fingerprint|incompatible"):
        run_phase_a_fold(
            replace(config, max_epochs=1),
            bundle,
            cache,
            fold=0,
            seed=20260723,
            device="cpu",
            backend=FailIfCalled(),
        )


def test_real_cpu_pipeline_uses_same_path_and_writes_only_phase_a_outputs(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path, max_epochs=2)
    artifacts = run_phase_a_fold(
        config,
        bundle,
        cache,
        fold=0,
        seed=20260723,
        device="cpu",
        batch_size=8,
    )

    assert np.isfinite(artifacts.predictions[["target", "prediction", "sample_weight"]]).all().all()
    assert artifacts.predictions.loc[artifacts.predictions.model == "G1", "gate"].between(0, 1).all()
    assert artifacts.checkpoint_paths.keys() == set(MODEL_SEQUENCE)
    assert artifacts.history_paths.keys() == set(MODEL_SEQUENCE)
    assert list(config.output_dir.rglob("*scaler*.npz"))
    assert not (tmp_path / "outputs" / "scheme1").exists()
    assert not (tmp_path / "outputs" / "scheme1_physics").exists()
