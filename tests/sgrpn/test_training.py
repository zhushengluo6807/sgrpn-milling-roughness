from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from roughness.scheme1.crossfit import make_group_inner_splits

from roughness.sgrpn.config import SGRPNConfig
from roughness.sgrpn.data import DataBundle
from roughness.sgrpn.order_spectrum import OrderSpectrumCache
from roughness.sgrpn.training import (
    ComponentFoldData,
    ComponentRefitData,
    EpochSelection,
    MODEL_SEQUENCE,
    TrainingResult,
    build_run_fingerprint,
    completed_fold_matches,
    refit_component,
    run_phase_a_fold,
    select_epochs_group_cv,
    set_global_seed,
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
    tmp_path.mkdir(parents=True, exist_ok=True)
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


def _expert_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith(("process_expert.", "residual_expert.")):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def _gate_hash(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if name.startswith("gate."):
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


class LabelSensitiveBackend:
    """Make selected state visibly depend on every label exposed to a fit."""

    def __init__(self, outer_train_count: int) -> None:
        self.outer_train_count = outer_train_count
        self.gate_expert_hashes: dict[tuple[str, ...], str] = {}
        self.gate_selected_hashes: dict[tuple[str, ...], str] = {}
        self.outer_p1_oof: dict[str, float] = {}
        self.outer_r1_refit_targets: dict[str, float] = {}
        self.final_p1_prediction: dict[str, float] = {}

    def fit(self, **kwargs) -> TrainingResult:
        stage = kwargs["stage"]
        model = kwargs["model"]
        train_batches = list(kwargs["train_loader"])
        validation_loader = kwargs["validation_loader"]
        valid_batches = [] if validation_loader is None else list(validation_loader)
        train_ids = [item for batch in train_batches for item in batch["sample_id"]]
        valid_ids = [item for batch in valid_batches for item in batch["sample_id"]]
        train_targets = torch.cat([batch["target"] for batch in train_batches])
        valid_targets = (
            torch.empty(0)
            if not valid_batches
            else torch.cat([batch["target"] for batch in valid_batches])
        )
        if stage == "G1" and valid_ids:
            self.gate_expert_hashes[tuple(sorted(valid_ids))] = _expert_hash(model)
        signal = 0.01 * float(torch.tanh(train_targets.mean()))
        if len(valid_targets):
            signal += 0.02 * float(torch.tanh(valid_targets.mean()))
        elif stage == "P1" and len(train_ids) == self.outer_train_count:
            signal += 0.05
        with torch.no_grad():
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.fill_(signal)
        if stage == "G1" and valid_ids:
            self.gate_selected_hashes[tuple(sorted(valid_ids))] = _gate_hash(model)
        if stage == "P1":
            model.eval()
            target_batches = valid_batches if valid_batches else train_batches
            with torch.no_grad():
                for batch in target_batches:
                    prediction = model(batch["process"]).detach().cpu().numpy()
                    if valid_batches and len(train_ids) + len(valid_ids) == self.outer_train_count:
                        destination = self.outer_p1_oof
                    elif not valid_batches and len(train_ids) == self.outer_train_count:
                        destination = self.final_p1_prediction
                    else:
                        continue
                    for sample_id, value in zip(batch["sample_id"], prediction, strict=True):
                        destination[str(sample_id)] = float(value)
        if stage == "R1" and not valid_batches and len(train_ids) == self.outer_train_count:
            for batch in train_batches:
                for sample_id, value in zip(
                    batch["sample_id"], batch["target"].numpy(), strict=True
                ):
                    self.outer_r1_refit_targets[str(sample_id)] = float(value)
        return TrainingResult(
            model=model,
            best_epoch=1,
            history=pd.DataFrame(
                [{"epoch": 1, "train_loss": signal, "validation_loss": signal}]
            ),
        )


def test_epoch_selection_uses_rounded_median_inner_best_epoch():
    assert EpochSelection(best_epochs=(3, 9, 5, 7)).refit_epochs == 6


class _AuditedFactory:
    def __init__(self, frame: pd.DataFrame, *, corrupt_scaler: bool = False) -> None:
        self.audit_frame = frame
        self.corrupt_scaler = corrupt_scaler
        self.calls = 0

    def __call__(self, train_index, valid_index, fold_number):
        self.calls += 1
        ids = self.audit_frame["sample_id"].astype(str).to_numpy()
        scaler_ids = tuple(ids[train_index])
        if self.corrupt_scaler and fold_number == 0:
            scaler_ids = scaler_ids[1:]
        return ComponentFoldData(
            train_data=("train", fold_number),
            validation_data=("valid", fold_number),
            train_sample_ids=tuple(ids[train_index]),
            validation_sample_ids=tuple(ids[valid_index]),
            scaler_source_ids=scaler_ids,
        )


def test_public_selection_preflights_groups_and_audits_scaler_boundaries():
    frame = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(8)], "group_id": [f"g{i}" for i in range(8)]}
    )
    splits = make_group_inner_splits(frame, n_splits=4, seed=20260723)
    factory = _AuditedFactory(frame)
    models = []

    def component_factory(_fold):
        model = nn.Linear(1, 1)
        models.append(model)
        return model

    def train_step(model, fold_data, validation_step, fold_number):
        del model, fold_data, validation_step
        return fold_number + 1

    selection = select_epochs_group_cv(
        component_factory, factory, splits, train_step, lambda *_args: None
    )
    assert tuple(selection.best_epochs) == (1, 2, 3, 4)
    assert selection.refit_epochs == 2
    assert factory.calls == 4
    assert len({id(model) for model in models}) == 4

    corrupt = _AuditedFactory(frame, corrupt_scaler=True)
    with pytest.raises(ValueError, match="scaler.*inner-train|boundary"):
        select_epochs_group_cv(component_factory, corrupt, splits, train_step, None)


@pytest.mark.parametrize("kind", ["overlap", "incomplete", "duplicate_coverage"])
def test_public_selection_rejects_invalid_splits_before_dataset_side_effects(kind):
    frame = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(8)], "group_id": [f"g{i}" for i in range(8)]}
    )
    splits = make_group_inner_splits(frame, n_splits=4, seed=20260723)
    malformed = [(train.copy(), valid.copy()) for train, valid in splits]
    if kind == "overlap":
        malformed[0] = (np.append(malformed[0][0], malformed[0][1][0]), malformed[0][1])
    elif kind == "incomplete":
        malformed[0] = (malformed[0][0][1:], malformed[0][1])
    else:
        malformed[3] = (malformed[3][0], malformed[0][1].copy())
    factory = _AuditedFactory(frame)

    with pytest.raises(ValueError, match="split|partition|assign|overlap|leakage"):
        select_epochs_group_cv(lambda _fold: nn.Linear(1, 1), factory, malformed, lambda *_args: 1, None)
    assert factory.calls == 0


def test_public_refit_requires_fresh_full_outer_train_scaler_boundary():
    frame = pd.DataFrame(
        {"sample_id": [f"s{i}" for i in range(8)], "group_id": [f"g{i}" for i in range(8)]}
    )
    ids = tuple(frame["sample_id"])

    class RefitFactory:
        audit_frame = frame

        def __init__(self):
            self.calls = 0

        def __call__(self):
            self.calls += 1
            return ComponentRefitData("all-outer-train", ids, ids)

    factory = RefitFactory()
    created = []

    def model_factory():
        model = nn.Linear(1, 1)
        created.append(model)
        return model

    def train_step(model, dataset, epochs):
        assert dataset == "all-outer-train"
        assert epochs == 2
        return model

    result = refit_component(
        model_factory, factory, EpochSelection((1, 2, 3, 4)), train_step
    )
    assert result is created[0]
    assert factory.calls == 1


def test_formal_signal_training_routes_through_hardened_selection_and_refit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    import roughness.sgrpn.training as training

    selection_calls = 0
    refit_calls = 0
    registered_selection = training.select_epochs_group_cv
    registered_refit = training.refit_component

    def selection_spy(*args, **kwargs):
        nonlocal selection_calls
        selection_calls += 1
        return registered_selection(*args, **kwargs)

    def refit_spy(*args, **kwargs):
        nonlocal refit_calls
        refit_calls += 1
        return registered_refit(*args, **kwargs)

    monkeypatch.setattr(training, "select_epochs_group_cv", selection_spy)
    monkeypatch.setattr(training, "refit_component", refit_spy)
    config, bundle, cache = _fixture(tmp_path, max_epochs=1)
    run_phase_a_fold(
        config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend(),
        output_root=config.output_dir,
    )

    assert selection_calls >= 2
    assert refit_calls >= 2


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


def test_fingerprint_binds_consumed_frames_and_source_provenance(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path)
    original = build_run_fingerprint(config, bundle, cache)

    changed_bundle = DataBundle(
        manifest=bundle.manifest.copy(), folds=bundle.folds.copy(),
        windows=bundle.windows, fold_audit=bundle.fold_audit,
        duration_audit=bundle.duration_audit,
    )
    changed_bundle.manifest.loc[0, "ra_mean"] += 1.0
    assert build_run_fingerprint(config, changed_bundle, cache).value != original.value

    config.manifest_path.write_text(
        config.manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
    )
    assert build_run_fingerprint(config, bundle, cache).value != original.value


def test_formal_seed_enables_strict_deterministic_algorithms(monkeypatch):
    calls = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda mode, *, warn_only=False: calls.append((mode, warn_only)),
    )
    set_global_seed(20260723)
    assert calls == [(True, False)]


def test_temporary_output_root_requires_explicit_injection(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path, max_epochs=1)
    with pytest.raises(ValueError, match="output.*root|outputs/sgrpn/phase_a"):
        run_phase_a_fold(
            config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend()
        )


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
        output_root=config.output_dir,
    )
    state = json.loads(
        (config.output_dir / "folds" / "fold_0" / "seed_20260723" / "state.json").read_text(encoding="utf-8")
    )
    assert state["completed_stages"] == list(MODEL_SEQUENCE)
    assert all(test_ids.isdisjoint(call["train_ids"]) for call in backend.calls)
    assert all(test_ids.isdisjoint(call["valid_ids"]) for call in backend.calls)
    assert set(artifacts.predictions.columns) == EXPECTED_COLUMNS
    assert set(artifacts.predictions["model"]) == set(MODEL_SEQUENCE)
    assert len(artifacts.predictions) == len(test_ids) * len(MODEL_SEQUENCE)
    assert not artifacts.predictions.duplicated(["sample_id", "model"]).any()
    assert np.isfinite(artifacts.predictions[["target", "prediction", "sample_weight"]]).all().all()
    gates = artifacts.predictions.loc[artifacts.predictions["model"] == "G1", "gate"]
    assert gates.between(0.0, 1.0).all()
    assert all(call["device"] == "cpu" for call in backend.calls)


def test_gate_fold_experts_are_invariant_to_that_validation_groups_labels(tmp_path: Path):
    first_config, first_bundle, first_cache = _fixture(tmp_path / "first", max_epochs=1)
    train = first_bundle.manifest.loc[first_bundle.folds["fold"].to_numpy() != 0].reset_index(drop=True)
    first_split = make_group_inner_splits(train, n_splits=4, seed=20260723)[0][1]
    held_out_ids = tuple(sorted(train.iloc[first_split]["sample_id"].astype(str)))
    first_backend = LabelSensitiveBackend(outer_train_count=len(train))
    run_phase_a_fold(
        first_config, first_bundle, first_cache, fold=0, seed=20260723,
        device="cpu", backend=first_backend, output_root=first_config.output_dir,
    )

    second_config, second_bundle, second_cache = _fixture(tmp_path / "second", max_epochs=1)
    mask = second_bundle.manifest["sample_id"].astype(str).isin(held_out_ids)
    for column in ("ra_1", "ra_2", "ra_3", "ra_mean"):
        second_bundle.manifest.loc[mask, column] += 1000.0
    second_bundle.manifest.to_csv(second_config.manifest_path, index=False)
    second_backend = LabelSensitiveBackend(outer_train_count=len(train))
    run_phase_a_fold(
        second_config, second_bundle, second_cache, fold=0, seed=20260723,
        device="cpu", backend=second_backend, output_root=second_config.output_dir,
    )

    assert first_backend.gate_expert_hashes[held_out_ids] == second_backend.gate_expert_hashes[held_out_ids]
    assert first_backend.gate_selected_hashes[held_out_ids] != second_backend.gate_selected_hashes[held_out_ids]


def test_r1_outer_refit_targets_are_sample_specific_p1_oof_residuals(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path, max_epochs=1)
    outer_train = bundle.manifest.loc[bundle.folds["fold"].to_numpy() != 0]
    backend = LabelSensitiveBackend(outer_train_count=len(outer_train))

    run_phase_a_fold(
        config, bundle, cache, fold=0, seed=20260723, device="cpu", backend=backend,
        output_root=config.output_dir,
    )

    expected = {
        str(row.sample_id): float(row.ra_mean) - backend.outer_p1_oof[str(row.sample_id)]
        for row in outer_train.itertuples(index=False)
    }
    assert backend.outer_r1_refit_targets == pytest.approx(expected)
    assert len(set(round(value, 6) for value in backend.outer_p1_oof.values())) > 1
    assert any(
        not np.isclose(backend.outer_p1_oof[sample_id], backend.final_p1_prediction[sample_id])
        for sample_id in expected
    )


def test_fold_checkpoint_is_atomic_and_exact_completed_fold_is_reused(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path)
    backend = RecordingBackend()
    first = run_phase_a_fold(
        config, bundle, cache, fold=0, seed=20260723, device="cpu", backend=backend,
        output_root=config.output_dir,
    )
    fold_dir = config.output_dir / "folds" / "fold_0" / "seed_20260723"
    marker = fold_dir / "complete.json"
    assert completed_fold_matches(marker, first.fingerprint.value, fold=0, seed=20260723)
    assert not list(config.output_dir.rglob("*.tmp-*"))
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    assert len(metadata["artifacts"]) == 15

    class FailIfCalled:
        def fit(self, **kwargs):
            raise AssertionError(f"resume retrained {kwargs['stage']}")

    resumed = run_phase_a_fold(
        config, bundle, cache, fold=0, seed=20260723, device="cpu", backend=FailIfCalled(),
        output_root=config.output_dir,
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
            output_root=config.output_dir,
        )


def test_resume_deeply_validates_all_stage_artifacts(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path, max_epochs=1)
    run_phase_a_fold(
        config, bundle, cache, fold=0, seed=20260723, device="cpu",
        backend=RecordingBackend(), output_root=config.output_dir,
    )
    fold_dir = config.output_dir / "folds" / "fold_0" / "seed_20260723"
    marker_path = fold_dir / "complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    checkpoint = fold_dir / "checkpoints" / "P1.pt"
    original_checkpoint = checkpoint.read_bytes()

    checkpoint.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="artifact|hash|checkpoint"):
        run_phase_a_fold(config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend(), output_root=config.output_dir)
    checkpoint.write_bytes(original_checkpoint)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload["model"] = "V1"
    torch.save(payload, checkpoint)
    relative = checkpoint.relative_to(fold_dir).as_posix()
    marker["artifacts"][relative] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    with pytest.raises(ValueError, match="metadata|stage|checkpoint"):
        run_phase_a_fold(config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend(), output_root=config.output_dir)


def test_resume_rejects_non_finite_scaler_even_with_matching_file_hash(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path, max_epochs=1)
    run_phase_a_fold(
        config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend(),
        output_root=config.output_dir,
    )
    fold_dir = config.output_dir / "folds" / "fold_0" / "seed_20260723"
    marker_path = fold_dir / "complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    scaler = fold_dir / "scalers" / "G1_scalers.npz"
    with np.load(scaler, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays["gate_quality_mean"] = np.array([np.nan])
    np.savez_compressed(scaler, **arrays)
    relative = scaler.relative_to(fold_dir).as_posix()
    marker["artifacts"][relative] = hashlib.sha256(scaler.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="scaler|schema|finite"):
        run_phase_a_fold(config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend(), output_root=config.output_dir)


def test_resume_preserves_leading_zero_ids_and_rejects_unexpected_oof_rows(tmp_path: Path):
    config, bundle, cache = _fixture(tmp_path, max_epochs=1)
    ids = [f"{index:03d}" for index in range(10)]
    bundle.manifest.loc[:, "sample_id"] = ids
    bundle.folds.loc[:, "sample_id"] = ids
    bundle.windows.loc[:, "segment_id"] = ids
    bundle.manifest.to_csv(config.manifest_path, index=False)
    bundle.folds.to_csv(config.folds_path, index=False)
    bundle.windows.to_csv(config.window_index_path, index=False)
    cache = OrderSpectrumCache(
        segment_ids=tuple(ids), spectra=cache.spectra, offsets=cache.offsets,
        quality=cache.quality, durations_s=cache.durations_s,
    )
    first = run_phase_a_fold(
        config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend(),
        output_root=config.output_dir,
    )
    resumed = run_phase_a_fold(
        config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend(),
        output_root=config.output_dir,
    )
    assert resumed.predictions["sample_id"].tolist() == first.predictions["sample_id"].tolist()
    assert all(len(value) == 3 for value in resumed.predictions["sample_id"])

    fold_dir = config.output_dir / "folds" / "fold_0" / "seed_20260723"
    predictions_path = fold_dir / "oof_predictions.csv"
    corrupted = pd.read_csv(predictions_path, dtype={"sample_id": str, "group_id": str, "version": str})
    corrupted = pd.concat([corrupted, corrupted.iloc[[0]]], ignore_index=True)
    corrupted.loc[len(corrupted) - 1, "sample_id"] = "unexpected"
    corrupted.to_csv(predictions_path, index=False)
    marker_path = fold_dir / "complete.json"
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["predictions_sha256"] = hashlib.sha256(predictions_path.read_bytes()).hexdigest()
    marker_path.write_text(json.dumps(marker), encoding="utf-8")

    with pytest.raises(ValueError, match="OOF|prediction|sample|Cartesian|unexpected"):
        run_phase_a_fold(
            config, bundle, cache, 0, 20260723, "cpu", backend=RecordingBackend(),
            output_root=config.output_dir,
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
        output_root=config.output_dir,
    )

    assert np.isfinite(artifacts.predictions[["target", "prediction", "sample_weight"]]).all().all()
    assert artifacts.predictions.loc[artifacts.predictions.model == "G1", "gate"].between(0, 1).all()
    assert artifacts.checkpoint_paths.keys() == set(MODEL_SEQUENCE)
    assert artifacts.history_paths.keys() == set(MODEL_SEQUENCE)
    assert list(config.output_dir.rglob("*scaler*.npz"))
    assert not (tmp_path / "outputs" / "scheme1").exists()
    assert not (tmp_path / "outputs" / "scheme1_physics").exists()
