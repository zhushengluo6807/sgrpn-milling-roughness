from __future__ import annotations

from dataclasses import asdict, replace
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from roughness.scheme1.crossfit import make_group_inner_splits
from roughness.sgrpn.config import PhaseAHandoff, PhaseBConfig, SGRPNConfig
from roughness.sgrpn.crossfit import ProcessScaler
from roughness.sgrpn.data import DataBundle, outer_indices
from roughness.sgrpn.models import (
    GlobalScale,
    ModelOutput,
    SelectiveGatedModel,
    VarianceHead,
)
from roughness.sgrpn.order_spectrum import (
    OrderSpectrumCache,
    QualityScaler,
    SpectrumScaler,
)
from roughness.sgrpn.phase_b_training import (
    SCALE_MODELS,
    build_nested_calibration,
    fit_scale_model,
    run_phase_b_fold,
)
from roughness.sgrpn.training import MeanPathArtifacts


EXPECTED_PHASE_B_PREDICTION_COLUMNS = (
    "sample_id",
    "group_id",
    "version",
    "fold",
    "seed",
    "scale_model",
    "target_mean",
    "ra_1",
    "ra_2",
    "ra_3",
    "sample_weight",
    "mu",
    "sigma",
    "gate",
    "correction",
    "raw_lower_90",
    "raw_upper_90",
    "raw_lower_95",
    "raw_upper_95",
    "conformal_q_90",
    "conformal_lower_90",
    "conformal_upper_90",
    "conformal_q_95",
    "conformal_lower_95",
    "conformal_upper_95",
)

EXPECTED_PHASE_B_MEAN_COLUMNS = (
    "sample_id",
    "group_id",
    "version",
    "fold",
    "seed",
    "model",
    "target_mean",
    "prediction",
    "sample_weight",
    "process_mean",
    "residual",
    "gate",
)

EXPECTED_CALIBRATION_SCORE_COLUMNS = (
    "group_id",
    "outer_fold",
    "inner_fold",
    "seed",
    "scale_model",
    "score",
    "region_count",
    "reading_count",
)


class TinyMean(SelectiveGatedModel):
    """Cheap deterministic G1-shaped mean used to exercise the real scale path."""

    def __init__(self, shift: float = 0.0) -> None:
        nn.Module.__init__(self)
        self.shift = nn.Parameter(torch.tensor(float(shift), dtype=torch.float32))

    def forward(self, spectrum, window_mask, process, quality):
        del window_mask
        channel_summary = spectrum.mean(dim=(1, 3))
        embedding = torch.cat(
            (
                channel_summary.repeat(1, 21),
                channel_summary[:, :1],
            ),
            dim=1,
        )
        process_mean = 0.1 * process[:, 0] + self.shift
        gate = torch.sigmoid(quality[:, 0])
        residual = 0.01 * (channel_summary[:, 0] - channel_summary[:, 1])
        prediction = process_mean + gate * residual
        return ModelOutput(
            prediction=prediction,
            process_mean=process_mean,
            residual=residual,
            gate=gate,
            embedding=embedding,
        )


class RecordingScale(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.raw = nn.Parameter(torch.zeros(()))
        self.seen: list[torch.Tensor] = []

    def forward(self, features):
        self.seen.append(features.detach().cpu().clone())
        return (torch.nn.functional.softplus(self.raw) + 1e-4).expand(len(features))


class NoRestoreScale(RecordingScale):
    def load_state_dict(self, state_dict, strict=True, assign=False):
        del state_dict, strict, assign
        raise AssertionError("exact-epoch refit must not restore a selected state")


def _scale_batch() -> dict[str, object]:
    spectrum = torch.zeros(2, 1, 3, 361)
    spectrum[0, 0, 0] = 2.0
    spectrum[0, 0, 1] = 6.0
    spectrum[1, 0, 0] = 8.0
    spectrum[1, 0, 1] = 4.0
    return {
        "spectrum": spectrum,
        "window_mask": torch.ones(2, 1, dtype=torch.bool),
        "process": torch.arange(18, dtype=torch.float32).reshape(2, 9) / 10,
        "quality": torch.arange(14, dtype=torch.float32).reshape(2, 7) / 20,
        "target": torch.tensor([0.5, 0.8]),
        "readings": torch.tensor([[0.4, 0.5, 0.6], [0.7, 0.8, 0.9]]),
        "sample_weight": torch.tensor([0.5, 0.25]),
        "sample_id": ["s0", "s1"],
        "group_id": ["g0", "g1"],
    }


@pytest.mark.parametrize("scale_model", [VarianceHead, GlobalScale])
def test_scale_fit_cannot_change_any_mean_parameter_or_buffer(scale_model):
    mean = TinyMean()
    loader = DataLoader([_scale_batch()], batch_size=None)
    before = {name: value.clone() for name, value in mean.state_dict().items()}

    result = fit_scale_model(
        mean_model=mean,
        scale_model=scale_model(),
        train_loader=loader,
        valid_loader=loader,
        max_epochs=2,
        patience=1,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device=torch.device("cpu"),
        seed=20260723,
    )

    assert all(torch.equal(before[name], value) for name, value in mean.state_dict().items())
    assert not any(parameter.requires_grad for parameter in mean.parameters())
    assert result.best_epoch in (1, 2)
    assert list(result.history.columns) == ["epoch", "train_nll", "validation_nll"]
    assert np.isfinite(result.history[["train_nll", "validation_nll"]]).all().all()
    assert len(result.mean_state_sha256) == 64


def test_scale_validation_features_use_swap_averaged_model_output_in_registered_order():
    mean = TinyMean()
    scale = RecordingScale()
    batch = _scale_batch()
    loader = DataLoader([batch], batch_size=None)

    fit_scale_model(
        mean_model=mean,
        scale_model=scale,
        train_loader=loader,
        valid_loader=loader,
        max_epochs=1,
        patience=1,
        learning_rate=1e-3,
        weight_decay=0.0,
        device=torch.device("cpu"),
        seed=20260723,
    )

    # First call is current-orientation training, second is swap-averaged validation.
    validation_features = scale.seen[1]
    np.testing.assert_allclose(validation_features[:, :9], batch["process"].numpy())
    np.testing.assert_allclose(validation_features[:, 73:80], batch["quality"].numpy())
    averaged_channels = batch["spectrum"].mean(dim=(1, 3)).clone()
    averaged_channels[:, :2] = averaged_channels[:, :2].mean(dim=1, keepdim=True)
    expected_embedding = torch.cat(
        (averaged_channels.repeat(1, 21), averaged_channels[:, :1]), dim=1
    )
    torch.testing.assert_close(validation_features[:, 9:73], expected_embedding)
    expected_gate = torch.sigmoid(batch["quality"][:, 0])
    torch.testing.assert_close(validation_features[:, 80], expected_gate)
    torch.testing.assert_close(validation_features[:, 81], torch.zeros(2))


def test_exact_epoch_scale_refit_trains_every_epoch_without_validation_or_restore(
    monkeypatch,
):
    from roughness.sgrpn import phase_b_training

    def forbidden_validation(*args, **kwargs):
        del args, kwargs
        raise AssertionError("exact-epoch refit must not run validation")

    monkeypatch.setattr(phase_b_training, "_validation_nll", forbidden_validation)
    scale = NoRestoreScale()
    result = phase_b_training._refit_scale_model_exact_epochs(
        mean_model=TinyMean(),
        scale_model=scale,
        train_loader=DataLoader([_scale_batch()], batch_size=None),
        epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        device="cpu",
        seed=20260723,
    )

    assert result.model is scale
    assert result.best_epoch == 3
    assert result.history["epoch"].tolist() == [1, 2, 3]
    assert list(result.history.columns) == ["epoch", "train_nll"]
    assert len(scale.seen) == 3


def test_public_phase_b_training_signatures_are_exactly_typed():
    nested = inspect.signature(build_nested_calibration)
    fold = inspect.signature(run_phase_b_fold)

    assert tuple(nested.parameters) == (
        "config",
        "phase_a_config",
        "bundle",
        "cache",
        "fold",
        "seed",
        "device",
        "backend",
        "batch_size",
    )
    assert tuple(fold.parameters) == (
        "config",
        "handoff",
        "phase_a_config",
        "bundle",
        "cache",
        "fold",
        "seed",
        "device",
        "backend",
        "batch_size",
        "output_root",
    )
    for signature, keyword_only in ((nested, 6), (fold, 7)):
        parameters = tuple(signature.parameters.values())
        assert all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters[:keyword_only]
        )
        assert all(
            parameter.kind is inspect.Parameter.KEYWORD_ONLY
            for parameter in parameters[keyword_only:]
        )
        assert all(parameter.annotation is not inspect.Parameter.empty for parameter in parameters)
        assert signature.return_annotation is not inspect.Signature.empty
    assert nested.parameters["device"].default is inspect.Parameter.empty
    assert nested.parameters["backend"].default is None
    assert nested.parameters["batch_size"].default == 8
    assert fold.parameters["device"].default is None
    assert fold.parameters["backend"].default is None
    assert fold.parameters["batch_size"].default == 8
    assert fold.parameters["output_root"].default is None


def _fixture(
    tmp_path: Path, *, group_count: int = 25
) -> tuple[PhaseBConfig, SGRPNConfig, PhaseAHandoff, DataBundle, OrderSpectrumCache]:
    group_numbers = np.concatenate((np.arange(group_count), np.array([1])))
    sample_count = len(group_numbers)
    sample_ids = [f"s{index:02d}" for index in range(sample_count)]
    group_ids = [f"g{index:02d}" for index in group_numbers]
    group_centers = np.linspace(0.2, 1.2, group_count)
    centers = group_centers[group_numbers]
    centers[-1] += 0.04
    group_sizes = pd.Series(group_ids).value_counts()
    split_count = np.asarray([group_sizes[group_id] for group_id in group_ids])
    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_id": group_ids,
            "signal_path": ["unused.csv"] * sample_count,
            "n_rpm": np.linspace(1.0, 2.0, sample_count),
            "fz_mm_per_tooth": np.linspace(0.03, 0.12, sample_count),
            "ap_mm": np.linspace(0.5, 2.0, sample_count),
            "ra_1": centers - 0.02,
            "ra_2": centers,
            "ra_3": centers + 0.02,
            "ra_mean": centers,
            "sample_weight": 1.0 / split_count,
            "split_count": split_count,
            "version": np.where(np.arange(sample_count) % 2, "v4", "v3"),
            "duration_s": np.ones(sample_count),
        }
    )
    folds = pd.DataFrame(
        {"sample_id": sample_ids, "group_id": group_ids, "fold": group_numbers % 5}
    )
    windows = pd.DataFrame(
        {
            "segment_id": sample_ids,
            "group_id": group_ids,
            "csv_path": ["unused.csv"] * sample_count,
            "window_id": np.zeros(sample_count, dtype=int),
            "start_sample": np.zeros(sample_count, dtype=int),
            "end_sample": np.full(sample_count, 25600, dtype=int),
            "is_tail_aligned": np.zeros(sample_count, dtype=bool),
        }
    )
    for name, frame in (("manifest", manifest), ("folds", folds), ("windows", windows)):
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    (tmp_path / "m0.csv").write_text("sample_id,prediction\n", encoding="utf-8")
    phase_a = SGRPNConfig(
        manifest_path=tmp_path / "manifest.csv",
        folds_path=tmp_path / "folds.csv",
        window_index_path=tmp_path / "windows.csv",
        m0_oof_path=tmp_path / "m0.csv",
        output_dir=tmp_path / "phase_a",
        sample_rate_hz=25600,
        window_samples=25600,
        order_min=0.0,
        order_max=90.0,
        order_step=0.25,
        seeds=(20260723,),
        inner_splits=4,
        max_epochs=1,
        patience=1,
        process_learning_rate=1e-3,
        signal_learning_rate=3e-4,
        gate_learning_rate=1e-3,
        weight_decay=1e-4,
        huber_delta_um=0.1,
        gate_penalty=1e-3,
        correction_penalty=1e-2,
        bootstrap_repetitions=10_000,
    )
    phase_b = PhaseBConfig(
        phase_a_config_path=tmp_path / "phase_a.yaml",
        phase_a_config_file_sha256="a" * 64,
        phase_a_acceptance_path=tmp_path / "acceptance.json",
        phase_a_run_manifest_path=tmp_path / "run_manifest.json",
        output_dir=tmp_path / "phase_b",
        seeds=(20260723, 20260724, 20260725),
        alphas=(0.10, 0.05),
        inner_splits=4,
        max_epochs=1,
        patience=1,
        variance_learning_rate=1e-3,
        weight_decay=1e-4,
        bootstrap_repetitions=10_000,
    )
    bundle = DataBundle(manifest, folds, windows, {}, pd.DataFrame())
    rng = np.random.default_rng(71)
    cache = OrderSpectrumCache(
        segment_ids=tuple(sample_ids),
        spectra=rng.normal(scale=0.1, size=(sample_count, 3, 361)).astype(np.float32),
        offsets=np.arange(sample_count + 1, dtype=np.int64),
        quality=rng.normal(scale=0.1, size=(sample_count, 7)).astype(np.float32),
        durations_s=np.ones(sample_count),
    )
    handoff = PhaseAHandoff(
        acceptance_sha256="b" * 64,
        run_manifest_sha256="c" * 64,
        phase_a_config_sha256="a" * 64,
        training_fingerprint="d" * 64,
        cache_sha256="e" * 64,
        input_sha256={"manifest": "f" * 64},
    )
    return phase_b, phase_a, handoff, bundle, cache


def _phase_a_fixture_hash(
    phase_a: SGRPNConfig,
    handoff: PhaseAHandoff,
    bundle: DataBundle,
    cache: OrderSpectrumCache,
) -> str:
    digest = hashlib.sha256()
    config_payload = {
        name: str(value) if isinstance(value, Path) else value
        for name, value in asdict(phase_a).items()
    }
    digest.update(json.dumps(config_payload, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(asdict(handoff), sort_keys=True).encode("utf-8"))
    for frame in (bundle.manifest, bundle.folds, bundle.windows):
        digest.update(frame.to_csv(index=False).encode("utf-8"))
    for value in (
        cache.segment_ids,
        cache.spectra,
        cache.offsets,
        cache.quality,
        cache.durations_s,
    ):
        if isinstance(value, tuple):
            digest.update("\0".join(value).encode("utf-8"))
        else:
            array = np.asarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(array.shape).encode("ascii"))
            digest.update(array.tobytes())
    return digest.hexdigest()


def _fake_mean_path(
    monkeypatch,
    calls: list[tuple[str, ...]],
    backend_calls: list[object] | None = None,
):
    from roughness.sgrpn import phase_b_training

    process_scaler = ProcessScaler(np.zeros(9), np.ones(9))
    spectrum_scaler = SpectrumScaler(np.zeros((3, 361)), np.ones((3, 361)))
    quality_scaler = QualityScaler(np.zeros(7), np.ones(7))

    def fake(config, bundle, cache, fold, seed, train_sample_ids, **kwargs):
        del config, cache
        if backend_calls is not None:
            backend_calls.append(kwargs.get("backend"))
        ids = tuple(map(str, train_sample_ids))
        calls.append(ids)
        indexed = bundle.manifest.assign(
            sample_id=bundle.manifest["sample_id"].astype(str)
        ).set_index("sample_id", drop=False)
        frame = indexed.loc[list(ids)].reset_index(drop=True)
        split = make_group_inner_splits(frame, 4, seed)
        shift = 0.05 * float(frame["ra_mean"].mean())
        models = tuple(TinyMean(shift + inner * 0.001) for inner in range(4))
        for model, (train_index, _valid_index) in zip(models, split, strict=True):
            model.checkpoint_metadata = {
                "train_sample_ids": tuple(frame.iloc[train_index]["sample_id"].astype(str)),
                "train_group_ids": tuple(frame.iloc[train_index]["group_id"].astype(str)),
            }
        final = TinyMean(shift)
        final.checkpoint_metadata = {
            "train_sample_ids": ids,
            "train_group_ids": tuple(frame["group_id"].astype(str)),
        }
        return MeanPathArtifacts(
            fold=fold,
            seed=seed,
            train_sample_ids=ids,
            model=final,
            process_scaler=process_scaler,
            spectrum_scaler=spectrum_scaler,
            quality_scaler=quality_scaler,
            inner_splits=tuple(split),
            inner_models=models,
            inner_scalers=tuple(
                (process_scaler, spectrum_scaler, quality_scaler) for _ in range(4)
            ),
            best_epochs={"P1": (1,) * 4, "R1": (1,) * 4, "G1": (1,) * 4},
            histories={name: pd.DataFrame({"epoch": [1]}) for name in ("P1", "R1", "G1")},
        )

    monkeypatch.setattr(phase_b_training, "fit_g1_mean_path", fake)


def test_nested_calibration_records_group_disjoint_mean_scale_and_scaler_sources(
    tmp_path, monkeypatch
):
    config, phase_a, _handoff, bundle, cache = _fixture(tmp_path)
    calls: list[tuple[str, ...]] = []
    _fake_mean_path(monkeypatch, calls)

    artifacts = build_nested_calibration(
        config,
        phase_a,
        bundle,
        cache,
        fold=0,
        seed=20260723,
        device="cpu",
        batch_size=32,
    )

    outer_train, _ = outer_indices(bundle, 0)
    outer_frame = bundle.manifest.iloc[outer_train]
    assert set(artifacts) == set(SCALE_MODELS)
    assert len(calls) == 4
    for scale_model, result in artifacts.items():
        assert result.predictions["sample_id"].is_unique
        assert set(result.predictions["sample_id"]) == set(outer_frame["sample_id"])
        assert tuple(result.group_scores.columns) == EXPECTED_CALIBRATION_SCORE_COLUMNS
        assert len(result.group_scores) == outer_frame["group_id"].nunique()
        assert result.group_scores["group_id"].is_unique
        assert (result.group_scores["reading_count"] == 3 * result.group_scores["region_count"]).all()
        assert set(result.group_scores["outer_fold"]) == {0}
        assert set(result.group_scores["inner_fold"]) == {0, 1, 2, 3}
        assert set(result.group_scores["seed"]) == {20260723}
        assert set(result.group_scores["scale_model"]) == {scale_model}
        assert np.isfinite(result.group_scores["score"]).all()
        assert (result.group_scores["score"] >= 0).all()
        multi_region = result.group_scores.query("region_count > 1")
        assert len(multi_region) == 1
        multi_group = str(multi_region.iloc[0]["group_id"])
        multi_predictions = result.predictions.query("group_id == @multi_group")
        expected_group_max = np.max(
            np.abs(
                multi_predictions[["ra_1", "ra_2", "ra_3"]].to_numpy()
                - multi_predictions["mu"].to_numpy()[:, None]
            )
            / multi_predictions["sigma"].to_numpy()[:, None]
        )
        assert multi_region.iloc[0]["score"] == pytest.approx(expected_group_max)
        assert multi_region.iloc[0]["reading_count"] == 6
        for alpha, quantile in result.quantiles.items():
            ordered = np.sort(result.group_scores["score"].to_numpy())
            assert quantile.alpha == alpha
            assert quantile.quantile == ordered[quantile.order_index - 1]

        summaries = result.inner_fold_definitions.query("record_type == 'predictor'")
        assert len(summaries) == 4
        for row in summaries.itertuples(index=False):
            held_out = set(row.validation_group_ids)
            assert held_out
            for field in (
                "train_group_ids",
                "scaler_source_group_ids",
                "mean_fit_group_ids",
                "scale_fit_group_ids",
                "epoch_selection_group_ids",
                "residual_source_group_ids",
            ):
                assert held_out.isdisjoint(set(getattr(row, field))), (scale_model, field)


def test_calibration_validation_labels_do_not_change_their_mu_or_sigma(
    tmp_path, monkeypatch
):
    config, phase_a, _handoff, bundle, cache = _fixture(tmp_path)
    _fake_mean_path(monkeypatch, [])
    first = build_nested_calibration(
        config, phase_a, bundle, cache, 0, 20260724, device="cpu", batch_size=32
    )
    selected = first["heteroscedastic"].predictions.iloc[0]
    selected_group = selected["group_id"]
    changed_manifest = bundle.manifest.copy()
    mask = changed_manifest["group_id"] == selected_group
    changed_manifest.loc[mask, ["ra_1", "ra_2", "ra_3", "ra_mean"]] += 50.0
    changed = replace(bundle, manifest=changed_manifest)
    _fake_mean_path(monkeypatch, [])
    second = build_nested_calibration(
        config, phase_a, changed, cache, 0, 20260724, device="cpu", batch_size=32
    )

    for scale_model in SCALE_MODELS:
        before = first[scale_model].predictions.query("group_id == @selected_group")
        after = second[scale_model].predictions.query("group_id == @selected_group")
        np.testing.assert_array_equal(before[["mu", "sigma"]], after[["mu", "sigma"]])
        before_score = first[scale_model].group_scores.query("group_id == @selected_group")["score"]
        after_score = second[scale_model].group_scores.query("group_id == @selected_group")["score"]
        assert not np.array_equal(before_score.to_numpy(), after_score.to_numpy())

        before_definition = first[scale_model].inner_fold_definitions.query(
            "record_type == 'predictor'"
        )
        before_definition = before_definition[
            before_definition["validation_group_ids"].map(
                lambda values: selected_group in values
            )
        ].iloc[0]
        after_definition = second[scale_model].inner_fold_definitions.query(
            "record_type == 'predictor'"
        )
        after_definition = after_definition[
            after_definition["validation_group_ids"].map(
                lambda values: selected_group in values
            )
        ].iloc[0]
        for field in (
            "train_sample_ids",
            "scaler_source_sample_ids",
            "mean_fit_sample_ids",
            "scale_fit_sample_ids",
        ):
            assert before_definition[field] == after_definition[field]
            assert not set(before.loc[:, "sample_id"]) & set(before_definition[field])


def test_confined_and_final_scale_refits_use_fresh_heads_for_exact_median_epochs(
    tmp_path, monkeypatch
):
    from roughness.sgrpn import phase_b_training

    config, phase_a, handoff, bundle, cache = _fixture(tmp_path)
    config = replace(config, max_epochs=4)
    _fake_mean_path(monkeypatch, [])
    selection_models: list[nn.Module] = []
    refit_models: list[nn.Module] = []
    refit_epochs: list[int] = []
    selection_count = 0

    def fake_selection(**kwargs):
        nonlocal selection_count
        model = kwargs["scale_model"]
        selection_models.append(model)
        selected_epoch = (1, 2, 3, 4)[selection_count % 4]
        selection_count += 1
        return phase_b_training.ScaleFit(
            model=model,
            best_epoch=selected_epoch,
            history=pd.DataFrame(
                {"epoch": [selected_epoch], "train_nll": [1.0], "validation_nll": [1.0]}
            ),
            mean_state_sha256="0" * 64,
        )

    def fake_exact_refit(**kwargs):
        model = kwargs["scale_model"]
        assert all(model is not selected for selected in selection_models)
        refit_models.append(model)
        refit_epochs.append(kwargs["epochs"])
        return phase_b_training.ScaleFit(
            model=model,
            best_epoch=kwargs["epochs"],
            history=pd.DataFrame(
                {"epoch": list(range(1, kwargs["epochs"] + 1)), "train_nll": [1.0] * kwargs["epochs"]}
            ),
            mean_state_sha256="0" * 64,
        )

    monkeypatch.setattr(phase_b_training, "fit_scale_model", fake_selection)
    monkeypatch.setattr(
        phase_b_training, "_refit_scale_model_exact_epochs", fake_exact_refit
    )
    run_phase_b_fold(
        config,
        handoff,
        phase_a,
        bundle,
        cache,
        0,
        20260723,
        device="cpu",
        batch_size=32,
        output_root=config.output_dir,
    )

    assert selection_count == 40
    assert len(refit_models) == 10
    assert len({id(model) for model in refit_models}) == 10
    assert refit_epochs == [2] * 10


def test_nested_calibration_forwards_backend_to_every_confined_mean_fit(
    tmp_path, monkeypatch
):
    config, phase_a, _handoff, bundle, cache = _fixture(tmp_path)
    backend = object()
    backend_calls: list[object] = []
    _fake_mean_path(monkeypatch, [], backend_calls)

    build_nested_calibration(
        config,
        phase_a,
        bundle,
        cache,
        0,
        20260723,
        device="cpu",
        backend=backend,
        batch_size=32,
    )

    assert backend_calls == [backend] * 4


def test_fold_accepts_positional_fold_seed_and_forwards_backend_to_all_mean_fits(
    tmp_path, monkeypatch
):
    config, phase_a, handoff, bundle, cache = _fixture(tmp_path)
    backend = object()
    backend_calls: list[object] = []
    _fake_mean_path(monkeypatch, [], backend_calls)

    run_phase_b_fold(
        config,
        handoff,
        phase_a,
        bundle,
        cache,
        0,
        20260724,
        device="cpu",
        backend=backend,
        batch_size=32,
        output_root=config.output_dir,
    )

    assert backend_calls == [backend] * 5


@pytest.mark.parametrize("seed", (20260723, 20260724, 20260725))
def test_one_fold_builds_exact_rows_without_outer_test_label_access(
    tmp_path, monkeypatch, seed
):
    from roughness.sgrpn import phase_b_training

    config, phase_a, handoff, bundle, cache = _fixture(tmp_path)
    fixture_hash = _phase_a_fixture_hash(phase_a, handoff, bundle, cache)
    files_before = {
        path.relative_to(tmp_path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    _fake_mean_path(monkeypatch, [])
    first = run_phase_b_fold(
        config,
        handoff,
        phase_a,
        bundle,
        cache,
        0,
        seed,
        device="cpu",
        batch_size=32,
        output_root=config.output_dir,
    )
    assert _phase_a_fixture_hash(phase_a, handoff, bundle, cache) == fixture_hash
    assert {
        path.relative_to(tmp_path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == files_before
    _, outer_test = outer_indices(bundle, 0)
    changed_manifest = bundle.manifest.copy()
    changed_manifest.loc[outer_test, ["ra_1", "ra_2", "ra_3", "ra_mean"]] += 100.0
    changed_bundle = replace(bundle, manifest=changed_manifest)
    changed_fixture_hash = _phase_a_fixture_hash(
        phase_a, handoff, changed_bundle, cache
    )
    _fake_mean_path(monkeypatch, [])
    second = run_phase_b_fold(
        config,
        handoff,
        phase_a,
        changed_bundle,
        cache,
        0,
        seed,
        device="cpu",
        batch_size=32,
        output_root=config.output_dir,
    )
    assert (
        _phase_a_fixture_hash(phase_a, handoff, changed_bundle, cache)
        == changed_fixture_hash
    )
    assert {
        path.relative_to(tmp_path)
        for path in tmp_path.rglob("*")
        if path.is_file()
    } == files_before

    assert phase_b_training.PHASE_B_PREDICTION_COLUMNS == EXPECTED_PHASE_B_PREDICTION_COLUMNS
    assert phase_b_training.PHASE_B_MEAN_COLUMNS == EXPECTED_PHASE_B_MEAN_COLUMNS
    assert phase_b_training.CALIBRATION_SCORE_COLUMNS == EXPECTED_CALIBRATION_SCORE_COLUMNS
    assert tuple(first.predictions.columns) == EXPECTED_PHASE_B_PREDICTION_COLUMNS
    assert len(first.predictions) == 2 * len(outer_test)
    assert not first.predictions.duplicated(["sample_id", "seed", "scale_model"]).any()
    assert set(first.predictions["seed"]) == {seed}
    assert set(first.predictions["scale_model"]) == set(SCALE_MODELS)
    numeric = first.predictions.select_dtypes(include=[np.number])
    assert np.isfinite(numeric).all().all()
    assert (first.predictions["sigma"] > 0).all()
    for suffix in ("90", "95"):
        assert (first.predictions[f"raw_lower_{suffix}"] <= first.predictions[f"raw_upper_{suffix}"]).all()
        assert (
            first.predictions[f"conformal_lower_{suffix}"]
            <= first.predictions[f"conformal_upper_{suffix}"]
        ).all()
        alpha = {"90": 0.10, "95": 0.05}[suffix]
        for scale_model in SCALE_MODELS:
            expected_q = first.calibration[scale_model].quantiles[alpha].quantile
            actual_q = first.predictions.query(
                "scale_model == @scale_model"
            )[f"conformal_q_{suffix}"]
            assert np.all(actual_q.to_numpy() == expected_q)

    invariant = [
        column
        for column in EXPECTED_PHASE_B_PREDICTION_COLUMNS
        if column not in {"target_mean", "ra_1", "ra_2", "ra_3"}
    ]
    pd.testing.assert_frame_equal(first.predictions[invariant], second.predictions[invariant])
    assert not np.array_equal(first.predictions["target_mean"], second.predictions["target_mean"])
    mean_predictions = first.predictions.attrs["mean_predictions"]
    assert tuple(mean_predictions.columns) == EXPECTED_PHASE_B_MEAN_COLUMNS
    assert len(mean_predictions) == 3 * len(outer_test)
    assert set(mean_predictions["model"]) == {"P1", "R1", "G1"}
    assert mean_predictions.notna().all().all()
    assert np.isfinite(
        mean_predictions.select_dtypes(include=[np.number]).to_numpy()
    ).all()
    reconstructed = (
        mean_predictions["process_mean"]
        + mean_predictions["gate"] * mean_predictions["residual"]
    )
    np.testing.assert_allclose(mean_predictions["prediction"], reconstructed)
    g1 = mean_predictions.query("model == 'G1'").set_index("sample_id")
    probability = first.predictions.set_index("sample_id")
    np.testing.assert_allclose(
        probability["correction"],
        probability.index.map(g1["prediction"] - g1["process_mean"]),
    )
    np.testing.assert_allclose(probability["gate"], probability.index.map(g1["gate"]))

    expected_labels = bundle.manifest.iloc[outer_test].set_index("sample_id")
    for row in first.predictions.itertuples(index=False):
        expected = expected_labels.loc[row.sample_id]
        assert row.version == expected["version"]
        assert row.sample_weight == pytest.approx(expected["sample_weight"])
        assert row.target_mean == pytest.approx(expected["ra_mean"])
        assert (row.ra_1, row.ra_2, row.ra_3) == pytest.approx(
            (expected["ra_1"], expected["ra_2"], expected["ra_3"])
        )
    assert first.checkpoint_paths.keys() == set(SCALE_MODELS)
    assert first.history_paths.keys() == set(SCALE_MODELS)
    assert all(path.is_relative_to(config.output_dir) for path in first.checkpoint_paths.values())
    assert not config.output_dir.exists()


@pytest.mark.parametrize("bad_seed", [7, True, 20260723.0])
def test_nested_protocol_rejects_unregistered_or_nonintegral_seed(
    tmp_path, monkeypatch, bad_seed
):
    config, phase_a, _handoff, bundle, cache = _fixture(tmp_path)
    _fake_mean_path(monkeypatch, [])
    with pytest.raises(ValueError, match="registered Phase B seed"):
        build_nested_calibration(
            config, phase_a, bundle, cache, 0, bad_seed, device="cpu"
        )


def test_nested_calibration_fails_closed_without_19_outer_train_groups(
    tmp_path, monkeypatch
):
    del monkeypatch
    config, phase_a, _handoff, bundle, cache = _fixture(tmp_path, group_count=20)

    with pytest.raises(ValueError, match="at least 19 calibration groups"):
        build_nested_calibration(
            config,
            phase_a,
            bundle,
            cache,
            fold=0,
            seed=20260723,
            device="cpu",
            batch_size=32,
        )
