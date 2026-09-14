import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import pytest
import torch

from roughness.sgrpn.corrective import (
    CorrectiveFittedPredictor,
    CorrectiveFoldResult,
    CorrectiveSplit,
    build_fixed_calibration,
    fit_corrective_predictor,
    load_corrective_config,
    load_validated_corrective_fold,
    make_corrective_split,
    publish_corrective_fold,
    run_corrective_all,
    run_corrective_fold,
    run_corrective_fold_in_memory,
)
from roughness.sgrpn.data import DataBundle
from roughness.sgrpn.phase_b_training import (
    CalibrationArtifacts,
    PHASE_B_MEAN_COLUMNS,
    PHASE_B_PREDICTION_COLUMNS,
)
from roughness.sgrpn.probability import GroupConformalResult


def _group_frame(group_count: int, rows_per_group: int = 2) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [
                f"g{group:03d}-s{row}"
                for group in range(group_count)
                for row in range(rows_per_group)
            ],
            "group_id": [
                f"g{group:03d}"
                for group in range(group_count)
                for _ in range(rows_per_group)
            ],
        }
    )


def test_corrective_split_is_deterministic_complete_and_group_disjoint():
    frame = _group_frame(80)

    first = make_corrective_split(frame, split_seed=20260723, calibration_fold=0)
    second = make_corrective_split(frame, split_seed=20260723, calibration_fold=0)

    assert first == second
    assert len(first.calibration_group_ids) == 20
    assert len(first.proper_train_group_ids) == 60
    assert set(first.calibration_group_ids).isdisjoint(first.proper_train_group_ids)
    assert set(first.calibration_sample_ids).isdisjoint(first.proper_train_sample_ids)
    assert set(first.calibration_sample_ids) | set(first.proper_train_sample_ids) == set(
        frame["sample_id"]
    )
    assert set(frame.loc[frame["group_id"].isin(first.calibration_group_ids), "sample_id"]) == set(
        first.calibration_sample_ids
    )


def test_corrective_split_rejects_too_few_calibration_groups_for_95_percent():
    frame = _group_frame(72)

    with pytest.raises(ValueError, match="at least 19 calibration groups"):
        make_corrective_split(frame, split_seed=20260723, calibration_fold=0)


def test_formal_corrective_splits_match_registered_real_group_counts():
    from roughness.sgrpn.config import load_phase_b_config, load_sgrpn_config
    from roughness.sgrpn.data import outer_indices

    project = Path(__file__).resolve().parents[2]
    corrective = load_corrective_config(
        project / "configs" / "sgrpn_phase_b_split_conformal_corrective.yaml"
    )
    phase_b = load_phase_b_config(corrective.phase_b_config_path)
    phase_a = load_sgrpn_config(phase_b.phase_a_config_path)
    manifest = pd.read_csv(
        phase_a.manifest_path,
        dtype={"sample_id": str, "group_id": str, "version": str},
    )
    folds = pd.read_csv(phase_a.folds_path, dtype={"sample_id": str})
    bundle = SimpleNamespace(manifest=manifest, folds=folds)

    observed = []
    for fold in range(5):
        train_index, test_index = outer_indices(bundle, fold)
        split = make_corrective_split(
            manifest.iloc[train_index].reset_index(drop=True),
            split_seed=corrective.split_seed,
            calibration_fold=corrective.calibration_fold,
        )
        observed.append(
            (
                len(split.proper_train_group_ids),
                len(split.calibration_group_ids),
                manifest.iloc[test_index]["group_id"].nunique(),
            )
        )
    assert observed == [
        (126, 43, 43),
        (126, 43, 43),
        (127, 43, 42),
        (127, 43, 42),
        (127, 43, 42),
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_corrective_config(tmp_path: Path) -> tuple[Path, Path]:
    source_config = tmp_path / "sgrpn_phase_b.yaml"
    source_manifest = tmp_path / "run_manifest.json"
    immutable_after = tmp_path / "immutable_hashes_after.json"
    source_config.write_text("phase: b\n", encoding="utf-8")
    source_manifest.write_text('{"status":"complete"}\n', encoding="utf-8")
    immutable_after.write_text('{"verified":true}\n', encoding="utf-8")
    output = tmp_path / "corrective-output"
    config = tmp_path / "corrective.yaml"
    config.write_text(
        "\n".join(
            (
                f"phase_b_config_path: {source_config.name}",
                f"phase_b_config_file_sha256: {_sha256(source_config)}",
                f"phase_b_run_manifest_path: {source_manifest.name}",
                f"phase_b_run_manifest_sha256: {_sha256(source_manifest)}",
                f"phase_b_immutable_after_path: {immutable_after.name}",
                f"phase_b_immutable_after_sha256: {_sha256(immutable_after)}",
                f"output_dir: {output.name}",
                "split_seed: 20260723",
                "calibration_fold: 0",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return config, output


def test_corrective_config_binds_prior_artifacts_and_exact_output_root(tmp_path: Path):
    config_path, output = _write_corrective_config(tmp_path)

    loaded = load_corrective_config(config_path, output_root=output)

    assert loaded.output_dir == output.resolve()
    assert loaded.split_seed == 20260723
    assert loaded.calibration_fold == 0
    assert loaded.phase_b_config_file_sha256 == _sha256(tmp_path / "sgrpn_phase_b.yaml")


def test_corrective_config_rejects_changed_prior_artifact(tmp_path: Path):
    config_path, output = _write_corrective_config(tmp_path)
    (tmp_path / "run_manifest.json").write_text('{"status":"changed"}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="phase_b_run_manifest_sha256"):
        load_corrective_config(config_path, output_root=output)


def test_fixed_calibration_uses_one_locked_predictor_and_only_reserved_groups(monkeypatch):
    from roughness.sgrpn import corrective, phase_b_training

    calibration = _group_frame(20, rows_per_group=1)
    calibration["ra_1"] = np.linspace(0.1, 0.3, len(calibration))
    calibration["ra_2"] = calibration["ra_1"] + 0.01
    calibration["ra_3"] = calibration["ra_1"] + 0.02
    locked_mean = object()
    mean_path = SimpleNamespace(model=locked_mean)
    locked_scales = {"heteroscedastic": object(), "homoscedastic": object()}
    seen: list[tuple[object, object, tuple[str, ...]]] = []

    def fake_batches(frame, cache, mean_path_arg, *, batch_size):
        assert cache == "cache"
        assert mean_path_arg is mean_path
        assert batch_size == 8
        return [tuple(frame["sample_id"])]

    def fake_predictions(
        batches,
        *,
        scale_model,
        mean_model,
        fold,
        seed,
        inner_fold,
        scale_name,
        device,
    ):
        del device
        sample_ids = tuple(batches[0])
        seen.append((mean_model, scale_model, sample_ids))
        indexed = calibration.set_index("sample_id")
        return pd.DataFrame(
            {
                "sample_id": sample_ids,
                "group_id": [indexed.loc[value, "group_id"] for value in sample_ids],
                "fold": fold,
                "seed": seed,
                "inner_fold": inner_fold,
                "scale_model": scale_name,
                "target_mean": [indexed.loc[value, ["ra_1", "ra_2", "ra_3"]].mean() for value in sample_ids],
                "ra_1": [indexed.loc[value, "ra_1"] for value in sample_ids],
                "ra_2": [indexed.loc[value, "ra_2"] for value in sample_ids],
                "ra_3": [indexed.loc[value, "ra_3"] for value in sample_ids],
                "mu": np.zeros(len(sample_ids)),
                "sigma": np.ones(len(sample_ids)),
            }
        )

    monkeypatch.setattr(phase_b_training, "_calibration_batches", fake_batches)
    monkeypatch.setattr(phase_b_training, "_calibration_predictions", fake_predictions)

    result = build_fixed_calibration(
        SimpleNamespace(alphas=(0.10, 0.05)),
        calibration,
        "cache",
        mean_path,
        locked_scales,
        fold=2,
        seed=20260724,
        device="cpu",
        batch_size=8,
    )

    expected_ids = tuple(calibration["sample_id"])
    assert seen == [
        (locked_mean, locked_scales["heteroscedastic"], expected_ids),
        (locked_mean, locked_scales["homoscedastic"], expected_ids),
    ]
    assert set(result) == set(locked_scales)
    for artifacts in result.values():
        assert set(artifacts.predictions["sample_id"]) == set(expected_ids)
        assert len(artifacts.group_scores) == 20
        assert set(artifacts.group_scores["inner_fold"]) == {0}
        assert set(artifacts.quantiles) == {0.10, 0.05}


def test_corrective_predictor_fits_only_proper_train_then_calibrates_locked_models(monkeypatch):
    from roughness.sgrpn import corrective, phase_b_training

    outer_train = _group_frame(80, rows_per_group=1)
    locked_mean = object()
    mean_path = SimpleNamespace(model=locked_mean)
    locked_scales = {"heteroscedastic": object(), "homoscedastic": object()}
    calibration_artifacts = {"heteroscedastic": object(), "homoscedastic": object()}
    events: list[tuple[str, tuple[str, ...]]] = []

    def fake_mean(config, bundle, cache, fold, seed, train_sample_ids, **kwargs):
        del config, bundle, cache, fold, seed, kwargs
        events.append(("mean", tuple(train_sample_ids)))
        return mean_path

    def fake_scales(config, train_frame, cache, mean_path_arg, **kwargs):
        del config, cache, kwargs
        assert mean_path_arg is mean_path
        events.append(("scales", tuple(train_frame["sample_id"])))
        return locked_scales, {"heteroscedastic": pd.DataFrame(), "homoscedastic": pd.DataFrame()}

    def fake_calibration(config, frame, cache, mean_path_arg, scale_models, **kwargs):
        del config, cache, kwargs
        assert mean_path_arg is mean_path
        assert scale_models is locked_scales
        events.append(("calibration", tuple(frame["sample_id"])))
        return calibration_artifacts

    monkeypatch.setattr(phase_b_training, "fit_g1_mean_path", fake_mean)
    monkeypatch.setattr(phase_b_training, "_select_and_refit_outer_scales", fake_scales)
    monkeypatch.setattr(corrective, "build_fixed_calibration", fake_calibration)

    result = fit_corrective_predictor(
        SimpleNamespace(split_seed=20260723, calibration_fold=0),
        SimpleNamespace(alphas=(0.10, 0.05)),
        object(),
        object(),
        outer_train,
        "cache",
        fold=1,
        seed=20260725,
        device="cpu",
    )

    proper_ids = set(result.split.proper_train_sample_ids)
    calibration_ids = set(result.split.calibration_sample_ids)
    assert proper_ids.isdisjoint(calibration_ids)
    assert proper_ids | calibration_ids == set(outer_train["sample_id"])
    assert events == [
        ("mean", result.split.proper_train_sample_ids),
        ("scales", result.split.proper_train_sample_ids),
        ("calibration", result.split.calibration_sample_ids),
    ]
    assert result.mean_path is mean_path
    assert result.scale_models is locked_scales
    assert result.calibration is calibration_artifacts


def test_corrective_fold_uses_locked_calibrated_predictor_for_outer_test(monkeypatch):
    from roughness.sgrpn import corrective, phase_b_training

    group_count = 100
    sample_ids = [f"s{index:03d}" for index in range(group_count)]
    group_ids = [f"g{index:03d}" for index in range(group_count)]
    centers = np.linspace(0.2, 1.0, group_count)
    manifest = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_id": group_ids,
            "version": ["v3"] * group_count,
            "sample_weight": np.ones(group_count),
            "split_count": np.ones(group_count, dtype=int),
            "ra_1": centers - 0.01,
            "ra_2": centers,
            "ra_3": centers + 0.01,
            "ra_mean": centers,
        }
    )
    folds = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "group_id": group_ids,
            "fold": np.arange(group_count) % 5,
        }
    )
    bundle = DataBundle(manifest, folds, pd.DataFrame(), {}, pd.DataFrame())
    expected_outer_train = manifest.loc[folds["fold"] != 0].reset_index(drop=True)
    actual_split = make_corrective_split(
        expected_outer_train, split_seed=20260723, calibration_fold=0
    )
    locked_mean = object()
    locked_scales = {"heteroscedastic": object(), "homoscedastic": object()}
    locked_calibration = {"heteroscedastic": object(), "homoscedastic": object()}
    fitted = SimpleNamespace(
        split=actual_split,
        mean_path=SimpleNamespace(model=locked_mean),
        scale_models=locked_scales,
        calibration=locked_calibration,
    )
    observed_test_ids: list[str] = []

    monkeypatch.setattr(corrective, "fit_corrective_predictor", lambda *args, **kwargs: fitted)

    def fake_inference_batches(frame, cache, mean_path, *, batch_size):
        del cache, batch_size
        assert mean_path is fitted.mean_path
        observed_test_ids.extend(frame["sample_id"].astype(str))
        return [frame.loc[:, ["sample_id", "group_id", "sample_weight"]].copy()]

    def fake_outer_inference(
        batches, *, mean_model, scale_models, calibration, fold, seed, device
    ):
        del device
        assert mean_model is locked_mean
        assert scale_models is locked_scales
        assert calibration is locked_calibration
        test = batches[0]
        probability_rows = []
        mean_rows = []
        for row in test.itertuples(index=False):
            for scale_name in ("heteroscedastic", "homoscedastic"):
                probability_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "group_id": row.group_id,
                        "fold": fold,
                        "seed": seed,
                        "scale_model": scale_name,
                        "sample_weight": row.sample_weight,
                        "mu": 0.5,
                        "sigma": 0.2,
                        "gate": 0.4,
                        "correction": 0.01,
                        "raw_lower_90": 0.1,
                        "raw_upper_90": 0.9,
                        "raw_lower_95": 0.0,
                        "raw_upper_95": 1.0,
                        "conformal_q_90": 2.0,
                        "conformal_lower_90": 0.1,
                        "conformal_upper_90": 0.9,
                        "conformal_q_95": 2.5,
                        "conformal_lower_95": 0.0,
                        "conformal_upper_95": 1.0,
                    }
                )
            for model, prediction, gate, correction in (
                ("P1", 0.49, 0.0, 0.0),
                ("R1", 0.51, 1.0, 0.02),
                ("G1", 0.5, 0.4, 0.01),
            ):
                mean_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "group_id": row.group_id,
                        "fold": fold,
                        "seed": seed,
                        "model": model,
                        "prediction": prediction,
                        "sample_weight": row.sample_weight,
                        "process_mean": 0.49,
                        "residual": 0.02,
                        "gate": gate,
                        "correction": correction,
                    }
                )
        return pd.DataFrame(probability_rows), pd.DataFrame(mean_rows)

    monkeypatch.setattr(phase_b_training, "_inference_batches", fake_inference_batches)
    monkeypatch.setattr(phase_b_training, "_outer_inference", fake_outer_inference)

    result = run_corrective_fold_in_memory(
        SimpleNamespace(split_seed=20260723, calibration_fold=0),
        SimpleNamespace(alphas=(0.10, 0.05)),
        object(),
        bundle,
        "cache",
        fold=0,
        seed=20260723,
        device="cpu",
    )

    expected_test = set(folds.loc[folds["fold"] == 0, "sample_id"])
    assert set(observed_test_ids) == expected_test
    assert tuple(result.predictions.columns) == PHASE_B_PREDICTION_COLUMNS
    assert tuple(result.mean_predictions.columns) == PHASE_B_MEAN_COLUMNS
    assert len(result.predictions) == 2 * len(expected_test)
    assert len(result.mean_predictions) == 3 * len(expected_test)


def _persistable_corrective_result() -> CorrectiveFoldResult:
    calibration_by_model = {}
    for scale_name in ("heteroscedastic", "homoscedastic"):
        scores = np.arange(1.0, 21.0) / 10.0
        predictions = pd.DataFrame(
            {
                "sample_id": [f"c{index:02d}" for index in range(20)],
                "group_id": [f"cg{index:02d}" for index in range(20)],
                "fold": [0] * 20,
                "seed": [20260723] * 20,
                "inner_fold": [0] * 20,
                "scale_model": [scale_name] * 20,
                "target_mean": scores,
                "ra_1": scores,
                "ra_2": scores,
                "ra_3": scores,
                "mu": np.zeros(20),
                "sigma": np.ones(20),
            }
        )
        group_scores = pd.DataFrame(
            {
                "group_id": predictions["group_id"],
                "outer_fold": [0] * 20,
                "inner_fold": [0] * 20,
                "seed": [20260723] * 20,
                "scale_model": [scale_name] * 20,
                "score": scores,
                "region_count": [1] * 20,
                "reading_count": [3] * 20,
            }
        )
        calibration_by_model[scale_name] = CalibrationArtifacts(
            predictions=predictions,
            group_scores=group_scores,
            quantiles={
                0.10: GroupConformalResult(0.10, 20, 19, 1.9),
                0.05: GroupConformalResult(0.05, 20, 20, 2.0),
            },
            inner_fold_definitions=pd.DataFrame(),
        )
    probability_rows = []
    for scale_name in ("heteroscedastic", "homoscedastic"):
        probability_rows.append(
            {
                "sample_id": "t0",
                "group_id": "tg0",
                "version": "v3",
                "fold": 0,
                "seed": 20260723,
                "scale_model": scale_name,
                "target_mean": 0.5,
                "ra_1": 0.4,
                "ra_2": 0.5,
                "ra_3": 0.6,
                "sample_weight": 1.0,
                "mu": 0.5,
                "sigma": 0.2,
                "gate": 0.4,
                "correction": 0.01,
                "raw_lower_90": 0.1,
                "raw_upper_90": 0.9,
                "raw_lower_95": 0.0,
                "raw_upper_95": 1.0,
                "conformal_q_90": 1.9,
                "conformal_lower_90": 0.12,
                "conformal_upper_90": 0.88,
                "conformal_q_95": 2.0,
                "conformal_lower_95": 0.1,
                "conformal_upper_95": 0.9,
            }
        )
    mean_rows = []
    for model, prediction, gate, correction in (
        ("P1", 0.49, 0.0, 0.0),
        ("R1", 0.51, 1.0, 0.02),
        ("G1", 0.5, 0.4, 0.01),
    ):
        mean_rows.append(
            {
                "sample_id": "t0",
                "group_id": "tg0",
                "version": "v3",
                "fold": 0,
                "seed": 20260723,
                "model": model,
                "target_mean": 0.5,
                "prediction": prediction,
                "sample_weight": 1.0,
                "process_mean": 0.49,
                "residual": 0.02,
                "gate": gate,
                "correction": correction,
            }
        )
    split = CorrectiveSplit(
        proper_train_sample_ids=tuple(f"p{index:02d}" for index in range(60)),
        calibration_sample_ids=tuple(f"c{index:02d}" for index in range(20)),
        proper_train_group_ids=tuple(f"pg{index:02d}" for index in range(60)),
        calibration_group_ids=tuple(f"cg{index:02d}" for index in range(20)),
    )
    mean_path = SimpleNamespace(
        model=torch.nn.Linear(1, 1),
        process_scaler=SimpleNamespace(mean=np.zeros(9), scale=np.ones(9)),
        spectrum_scaler=SimpleNamespace(mean=np.zeros((3, 361)), scale=np.ones((3, 361))),
        quality_scaler=SimpleNamespace(mean=np.zeros(7), scale=np.ones(7)),
    )
    fitted = CorrectiveFittedPredictor(
        split=split,
        mean_path=mean_path,
        scale_models={
            "heteroscedastic": torch.nn.Linear(1, 1),
            "homoscedastic": torch.nn.Linear(1, 1),
        },
        scale_histories={
            "heteroscedastic": pd.DataFrame({"epoch": [1], "train_nll": [0.1]}),
            "homoscedastic": pd.DataFrame({"epoch": [1], "train_nll": [0.1]}),
        },
        calibration=calibration_by_model,
    )
    return CorrectiveFoldResult(
        predictions=pd.DataFrame(probability_rows).loc[:, PHASE_B_PREDICTION_COLUMNS],
        mean_predictions=pd.DataFrame(mean_rows).loc[:, PHASE_B_MEAN_COLUMNS],
        fitted=fitted,
    )


def test_corrective_fold_publication_is_hash_closed_and_rejects_tampering(tmp_path: Path):
    result = _persistable_corrective_result()

    marker = publish_corrective_fold(
        tmp_path,
        result,
        fingerprint="f" * 64,
        fold=0,
        seed=20260723,
        selected_device="cpu",
    )

    assert marker.name == "complete.json"
    loaded = load_validated_corrective_fold(
        marker.parent,
        fingerprint="f" * 64,
        fold=0,
        seed=20260723,
    )
    pd.testing.assert_frame_equal(loaded.predictions, result.predictions, check_dtype=False)
    score_path = marker.parent / "calibration" / "group_scores.csv"
    score_path.write_text(score_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifact hash"):
        load_validated_corrective_fold(
            marker.parent,
            fingerprint="f" * 64,
            fold=0,
            seed=20260723,
        )


def test_corrective_deep_validation_recomputes_scores_after_hash_refresh(tmp_path: Path):
    result = _persistable_corrective_result()
    marker_path = publish_corrective_fold(
        tmp_path,
        result,
        fingerprint="e" * 64,
        fold=0,
        seed=20260723,
        selected_device="cpu",
    )
    score_path = marker_path.parent / "calibration" / "group_scores.csv"
    scores = pd.read_csv(score_path)
    scores.loc[0, "score"] = float(scores.loc[0, "score"]) + 0.5
    scores.to_csv(score_path, index=False)
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifacts"]["calibration/group_scores.csv"] = _sha256(score_path)
    marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration scores do not derive"):
        load_validated_corrective_fold(
            marker_path.parent,
            fingerprint="e" * 64,
            fold=0,
            seed=20260723,
        )


def test_corrective_deep_validation_rejects_forged_checkpoint_provenance(
    tmp_path: Path,
):
    result = _persistable_corrective_result()
    marker_path = publish_corrective_fold(
        tmp_path,
        result,
        fingerprint="a" * 64,
        fold=0,
        seed=20260723,
        selected_device="cpu",
    )
    provenance_path = marker_path.parent / "provenance.json"
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance["checkpoint_sha256"]["mean"] = "0" * 64
    provenance_path.write_text(json.dumps(provenance, sort_keys=True), encoding="utf-8")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker["artifacts"]["provenance.json"] = _sha256(provenance_path)
    marker_path.write_text(json.dumps(marker, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="checkpoint provenance hashes"):
        load_validated_corrective_fold(
            marker_path.parent,
            fingerprint="a" * 64,
            fold=0,
            seed=20260723,
        )


def test_corrective_completed_unit_resumes_without_retraining(tmp_path: Path, monkeypatch):
    from roughness.sgrpn import corrective

    result = _persistable_corrective_result()
    fingerprint = "d" * 64
    publish_corrective_fold(
        tmp_path,
        result,
        fingerprint=fingerprint,
        fold=0,
        seed=20260723,
        selected_device="cpu",
    )
    monkeypatch.setattr(corrective, "corrective_fingerprint", lambda *args, **kwargs: fingerprint)
    monkeypatch.setattr(
        corrective,
        "run_corrective_fold_in_memory",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrained")),
    )

    loaded = run_corrective_fold(
        SimpleNamespace(output_dir=tmp_path),
        object(),
        object(),
        object(),
        object(),
        fold=0,
        seed=20260723,
        device="cpu",
    )

    pd.testing.assert_frame_equal(loaded.predictions, result.predictions, check_dtype=False)


def test_corrective_training_seal_blocks_all_training_writes(tmp_path: Path):
    from roughness.sgrpn import corrective

    (tmp_path / "training_seal.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="training root is sealed"):
        run_corrective_fold(
            SimpleNamespace(output_dir=tmp_path),
            object(),
            object(),
            object(),
            object(),
            fold=0,
            seed=20260723,
            device="cpu",
        )
    with pytest.raises(ValueError, match="training root is sealed"):
        corrective.run_corrective_all(
            SimpleNamespace(output_dir=tmp_path),
            SimpleNamespace(seeds=(20260723, 20260724, 20260725)),
            object(),
            object(),
            object(),
            device="cpu",
        )


def test_corrective_training_invocation_is_atomic_and_exclusive(tmp_path: Path):
    from roughness.sgrpn.corrective import _acquire_corrective_training_invocation

    def acquire():
        try:
            _acquire_corrective_training_invocation(
                tmp_path,
                fold=0,
                seed=20260723,
                fingerprint="f" * 64,
            )
            return "acquired"
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: acquire(), range(2)))
    assert outcomes.count("acquired") == 1
    assert sum("already been invoked" in outcome for outcome in outcomes) == 1


def test_corrective_all_runs_exact_cartesian_and_publishes_combined_tables(
    tmp_path: Path, monkeypatch
):
    from roughness.sgrpn import corrective, phase_b_training

    calls: list[tuple[int, int]] = []

    def fake_fold(*args, fold, seed, **kwargs):
        del args, kwargs
        calls.append((fold, seed))
        base = _persistable_corrective_result()
        probability = base.predictions.copy()
        probability["sample_id"] = f"t{fold}"
        probability["group_id"] = f"tg{fold}"
        probability["fold"] = fold
        probability["seed"] = seed
        mean = base.mean_predictions.copy()
        mean["sample_id"] = f"t{fold}"
        mean["group_id"] = f"tg{fold}"
        mean["fold"] = fold
        mean["seed"] = seed
        calibration = pd.concat(
            [value.predictions for value in base.fitted.calibration.values()],
            ignore_index=True,
        )
        scores = pd.concat(
            [value.group_scores for value in base.fitted.calibration.values()],
            ignore_index=True,
        )
        calibration["fold"] = fold
        calibration["seed"] = seed
        scores["outer_fold"] = fold
        scores["seed"] = seed
        quantiles = {
            name: {
                f"{alpha:.2f}": {
                    "alpha": alpha,
                    "group_count": 20,
                    "order_index": 19 if alpha == 0.10 else 20,
                    "quantile": 1.9 if alpha == 0.10 else 2.0,
                }
                for alpha in (0.10, 0.05)
            }
            for name in ("heteroscedastic", "homoscedastic")
        }
        return SimpleNamespace(
            predictions=probability,
            mean_predictions=mean,
            calibration_predictions=calibration,
            group_scores=scores,
            quantiles=quantiles,
        )

    validated: list[tuple[int, int]] = []

    def fake_validate(probability, mean, bundle):
        del bundle
        validated.append((len(probability), len(mean)))

    monkeypatch.setattr(corrective, "run_corrective_fold", fake_fold)
    monkeypatch.setattr(phase_b_training, "_validate_phase_b_oof_cartesian", fake_validate)

    result = run_corrective_all(
        SimpleNamespace(
            output_dir=tmp_path,
            phase_b_config_file_sha256="a" * 64,
            phase_b_run_manifest_sha256="b" * 64,
            phase_b_immutable_after_sha256="c" * 64,
            split_seed=20260723,
            calibration_fold=0,
        ),
        SimpleNamespace(seeds=(20260723, 20260724, 20260725)),
        object(),
        object(),
        object(),
        device="cpu",
    )

    assert calls == [
        (fold, seed)
        for fold in range(5)
        for seed in (20260723, 20260724, 20260725)
    ]
    assert validated == [(30, 45)]
    assert len(result.quantiles) == 60
    assert (tmp_path / "predictions" / "oof_probability_predictions.csv").is_file()
    assert (tmp_path / "predictions" / "oof_mean_predictions.csv").is_file()
    manifest = json.loads((tmp_path / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_status"] == "complete"
    assert manifest["completed_units"] == 15
