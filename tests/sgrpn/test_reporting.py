import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roughness.sgrpn.config import PhaseAHandoff, PhaseBConfig
from roughness.sgrpn.data import DataBundle
from roughness.sgrpn.reporting import (
    validate_phase_b_outputs,
    write_phase_a_report,
    write_phase_b_report,
)


def _synthetic_predictions() -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest_rows = []
    prediction_rows = []
    offsets = {"M0": 0.10, "P1": 0.10, "V1": 0.15, "F1": 0.13, "R1": 0.12, "G1": 0.05}
    for index in range(10):
        sample = f"{index:03d}"
        group = f"g{index:02d}"
        fold = index % 5
        target = 0.5 + index * 0.1
        manifest_rows.append(
            {
                "sample_id": sample,
                "group_id": group,
                "version": "v3" if index < 5 else "v4",
                "n_rpm": 4000.0 + fold * 1000,
                "fz_mm_per_tooth": 0.03 + fold * 0.01,
                "ap_mm": 0.5 + fold * 0.25,
                "region_index": 1,
            }
        )
        for model, offset in offsets.items():
            prediction_rows.append(
                {
                    "sample_id": sample,
                    "group_id": group,
                    "version": "v3" if index < 5 else "v4",
                    "fold": fold,
                    "seed": 20260723,
                    "model": model,
                    "target": target,
                    "prediction": target + offset,
                    "sample_weight": 1.0,
                    "process_mean": target + 0.10 if model in {"P1", "R1", "G1"} else np.nan,
                    "residual": offset - 0.10 if model in {"R1", "G1"} else np.nan,
                    "gate": 0.5 if model == "G1" else np.nan,
                }
            )
    return pd.DataFrame(prediction_rows), pd.DataFrame(manifest_rows)


def _hash_tree(root: Path) -> dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_report_is_atomic_machine_readable_and_regenerates_without_checkpoints(tmp_path: Path):
    predictions, manifest = _synthetic_predictions()
    canonical_duration = pd.DataFrame(
        {"sample_id": manifest["sample_id"], "duration_recomputed_s": 1.0}
    )
    legacy = tmp_path / "outputs" / "scheme1"
    legacy.mkdir(parents=True)
    (legacy / "keep.bin").write_bytes(b"legacy")
    old_hashes = _hash_tree(legacy)
    output = tmp_path / "outputs" / "sgrpn" / "phase_a"

    written = write_phase_a_report(
        output,
        predictions,
        manifest,
        output_root=output,
        duration_audit=canonical_duration,
        bootstrap_repetitions=50,
        bootstrap_seed=20260723,
        run_manifest={"device": "cpu", "fingerprint": "fixture"},
    )

    expected = {
        "predictions/oof_predictions.csv",
        "evaluation/summary_metrics.csv",
        "evaluation/fold_metrics.csv",
        "evaluation/paired_bootstrap.csv",
        "evaluation/negative_transfer.csv",
        "evaluation/gate_statistics.csv",
        "evaluation/acceptance.json",
        "evaluation/method_notes.json",
        "evaluation/figures/prediction_scatter.png",
        "evaluation/figures/residual_plot.png",
        "evaluation/figures/gate_distribution.png",
        "evaluation/figures/gate_condition_heatmap.png",
    }
    assert expected <= {str(path.relative_to(output)).replace("\\", "/") for path in written.values()}
    assert _hash_tree(legacy) == old_hashes
    acceptance = json.loads((output / "evaluation" / "acceptance.json").read_text(encoding="utf-8"))
    assert isinstance(acceptance["proceed_to_phase_b"], bool)
    assert acceptance["decision"]["proceed_to_phase_b"] == acceptance["proceed_to_phase_b"]
    assert acceptance["subjective_override_allowed"] is False
    assert "transfer_reduction_comparison_atol" not in acceptance["thresholds"]
    assert acceptance["bootstrap"]["seed"] == 20260723
    assert acceptance["bootstrap"]["repetitions"] == 50
    notes = json.loads((output / "evaluation" / "method_notes.json").read_text(encoding="utf-8"))
    assert notes["version_input"] == "excluded"
    assert notes["phase_a_seed_count"] == 1
    assert "composite-domain" in notes["v3_v4_stress_test"]
    assert "uncertain" in notes["ch9_ch10_orientation"]

    for name in ("prediction_scatter.png", "residual_plot.png", "gate_distribution.png", "gate_condition_heatmap.png"):
        (output / "evaluation" / "figures" / name).unlink()
    registered_duration = (output / "audit" / "duration_audit.csv").read_bytes()
    write_phase_a_report(
        output,
        pd.read_csv(output / "predictions" / "oof_predictions.csv", dtype={"sample_id": str, "group_id": str, "version": str}),
        manifest,
        output_root=output,
        duration_audit=canonical_duration,
        bootstrap_repetitions=50,
        bootstrap_seed=20260723,
    )
    assert all((output / "evaluation" / "figures" / name).is_file() for name in ("prediction_scatter.png", "residual_plot.png", "gate_distribution.png", "gate_condition_heatmap.png"))
    assert not list(output.rglob("*.tmp-*"))
    regenerated_manifest = json.loads(
        (output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert regenerated_manifest["fingerprint"] == "fixture"
    assert regenerated_manifest["device"] == "cpu"
    assert regenerated_manifest["duration_audit"]["provenance"] == "existing_registered"
    assert regenerated_manifest["duration_audit"]["row_count"] == len(canonical_duration)
    assert (output / "audit" / "duration_audit.csv").read_bytes() == registered_duration
    assert {
        "python_version", "python_executable", "platform", "package_version"
    } <= set(regenerated_manifest["environment"])


def test_report_contains_segment_and_group_metrics_and_gate_diagnostics(tmp_path: Path):
    predictions, manifest = _synthetic_predictions()
    output = tmp_path / "phase_a"
    quality = pd.DataFrame(
        {
            "sample_id": manifest["sample_id"],
            **{f"quality_{index}": np.linspace(0.0, 1.0, len(manifest)) for index in range(7)},
        }
    )

    write_phase_a_report(
        output,
        predictions,
        manifest,
        output_root=output,
        duration_audit=pd.DataFrame(
            {"sample_id": manifest["sample_id"], "duration_recomputed_s": 1.0}
        ),
        quality_features=quality,
        bootstrap_repetitions=20,
    )

    summary = pd.read_csv(output / "evaluation" / "summary_metrics.csv")
    assert set(summary["aggregation_unit"]) == {"segment", "group"}
    assert set(summary["model"]) == {"M0", "P1", "V1", "F1", "R1", "G1"}
    gate = pd.read_csv(output / "evaluation" / "gate_statistics.csv")
    assert {
        "overall", "n_rpm", "fz_mm_per_tooth", "ap_mm", "version", "quality_0", "quality_6"
    } <= set(gate["dimension"])


def test_direct_reporting_rejects_arbitrary_phase_a_suffix(tmp_path: Path):
    predictions, manifest = _synthetic_predictions()
    ambiguous = tmp_path / "attacker" / "outputs" / "sgrpn" / "phase_a"

    with pytest.raises(ValueError, match="exact project.*output root"):
        write_phase_a_report(
            ambiguous,
            predictions,
            manifest,
            duration_audit=pd.DataFrame(
                {"sample_id": manifest["sample_id"], "duration_recomputed_s": 1.0}
            ),
            bootstrap_repetitions=20,
        )

    assert not ambiguous.exists()


@pytest.mark.parametrize(
    "tampering", ("rewritten_and_rehashed", "wrong_hash", "missing_row_count")
)
def test_report_reconstructs_duration_from_current_canonical_data_when_registration_is_invalid(
    tmp_path: Path, tampering: str
):
    predictions, manifest = _synthetic_predictions()
    output = tmp_path / "phase_a"
    canonical = pd.DataFrame(
        {"sample_id": manifest["sample_id"], "duration_recomputed_s": 1.0}
    )
    write_phase_a_report(
        output,
        predictions,
        manifest,
        output_root=output,
        duration_audit=canonical,
        bootstrap_repetitions=20,
    )
    duration_path = output / "audit" / "duration_audit.csv"
    manifest_path = output / "run_manifest.json"
    registered = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tampering == "rewritten_and_rehashed":
        stale = canonical.assign(duration_recomputed_s=2.0)
        duration_path.write_bytes(stale.to_csv(index=False).encode("utf-8"))
        registered["duration_audit"]["sha256"] = hashlib.sha256(
            duration_path.read_bytes()
        ).hexdigest()
    elif tampering == "wrong_hash":
        registered["duration_audit"]["sha256"] = "0" * 64
    else:
        registered["duration_audit"].pop("row_count")
    manifest_path.write_text(json.dumps(registered), encoding="utf-8")

    write_phase_a_report(
        output,
        predictions,
        manifest,
        output_root=output,
        duration_audit=canonical,
        bootstrap_repetitions=20,
    )

    assert duration_path.read_bytes() == canonical.to_csv(index=False).encode("utf-8")
    repaired = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert repaired["duration_audit"]["provenance"] == "canonical_current_data"
    assert repaired["duration_audit"]["row_count"] == len(canonical)
    assert repaired["duration_audit"]["sha256"] == hashlib.sha256(
        duration_path.read_bytes()
    ).hexdigest()


def _phase_b_report_fixture(tmp_path: Path) -> tuple[
    PhaseBConfig,
    PhaseAHandoff,
    DataBundle,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Build the exact registered Phase B Cartesian product without checkpoints."""
    groups = [f"g{index:03d}" for index in range(212)]
    sample_rows: list[dict[str, object]] = []
    for group_index, group_id in enumerate(groups):
        count = 3 if group_index < 162 else 2
        for region_index in range(count):
            sample_index = len(sample_rows)
            reading = 1.0 + sample_index / 1_000.0
            sample_rows.append(
                {
                    "sample_id": f"s{sample_index:03d}",
                    "group_id": group_id,
                    "version": "v3" if group_index % 2 == 0 else "v4",
                    "fold": group_index % 5,
                    "ra_1": reading - 0.02,
                    "ra_2": reading,
                    "ra_3": reading + 0.02,
                    "ra_mean": reading,
                    "sample_weight": 1.0 / count,
                    "split_count": count,
                    "n_rpm": 4_000.0 + 1_000.0 * (group_index % 5),
                    "fz_mm_per_tooth": 0.03 + 0.01 * (group_index % 4),
                    "ap_mm": 0.5 + 0.25 * (group_index % 3),
                }
            )
    manifest = pd.DataFrame(sample_rows)
    folds = manifest.loc[:, ["sample_id", "group_id", "fold"]].copy()
    bundle = DataBundle(
        manifest=manifest.drop(columns="fold"),
        folds=folds,
        windows=pd.DataFrame(),
        fold_audit={"n_folds": 5},
        duration_audit=pd.DataFrame(),
    )
    bundle.manifest.attrs["quality_features"] = pd.DataFrame(
        {
            "sample_id": bundle.manifest["sample_id"],
            **{
                f"quality_{index}": np.full(len(bundle.manifest), float(index))
                for index in range(7)
            },
        }
    )

    mean_rows: list[dict[str, object]] = []
    probability_rows: list[dict[str, object]] = []
    z_values = {90: 1.6448536269514722, 95: 1.959963984540054}
    for row in manifest.itertuples(index=False):
        for seed in (20260723, 20260724, 20260725):
            process_mean = float(row.ra_mean) - 0.10
            residual = 0.15
            correction = 0.5 * residual
            for model, prediction, gate, model_correction in (
                ("P1", process_mean, 0.0, 0.0),
                ("R1", process_mean + residual, 1.0, residual),
                ("G1", process_mean + correction, 0.5, correction),
            ):
                mean_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "group_id": row.group_id,
                        "version": row.version,
                        "fold": row.fold,
                        "seed": seed,
                        "model": model,
                        "target_mean": row.ra_mean,
                        "prediction": prediction,
                        "sample_weight": row.sample_weight,
                        "process_mean": process_mean,
                        "residual": residual,
                        "gate": gate,
                        "correction": model_correction,
                    }
                )
            for scale_model, sigma in (("heteroscedastic", 0.10), ("homoscedastic", 0.12)):
                probability_rows.append(
                    {
                        "sample_id": row.sample_id,
                        "group_id": row.group_id,
                        "version": row.version,
                        "fold": row.fold,
                        "seed": seed,
                        "scale_model": scale_model,
                        "target_mean": row.ra_mean,
                        "ra_1": row.ra_1,
                        "ra_2": row.ra_2,
                        "ra_3": row.ra_3,
                        "sample_weight": row.sample_weight,
                        "mu": process_mean + correction,
                        "sigma": sigma,
                        "gate": 0.5,
                        "correction": correction,
                        "raw_lower_90": process_mean + correction - z_values[90] * sigma,
                        "raw_upper_90": process_mean + correction + z_values[90] * sigma,
                        "raw_lower_95": process_mean + correction - z_values[95] * sigma,
                        "raw_upper_95": process_mean + correction + z_values[95] * sigma,
                        "conformal_q_90": 1.0,
                        "conformal_lower_90": process_mean + correction - sigma,
                        "conformal_upper_90": process_mean + correction + sigma,
                        "conformal_q_95": 1.0,
                        "conformal_lower_95": process_mean + correction - sigma,
                        "conformal_upper_95": process_mean + correction + sigma,
                    }
                )

    scores: list[dict[str, object]] = []
    quantiles: list[dict[str, object]] = []
    group_folds = manifest.groupby("group_id", sort=True)["fold"].first()
    group_counts = manifest.groupby("group_id", sort=True).size()
    for outer_fold in range(5):
        calibration_groups = group_folds.index[group_folds != outer_fold]
        for seed in (20260723, 20260724, 20260725):
            for scale_model in ("heteroscedastic", "homoscedastic"):
                for group_id in calibration_groups:
                    count = int(group_counts.loc[group_id])
                    scores.append(
                        {
                            "group_id": group_id,
                            "outer_fold": outer_fold,
                            "inner_fold": int(int(group_id[1:]) % 4),
                            "seed": seed,
                            "scale_model": scale_model,
                            "score": 1.0,
                            "region_count": count,
                            "reading_count": 3 * count,
                        }
                    )
                for alpha in (0.10, 0.05):
                    count = len(calibration_groups)
                    quantiles.append(
                        {
                            "fold": outer_fold,
                            "seed": seed,
                            "scale_model": scale_model,
                            "alpha": alpha,
                            "group_count": count,
                            "order_index": int(np.ceil((count + 1) * (1 - alpha))),
                            "quantile": 1.0,
                        }
                    )

    config_path = tmp_path / "configs" / "sgrpn_phase_a.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text("phase-a fixture", encoding="utf-8")
    phase_a_root = tmp_path / "outputs" / "sgrpn" / "phase_a"
    phase_a_root.mkdir(parents=True)
    (phase_a_root / "immutable.bin").write_bytes(b"phase-a")
    for legacy in (tmp_path / "outputs" / "scheme1", tmp_path / "outputs" / "scheme1_physics"):
        legacy.mkdir(parents=True)
        (legacy / "immutable.bin").write_bytes(legacy.name.encode("utf-8"))
    config = PhaseBConfig(
        phase_a_config_path=config_path,
        phase_a_config_file_sha256="a" * 64,
        phase_a_acceptance_path=phase_a_root / "evaluation" / "acceptance.json",
        phase_a_run_manifest_path=phase_a_root / "run_manifest.json",
        output_dir=tmp_path / "outputs" / "sgrpn" / "phase_b",
        seeds=(20260723, 20260724, 20260725),
        alphas=(0.10, 0.05),
        inner_splits=4,
        max_epochs=200,
        patience=20,
        variance_learning_rate=1e-3,
        weight_decay=1e-4,
        bootstrap_repetitions=10_000,
    )
    handoff = PhaseAHandoff(
        acceptance_sha256="b" * 64,
        run_manifest_sha256="c" * 64,
        phase_a_config_sha256="a" * 64,
        training_fingerprint="fixture-training",
        cache_sha256="d" * 64,
        input_sha256={"manifest": "e" * 64},
    )
    return (
        config,
        handoff,
        bundle,
        pd.DataFrame(mean_rows),
        pd.DataFrame(probability_rows),
        pd.DataFrame(scores),
        pd.DataFrame(quantiles),
    )


def test_phase_b_report_regenerates_tables_and_figures_without_checkpoints(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config, handoff, bundle, mean, probability, scores, quantiles = _phase_b_report_fixture(tmp_path)
    import roughness.sgrpn.reporting as reporting

    monkeypatch.setattr(reporting, "validate_phase_b_output_root", lambda path: Path(path))
    monkeypatch.setattr(reporting, "_phase_b_fingerprint_map", lambda *args: {}, raising=False)
    immutable_roots = [
        config.phase_a_run_manifest_path.parent,
        tmp_path / "outputs" / "scheme1",
        tmp_path / "outputs" / "scheme1_physics",
    ]
    before = {str(root): _hash_tree(root) for root in immutable_roots}
    written = write_phase_b_report(config, handoff, bundle, mean, probability, scores, quantiles)
    table_bytes = {
        path.relative_to(config.output_dir).as_posix(): path.read_bytes()
        for path in config.output_dir.rglob("*.csv")
    }
    figure_bytes = {
        path.relative_to(config.output_dir).as_posix(): path.read_bytes()
        for path in (config.output_dir / "evaluation" / "figures").glob("*.png")
    }
    assert len(table_bytes) > 10
    assert all(path.is_file() for name, path in written.items() if name.startswith("figure_"))
    assert json.loads((config.output_dir / "evaluation" / "method_notes.json").read_text(encoding="utf-8")) == reporting._PHASE_B_METHOD_NOTES

    for path in list(config.output_dir.rglob("*.csv")) + list(
        (config.output_dir / "evaluation" / "figures").glob("*.png")
    ):
        path.unlink()
    write_phase_b_report(config, handoff, bundle, mean, probability, scores, quantiles)

    assert {
        path.relative_to(config.output_dir).as_posix(): path.read_bytes()
        for path in config.output_dir.rglob("*.csv")
    } == table_bytes
    assert {
        path.relative_to(config.output_dir).as_posix(): path.read_bytes()
        for path in (config.output_dir / "evaluation" / "figures").glob("*.png")
    } == figure_bytes
    assert {str(root): _hash_tree(root) for root in immutable_roots} == before


def _phase_b_complete_report_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> tuple[PhaseBConfig, PhaseAHandoff, DataBundle, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str]]:
    config, handoff, bundle, mean, probability, scores, quantiles = _phase_b_report_fixture(tmp_path)
    import roughness.sgrpn.reporting as reporting

    monkeypatch.setattr(reporting, "validate_phase_b_output_root", lambda path: Path(path))
    fingerprints: dict[str, str] = {}
    devices: dict[str, str] = {}
    inner_frames: list[pd.DataFrame] = []
    for fold in range(5):
        for seed in config.seeds:
            key = f"fold_{fold}/seed_{seed}"
            fingerprints[key] = hashlib.sha256(key.encode("ascii")).hexdigest()
            devices[key] = "cpu"
            fold_dir = config.output_dir / "folds" / f"fold_{fold}" / f"seed_{seed}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            (fold_dir / "complete.json").write_text(
                json.dumps({"fixture": key}), encoding="utf-8"
            )
            inner = pd.DataFrame(
                {"record_type": ["fixture"], "fold": [fold], "seed": [seed]}
            )
            inner_path = fold_dir / "calibration" / "inner_folds.csv"
            inner_path.parent.mkdir(parents=True, exist_ok=True)
            inner.to_csv(inner_path, index=False)
            inner_frames.append(inner)
            nested = {"fold": fold, "seed": seed, "quantiles": {}}
            for scale_model in ("heteroscedastic", "homoscedastic"):
                nested["quantiles"][scale_model] = {}
                for alpha in (0.10, 0.05):
                    row = quantiles.loc[
                        (quantiles["fold"] == fold)
                        & (quantiles["seed"] == seed)
                        & (quantiles["scale_model"] == scale_model)
                        & (quantiles["alpha"] == alpha)
                    ].iloc[0]
                    nested["quantiles"][scale_model][f"{alpha:.2f}"] = {
                        "group_count": int(row["group_count"]),
                        "order_index": int(row["order_index"]),
                        "quantile": float(row["quantile"]),
                    }
            (fold_dir / "calibration" / "quantiles.json").write_text(
                json.dumps(nested), encoding="utf-8"
            )
    (config.output_dir / "run_manifest.json").write_text(
        json.dumps({"selected_device_by_fold_seed": devices}), encoding="utf-8"
    )
    monkeypatch.setattr(reporting, "_phase_b_fingerprint_map", lambda *args: fingerprints, raising=False)
    write_phase_b_report(config, handoff, bundle, mean, probability, scores, quantiles)
    return config, handoff, bundle, mean, probability, scores, quantiles, fingerprints


def test_phase_b_manifest_is_closed_and_deep_validation_rebuilds_nested_quantiles_and_pngs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    (
        config,
        handoff,
        bundle,
        mean,
        probability,
        scores,
        quantiles,
        fingerprints,
    ) = _phase_b_complete_report_fixture(monkeypatch, tmp_path)
    import roughness.sgrpn.reporting as reporting

    inner = pd.concat(
        [
            pd.read_csv(
                config.output_dir / "folds" / f"fold_{fold}" / f"seed_{seed}" / "calibration" / "inner_folds.csv"
            )
            for fold in range(5)
            for seed in config.seeds
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(reporting, "validate_phase_b_handoff", lambda value: handoff)
    monkeypatch.setattr(reporting, "load_sgrpn_config", lambda path: object())
    monkeypatch.setattr(reporting, "load_data_bundle", lambda value: bundle)
    monkeypatch.setattr(
        reporting,
        "_phase_b_saved_fold_inputs",
        lambda *args: (probability, mean, scores, inner),
    )
    monkeypatch.setattr(reporting, "_phase_b_fingerprint_map", lambda *args: fingerprints, raising=False)

    manifest_path = config.output_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["fold_fingerprints"] == fingerprints
    assert set(manifest["selected_device_by_fold_seed"]) == set(fingerprints)
    assert manifest["environment"] and all(isinstance(value, str) and value for value in manifest["environment"].values())
    validate_phase_b_outputs(config)

    original_manifest = manifest_path.read_bytes()
    original_png = (config.output_dir / "evaluation" / "figures" / "coverage_width.png").read_bytes()
    original_quantiles = (
        config.output_dir / "folds" / "fold_0" / "seed_20260723" / "calibration" / "quantiles.json"
    ).read_bytes()
    original_table = (config.output_dir / "evaluation" / "mean_metrics.csv").read_bytes()
    original_marker = (config.output_dir / "folds" / "fold_0" / "seed_20260723" / "complete.json").read_bytes()

    manifest["phase_b_config"]["max_epochs"] = 1
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest|config"):
        validate_phase_b_outputs(config)
    manifest_path.write_bytes(original_manifest)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["fold_fingerprints"]["fold_0/seed_20260723"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        validate_phase_b_outputs(config)
    manifest_path.write_bytes(original_manifest)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["selected_device_by_fold_seed"]["fold_0/seed_20260723"] = "cuda"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="device"):
        validate_phase_b_outputs(config)
    manifest_path.write_bytes(original_manifest)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["evaluation/mean_metrics.csv"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="hash"):
        validate_phase_b_outputs(config)
    manifest_path.write_bytes(original_manifest)

    png_path = config.output_dir / "evaluation" / "figures" / "coverage_width.png"
    png_path.write_bytes(b"replaced")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["evaluation/figures/coverage_width.png"] = hashlib.sha256(
        png_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="figure|PNG"):
        validate_phase_b_outputs(config)
    png_path.write_bytes(original_png)
    manifest_path.write_bytes(original_manifest)

    quantile_path = config.output_dir / "folds" / "fold_0" / "seed_20260723" / "calibration" / "quantiles.json"
    nested = json.loads(quantile_path.read_text(encoding="utf-8"))
    nested["quantiles"]["heteroscedastic"]["0.10"]["quantile"] = 3.0
    quantile_path.write_text(json.dumps(nested), encoding="utf-8")
    with pytest.raises(ValueError, match="quantile"):
        validate_phase_b_outputs(config)
    quantile_path.write_bytes(original_quantiles)

    table_path = config.output_dir / "evaluation" / "mean_metrics.csv"
    table_path.write_text("tampered\n", encoding="utf-8")
    with pytest.raises(ValueError, match="metrics|unreadable"):
        validate_phase_b_outputs(config)
    table_path.write_bytes(original_table)

    marker_path = config.output_dir / "folds" / "fold_0" / "seed_20260723" / "complete.json"
    marker_path.write_text(json.dumps({"fixture": "tampered"}), encoding="utf-8")
    with pytest.raises(ValueError, match="completion"):
        validate_phase_b_outputs(config)
    marker_path.write_bytes(original_marker)


def test_phase_b_coverage_width_figure_includes_width_series(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config, handoff, bundle, mean, probability, scores, quantiles = _phase_b_report_fixture(tmp_path)
    import roughness.sgrpn.reporting as reporting

    monkeypatch.setattr(reporting, "validate_phase_b_output_root", lambda path: Path(path))
    monkeypatch.setattr(reporting, "_phase_b_fingerprint_map", lambda *args: {}, raising=False)
    write_phase_b_report(config, handoff, bundle, mean, probability, scores, quantiles)
    captured: dict[str, object] = {}
    coverage_width_figure = reporting._phase_b_coverage_width_figure

    def capture(coverage: pd.DataFrame, width: pd.DataFrame):
        figure = coverage_width_figure(coverage, width)
        captured["labels"] = [axis.get_ylabel() for axis in figure.axes]
        return figure

    monkeypatch.setattr(reporting, "_phase_b_coverage_width_figure", capture)
    reporting._write_phase_b_figures_from_tables(config.output_dir)
    assert captured["labels"] == ["Coverage", "Mean interval width (μm)"]
