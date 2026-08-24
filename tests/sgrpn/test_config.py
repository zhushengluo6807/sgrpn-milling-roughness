from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from roughness.sgrpn.config import (
    PhaseBConfig,
    load_phase_b_config,
    load_sgrpn_config,
    validate_phase_b_handoff,
    validate_phase_b_output_root,
)
from roughness.sgrpn.data import load_data_bundle
from roughness.sgrpn.models import (
    DirectFusionModel,
    ProcessMLP,
    ResidualExpert,
    SelectiveGatedModel,
    VibrationOnlyModel,
)
from roughness.sgrpn.order_spectrum import (
    OrderSpectrumCache,
    load_order_cache,
    save_order_cache,
)
from roughness.sgrpn.training import (
    MODEL_SEQUENCE,
    OOF_COLUMNS,
    PHASE_A_PROTOCOL,
    build_run_fingerprint,
)


def test_phase_a_protocol_is_locked():
    cfg = load_sgrpn_config(Path("configs/sgrpn_phase_a.yaml"))
    assert cfg.sample_rate_hz == 25600
    assert cfg.window_samples == 25600
    assert cfg.order_min == 0.0
    assert cfg.order_max == 90.0
    assert cfg.order_step == 0.25
    assert cfg.seeds == (20260723,)
    assert cfg.inner_splits == 4
    assert cfg.bootstrap_repetitions == 10000


def test_rejects_phase_a_with_multiple_seeds(tmp_path):
    text = Path("configs/sgrpn_phase_a.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad.yaml"
    path.write_text(text.replace("seeds: [20260723]", "seeds: [1, 2]"), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly seed 20260723"):
        load_sgrpn_config(path)


def test_rejects_non_fixed_phase_a_sample_rate(tmp_path):
    text = Path("configs/sgrpn_phase_a.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad-rate.yaml"
    path.write_text(
        text.replace("sample_rate_hz: 25600", "sample_rate_hz: 12800"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_rate_hz.*25600"):
        load_sgrpn_config(path)


@pytest.mark.parametrize(
    "sample_rate",
    ["25600.5", '"25600"', "true", ".inf"],
)
def test_rejects_non_integral_or_non_numeric_phase_a_sample_rate(
    tmp_path, sample_rate
):
    text = Path("configs/sgrpn_phase_a.yaml").read_text(encoding="utf-8")
    path = tmp_path / "bad-rate.yaml"
    path.write_text(
        text.replace("sample_rate_hz: 25600", f"sample_rate_hz: {sample_rate}"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="sample_rate_hz.*25600"):
        load_sgrpn_config(path)


def test_phase_b_protocol_is_exactly_registered():
    config = load_phase_b_config("configs/sgrpn_phase_b.yaml")

    assert config.phase_a_config_file_sha256 == (
        "53768589f6e2311bd3386997ef9295b441290b5607a16f7d688696709110d192"
    )
    assert tuple(config.seeds) == (20260723, 20260724, 20260725)
    assert tuple(config.alphas) == (0.10, 0.05)
    assert config.inner_splits == 4
    assert config.max_epochs == 200
    assert config.patience == 20
    assert config.variance_learning_rate == 1e-3
    assert config.weight_decay == 1e-4
    assert config.bootstrap_repetitions == 10000
    assert config.output_dir.name == "phase_b"


def test_phase_b_rejects_unregistered_seed(tmp_path):
    source = Path("configs/sgrpn_phase_b.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["seeds"] = [20260723, 7, 20260725]
    path = tmp_path / "bad_phase_b.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="exactly 20260723, 20260724, 20260725"):
        load_phase_b_config(path)


@pytest.mark.parametrize("value", ["A" * 64, "0" * 63, "g" * 64])
def test_phase_b_rejects_noncanonical_phase_a_config_file_sha256(tmp_path, value):
    source = Path("configs/sgrpn_phase_b.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload["phase_a_config_file_sha256"] = value
    path = tmp_path / "bad_phase_b.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="phase_a_config_file_sha256"):
        load_phase_b_config(path)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("alphas", [0.05, 0.10], "alphas.*0.10, 0.05"),
        ("inner_splits", True, "inner_splits.*four"),
        ("max_epochs", True, "max_epochs.*200"),
        ("patience", True, "patience.*20"),
        ("variance_learning_rate", True, "variance_learning_rate.*0.001"),
        ("weight_decay", True, "weight_decay.*0.0001"),
        ("bootstrap_repetitions", True, "bootstrap_repetitions.*10000"),
    ],
)
def test_phase_b_rejects_noncanonical_or_boolean_numeric_values(
    tmp_path, field, value, message
):
    source = Path("configs/sgrpn_phase_b.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    payload[field] = value
    path = tmp_path / "bad_phase_b.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_phase_b_config(path)


@pytest.mark.parametrize("change", ["missing", "extra"])
def test_phase_b_rejects_missing_or_extra_keys(tmp_path, change):
    source = Path("configs/sgrpn_phase_b.yaml")
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if change == "missing":
        del payload["patience"]
    else:
        payload["unregistered"] = 1
    path = tmp_path / "bad_phase_b.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="keys"):
        load_phase_b_config(path)


def test_phase_b_rejects_output_root_escape(tmp_path):
    with pytest.raises(ValueError, match="exact Phase B output root"):
        validate_phase_b_output_root(
            tmp_path / "outside",
            output_root=tmp_path / "outputs" / "sgrpn" / "phase_b",
        )


def test_phase_b_accepts_explicit_exact_test_output_root(tmp_path):
    output = tmp_path / "outputs" / "sgrpn" / "phase_b"

    assert validate_phase_b_output_root(output, output_root=output) == output.resolve()
    assert not output.exists()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")


def _phase_a_fold_dir(config: PhaseBConfig, fold: int = 0) -> Path:
    phase_a = load_sgrpn_config(config.phase_a_config_path)
    return phase_a.output_dir / "folds" / f"fold_{fold}" / "seed_20260723"


def _minimal_oof_row(source: pd.Series, fold: int, model: str) -> dict:
    target = float(source["ra_mean"])
    row = {
        "sample_id": str(source["sample_id"]),
        "group_id": str(source["group_id"]),
        "version": str(source["version"]),
        "fold": fold,
        "seed": 20260723,
        "model": model,
        "target": target,
        "prediction": target,
        "sample_weight": float(source["sample_weight"]),
        "process_mean": np.nan,
        "residual": np.nan,
        "gate": np.nan,
    }
    if model == "P1":
        row["process_mean"] = target
    elif model == "R1":
        row["process_mean"] = target
        row["residual"] = 0.0
    elif model == "G1":
        row["process_mean"] = target
        row["residual"] = 0.0
        row["gate"] = 0.5
    return row


@pytest.fixture
def valid_phase_a_v2_fixture(tmp_path, monkeypatch):
    project = tmp_path / "fixture_project"
    configs = project / "configs"
    inputs = project / "inputs"
    signals = inputs / "signals"
    phase_a_output = project / "outputs" / "sgrpn" / "phase_a"
    for path in (configs, inputs, signals):
        path.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    fold_rows = []
    window_rows = []
    for fold in range(5):
        sample_id = f"sample-{fold}"
        group_id = f"group-{fold}"
        signal_path = signals / f"{sample_id}.csv"
        pd.DataFrame({"Ch9_g": [0.1], "Ch10_g": [0.2], "Ch11_g": [0.3]}).to_csv(
            signal_path, index=False
        )
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "group_id": group_id,
                "signal_path": str(signal_path),
                "n_rpm": 1000.0 + fold,
                "fz_mm_per_tooth": 0.1,
                "ap_mm": 0.2,
                "ra_1": 0.4 + fold / 100,
                "ra_2": 0.5 + fold / 100,
                "ra_3": 0.6 + fold / 100,
                "ra_mean": 0.5 + fold / 100,
                "sample_weight": 1.0,
                "split_count": 1.0,
                "version": "v3",
                "duration_s": 1.0 / 25600.0,
            }
        )
        fold_rows.append({"sample_id": sample_id, "group_id": group_id, "fold": fold})
        window_rows.append(
            {
                "segment_id": sample_id,
                "group_id": group_id,
                "csv_path": str(signal_path),
                "window_id": 0,
                "start_sample": 0,
                "end_sample": 1,
                "is_tail_aligned": False,
            }
        )

    manifest_path = inputs / "manifest.csv"
    folds_path = inputs / "folds.csv"
    windows_path = inputs / "window_index.csv"
    m0_path = inputs / "m0_oof.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    pd.DataFrame(fold_rows).to_csv(folds_path, index=False)
    pd.DataFrame(window_rows).to_csv(windows_path, index=False)
    pd.DataFrame({"sample_id": ["unused"], "prediction": [0.5]}).to_csv(
        m0_path, index=False
    )

    phase_a_path = configs / "sgrpn_phase_a.yaml"
    phase_a_payload = {
        "manifest_path": str(manifest_path),
        "folds_path": str(folds_path),
        "window_index_path": str(windows_path),
        "m0_oof_path": str(m0_path),
        "output_dir": str(phase_a_output),
        "sample_rate_hz": 25600,
        "window_samples": 25600,
        "order_min": 0.0,
        "order_max": 90.0,
        "order_step": 0.25,
        "seeds": [20260723],
        "inner_splits": 4,
        "max_epochs": 200,
        "patience": 20,
        "process_learning_rate": 0.001,
        "signal_learning_rate": 0.0003,
        "gate_learning_rate": 0.001,
        "weight_decay": 0.0001,
        "huber_delta_um": 0.10,
        "gate_penalty": 0.001,
        "correction_penalty": 0.01,
        "bootstrap_repetitions": 10000,
    }
    phase_a_path.write_text(yaml.safe_dump(phase_a_payload, sort_keys=False), encoding="utf-8")
    phase_a = load_sgrpn_config(phase_a_path)
    bundle = load_data_bundle(phase_a)
    cache = OrderSpectrumCache(
        segment_ids=tuple(pd.DataFrame(manifest_rows)["sample_id"]),
        spectra=np.zeros((5, 3, 361), dtype=np.float32),
        offsets=np.arange(6, dtype=np.int64),
        quality=np.zeros((5, 7), dtype=np.float32),
        durations_s=np.full(5, 1.0 / 25600.0),
    )
    save_order_cache(cache, bundle, phase_a)
    cache = load_order_cache(bundle, phase_a)
    fingerprint = build_run_fingerprint(phase_a, bundle, cache)

    for fold in range(5):
        fold_dir = phase_a_output / "folds" / f"fold_{fold}" / "seed_20260723"
        p1 = ProcessMLP()
        r1 = ResidualExpert()
        models = {
            "P1": p1,
            "V1": VibrationOnlyModel(),
            "F1": DirectFusionModel(),
            "R1": r1,
            "G1": SelectiveGatedModel(process_expert=p1, residual_expert=r1),
        }
        artifact_hashes = {}
        for stage in MODEL_SEQUENCE:
            metadata = {
                "protocol": PHASE_A_PROTOCOL,
                "fingerprint": fingerprint.value,
                "fold": fold,
                "seed": 20260723,
                "stage": stage,
                "best_epochs": [1, 1, 1, 1],
                "refit_epochs": 1,
            }
            checkpoint = fold_dir / "checkpoints" / f"{stage}.pt"
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    **metadata,
                    "model": stage,
                    "completed_stage": stage,
                    "device": "cpu",
                    "model_state": models[stage].state_dict(),
                },
                checkpoint,
            )
            history = fold_dir / "history" / f"{stage}.csv"
            history.parent.mkdir(parents=True, exist_ok=True)
            history_rows = [
                {
                    "phase": "selection",
                    "inner_fold": inner_fold,
                    "epoch": 1,
                    "train_loss": 1.0,
                    "validation_loss": 1.0,
                    **{key: value for key, value in metadata.items() if key != "best_epochs"},
                    "best_epochs": "[1,1,1,1]",
                }
                for inner_fold in range(4)
            ]
            history_rows.append(
                {
                    "phase": "refit",
                    "inner_fold": -1,
                    "epoch": 1,
                    "train_loss": 1.0,
                    "validation_loss": 1.0,
                    **{key: value for key, value in metadata.items() if key != "best_epochs"},
                    "best_epochs": "[1,1,1,1]",
                }
            )
            pd.DataFrame(history_rows).to_csv(history, index=False)
            scaler = fold_dir / "scalers" / f"{stage}_scalers.npz"
            scaler.parent.mkdir(parents=True, exist_ok=True)
            arrays = {
                "process_mean": np.zeros(9, dtype=np.float32),
                "process_scale": np.ones(9, dtype=np.float32),
            }
            if stage != "P1":
                arrays.update(
                    spectrum_mean=np.zeros((3, 361), dtype=np.float32),
                    spectrum_scale=np.ones((3, 361), dtype=np.float32),
                    quality_mean=np.zeros(7, dtype=np.float32),
                    quality_scale=np.ones(7, dtype=np.float32),
                )
            np.savez_compressed(
                scaler,
                **arrays,
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
            )
            for artifact_path in (checkpoint, history, scaler):
                artifact_hashes[artifact_path.relative_to(fold_dir).as_posix()] = sha256_file(
                    artifact_path
                )

        source = bundle.manifest.loc[bundle.folds["fold"].to_numpy() == fold].iloc[0]
        predictions = pd.DataFrame(
            [_minimal_oof_row(source, fold, model) for model in MODEL_SEQUENCE],
            columns=OOF_COLUMNS,
        )
        prediction_path = fold_dir / "oof_predictions.csv"
        predictions.to_csv(prediction_path, index=False)
        common = {
            "status": "complete",
            "protocol": PHASE_A_PROTOCOL,
            "fingerprint": fingerprint.value,
            "fold": fold,
            "seed": 20260723,
            "models": list(MODEL_SEQUENCE),
            "completed_stages": list(MODEL_SEQUENCE),
            "next_stage": None,
            "device": "cpu",
        }
        _write_json(fold_dir / "state.json", common)
        _write_json(
            fold_dir / "complete.json",
            {
                **common,
                "artifacts": artifact_hashes,
                "prediction_rows": len(predictions),
                "predictions_sha256": sha256_file(prediction_path),
            },
        )

    acceptance_path = phase_a_output / "evaluation" / "acceptance.json"
    _write_json(
        acceptance_path,
        {
            "schema_version": "sgrpn-phase-a-acceptance-v1",
            "proceed_to_phase_b": True,
            "phase_b_executed": False,
            "subjective_override_allowed": False,
            "decision": {"proceed_to_phase_b": True, "path": "transfer_safety"},
        },
    )
    run_manifest_path = phase_a_output / "run_manifest.json"
    _write_json(
        run_manifest_path,
        {
            "artifacts": {"evaluation/acceptance.json": sha256_file(acceptance_path)},
            "input_sha256": {
                "manifest": sha256_file(manifest_path),
                "folds": sha256_file(folds_path),
                "window_index": sha256_file(windows_path),
                "m0_oof": sha256_file(m0_path),
            },
            "config_sha256": fingerprint.config_sha256,
            "manifest_sha256": fingerprint.manifest_sha256,
            "folds_sha256": fingerprint.folds_sha256,
            "cache_sha256": fingerprint.cache_sha256,
            "training_fingerprint": fingerprint.value,
            "phase_b_executed": False,
        },
    )

    output_dir = project / "outputs" / "sgrpn" / "phase_b"
    monkeypatch.setattr("roughness.sgrpn.config._PROJECT_PHASE_B_OUTPUT", output_dir.resolve())
    assert not output_dir.exists()
    config = PhaseBConfig(
        phase_a_config_path=phase_a_path,
        phase_a_config_file_sha256=sha256_file(phase_a_path),
        phase_a_acceptance_path=acceptance_path,
        phase_a_run_manifest_path=run_manifest_path,
        output_dir=output_dir,
        seeds=(20260723, 20260724, 20260725),
        alphas=(0.10, 0.05),
        inner_splits=4,
        max_epochs=200,
        patience=20,
        variance_learning_rate=0.001,
        weight_decay=0.0001,
        bootstrap_repetitions=10000,
    )
    return config, output_dir


def _artifact_paths(config: PhaseBConfig, artifact: str) -> tuple[Path, Path, str]:
    fold_dir = _phase_a_fold_dir(config)
    relative = {
        "checkpoint": "checkpoints/P1.pt",
        "history": "history/P1.csv",
        "scaler": "scalers/P1_scalers.npz",
    }[artifact]
    return fold_dir / relative, fold_dir / "complete.json", relative


def mutate_deep_phase_a_metadata(
    config: PhaseBConfig, *, artifact: str, mutation: str
) -> tuple[Path, Path, str]:
    path, marker_path, relative = _artifact_paths(config, artifact)
    field = "fingerprint" if mutation == "fingerprint_mismatch" else "protocol"
    value = {
        "v1_protocol": "sgrpn-phase-a-v1",
        "protocol_mismatch": "unregistered-protocol",
        "fingerprint_mismatch": "0" * 64,
    }.get(mutation)
    if artifact == "checkpoint":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if mutation == "missing_protocol":
            payload.pop("protocol")
        else:
            payload[field] = value
        torch.save(payload, path)
    elif artifact == "history":
        frame = pd.read_csv(path, dtype=str)
        frame[field] = "" if mutation == "missing_protocol" else value
        frame.to_csv(path, index=False)
    else:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(arrays.pop("metadata_json").item()))
        if mutation == "missing_protocol":
            metadata.pop("protocol")
        else:
            metadata[field] = value
        np.savez_compressed(
            path,
            **arrays,
            metadata_json=np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":"))),
        )
    return path, marker_path, relative


def update_marker_artifact_sha256(marker_path: Path, relative: str, value: str) -> None:
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    before = dict(marker["artifacts"])
    marker["artifacts"][relative] = value
    assert {
        key: registered
        for key, registered in marker["artifacts"].items()
        if key != relative
    } == {key: registered for key, registered in before.items() if key != relative}
    _write_json(marker_path, marker)


def mutate_registered_artifact_byte(
    config: PhaseBConfig, *, artifact: str
) -> tuple[Path, Path, str]:
    path, marker_path, relative = _artifact_paths(config, artifact)
    path.write_bytes(path.read_bytes() + b"x")
    return path, marker_path, relative


def test_phase_b_handoff_accepts_fully_consistent_v2_without_output(
    valid_phase_a_v2_fixture,
):
    config, output_dir = valid_phase_a_v2_fixture
    phase_a = load_sgrpn_config(config.phase_a_config_path)
    bundle = load_data_bundle(phase_a)
    cache = load_order_cache(bundle, phase_a)
    expected = build_run_fingerprint(phase_a, bundle, cache)

    handoff = validate_phase_b_handoff(config)

    assert handoff.phase_a_config_sha256 == config.phase_a_config_file_sha256
    assert handoff.training_fingerprint == expected.value
    assert handoff.cache_sha256 == expected.cache_sha256
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("artifact", "mutation"),
    [
        ("checkpoint", "missing_protocol"),
        ("checkpoint", "v1_protocol"),
        ("checkpoint", "protocol_mismatch"),
        ("checkpoint", "fingerprint_mismatch"),
        ("history", "missing_protocol"),
        ("history", "v1_protocol"),
        ("history", "protocol_mismatch"),
        ("history", "fingerprint_mismatch"),
        ("scaler", "missing_protocol"),
        ("scaler", "v1_protocol"),
        ("scaler", "protocol_mismatch"),
        ("scaler", "fingerprint_mismatch"),
    ],
)
def test_phase_b_handoff_rejects_deep_metadata_after_hash_is_reregistered(
    valid_phase_a_v2_fixture, artifact, mutation
):
    config, output_dir = valid_phase_a_v2_fixture
    artifact_path, marker_path, relative_name = mutate_deep_phase_a_metadata(
        config, artifact=artifact, mutation=mutation
    )
    update_marker_artifact_sha256(marker_path, relative_name, sha256_file(artifact_path))
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["artifacts"][relative_name] == sha256_file(artifact_path)

    with pytest.raises(ValueError, match=rf"{artifact}.*(protocol|fingerprint|metadata)"):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


@pytest.mark.parametrize("artifact", ["checkpoint", "history", "scaler"])
def test_phase_b_handoff_rejects_registered_hash_mismatch_before_metadata(
    valid_phase_a_v2_fixture, artifact
):
    config, output_dir = valid_phase_a_v2_fixture
    artifact_path, marker_path, relative_name = mutate_registered_artifact_byte(
        config, artifact=artifact
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    assert marker["artifacts"][relative_name] != sha256_file(artifact_path)

    with pytest.raises(ValueError, match=rf"hash mismatch: .*{artifact}"):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("filename", "mutation"),
    [
        ("complete.json", "missing_protocol"),
        ("complete.json", "v1_protocol"),
        ("complete.json", "fingerprint_mismatch"),
        ("state.json", "missing_protocol"),
        ("state.json", "v1_protocol"),
        ("state.json", "fingerprint_mismatch"),
    ],
)
def test_phase_b_handoff_rejects_marker_or_state_identity(
    valid_phase_a_v2_fixture, filename, mutation
):
    config, output_dir = valid_phase_a_v2_fixture
    path = _phase_a_fold_dir(config) / filename
    payload = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "missing_protocol":
        payload.pop("protocol")
    elif mutation == "v1_protocol":
        payload["protocol"] = "sgrpn-phase-a-v1"
    else:
        payload["fingerprint"] = "0" * 64
    _write_json(path, payload)

    identity_name = "marker" if filename == "complete.json" else "state"
    with pytest.raises(ValueError, match=rf"{identity_name}.*(protocol|fingerprint)"):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


def test_phase_b_handoff_rejects_closed_acceptance_before_output(
    valid_phase_a_v2_fixture,
):
    config, output_dir = valid_phase_a_v2_fixture
    acceptance = json.loads(config.phase_a_acceptance_path.read_text(encoding="utf-8"))
    acceptance["proceed_to_phase_b"] = False
    _write_json(config.phase_a_acceptance_path, acceptance)

    with pytest.raises(ValueError, match="Phase A acceptance gate"):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("nested_proceed", False),
        ("subjective_override_allowed", True),
        ("phase_b_executed", True),
    ],
)
def test_phase_b_handoff_rejects_each_acceptance_gate_invariant(
    valid_phase_a_v2_fixture, field, value
):
    config, output_dir = valid_phase_a_v2_fixture
    acceptance = json.loads(config.phase_a_acceptance_path.read_text(encoding="utf-8"))
    if field == "nested_proceed":
        acceptance["decision"]["proceed_to_phase_b"] = value
    else:
        acceptance[field] = value
    _write_json(config.phase_a_acceptance_path, acceptance)

    with pytest.raises(ValueError, match="Phase A acceptance gate"):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


def test_phase_b_handoff_rejects_manifest_already_executed_before_output(
    valid_phase_a_v2_fixture,
):
    config, output_dir = valid_phase_a_v2_fixture
    manifest = json.loads(config.phase_a_run_manifest_path.read_text(encoding="utf-8"))
    manifest["phase_b_executed"] = True
    _write_json(config.phase_a_run_manifest_path, manifest)

    with pytest.raises(ValueError, match="already executed"):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


@pytest.mark.parametrize("source", ["acceptance", "manifest", "folds", "cache"])
def test_phase_b_handoff_rejects_source_byte_mutation_before_output(
    valid_phase_a_v2_fixture, source
):
    config, output_dir = valid_phase_a_v2_fixture
    phase_a = load_sgrpn_config(config.phase_a_config_path)
    path = {
        "acceptance": config.phase_a_acceptance_path,
        "manifest": phase_a.manifest_path,
        "folds": phase_a.folds_path,
        "cache": phase_a.output_dir / "features" / "order_spectrum_cache.npz",
    }[source]
    path.write_bytes(path.read_bytes() + b" ")

    with pytest.raises(ValueError, match=source):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


def test_phase_b_handoff_rejects_wrong_marker_artifact_hash(
    valid_phase_a_v2_fixture,
):
    config, output_dir = valid_phase_a_v2_fixture
    _, marker_path, relative = _artifact_paths(config, "checkpoint")
    update_marker_artifact_sha256(marker_path, relative, "0" * 64)

    with pytest.raises(ValueError, match="hash mismatch: .*checkpoint"):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


@pytest.mark.parametrize(
    "field",
    [
        "config_sha256",
        "manifest_sha256",
        "folds_sha256",
        "cache_sha256",
        "training_fingerprint",
    ],
)
def test_phase_b_handoff_independently_recomputes_each_fingerprint_component(
    valid_phase_a_v2_fixture, field
):
    config, output_dir = valid_phase_a_v2_fixture
    manifest = json.loads(config.phase_a_run_manifest_path.read_text(encoding="utf-8"))
    manifest[field] = "0" * 64
    _write_json(config.phase_a_run_manifest_path, manifest)

    with pytest.raises(ValueError, match=field):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


def test_phase_b_handoff_rejects_yaml_byte_only_change_before_output(
    valid_phase_a_v2_fixture,
):
    config, output_dir = valid_phase_a_v2_fixture
    path = config.phase_a_config_path
    phase_a_before = load_sgrpn_config(path)
    bundle_before = load_data_bundle(phase_a_before)
    cache_before = load_order_cache(bundle_before, phase_a_before)
    before = build_run_fingerprint(phase_a_before, bundle_before, cache_before)
    path.write_text(
        path.read_text(encoding="utf-8") + "\n# byte-only change\n", encoding="utf-8"
    )
    phase_a_after = load_sgrpn_config(path)
    bundle_after = load_data_bundle(phase_a_after)
    cache_after = load_order_cache(bundle_after, phase_a_after)
    after = build_run_fingerprint(phase_a_after, bundle_after, cache_after)
    assert after.config_sha256 == before.config_sha256
    assert after.value == before.value
    assert sha256_file(path) != config.phase_a_config_file_sha256

    with pytest.raises(ValueError, match="Phase A config file SHA-256"):
        validate_phase_b_handoff(config)
    assert not output_dir.exists()


def test_phase_b_handoff_rejects_wrong_registered_config_file_hash(
    valid_phase_a_v2_fixture,
):
    config, output_dir = valid_phase_a_v2_fixture
    wrong = replace(config, phase_a_config_file_sha256="0" * 64)

    with pytest.raises(ValueError, match="Phase A config file SHA-256"):
        validate_phase_b_handoff(wrong)
    assert not output_dir.exists()
