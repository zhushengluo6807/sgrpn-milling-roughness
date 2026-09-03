from pathlib import Path
import hashlib
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from roughness.sgrpn import cli
from roughness.sgrpn.config import SGRPNConfig
from roughness.sgrpn.data import DataBundle
from roughness.sgrpn.order_spectrum import OrderSpectrumCache, save_order_cache
from roughness.sgrpn.reporting import _atomic_csv
from roughness.sgrpn.training import MODEL_SEQUENCE, RunFingerprint


def test_parser_exposes_registered_phase_a_and_phase_b_commands():
    parser = cli.build_parser()
    choices = parser._subparsers._group_actions[0].choices
    assert set(choices) == {
        "audit",
        "features",
        "train-phase-a",
        "evaluate-phase-a",
        "run-phase-a",
        "preflight-phase-b",
        "train-phase-b",
        "evaluate-phase-b",
        "run-phase-b",
    }
    train = parser.parse_args(["train-phase-a", "--config", "x.yaml", "--fold", "0", "--device", "auto", "--resume"])
    assert train.fold == 0
    assert train.device == "auto"
    assert train.resume is True
    phase_b = parser.parse_args(
        ["train-phase-b", "--config", "x.yaml", "--fold", "0", "--seed", "20260723", "--device", "auto", "--resume"]
    )
    assert (phase_b.fold, phase_b.seed, phase_b.device, phase_b.resume) == (
        0,
        20260723,
        "auto",
        True,
    )


def test_evaluate_never_calls_training(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    events = []
    monkeypatch.setattr(cli, "load_sgrpn_config", lambda path: object())
    monkeypatch.setattr(cli, "_evaluate", lambda config: events.append("evaluate"))
    monkeypatch.setattr(cli, "_train", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("retrained")))

    assert cli.main(["evaluate-phase-a", "--config", str(tmp_path / "cfg.yaml")]) == 0
    assert events == ["evaluate"]


def test_run_phase_a_orders_all_steps_and_never_runs_phase_b(monkeypatch: pytest.MonkeyPatch):
    events = []
    config = object()
    monkeypatch.setattr(cli, "load_sgrpn_config", lambda path: config)
    monkeypatch.setattr(cli, "_audit", lambda value: events.append("audit"))
    monkeypatch.setattr(cli, "_features", lambda value: events.append("features"))
    monkeypatch.setattr(cli, "_train", lambda value, **kwargs: events.append("train"))
    monkeypatch.setattr(cli, "_evaluate", lambda value: events.append("evaluate"))

    assert cli.main(["run-phase-a", "--config", "cfg.yaml", "--device", "cpu", "--resume"]) == 0
    assert events == ["audit", "features", "train", "evaluate"]


def test_cli_returns_nonzero_for_incompatible_formal_artifacts(monkeypatch: pytest.MonkeyPatch, capsys):
    monkeypatch.setattr(cli, "load_sgrpn_config", lambda path: object())
    monkeypatch.setattr(cli, "_evaluate", lambda config: (_ for _ in ()).throw(ValueError("OOF Cartesian incomplete")))

    assert cli.main(["evaluate-phase-a", "--config", "cfg.yaml"]) != 0
    assert "OOF Cartesian incomplete" in capsys.readouterr().err


def _canonical_bundle() -> DataBundle:
    manifest = pd.DataFrame(
        {
            "sample_id": [f"s{fold}" for fold in range(5)],
            "group_id": [f"g{fold}" for fold in range(5)],
            "version": ["v3"] * 5,
            "sample_weight": [1.0] * 5,
            "ra_mean": np.linspace(0.5, 0.9, 5),
            "n_rpm": [4000.0] * 5,
            "fz_mm_per_tooth": [0.05] * 5,
            "ap_mm": [1.0] * 5,
        }
    )
    folds = pd.DataFrame(
        {
            "sample_id": manifest["sample_id"],
            "group_id": manifest["group_id"],
            "fold": range(5),
        }
    )
    return DataBundle(
        manifest=manifest,
        folds=folds,
        windows=pd.DataFrame(),
        fold_audit={"n_folds": 5},
        duration_audit=pd.DataFrame(
            {"sample_id": manifest["sample_id"], "duration_recomputed_s": 1.0}
        ),
    )


def test_formal_loader_uses_canonical_current_bundle_cache_and_fingerprint(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / "outputs" / "sgrpn" / "phase_a"
    bundle = _canonical_bundle()
    cache = object()
    expected = RunFingerprint("current", "config", "manifest", "folds", "cache")
    config = SimpleNamespace(output_dir=output)
    events = []

    monkeypatch.setattr(cli, "validate_phase_a_output_root", lambda value, output_root=None: output)
    monkeypatch.setattr(cli, "load_data_bundle", lambda value: events.append("bundle") or bundle)
    monkeypatch.setattr(cli, "load_order_cache", lambda loaded, value: events.append("cache") or cache)
    monkeypatch.setattr(cli, "build_run_fingerprint", lambda value, loaded, cached: events.append("fingerprint") or expected)

    class StopAfterDurationBoundary(Exception):
        pass

    def verify(root, fold, expected_frame, expected_fingerprint):
        events.append(("verify", fold, expected_fingerprint.value))
        return expected_fingerprint.value, "cpu", pd.DataFrame()

    def validate_manifest(root, value, fingerprint, current_duration):
        events.append(("duration", fingerprint.value))
        pd.testing.assert_frame_equal(current_duration, bundle.duration_audit)
        raise StopAfterDurationBoundary

    monkeypatch.setattr(cli, "_verify_fold_artifacts", verify)
    monkeypatch.setattr(cli, "_validate_formal_run_manifest", validate_manifest)
    with pytest.raises(StopAfterDurationBoundary):
        cli.load_formal_phase_a_predictions(config, output_root=output)
    assert events == [
        "bundle",
        "cache",
        "fingerprint",
        *(("verify", fold, "current") for fold in range(5)),
        ("duration", "current"),
    ]


@pytest.mark.parametrize(
    "malformation",
    [
        "malformed-spectra",
        "malformed-offsets",
        "malformed-duration",
        "malformed-window-binding",
        "rewritten-cache-metadata",
    ],
)
def test_formal_loader_stops_on_canonical_current_artifact_failure_before_markers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, malformation: str
):
    output = tmp_path / "outputs" / "sgrpn" / "phase_a"
    config = SimpleNamespace(output_dir=output)
    bundle = _canonical_bundle()
    marker_reads = []
    monkeypatch.setattr(cli, "validate_phase_a_output_root", lambda value, output_root=None: output)
    if malformation == "malformed-window-binding":
        monkeypatch.setattr(
            cli, "load_data_bundle", lambda value: (_ for _ in ()).throw(ValueError(malformation))
        )
    else:
        monkeypatch.setattr(cli, "load_data_bundle", lambda value: bundle)
        monkeypatch.setattr(
            cli, "load_order_cache", lambda loaded, value: (_ for _ in ()).throw(ValueError(malformation))
        )
    monkeypatch.setattr(cli, "_verify_fold_artifacts", lambda *args: marker_reads.append(args))

    with pytest.raises(ValueError, match=malformation):
        cli.load_formal_phase_a_predictions(config, output_root=output)
    assert marker_reads == []


@pytest.mark.parametrize(
    "current_state",
    ["changed-config", "changed-manifest", "changed-folds", "changed-cache"],
)
def test_rewritten_marker_and_hash_cannot_self_attest_old_fingerprint(
    tmp_path: Path, current_state: str
):
    output = tmp_path / "outputs" / "sgrpn" / "phase_a"
    fold_dir = output / "folds" / "fold_0" / "seed_20260723"
    fold_dir.mkdir(parents=True)
    (fold_dir / "complete.json").write_text(
        json.dumps(
            {
                "status": "complete",
                "fingerprint": "attacker-rewritten-old-state",
                "fold": 0,
                "seed": 20260723,
                "models": list(MODEL_SEQUENCE),
                "completed_stages": list(MODEL_SEQUENCE),
                "predictions_sha256": "attacker-rewritten-hash",
                "artifacts": {},
            }
        ),
        encoding="utf-8",
    )
    expected = RunFingerprint(f"current-recomputed-{current_state}", "", "", "", "")

    with pytest.raises(ValueError, match="current.*fingerprint|fingerprint.*current"):
        cli._verify_fold_artifacts(output, 0, pd.DataFrame(), expected)


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("fold", "0.0"),
        ("fold", "+0"),
        ("fold", "00"),
        ("fold", "0e0"),
        ("fold", "true"),
        ("seed", "20260723.0"),
        ("seed", "+20260723"),
        ("seed", "020260723"),
        ("seed", "2.0260723e7"),
        ("seed", "false"),
    ],
)
def test_m0_rejects_noncanonical_fold_and_seed_text(tmp_path: Path, column: str, value: str):
    path = tmp_path / "m0.csv"
    row = {
        "sample_id": "001",
        "group_id": "01",
        "model": "M0",
        "fold": "0",
        "seed": "20260723",
        "y_true": 1.0,
        "y_pred": 1.0,
        "sample_weight": 1.0,
    }
    row[column] = value
    pd.DataFrame([row]).to_csv(path, index=False)
    manifest = pd.DataFrame(
        {
            "sample_id": ["001"],
            "group_id": ["01"],
            "version": ["v3"],
            "ra_mean": [1.0],
            "sample_weight": [1.0],
        }
    )
    folds = pd.DataFrame({"sample_id": ["001"], "group_id": ["01"], "fold": [0]})

    with pytest.raises(ValueError, match="canonical integral"):
        cli._load_m0(SimpleNamespace(m0_oof_path=path), manifest, folds)


def test_cli_output_guard_rejects_arbitrary_matching_suffix(tmp_path: Path):
    ambiguous = tmp_path / "attacker" / "outputs" / "sgrpn" / "phase_a"
    with pytest.raises(ValueError, match="exact project.*output root"):
        cli._validate_output_dir(SimpleNamespace(output_dir=ambiguous))


def test_formal_run_manifest_requires_complete_current_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / "outputs" / "sgrpn" / "phase_a"
    output.mkdir(parents=True)
    inputs = {}
    for name in ("manifest", "folds", "window_index", "m0_oof"):
        path = tmp_path / f"{name}.csv"
        path.write_text(name, encoding="utf-8")
        inputs[name] = path
    duration = output / "audit" / "duration_audit.csv"
    duration.parent.mkdir(parents=True)
    duration.write_text("sample_id,duration_recomputed_s\ns0,1.0\n", encoding="utf-8")
    config = SimpleNamespace(
        manifest_path=inputs["manifest"],
        folds_path=inputs["folds"],
        window_index_path=inputs["window_index"],
        m0_oof_path=inputs["m0_oof"],
    )
    expected = RunFingerprint("current", "", "", "", "")
    monkeypatch.setattr(cli, "config_fingerprint", lambda value, phase: "config-current")
    manifest = {
        "audit_status": "complete",
        "feature_status": "complete",
        "training_status": "complete",
        "config_fingerprint": "config-current",
        "training_fingerprint": "current",
        "selected_device": "cpu",
        "input_sha256": {name: cli._sha256(path) for name, path in inputs.items()},
        "duration_audit": {
            "provenance": "canonical_current_data",
            "row_count": 1,
            "sha256": cli._sha256(duration),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    current_duration = pd.DataFrame(
        {"sample_id": ["s0"], "duration_recomputed_s": [1.0]}
    )
    validated = cli._validate_formal_run_manifest(
        output, config, expected, current_duration
    )
    assert validated["selected_device"] == "cpu"

    inputs["m0_oof"].write_text("attacker changed M0", encoding="utf-8")
    with pytest.raises(ValueError, match="input fingerprint"):
        cli._validate_formal_run_manifest(output, config, expected, current_duration)

    inputs["m0_oof"].write_text("m0_oof", encoding="utf-8")
    manifest.pop("audit_status")
    (output / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        cli._validate_formal_run_manifest(output, config, expected, current_duration)


def test_formal_duration_validation_round_trips_atomic_csv_floats_exactly(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    output = tmp_path / "outputs" / "sgrpn" / "phase_a"
    inputs = {}
    for name in ("manifest", "folds", "window_index", "m0_oof"):
        path = tmp_path / f"{name}.csv"
        path.write_text(name, encoding="utf-8")
        inputs[name] = path
    canonical = pd.DataFrame(
        {
            "sample_id": ["v3_34_seg1"],
            "row_count": [79104],
            "duration_source_s": [2.95668],
            "duration_recomputed_s": [3.09],
            "absolute_difference_s": [0.13331999999999988],
            "mismatch_over_1ms": [True],
        }
    )
    duration_path = _atomic_csv(output / "audit" / "duration_audit.csv", canonical)
    config = SimpleNamespace(
        manifest_path=inputs["manifest"],
        folds_path=inputs["folds"],
        window_index_path=inputs["window_index"],
        m0_oof_path=inputs["m0_oof"],
    )
    expected = RunFingerprint("current", "", "", "", "")
    monkeypatch.setattr(cli, "config_fingerprint", lambda value, phase: "config-current")
    run_manifest = {
        "audit_status": "complete",
        "feature_status": "complete",
        "training_status": "complete",
        "config_fingerprint": "config-current",
        "training_fingerprint": "current",
        "selected_device": "cpu",
        "input_sha256": {name: cli._sha256(path) for name, path in inputs.items()},
        "duration_audit": {
            "provenance": "canonical_current_data",
            "row_count": 1,
            "sha256": cli._sha256(duration_path),
        },
    }
    (output / "run_manifest.json").write_text(json.dumps(run_manifest), encoding="utf-8")

    validated = cli._validate_formal_run_manifest(
        output, config, expected, canonical
    )

    assert validated["duration_audit"]["sha256"] == cli._sha256(duration_path)


@pytest.mark.parametrize(
    "tampering",
    ("stale", "rewritten_and_rehashed", "missing_row_count", "wrong_row_count"),
)
def test_formal_run_manifest_rejects_duration_not_exactly_bound_to_current_canonical_data(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tampering: str
):
    output = tmp_path / "outputs" / "sgrpn" / "phase_a"
    output.mkdir(parents=True)
    inputs = {}
    for name in ("manifest", "folds", "window_index", "m0_oof"):
        path = tmp_path / f"{name}.csv"
        path.write_text(name, encoding="utf-8")
        inputs[name] = path
    duration = output / "audit" / "duration_audit.csv"
    duration.parent.mkdir(parents=True)
    canonical = pd.DataFrame(
        {"sample_id": ["s0"], "duration_recomputed_s": [1.0]}
    )
    duration.write_bytes(canonical.to_csv(index=False).encode("utf-8"))
    config = SimpleNamespace(
        manifest_path=inputs["manifest"],
        folds_path=inputs["folds"],
        window_index_path=inputs["window_index"],
        m0_oof_path=inputs["m0_oof"],
    )
    expected = RunFingerprint("current", "", "", "", "")
    monkeypatch.setattr(cli, "config_fingerprint", lambda value, phase: "config-current")
    run_manifest = {
        "audit_status": "complete",
        "feature_status": "complete",
        "training_status": "complete",
        "config_fingerprint": "config-current",
        "training_fingerprint": "current",
        "selected_device": "cpu",
        "input_sha256": {name: cli._sha256(path) for name, path in inputs.items()},
        "duration_audit": {
            "provenance": "canonical_current_data",
            "row_count": 1,
            "sha256": cli._sha256(duration),
        },
    }
    if tampering in {"stale", "rewritten_and_rehashed"}:
        duration.write_text(
            "sample_id,duration_recomputed_s\ns0,2.0\n", encoding="utf-8"
        )
        if tampering == "rewritten_and_rehashed":
            run_manifest["duration_audit"]["sha256"] = cli._sha256(duration)
    elif tampering == "missing_row_count":
        run_manifest["duration_audit"].pop("row_count")
    else:
        run_manifest["duration_audit"]["row_count"] = 2
    (output / "run_manifest.json").write_text(
        json.dumps(run_manifest), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="duration audit"):
        cli._validate_formal_run_manifest(output, config, expected, canonical)


def _cache_bound_formal_fixture(
    tmp_path: Path,
) -> tuple[SGRPNConfig, DataBundle, Path, Path]:
    bundle = _canonical_bundle()
    manifest = bundle.manifest.assign(
        signal_path="unused.csv",
        duration_s=1.0,
    )
    bundle = DataBundle(
        manifest=manifest,
        folds=bundle.folds,
        windows=pd.DataFrame(
            {
                "segment_id": bundle.manifest["sample_id"],
                "group_id": bundle.manifest["group_id"],
                "csv_path": ["unused.csv"] * 5,
                "window_id": [0] * 5,
                "start_sample": [0] * 5,
                "end_sample": [25600] * 5,
                "is_tail_aligned": [False] * 5,
            }
        ),
        fold_audit=bundle.fold_audit,
        duration_audit=bundle.duration_audit,
    )
    manifest_path = tmp_path / "manifest.csv"
    folds_path = tmp_path / "folds.csv"
    windows_path = tmp_path / "windows.csv"
    m0_path = tmp_path / "m0.csv"
    bundle.manifest.to_csv(manifest_path, index=False)
    bundle.folds.to_csv(folds_path, index=False)
    bundle.windows.to_csv(windows_path, index=False)
    m0_path.write_text("unused", encoding="utf-8")
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
        max_epochs=200,
        patience=20,
        process_learning_rate=1e-3,
        signal_learning_rate=3e-4,
        gate_learning_rate=1e-3,
        weight_decay=1e-4,
        huber_delta_um=0.10,
        gate_penalty=1e-3,
        correction_penalty=1e-2,
        bootstrap_repetitions=10_000,
    )
    cache = OrderSpectrumCache(
        segment_ids=tuple(bundle.manifest["sample_id"]),
        spectra=np.zeros((5, 3, 361), dtype=np.float32),
        offsets=np.arange(6, dtype=np.int64),
        quality=np.zeros((5, 7), dtype=np.float32),
        durations_s=np.ones(5, dtype=np.float64),
    )
    npz_path, metadata_path = save_order_cache(cache, bundle, config)
    return config, bundle, npz_path, metadata_path


@pytest.mark.parametrize(
    "malformation",
    ["spectra", "offsets", "duration", "window-binding", "cache-metadata"],
)
def test_formal_loader_rejects_rehashed_malformed_full_cache_before_marker_trust(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, malformation: str
):
    config, bundle, npz_path, metadata_path = _cache_bound_formal_fixture(tmp_path)
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if malformation == "spectra":
        arrays["spectra"] = arrays["spectra"][:, :2, :]
    elif malformation == "offsets":
        arrays["offsets"][-1] = arrays["offsets"][-2]
    elif malformation == "duration":
        arrays["durations_s"][0] = np.nan
    elif malformation == "window-binding":
        bundle.windows.loc[len(bundle.windows)] = bundle.windows.iloc[0]
    else:
        metadata["segment_ids"] = list(reversed(metadata["segment_ids"]))
    if malformation != "window-binding":
        np.savez_compressed(npz_path, **arrays)
        metadata["cache_sha256"] = cli._sha256(npz_path)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    marker_reads = []
    monkeypatch.setattr(cli, "load_data_bundle", lambda value: bundle)
    monkeypatch.setattr(cli, "_verify_fold_artifacts", lambda *args: marker_reads.append(args))

    with pytest.raises(ValueError, match="cache|spectr|offset|duration|binding"):
        cli.load_formal_phase_a_predictions(config, output_root=config.output_dir)
    assert marker_reads == []


@pytest.mark.parametrize("command", ("train-phase-b", "evaluate-phase-b", "run-phase-b"))
def test_phase_b_gate_runs_before_any_output_root_creation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, command: str
):
    output = tmp_path / "outputs" / "sgrpn" / "phase_b"
    config = SimpleNamespace(output_dir=output)
    events: list[str] = []
    monkeypatch.setattr(cli, "load_phase_b_config", lambda path: config)

    def closed_gate(value):
        events.append("preflight")
        raise ValueError("Phase A gate is closed")

    monkeypatch.setattr(cli, "_preflight_phase_b", closed_gate)
    monkeypatch.setattr(
        cli,
        "_train_phase_b",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trained")),
    )
    monkeypatch.setattr(
        cli,
        "_evaluate_phase_b",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluated")),
    )

    assert cli.main([command, "--config", "phase-b.yaml"]) != 0
    assert events == ["preflight"]
    assert not output.exists()


def test_phase_b_run_orders_preflight_train_evaluate_then_deep_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config = SimpleNamespace(output_dir=tmp_path / "outputs" / "sgrpn" / "phase_b")
    events: list[str] = []
    monkeypatch.setattr(cli, "load_phase_b_config", lambda path: config)
    monkeypatch.setattr(cli, "_preflight_phase_b", lambda value: events.append("preflight") or object())
    monkeypatch.setattr(cli, "_phase_b_evaluation_complete", lambda value: False)
    monkeypatch.setattr(
        cli,
        "_train_phase_b",
        lambda value, context, **kwargs: events.append("train"),
    )
    monkeypatch.setattr(
        cli,
        "_evaluate_phase_b",
        lambda value, context: events.append("evaluate"),
    )
    monkeypatch.setattr(
        cli,
        "validate_phase_b_outputs",
        lambda value: events.append("validate"),
    )

    assert cli.main(["run-phase-b", "--config", "phase-b.yaml", "--device", "cpu"]) == 0
    assert events == ["preflight", "train", "evaluate", "validate"]


def test_phase_b_resume_of_completed_evaluation_only_deep_validates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    config = SimpleNamespace(output_dir=tmp_path / "outputs" / "sgrpn" / "phase_b")
    events: list[str] = []
    monkeypatch.setattr(cli, "load_phase_b_config", lambda path: config)
    monkeypatch.setattr(cli, "_preflight_phase_b", lambda value: events.append("preflight") or object())
    monkeypatch.setattr(cli, "_phase_b_evaluation_complete", lambda value: True)
    monkeypatch.setattr(
        cli,
        "_train_phase_b",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("trained")),
    )
    monkeypatch.setattr(
        cli,
        "_evaluate_phase_b",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("evaluated")),
    )
    monkeypatch.setattr(
        cli,
        "validate_phase_b_outputs",
        lambda value: events.append("validate"),
    )

    assert cli.main(["run-phase-b", "--config", "phase-b.yaml", "--resume"]) == 0
    assert events == ["preflight", "validate"]


@pytest.mark.parametrize(
    "arguments",
    (
        ["train-phase-b", "--config", "phase-b.yaml", "--fold", "5"],
        ["train-phase-b", "--config", "phase-b.yaml", "--seed", "7"],
    ),
)
def test_phase_b_parser_rejects_unregistered_fold_and_seed(arguments: list[str]):
    with pytest.raises(SystemExit) as error:
        cli.main(arguments)
    assert error.value.code == 2


def _persisted_cli_phase_b_fold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from tests.sgrpn.test_phase_b_training import _persisted_phase_b_fold

    return _persisted_phase_b_fold(tmp_path, monkeypatch)


def test_phase_b_train_resume_uses_marker_device_after_interruption(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    marker_path, _artifacts, config, phase_a, handoff, bundle, cache = (
        _persisted_cli_phase_b_fold(tmp_path, monkeypatch)
    )
    output = marker_path.parents[3]
    keys = [
        f"fold_{fold}/seed_{seed}"
        for fold in range(5)
        for seed in (20260723, 20260724, 20260725)
    ]
    existing_key = keys[0]
    assert not (output / "run_manifest.json").exists()
    calls: list[str] = []
    monkeypatch.setattr(cli, "validate_phase_b_output_root", lambda path: Path(path))
    monkeypatch.setattr(cli, "_selected_device", lambda requested: "cuda")
    monkeypatch.setattr(
        cli, "run_phase_b", lambda *args, **kwargs: calls.append(kwargs["device"])
    )

    cli._train_phase_b(
        config, (handoff, phase_a, bundle, cache), device="cuda", resume=True
    )

    recorded = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))[
        "selected_device_by_fold_seed"
    ]
    assert recorded[existing_key] == "cpu"
    assert all(recorded[key] == "cuda" for key in keys[1:])
    assert calls == ["cuda"]


def test_phase_b_train_rejects_manifest_missing_reused_marker_device(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    marker_path, _artifacts, config, phase_a, handoff, bundle, cache = (
        _persisted_cli_phase_b_fold(tmp_path, monkeypatch)
    )
    manifest_path = marker_path.parents[3] / "run_manifest.json"
    manifest_path.write_text(
        json.dumps({"selected_device_by_fold_seed": {}}), encoding="utf-8"
    )
    before = manifest_path.read_bytes()
    calls: list[str] = []
    monkeypatch.setattr(cli, "validate_phase_b_output_root", lambda path: Path(path))
    monkeypatch.setattr(cli, "_selected_device", lambda requested: "cuda")
    monkeypatch.setattr(
        cli, "run_phase_b", lambda *args, **kwargs: calls.append("all")
    )
    monkeypatch.setattr(
        cli, "run_phase_b_fold", lambda *args, **kwargs: calls.append("one")
    )

    with pytest.raises(ValueError, match="device provenance"):
        cli._train_phase_b(
            config, (handoff, phase_a, bundle, cache), device="cuda", resume=True
        )
    assert calls == []
    assert manifest_path.read_bytes() == before


def test_phase_b_train_rejects_manifest_device_that_disagrees_with_marker(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    marker_path, _artifacts, config, phase_a, handoff, bundle, cache = (
        _persisted_cli_phase_b_fold(tmp_path, monkeypatch)
    )
    output = marker_path.parents[3]
    existing_key = "fold_0/seed_20260723"
    (output / "run_manifest.json").write_text(
        json.dumps({"selected_device_by_fold_seed": {existing_key: "cuda"}}),
        encoding="utf-8",
    )
    calls: list[str] = []
    monkeypatch.setattr(cli, "validate_phase_b_output_root", lambda path: Path(path))
    monkeypatch.setattr(cli, "_selected_device", lambda requested: "cuda")
    monkeypatch.setattr(
        cli, "run_phase_b", lambda *args, **kwargs: calls.append(kwargs["device"])
    )

    with pytest.raises(ValueError, match="device provenance"):
        cli._train_phase_b(
            config, (handoff, phase_a, bundle, cache), device="cuda", resume=True
        )
    assert calls == []


def test_phase_b_train_rejects_legacy_marker_without_labelling_it_from_request(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    marker_path, _artifacts, config, phase_a, handoff, bundle, cache = (
        _persisted_cli_phase_b_fold(tmp_path, monkeypatch)
    )
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    marker.pop("selected_device", None)
    marker_path.write_text(json.dumps(marker), encoding="utf-8")
    calls: list[str] = []
    monkeypatch.setattr(cli, "validate_phase_b_output_root", lambda path: Path(path))
    monkeypatch.setattr(cli, "_selected_device", lambda requested: "cuda")
    monkeypatch.setattr(
        cli, "run_phase_b", lambda *args, **kwargs: calls.append(kwargs["device"])
    )

    with pytest.raises(ValueError, match="incompatible completed Phase B fold"):
        cli._train_phase_b(
            config, (handoff, phase_a, bundle, cache), device="cuda", resume=True
        )
    assert calls == []
    assert not (marker_path.parents[3] / "run_manifest.json").exists()


def test_phase_b_train_resume_preserves_final_manifest_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
):
    from dataclasses import asdict

    import roughness.sgrpn.reporting as reporting
    from roughness.sgrpn.config import PhaseAHandoff, PhaseBConfig

    output = tmp_path / "outputs" / "sgrpn" / "phase_b"
    output.mkdir(parents=True)
    config = PhaseBConfig(
        phase_a_config_path=tmp_path / "phase-a.yaml",
        phase_a_config_file_sha256="a" * 64,
        phase_a_acceptance_path=tmp_path / "acceptance.json",
        phase_a_run_manifest_path=tmp_path / "run-manifest.json",
        output_dir=output,
        seeds=(20260723, 20260724, 20260725),
        alphas=(0.10, 0.05),
        inner_splits=4,
        max_epochs=200,
        patience=20,
        variance_learning_rate=0.001,
        weight_decay=0.0001,
        bootstrap_repetitions=10000,
    )
    handoff = PhaseAHandoff(
        acceptance_sha256="b" * 64,
        run_manifest_sha256="c" * 64,
        phase_a_config_sha256="d" * 64,
        training_fingerprint="fixture-training",
        cache_sha256="e" * 64,
        input_sha256={"manifest": "f" * 64},
    )
    keys = tuple(
        f"fold_{fold}/seed_{seed}"
        for fold in range(5)
        for seed in config.seeds
    )
    fingerprints = {
        key: hashlib.sha256(f"fingerprint/{key}".encode("ascii")).hexdigest()
        for key in keys
    }
    completion_hashes = {
        key: hashlib.sha256(f"completion/{key}".encode("ascii")).hexdigest()
        for key in keys
    }
    devices = {key: "cpu" for key in keys}
    handoff_file_sha256 = "1" * 64
    manifest = {
        "schema_version": "sgrpn-phase-b-run-v1",
        "phase_b_config": reporting._jsonable(asdict(config)),
        "phase_b_config_sha256": reporting._phase_b_config_hash(config),
        "phase_a_handoff": asdict(handoff),
        "handoff_file_sha256": handoff_file_sha256,
        "protocol": reporting.PHASE_B_PROTOCOL,
        "status": "complete",
        "seeds": list(config.seeds),
        "alphas": list(config.alphas),
        "input_fingerprints": dict(handoff.input_sha256),
        "cache_sha256": handoff.cache_sha256,
        "training_fingerprint": handoff.training_fingerprint,
        "fold_fingerprints": fingerprints,
        "fold_completion_sha256": completion_hashes,
        "selected_device_by_fold_seed": devices,
        "device_provenance_sha256": reporting._phase_b_device_provenance_hash(
            devices, fingerprints, completion_hashes
        ),
        "bootstrap": {
            "repetitions": reporting.BOOTSTRAP_REPETITIONS,
            "seed": reporting.BOOTSTRAP_SEED,
            "resampling_unit": "group_id",
        },
        "environment": {
            "python_version": "3.12.13",
            "python_executable": "fixture-python",
            "platform": "fixture-platform",
        },
        "artifacts": {},
    }
    assert set(manifest) == reporting._PHASE_B_MANIFEST_KEYS
    manifest_path = output / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    before = manifest_path.read_bytes()
    validation_events: list[PhaseBConfig] = []

    monkeypatch.setattr(cli, "load_phase_b_config", lambda path: config)
    monkeypatch.setattr(
        cli, "_preflight_phase_b", lambda value: (handoff, object(), object(), object())
    )
    monkeypatch.setattr(cli, "validate_phase_b_output_root", lambda path: Path(path))
    monkeypatch.setattr(cli, "_selected_device", lambda requested: "cpu")
    monkeypatch.setattr(
        cli,
        "_phase_b_existing_completed_devices",
        lambda *args, **kwargs: {key: "cpu" for key in keys},
        raising=False,
    )
    monkeypatch.setattr(cli, "run_phase_b", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        cli, "validate_phase_b_outputs", lambda value: validation_events.append(value)
    )

    assert cli.main(["train-phase-b", "--config", "phase-b.yaml", "--resume"]) == 0
    assert manifest_path.read_bytes() == before
    assert validation_events == [config]
    reporting._phase_b_validate_manifest_contract(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        config=config,
        handoff=handoff,
        handoff_file_sha256=handoff_file_sha256,
        fingerprints=fingerprints,
        completion_hashes=completion_hashes,
    )
