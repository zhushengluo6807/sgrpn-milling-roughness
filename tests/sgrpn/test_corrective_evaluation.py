import json
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import pytest


def _fake_artifacts() -> dict[str, object]:
    return {
        "probability_metrics": pd.DataFrame({"metric": ["coverage"], "value": [0.9]}),
        "mean_metrics": pd.DataFrame({"metric": ["mae"], "value": [0.1]}),
        "method_notes": {"analysis_status": "post_audit_corrective_reanalysis"},
    }


def _write_training_seal(root: Path):
    import hashlib

    from roughness.sgrpn.corrective import CORRECTIVE_PROTOCOL
    from roughness.sgrpn import corrective_evaluation as module

    for fold in range(5):
        for seed in (20260723, 20260724, 20260725):
            marker = root / "folds" / f"fold_{fold}" / f"seed_{seed}" / "complete.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}", encoding="utf-8")
    run_manifest = root / "run_manifest.json"
    run_manifest.write_text(
        json.dumps({"training_status": "complete", "completed_units": 15}),
        encoding="utf-8",
    )
    phase_a = root.parent / f"{root.name}-phase-a.yaml"
    phase_a.write_text("frozen: true\n", encoding="utf-8")
    phase_a_hash = hashlib.sha256(phase_a.read_bytes()).hexdigest()
    config = SimpleNamespace(
        output_dir=root,
        phase_b_config_file_sha256="e" * 64,
        phase_b_run_manifest_sha256="f" * 64,
        phase_b_immutable_after_sha256="1" * 64,
    )
    phase_b = SimpleNamespace(
        phase_a_config_path=phase_a,
        phase_a_config_file_sha256=phase_a_hash,
    )

    digest, file_count, total_bytes = module._training_tree_inventory(root)
    markers = {
        path.relative_to(root).as_posix(): module._sha256_file(path)
        for path in sorted(root.rglob("complete.json"))
    }
    (root / "training_seal.json").write_text(
        json.dumps(
            {
                "protocol": CORRECTIVE_PROTOCOL,
                "status": "sealed",
                "training_tree_sha256": digest,
                "file_count": file_count,
                "total_bytes": total_bytes,
                "completed_units": 15,
                "phase_a_config_file_sha256": phase_a_hash,
                "phase_b_config_file_sha256": config.phase_b_config_file_sha256,
                "phase_b_run_manifest_sha256": config.phase_b_run_manifest_sha256,
                "phase_b_immutable_after_sha256": config.phase_b_immutable_after_sha256,
                "source_revision": module._current_source_revision(),
                "source_files": module._source_file_hashes(),
                "unit_completion_sha256": markers,
                "run_manifest_sha256": module._sha256_file(run_manifest),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return digest, config, phase_b


def test_create_training_seal_is_exclusive_and_keeps_tree_digest(tmp_path: Path):
    from roughness.sgrpn.corrective_evaluation import (
        corrective_training_tree_digest,
        create_corrective_training_seal,
    )

    phase_a = tmp_path / "phase-a.yaml"
    phase_a.write_text("frozen: true\n", encoding="utf-8")
    phase_a_hash = __import__("hashlib").sha256(phase_a.read_bytes()).hexdigest()
    root = tmp_path / "formal"
    for fold in range(5):
        for seed in (20260723, 20260724, 20260725):
            marker = root / "folds" / f"fold_{fold}" / f"seed_{seed}" / "complete.json"
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text("{}", encoding="utf-8")
    (root / "run_manifest.json").write_text(
        json.dumps({"training_status": "complete", "completed_units": 15}),
        encoding="utf-8",
    )
    before = corrective_training_tree_digest(root)
    config = SimpleNamespace(
        output_dir=root,
        phase_b_config_file_sha256="a" * 64,
        phase_b_run_manifest_sha256="b" * 64,
        phase_b_immutable_after_sha256="c" * 64,
    )
    phase_b = SimpleNamespace(
        phase_a_config_path=phase_a,
        phase_a_config_file_sha256=phase_a_hash,
    )

    seal = create_corrective_training_seal(config, phase_b)
    assert seal.is_file()
    assert corrective_training_tree_digest(root) == before
    with pytest.raises(ValueError, match="already exists"):
        create_corrective_training_seal(config, phase_b)


def _claim_metrics(*, hetero_coverage_90: float = 0.90) -> pd.DataFrame:
    rows = []
    for model, winkler in (("heteroscedastic", 0.5), ("homoscedastic", 0.6)):
        for coverage in (0.90, 0.95):
            rows.extend(
                [
                    {
                        "aggregation": "all_seed",
                        "interval_type": "conformal",
                        "scale_model": model,
                        "nominal_coverage": coverage,
                        "metric": "simultaneous_group_coverage",
                        "value": (
                            hetero_coverage_90
                            if model == "heteroscedastic" and coverage == 0.90
                            else coverage
                        ),
                    },
                    {
                        "aggregation": "all_seed",
                        "interval_type": "conformal",
                        "scale_model": model,
                        "nominal_coverage": coverage,
                        "metric": "winkler_score",
                        "value": winkler,
                    },
                ]
            )
    return pd.DataFrame(rows)


def test_corrective_claim_requires_coverage_and_two_winkler_wins():
    from roughness.sgrpn.corrective_evaluation import _build_claim_decision

    accepted = _build_claim_decision(_claim_metrics())
    rejected = _build_claim_decision(_claim_metrics(hetero_coverage_90=0.89))

    assert accepted["emphasize_heteroscedasticity"] is True
    assert rejected["emphasize_heteroscedasticity"] is False
    assert rejected["retain_both_scale_variants"] is True
    assert rejected["retain_group_split_conformal_method"] is True


def test_corrective_evaluation_is_one_shot_and_preserves_training_tree(
    tmp_path: Path, monkeypatch
):
    from roughness.sgrpn import corrective_evaluation as module

    training = tmp_path / "training.bin"
    training.write_bytes(b"frozen-training")
    _, config, phase_b = _write_training_seal(tmp_path)
    monkeypatch.setattr(
        module, "_load_training_predictions", lambda *args: (object(), object())
    )
    monkeypatch.setattr(module, "_build_non_claim_artifacts", lambda *args: _fake_artifacts())
    monkeypatch.setattr(
        module, "_build_claim_decision", lambda *args: {"retain_group_conformal": True}
    )

    written = module.evaluate_corrective_once(
        config,
        phase_b,
        object(),
        object(),
    )

    assert training.read_bytes() == b"frozen-training"
    assert (tmp_path / "evaluation" / "evaluation_invocation.json").is_file()
    assert (tmp_path / "evaluation" / "evaluation_complete.json").is_file()
    assert written["claim_decision"].name == "claim_decision.json"
    with pytest.raises(ValueError, match="already been invoked"):
        module.evaluate_corrective_once(
            config,
            phase_b,
            object(),
            object(),
        )


def test_corrective_evaluation_failure_consumes_the_single_invocation(
    tmp_path: Path, monkeypatch
):
    from roughness.sgrpn import corrective_evaluation as module

    (tmp_path / "training.bin").write_bytes(b"frozen-training")
    _, config, phase_b = _write_training_seal(tmp_path)
    monkeypatch.setattr(
        module, "_load_training_predictions", lambda *args: (object(), object())
    )

    def fail(*args):
        raise RuntimeError("synthetic evaluation failure")

    monkeypatch.setattr(module, "_build_non_claim_artifacts", fail)
    monkeypatch.setattr(module, "_build_claim_decision", lambda *args: {})
    with pytest.raises(RuntimeError, match="synthetic evaluation failure"):
        module.evaluate_corrective_once(
            config,
            phase_b,
            object(),
            object(),
        )
    invocation = json.loads(
        (tmp_path / "evaluation" / "evaluation_invocation.json").read_text(
            encoding="utf-8"
        )
    )
    assert invocation["evaluation_invocations"] == 1
    assert not (tmp_path / "evaluation" / "evaluation_complete.json").exists()
    with pytest.raises(ValueError, match="already been invoked"):
        module.evaluate_corrective_once(
            config,
            phase_b,
            object(),
            object(),
        )


def test_corrective_evaluation_rejects_changed_training_tree(tmp_path: Path):
    from roughness.sgrpn import corrective_evaluation as module

    (tmp_path / "training.bin").write_bytes(b"before")
    _, config, phase_b = _write_training_seal(tmp_path)
    (tmp_path / "training.bin").write_bytes(b"changed")
    with pytest.raises(ValueError, match="training tree hash"):
        module.evaluate_corrective_once(
            config,
            phase_b,
            object(),
            object(),
        )
    assert (tmp_path / "evaluation" / "evaluation_invocation.json").is_file()


def test_corrective_evaluation_loader_failure_consumes_invocation(tmp_path: Path, monkeypatch):
    from roughness.sgrpn import corrective_evaluation as module

    (tmp_path / "training.bin").write_bytes(b"frozen-training")
    _, config, phase_b = _write_training_seal(tmp_path)

    def fail(*args):
        raise RuntimeError("synthetic load failure")

    monkeypatch.setattr(module, "_load_training_predictions", fail)
    with pytest.raises(RuntimeError, match="synthetic load failure"):
        module.evaluate_corrective_once(
            config,
            phase_b,
            object(),
            object(),
        )
    assert (tmp_path / "evaluation" / "evaluation_invocation.json").is_file()


def test_corrective_evaluation_atomic_invocation_allows_one_concurrent_caller(
    tmp_path: Path, monkeypatch
):
    from roughness.sgrpn import corrective_evaluation as module

    (tmp_path / "training.bin").write_bytes(b"frozen-training")
    _, config, phase_b = _write_training_seal(tmp_path)
    monkeypatch.setattr(
        module, "_load_training_predictions", lambda *args: (object(), object())
    )
    monkeypatch.setattr(module, "_build_non_claim_artifacts", lambda *args: _fake_artifacts())
    monkeypatch.setattr(module, "_build_claim_decision", lambda *args: {})

    def invoke():
        try:
            module.evaluate_corrective_once(
                config,
                phase_b,
                object(),
                object(),
            )
            return "complete"
        except ValueError as error:
            return str(error)

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(lambda _: invoke(), range(2)))
    assert outcomes.count("complete") == 1
    assert sum("already been invoked" in outcome for outcome in outcomes) == 1


def test_corrective_evaluation_validator_recomputes_and_rejects_tamper(
    tmp_path: Path, monkeypatch
):
    from roughness.sgrpn import corrective_evaluation as module

    (tmp_path / "training.bin").write_bytes(b"frozen-training")
    _, config, phase_b = _write_training_seal(tmp_path)
    monkeypatch.setattr(
        module, "_load_training_predictions", lambda *args: (object(), object())
    )
    monkeypatch.setattr(module, "_build_non_claim_artifacts", lambda *args: _fake_artifacts())
    monkeypatch.setattr(module, "_build_claim_decision", lambda *args: {})
    module.evaluate_corrective_once(
        config,
        phase_b,
        object(),
        object(),
    )

    module.validate_corrective_evaluation_outputs(
        config,
        phase_b,
        object(),
        object(),
    )
    monkeypatch.setattr(
        module,
        "_build_claim_decision",
        lambda *args: (_ for _ in ()).throw(AssertionError("claim assessor reached")),
    )
    module.validate_corrective_evaluation_outputs(
        config,
        phase_b,
        object(),
        object(),
    )
    (tmp_path / "evaluation" / "probability_metrics.csv").write_text(
        "metric,value\ncoverage,0.1\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="artifact hash closure"):
        module.validate_corrective_evaluation_outputs(
            config,
            phase_b,
            object(),
            object(),
        )
