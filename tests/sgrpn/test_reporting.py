import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from roughness.sgrpn.reporting import write_phase_a_report


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
