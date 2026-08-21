import json

import pandas as pd

from roughness.scheme1_physics.reporting import (
    write_formula_difference_report,
    write_gate_reports,
    write_method_notes,
)


def test_formula_report_contains_absolute_and_relative_error(tmp_path):
    formula_predictions = pd.DataFrame(
        {
            "sample_id": ["s0", "s1"],
            "radius_mm": [0.2, 0.2],
            "fz_mm_per_tooth": [0.02, 0.04],
            "exact_um": [0.06251, 0.25031],
            "word_um": [0.06250, 0.25000],
        }
    )
    paths = write_formula_difference_report(formula_predictions, tmp_path)
    table = pd.read_csv(paths["table"])
    assert {
        "sample_id",
        "radius_mm",
        "exact_um",
        "word_um",
        "absolute_difference_um",
        "relative_difference",
    } <= set(table)
    assert paths["figure"].stat().st_size > 0


def test_gate_report_only_uses_gated_models(tmp_path):
    gated_oof = pd.DataFrame(
        {
            "sample_id": ["s0", "s1", "s0", "s1"],
            "model": ["PW2", "PW2", "PE2", "PE2"],
            "gate": [0.2, 0.8, 0.3, 0.7],
            "n_rpm": [6000, 7000, 6000, 7000],
            "fz_mm_per_tooth": [0.02, 0.04, 0.02, 0.04],
            "ap_mm": [0.5, 1.0, 0.5, 1.0],
            "base_ra": [0.05, 0.1, 0.051, 0.101],
        }
    )
    paths = write_gate_reports(gated_oof, tmp_path)
    summary = pd.read_csv(paths["summary"])
    assert set(summary["model"]) == {"PW2", "PE2"}
    assert summary["gate_min"].between(0, 1).all()
    assert summary["gate_max"].between(0, 1).all()
    assert all(path.stat().st_size > 0 for path in paths.values())


def test_method_notes_state_interpretation_limits(tmp_path):
    path = write_method_notes(tmp_path)
    notes = json.loads(path.read_text(encoding="utf-8"))
    assert "sensitivity scales" in notes["effective_radius"]
    assert "not calibrated physical probabilities" in notes["gate_meaning"]
