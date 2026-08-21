import json
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from roughness.scheme1.cli import (
    completed_run_matches,
    main,
    validate_complete_neural_oof,
)
from roughness.scheme1.training import compute_run_fingerprint


def test_cli_help_exits_successfully(capsys):
    with pytest.raises(SystemExit) as error:
        main(["--help"])

    assert error.value.code == 0
    assert "audit" in capsys.readouterr().out


def test_module_help_does_not_initialize_plotting_cache():
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "roughness.scheme1.cli", "--help"],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "font_manager cache" not in result.stderr


def test_completed_run_requires_matching_formal_epoch_request(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "oof_predictions.csv").write_text("sample_id\ns1\n")
    (run_dir / "run_metadata.json").write_text(
        json.dumps({"status": "complete", "requested_max_epochs": 200}),
        encoding="utf-8",
    )

    assert completed_run_matches(run_dir, requested_max_epochs=200)
    assert not completed_run_matches(run_dir, requested_max_epochs=2)


def test_run_fingerprint_changes_when_an_input_file_changes(tmp_path):
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("a", encoding="utf-8")
    second.write_text("b", encoding="utf-8")

    original = compute_run_fingerprint(
        {"model": "N1", "fold": 0, "seed": 1},
        [first, second],
    )
    second.write_text("changed", encoding="utf-8")
    changed = compute_run_fingerprint(
        {"model": "N1", "fold": 0, "seed": 1},
        [first, second],
    )

    assert original != changed


def test_complete_neural_oof_rejects_duplicate_or_missing_predictions():
    frame = pd.DataFrame(
        {
            "sample_id": ["s1", "s1"],
            "model": ["N1", "N1"],
            "seed": [1, 1],
            "y_pred": [0.8, 0.9],
        }
    )

    with pytest.raises(ValueError, match="duplicate"):
        validate_complete_neural_oof(
            frame,
            expected_sample_ids={"s1", "s2"},
            models=["N1"],
            seeds=[1],
        )
