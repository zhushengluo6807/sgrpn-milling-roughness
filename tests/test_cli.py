from pathlib import Path

from roughness.cli import run


def test_run_writes_required_outputs(tmp_path: Path):
    result = run(Path("configs/first_round.yaml"), output_override=tmp_path)

    expected = {
        "manifest.csv",
        "audit.json",
        "folds.csv",
        "oof_predictions.csv",
        "fold_metrics.csv",
        "summary_metrics.csv",
        "run_metadata.json",
        "prediction_scatter.png",
        "residual_plot.png",
    }
    assert expected == {path.name for path in tmp_path.iterdir()}
    assert result["rows"] == 586
    assert result["groups"] == 212
