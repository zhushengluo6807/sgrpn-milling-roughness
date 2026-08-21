import os
from pathlib import Path

_matplotlib_cache = Path.cwd() / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def write_prediction_figures(
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    required = {"model", "y_true", "y_pred"}
    missing = sorted(required - set(predictions.columns))
    if missing:
        raise ValueError(f"Predictions missing columns: {missing}")
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    scatter_path = destination / "prediction_scatter.png"
    residual_path = destination / "residual_plot.png"

    figure, axis = plt.subplots(figsize=(7, 6))
    for model, group in predictions.groupby("model"):
        axis.scatter(group["y_true"], group["y_pred"], s=16, alpha=0.6, label=model)
    low = float(min(predictions["y_true"].min(), predictions["y_pred"].min()))
    high = float(max(predictions["y_true"].max(), predictions["y_pred"].max()))
    axis.plot([low, high], [low, high], "k--", linewidth=1)
    axis.set_xlabel("Measured Ra (μm)")
    axis.set_ylabel("Predicted Ra (μm)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(scatter_path, dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 6))
    residual = predictions["y_true"] - predictions["y_pred"]
    for model, indices in predictions.groupby("model").groups.items():
        axis.scatter(
            predictions.loc[indices, "y_pred"],
            residual.loc[indices],
            s=16,
            alpha=0.6,
            label=model,
        )
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set_xlabel("Predicted Ra (μm)")
    axis.set_ylabel("Residual: measured - predicted (μm)")
    axis.legend()
    figure.tight_layout()
    figure.savefig(residual_path, dpi=180)
    plt.close(figure)
    return {"scatter": scatter_path, "residual": residual_path}


def write_error_breakdowns(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, dict[str, Path]]:
    prediction_required = {
        "sample_id",
        "y_true",
        "y_pred",
        "sample_weight",
    }
    dimensions = [
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
        "version",
        "region_index",
    ]
    missing_predictions = sorted(
        prediction_required - set(predictions.columns)
    )
    missing_manifest = sorted(
        {"sample_id", *dimensions} - set(manifest.columns)
    )
    if missing_predictions:
        raise ValueError(f"Predictions missing columns: {missing_predictions}")
    if missing_manifest:
        raise ValueError(f"Manifest missing columns: {missing_manifest}")
    data = predictions.merge(
        manifest[["sample_id", *dimensions]],
        on="sample_id",
        validate="one_to_one",
    )
    data["weighted_absolute_error"] = (
        np.abs(data["y_true"] - data["y_pred"]) * data["sample_weight"]
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    written: dict[str, dict[str, Path]] = {}
    for dimension in dimensions:
        grouped = (
            data.groupby(dimension, dropna=False, sort=True)
            .agg(
                weighted_error_sum=("weighted_absolute_error", "sum"),
                weight_sum=("sample_weight", "sum"),
                segments=("sample_id", "size"),
            )
            .reset_index()
        )
        grouped["weighted_mae"] = (
            grouped["weighted_error_sum"] / grouped["weight_sum"]
        )
        csv_path = destination / f"error_by_{dimension}.csv"
        figure_path = destination / f"error_by_{dimension}.png"
        grouped.to_csv(csv_path, index=False)
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.bar(
            grouped[dimension].astype(str),
            grouped["weighted_mae"],
            color="#4472C4",
        )
        axis.set_xlabel(dimension)
        axis.set_ylabel("Weighted MAE (μm)")
        axis.tick_params(axis="x", rotation=45)
        figure.tight_layout()
        figure.savefig(figure_path, dpi=180)
        plt.close(figure)
        written[dimension] = {"csv": csv_path, "figure": figure_path}
    return written
