import os
from pathlib import Path

_MATPLOTLIB_CACHE = Path.cwd() / ".cache" / "matplotlib"
_MATPLOTLIB_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MATPLOTLIB_CACHE))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def summarize_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "mae",
        "rmse",
        "r2",
        "weighted_mae",
        "weighted_rmse",
        "weighted_r2",
        "fit_seconds",
        "predict_seconds",
    ]
    summary = fold_metrics.groupby("model")[columns].agg(["mean", "std"])
    summary.columns = [
        f"{metric}_{statistic}" for metric, statistic in summary.columns
    ]
    return summary.reset_index()


def save_diagnostic_plots(
    predictions: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> str:
    best_model = str(
        summary.sort_values("weighted_mae_mean").iloc[0]["model"]
    )
    chosen = predictions[predictions["model"] == best_model]

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(chosen["y_true"], chosen["y_pred"], s=18, alpha=0.7)
    limits = [
        min(chosen["y_true"].min(), chosen["y_pred"].min()),
        max(chosen["y_true"].max(), chosen["y_pred"].max()),
    ]
    axis.plot(limits, limits, "k--", linewidth=1)
    axis.set(
        xlabel="Measured Ra (um)",
        ylabel="Predicted Ra (um)",
        title=f"OOF Prediction: {best_model}",
    )
    figure.tight_layout()
    figure.savefig(output / "prediction_scatter.png", dpi=180)
    plt.close(figure)

    residual = chosen["y_pred"] - chosen["y_true"]
    figure, axis = plt.subplots(figsize=(7, 4))
    axis.scatter(chosen["y_pred"], residual, s=18, alpha=0.7)
    axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
    axis.set(
        xlabel="Predicted Ra (um)",
        ylabel="Residual (um)",
        title=f"OOF Residuals: {best_model}",
    )
    figure.tight_layout()
    figure.savefig(output / "residual_plot.png", dpi=180)
    plt.close(figure)
    return best_model
