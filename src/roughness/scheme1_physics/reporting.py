import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

_matplotlib_cache = Path.cwd() / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2F5D8A"
GOLD = "#C6922D"
INK = "#242A30"
GRID = "#D8DEE5"


def _style_axis(axis, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", color=INK, fontsize=12, pad=18)
    axis.text(
        0,
        1.01,
        subtitle,
        transform=axis.transAxes,
        color="#5C6670",
        fontsize=9,
        va="bottom",
    )
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.7)
    axis.set_axisbelow(True)


def write_formula_difference_report(
    formula_predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    required = {
        "sample_id",
        "radius_mm",
        "fz_mm_per_tooth",
        "exact_um",
        "word_um",
    }
    missing = sorted(required - set(formula_predictions.columns))
    if missing:
        raise ValueError(f"Formula predictions missing columns: {missing}")
    destination = Path(output_dir)
    figures = destination / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    table = formula_predictions.copy()
    table["absolute_difference_um"] = np.abs(
        table["exact_um"] - table["word_um"]
    )
    table["relative_difference"] = table["absolute_difference_um"] / np.maximum(
        np.abs(table["exact_um"]), 1e-12
    )
    table_path = destination / "formula_difference_by_candidate.csv"
    summary_path = destination / "formula_difference_by_candidate_summary.csv"
    figure_path = figures / "formula_relative_difference.png"
    table.to_csv(table_path, index=False)
    summary = (
        table.groupby("radius_mm", as_index=False)
        .agg(
            samples=("sample_id", "size"),
            mean_absolute_difference_um=("absolute_difference_um", "mean"),
            max_absolute_difference_um=("absolute_difference_um", "max"),
            mean_relative_difference=("relative_difference", "mean"),
            max_relative_difference=("relative_difference", "max"),
        )
        .sort_values("radius_mm")
    )
    summary.to_csv(summary_path, index=False)

    figure, axis = plt.subplots(figsize=(8, 5.2))
    for radius, group in table.groupby("radius_mm", sort=True):
        ordered = group.sort_values("fz_mm_per_tooth")
        axis.plot(
            ordered["fz_mm_per_tooth"],
            100.0 * ordered["relative_difference"],
            marker="o",
            markersize=3,
            linewidth=1.2,
            label=f"{radius:g} mm",
        )
    _style_axis(
        axis,
        "Exact and Word formula relative difference",
        "Difference at shared candidate radius; denominator is exact Ra",
    )
    axis.set_xlabel("Feed per tooth, fz (mm/tooth)")
    axis.set_ylabel("Relative difference (%)")
    axis.legend(title="Candidate radius", frameon=False, ncol=2)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=180, facecolor="white")
    plt.close(figure)
    return {
        "table": table_path,
        "summary": summary_path,
        "figure": figure_path,
    }


def write_gate_reports(
    gated_oof: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    required = {
        "model",
        "gate",
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
    }
    missing = sorted(required - set(gated_oof.columns))
    if missing:
        raise ValueError(f"Gate OOF missing columns: {missing}")
    data = gated_oof[gated_oof["model"].isin(["PW2", "PE2"])].copy()
    if data.empty or not data["gate"].between(0, 1).all():
        raise ValueError("Gate values must be present in [0, 1]")
    destination = Path(output_dir)
    gate_dir = destination / "gates"
    figures = destination / "figures"
    gate_dir.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    summary_path = gate_dir / "gate_summary.csv"
    bins_path = gate_dir / "gate_by_process_bin.csv"
    summary = (
        data.groupby("model", as_index=False)
        .agg(
            samples=("gate", "size"),
            gate_mean=("gate", "mean"),
            gate_std=("gate", "std"),
            gate_min=("gate", "min"),
            gate_q25=("gate", lambda values: values.quantile(0.25)),
            gate_median=("gate", "median"),
            gate_q75=("gate", lambda values: values.quantile(0.75)),
            gate_max=("gate", "max"),
        )
    )
    summary.to_csv(summary_path, index=False)
    process_summary = (
        data.groupby(
            ["model", "n_rpm", "fz_mm_per_tooth", "ap_mm"],
            as_index=False,
        )
        .agg(gate_mean=("gate", "mean"), samples=("gate", "size"))
    )
    process_summary.to_csv(bins_path, index=False)

    paths = {"summary": summary_path, "process_bins": bins_path}
    colors = {"PW2": BLUE, "PE2": GOLD}
    for model in ("PW2", "PE2"):
        figure_path = figures / f"gate_distribution_{model}.png"
        figure, axis = plt.subplots(figsize=(7, 4.8))
        values = data.loc[data["model"] == model, "gate"]
        axis.hist(
            values,
            bins=np.linspace(0, 1, 21),
            color=colors[model],
            edgecolor=INK,
            linewidth=0.5,
        )
        _style_axis(
            axis,
            f"{model} gate distribution",
            "Gate is residual shrinkage confidence, not a calibrated probability",
        )
        axis.set_xlim(0, 1)
        axis.set_xlabel("Physical-confidence gate")
        axis.set_ylabel("OOF observations")
        figure.tight_layout()
        figure.savefig(figure_path, dpi=180, facecolor="white")
        plt.close(figure)
        paths[f"distribution_{model}"] = figure_path

    process_path = figures / "gate_vs_process.png"
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, column, label in zip(
        axes,
        ("n_rpm", "fz_mm_per_tooth", "ap_mm"),
        ("Spindle speed (rpm)", "Feed per tooth (mm/tooth)", "Axial depth (mm)"),
    ):
        for model in ("PW2", "PE2"):
            subset = process_summary[process_summary["model"] == model]
            axis.scatter(
                subset[column],
                subset["gate_mean"],
                s=22,
                alpha=0.75,
                color=colors[model],
                label=model,
            )
        axis.set_xlabel(label)
        axis.set_ylim(0, 1)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=GRID, linewidth=0.6, alpha=0.7)
    axes[0].set_ylabel("Mean gate")
    axes[0].legend(frameon=False)
    figure.suptitle("Gate behavior across process settings", x=0.04, ha="left")
    figure.tight_layout()
    figure.savefig(process_path, dpi=180, facecolor="white")
    plt.close(figure)
    paths["process_figure"] = process_path
    return paths


def write_physics_prediction_figures(
    predictions: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    averaged = (
        predictions.groupby(["sample_id", "model"], as_index=False)
        .agg(y_true=("y_true", "first"), y_pred=("y_pred", "mean"))
    )
    models = sorted(
        averaged["model"].unique(),
        key=lambda value: (
            ["M0", "PW0", "PW1", "PW2", "PE0", "PE1", "PE2"].index(value)
            if value in ["M0", "PW0", "PW1", "PW2", "PE0", "PE1", "PE2"]
            else 99
        ),
    )
    low = float(min(averaged["y_true"].min(), averaged["y_pred"].min()))
    high = float(max(averaged["y_true"].max(), averaged["y_pred"].max()))
    scatter_path = destination / "prediction_scatter.png"
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, models):
        subset = averaged[averaged["model"] == model]
        axis.scatter(
            subset["y_true"],
            subset["y_pred"],
            s=10,
            alpha=0.45,
            color=BLUE,
            edgecolors="none",
        )
        axis.plot([low, high], [low, high], color=INK, linestyle="--")
        axis.set_title(model, loc="left")
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=GRID, linewidth=0.6, alpha=0.7)
    for axis in axes.flat[len(models) :]:
        axis.set_visible(False)
    figure.suptitle(
        "OOF predicted versus measured roughness by model",
        x=0.06,
        ha="left",
        fontsize=14,
    )
    figure.supxlabel("Measured Ra (µm)")
    figure.supylabel("Predicted Ra (µm)")
    figure.tight_layout(rect=(0.03, 0.03, 1, 0.94))
    figure.savefig(scatter_path, dpi=180, facecolor="white")
    plt.close(figure)

    residual_path = destination / "residual_plot.png"
    averaged["residual"] = averaged["y_true"] - averaged["y_pred"]
    residual_limit = float(np.abs(averaged["residual"]).max())
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), sharex=True, sharey=True)
    for axis, model in zip(axes.flat, models):
        subset = averaged[averaged["model"] == model]
        axis.scatter(
            subset["y_pred"],
            subset["residual"],
            s=10,
            alpha=0.45,
            color=BLUE,
            edgecolors="none",
        )
        axis.axhline(0, color=INK, linestyle="--")
        axis.set_title(model, loc="left")
        axis.set_ylim(-residual_limit, residual_limit)
        axis.spines[["top", "right"]].set_visible(False)
        axis.grid(color=GRID, linewidth=0.6, alpha=0.7)
    for axis in axes.flat[len(models) :]:
        axis.set_visible(False)
    figure.suptitle(
        "OOF residuals by model",
        x=0.06,
        ha="left",
        fontsize=14,
    )
    figure.supxlabel("Predicted Ra (µm)")
    figure.supylabel("Residual: measured − predicted (µm)")
    figure.tight_layout(rect=(0.03, 0.03, 1, 0.94))
    figure.savefig(residual_path, dpi=180, facecolor="white")
    plt.close(figure)
    paths = {"scatter": scatter_path, "residual": residual_path}
    values = predictions.copy()
    values["weighted_error"] = (
        np.abs(values["y_true"] - values["y_pred"])
        * values["sample_weight"]
    )
    summary = (
        values.groupby("model", as_index=False)
        .agg(error=("weighted_error", "sum"), weight=("sample_weight", "sum"))
    )
    summary["weighted_mae"] = summary["error"] / summary["weight"]
    summary = summary.sort_values("weighted_mae")
    mae_path = destination / "weighted_mae_by_model.png"
    figure, axis = plt.subplots(figsize=(8, 4.8))
    axis.barh(summary["model"], summary["weighted_mae"], color=BLUE)
    _style_axis(
        axis,
        "OOF weighted MAE by model",
        "Lower is better; all available folds and seeds",
    )
    axis.set_xlabel("Weighted MAE (µm)")
    axis.set_ylabel("")
    figure.tight_layout()
    figure.savefig(mae_path, dpi=180, facecolor="white")
    plt.close(figure)
    paths["weighted_mae"] = mae_path
    return paths


def write_physics_error_breakdowns(
    predictions: pd.DataFrame,
    manifest: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    dimensions = ("n_rpm", "fz_mm_per_tooth", "ap_mm", "version")
    missing = sorted({"sample_id", *dimensions} - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest missing columns: {missing}")
    data = predictions.merge(
        manifest[["sample_id", *dimensions]],
        on="sample_id",
        validate="many_to_one",
    )
    data["weighted_error"] = (
        np.abs(data["y_true"] - data["y_pred"]) * data["sample_weight"]
    )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = {}
    for dimension in dimensions:
        grouped = (
            data.groupby(["model", dimension], as_index=False, dropna=False)
            .agg(
                weighted_error=("weighted_error", "sum"),
                weight=("sample_weight", "sum"),
                samples=("sample_id", "size"),
            )
        )
        grouped["weighted_mae"] = (
            grouped["weighted_error"] / grouped["weight"]
        )
        path = destination / f"error_by_{dimension}.csv"
        grouped.to_csv(path, index=False)
        paths[dimension] = path
    return paths


def write_method_notes(output_dir: str | Path) -> Path:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / "method_notes.json"
    notes = {
        "effective_radius": (
            "Candidate radii are sensitivity scales, not measured tool radii."
        ),
        "formula_relation": (
            "The Word formula is the small-feed approximation of the exact "
            "circular formula."
        ),
        "gate_meaning": (
            "Gate values are internal residual-shrinkage coefficients, "
            "not calibrated physical probabilities."
        ),
        "causal_limit": (
            "No exact matched no-physics architecture was added, so "
            "independent causal attribution to the formula is not claimed."
        ),
        "sensor_direction": (
            "Horizontal channels retain the source Scheme 1 raw ordering "
            "and are not relabeled as measured X/Y directions."
        ),
    }
    path.write_text(
        json.dumps(notes, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
