"""Generate publication-ready SGRPN tables and figures from frozen outputs.

The module is deliberately downstream-only: it reads persisted summary tables and
manifests, never imports the training or evaluation entry points, and never reads
the one-time formal claim-decision artifact.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, Mapping

_matplotlib_cache = Path.cwd() / ".cache" / "matplotlib"
_matplotlib_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_matplotlib_cache))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd


TABLE_FILENAMES = {
    "table_1_dataset_structure": "table_1_dataset_structure.csv",
    "table_2_model_matrix": "table_2_model_matrix.csv",
    "table_3_point_and_transfer": "table_3_point_and_transfer.csv",
    "table_4_conformal_results": "table_4_conformal_results.csv",
    "table_5_scale_ablation_bootstrap": "table_5_scale_ablation_bootstrap.csv",
}

BLUE = "#1F5A7A"
GOLD = "#D99032"
INK = "#20262E"
GRID = "#D9DEE3"
REGISTERED_PHASE_B = "registered_phase_b"
SPLIT_CONFORMAL_CORRECTIVE = "split_conformal_corrective"


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen result is missing: {path}")
    frame = pd.read_csv(path)
    if frame.empty:
        raise ValueError(f"Required frozen result is empty: {path}")
    return frame


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen manifest is missing: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _one_row(frame: pd.DataFrame, mask: pd.Series, name: str) -> pd.Series:
    selected = frame.loc[mask]
    if len(selected) != 1:
        raise ValueError(f"{name} must resolve to exactly one row, found {len(selected)}")
    return selected.iloc[0]


def _format_ci(lower: float, upper: float) -> str:
    return f"[{float(lower):.6f}, {float(upper):.6f}]"


def _phase_b_evaluation_dir(project_root: Path, analysis: str) -> Path:
    roots = {
        REGISTERED_PHASE_B: "phase_b",
        SPLIT_CONFORMAL_CORRECTIVE: "phase_b_split_conformal_corrective",
    }
    if analysis not in roots:
        raise ValueError(f"Unknown paper analysis: {analysis}")
    return project_root / "outputs" / "sgrpn" / roots[analysis] / "evaluation"


def _unique_numeric_values(path: Path, column: str) -> list[float]:
    frame = _read_csv(path)
    _require_columns(frame, [column], path.name)
    values = pd.to_numeric(frame[column], errors="raise").drop_duplicates().sort_values()
    return [float(value) for value in values]


def _build_table_1(project_root: Path) -> pd.DataFrame:
    phase_a = project_root / "outputs" / "sgrpn" / "phase_a"
    manifest = _read_json(phase_a / "run_manifest.json")
    fold_audit = manifest.get("fold_audit")
    if not isinstance(fold_audit, dict):
        raise ValueError("Phase A run manifest is missing fold_audit")

    breakdowns = phase_a / "evaluation" / "breakdowns"
    speeds = _unique_numeric_values(breakdowns / "error_by_n_rpm.csv", "n_rpm")
    feeds = _unique_numeric_values(
        breakdowns / "error_by_fz_mm_per_tooth.csv", "fz_mm_per_tooth"
    )
    depths = _unique_numeric_values(breakdowns / "error_by_ap_mm.csv", "ap_mm")
    versions_frame = _read_csv(breakdowns / "error_by_version.csv")
    _require_columns(versions_frame, ["version"], "error_by_version.csv")
    versions = sorted(versions_frame["version"].astype(str).unique())

    def range_text(values: list[float], unit: str, decimals: int) -> str:
        rendered = ", ".join(f"{value:.{decimals}f}" for value in values)
        return f"{len(values)} levels ({rendered} {unit})"

    rows = [
        (
            "Independent machining groups",
            str(int(fold_audit["n_groups"])),
            "Outer splitting and Bootstrap resampling unit",
            "Phase A run manifest",
        ),
        (
            "Surface regions",
            str(int(fold_audit["n_samples"])),
            "Prediction unit; regions remain nested in machining groups",
            "Phase A run manifest",
        ),
        (
            "Ra readings per region",
            "3",
            "Repeated readings share one region-level mean and uncertainty model",
            "Locked SGRPN protocol",
        ),
        (
            "Vibration channels",
            "3 (Ch9, Ch10, Ch11)",
            "Ch11 is axial; Ch9/Ch10 orientation is unresolved and symmetrized",
            "Locked SGRPN protocol",
        ),
        (
            "Vibration sampling rate",
            "25.6 kHz",
            "Each retained model window contains 25,600 samples (1 s)",
            "Locked SGRPN protocol",
        ),
        (
            "Spindle-speed grid",
            range_text(speeds, "rpm", 0),
            "Observed levels in the frozen Phase A breakdown",
            "error_by_n_rpm.csv",
        ),
        (
            "Feed-per-tooth grid",
            range_text(feeds, "mm/tooth", 2),
            "Observed levels in the frozen Phase A breakdown",
            "error_by_fz_mm_per_tooth.csv",
        ),
        (
            "Axial-depth grid",
            range_text(depths, "mm", 1),
            "Observed levels in the frozen Phase A breakdown",
            "error_by_ap_mm.csv",
        ),
        (
            "Observed process combinations",
            "189 / 200",
            "Observed combinations relative to the full 10×5×4 grid",
            "Locked SGRPN data specification",
        ),
        (
            "Acquisition versions",
            f"{len(versions)} ({', '.join(versions)})",
            "Composite-domain stress test; version is excluded from model inputs",
            "error_by_version.csv",
        ),
        (
            "Outer cross-validation",
            f"{int(fold_audit['n_folds'])} group-disjoint folds",
            f"Recorded group overlap count: {int(fold_audit['group_overlap_count'])}",
            "Phase A run manifest",
        ),
        (
            "Registered random seeds",
            "1 in Phase A; 3 in Phase B",
            "Phase A feasibility followed by frozen three-seed replication",
            "Phase A/Phase B protocols",
        ),
    ]
    return pd.DataFrame(rows, columns=["item", "value", "definition_or_scope", "source"])


def _build_table_2() -> pd.DataFrame:
    rows = [
        (
            "M0",
            "Quadratic process ridge regression",
            "Process parameters and registered quadratic terms",
            "Region-mean Ra",
            "Strong classical process baseline",
        ),
        (
            "P1",
            "Process MLP",
            "Nine scaled process features",
            "Region-mean Ra",
            "Neural process expert and safe fallback",
        ),
        (
            "V1",
            "Vibration-only CNN",
            "Three-channel order spectra",
            "Region-mean Ra",
            "Tests vibration without process context",
        ),
        (
            "F1",
            "Direct process-vibration fusion",
            "Process features and order-spectrum embedding",
            "Region-mean Ra",
            "Naive fusion comparator",
        ),
        (
            "R1",
            "Ungated residual fusion",
            "P1 process mean and order-spectrum embedding",
            "Group-safe P1 OOF residual",
            "Tests the full learned vibration correction",
        ),
        (
            "G1",
            "Selective gated residual fusion",
            "P1 mean, residual embedding, process and seven quality features",
            "Group-safe P1 OOF residual with gated correction",
            "Proposed safe fusion model",
        ),
    ]
    return pd.DataFrame(
        rows,
        columns=["model", "architecture", "inputs", "training_target", "role"],
    )


def _build_table_3(project_root: Path, analysis: str) -> pd.DataFrame:
    outputs = project_root / "outputs" / "sgrpn"
    phase_a_eval = outputs / "phase_a" / "evaluation"
    phase_b_eval = _phase_b_evaluation_dir(project_root, analysis)
    summary_a = _read_csv(phase_a_eval / "summary_metrics.csv")
    folds_a = _read_csv(phase_a_eval / "fold_metrics.csv")
    transfer_a = _read_csv(phase_a_eval / "negative_transfer.csv")
    bootstrap_a = _read_csv(phase_a_eval / "paired_bootstrap.csv")
    means_b = _read_csv(phase_b_eval / "mean_metrics.csv")
    transfer_b = _read_csv(phase_b_eval / "negative_transfer.csv")
    bootstrap_b = _read_csv(phase_b_eval / "paired_bootstrap.csv")

    _require_columns(
        summary_a,
        ["aggregation_unit", "model", "weighted_mae", "weighted_rmse", "weighted_r2"],
        "Phase A summary metrics",
    )
    _require_columns(
        folds_a, ["aggregation_unit", "model", "fold", "weighted_mae"], "Phase A folds"
    )
    _require_columns(
        transfer_a, ["model", "material_rate", "material_margin_um"], "Phase A transfer"
    )
    _require_columns(
        bootstrap_a, ["baseline", "candidate", "point_estimate", "lower", "upper"],
        "Phase A bootstrap",
    )

    segment_a = summary_a.loc[summary_a["aggregation_unit"].eq("segment")].copy()
    if set(segment_a["model"]) != {"M0", "P1", "V1", "F1", "R1", "G1"}:
        raise ValueError("Phase A segment summary must contain M0/P1/V1/F1/R1/G1 exactly")
    folds_segment = folds_a.loc[folds_a["aggregation_unit"].eq("segment")]

    references_a: Mapping[str, str | None] = {
        "M0": None,
        "P1": "M0",
        "V1": "P1",
        "F1": "P1",
        "R1": "P1",
        "G1": "P1",
    }
    rows: list[dict[str, object]] = []
    for model in ("M0", "P1", "V1", "F1", "R1", "G1"):
        metric = _one_row(segment_a, segment_a["model"].eq(model), f"Phase A {model}")
        reference = references_a[model]
        wins: int | pd._libs.missing.NAType = pd.NA
        improvement: float | pd._libs.missing.NAType = pd.NA
        ci = "—"
        if reference is not None:
            pivot = folds_segment.loc[
                folds_segment["model"].isin([reference, model]),
                ["fold", "model", "weighted_mae"],
            ].pivot(index="fold", columns="model", values="weighted_mae")
            if len(pivot) != 5 or set(pivot.columns) != {reference, model}:
                raise ValueError(f"Phase A fold comparison is incomplete: {reference} vs {model}")
            wins = int((pivot[model] < pivot[reference]).sum())
            comparison = _one_row(
                bootstrap_a,
                bootstrap_a["baseline"].eq(reference) & bootstrap_a["candidate"].eq(model),
                f"Phase A bootstrap {reference} vs {model}",
            )
            improvement = float(comparison["point_estimate"])
            ci = _format_ci(comparison["lower"], comparison["upper"])

        transfer_rows = transfer_a.loc[transfer_a["model"].eq(model)]
        transfer: float | pd._libs.missing.NAType = pd.NA
        if len(transfer_rows) == 1:
            transfer = float(transfer_rows.iloc[0]["material_rate"])
        elif len(transfer_rows) > 1:
            raise ValueError(f"Phase A transfer has duplicate {model} rows")
        rows.append(
            {
                "phase": "Phase A",
                "seed_scope": "20260723",
                "model": model,
                "reference": reference or "—",
                "mae_um": float(metric["weighted_mae"]),
                "rmse_um": float(metric["weighted_rmse"]),
                "r2": float(metric["weighted_r2"]),
                "fold_wins_vs_reference": wins,
                "material_negative_transfer_rate": transfer,
                "mae_improvement_um": improvement,
                "mae_improvement_ci95": ci,
            }
        )

    _require_columns(means_b, ["aggregation", "model", "metric", "value"], "Phase B means")
    all_seed_b = means_b.loc[means_b["aggregation"].eq("all_seed")]
    metric_matrix = all_seed_b.pivot(index="model", columns="metric", values="value")
    expected_metrics = {"mean_mae", "mean_rmse", "mean_r2"}
    if set(metric_matrix.index) != {"P1", "R1", "G1"} or not expected_metrics.issubset(
        metric_matrix.columns
    ):
        raise ValueError("Phase B all-seed mean metrics are incomplete")

    for model in ("P1", "R1", "G1"):
        reference = "P1" if model != "P1" else "—"
        transfer: float | pd._libs.missing.NAType = pd.NA
        transfer_rows = transfer_b.loc[
            transfer_b["aggregation"].eq("all_seed") & transfer_b["candidate"].eq(model)
        ]
        if len(transfer_rows) == 1:
            transfer = float(transfer_rows.iloc[0]["material_rate"])
        elif len(transfer_rows) > 1:
            raise ValueError(f"Phase B transfer has duplicate {model} rows")

        improvement: float | pd._libs.missing.NAType = pd.NA
        ci = "—"
        if model != "P1":
            comparison = _one_row(
                bootstrap_b,
                bootstrap_b["baseline"].eq("P1")
                & bootstrap_b["candidate"].eq(model)
                & bootstrap_b["metric"].eq("weighted_mae_improvement_um"),
                f"Phase B bootstrap P1 vs {model}",
            )
            improvement = float(comparison["point_estimate"])
            ci = _format_ci(comparison["lower"], comparison["upper"])

        rows.append(
            {
                "phase": (
                    "Corrective Phase B"
                    if analysis == SPLIT_CONFORMAL_CORRECTIVE
                    else "Phase B"
                ),
                "seed_scope": "mean of 3 registered seeds",
                "model": model,
                "reference": reference,
                "mae_um": float(metric_matrix.loc[model, "mean_mae"]),
                "rmse_um": float(metric_matrix.loc[model, "mean_rmse"]),
                "r2": float(metric_matrix.loc[model, "mean_r2"]),
                "fold_wins_vs_reference": pd.NA,
                "material_negative_transfer_rate": transfer,
                "mae_improvement_um": improvement,
                "mae_improvement_ci95": ci,
            }
        )
    result = pd.DataFrame(rows)
    result["fold_wins_vs_reference"] = result["fold_wins_vs_reference"].astype("Int64")
    if analysis == SPLIT_CONFORMAL_CORRECTIVE:
        historical = _build_table_3(project_root, REGISTERED_PHASE_B)
        result = pd.concat(
            [
                historical,
                result.loc[result["phase"].eq("Corrective Phase B")],
            ],
            ignore_index=True,
        )
        result["fold_wins_vs_reference"] = result["fold_wins_vs_reference"].astype(
            "Int64"
        )
    return result


def _build_table_4(project_root: Path, analysis: str) -> pd.DataFrame:
    path = _phase_b_evaluation_dir(project_root, analysis) / "probability_metrics.csv"
    metrics = _read_csv(path)
    _require_columns(
        metrics,
        ["aggregation", "scale_model", "interval_type", "nominal_coverage", "metric", "value"],
        "Phase B probability metrics",
    )
    conformal = metrics.loc[
        metrics["aggregation"].eq("all_seed") & metrics["interval_type"].eq("conformal")
    ].copy()
    pivot = conformal.pivot(
        index=["scale_model", "nominal_coverage"], columns="metric", values="value"
    )
    required = {
        "single_reading_coverage",
        "simultaneous_group_coverage",
        "mean_interval_width",
        "winkler_score",
    }
    if len(pivot) != 4 or not required.issubset(pivot.columns):
        raise ValueError("Phase B all-seed conformal metrics must contain 2 models × 2 levels")
    gaussian = metrics.loc[
        metrics["aggregation"].eq("all_seed")
        & metrics["metric"].isin(["gaussian_nll", "gaussian_crps"])
    ].pivot(index="scale_model", columns="metric", values="value")
    if set(gaussian.index) != {"homoscedastic", "heteroscedastic"}:
        raise ValueError("Phase B all-seed Gaussian metrics are incomplete")

    rows = []
    for scale_model in ("homoscedastic", "heteroscedastic"):
        for nominal in (0.90, 0.95):
            value = pivot.loc[(scale_model, nominal)]
            rows.append(
                {
                    "scale_model": scale_model.capitalize(),
                    "nominal_coverage": nominal,
                    "single_reading_coverage": float(value["single_reading_coverage"]),
                    "simultaneous_group_coverage": float(
                        value["simultaneous_group_coverage"]
                    ),
                    "mean_interval_width_um": float(value["mean_interval_width"]),
                    "winkler_score": float(value["winkler_score"]),
                    "gaussian_nll": float(gaussian.loc[scale_model, "gaussian_nll"]),
                    "gaussian_crps": float(gaussian.loc[scale_model, "gaussian_crps"]),
                }
            )
    return pd.DataFrame(rows)


def _build_table_5(project_root: Path, analysis: str) -> pd.DataFrame:
    path = _phase_b_evaluation_dir(project_root, analysis) / "paired_bootstrap.csv"
    bootstrap = _read_csv(path)
    _require_columns(
        bootstrap,
        [
            "baseline",
            "candidate",
            "metric",
            "point_estimate",
            "lower",
            "upper",
            "interval_type",
            "nominal_coverage",
        ],
        "Phase B paired bootstrap",
    )
    values = bootstrap.loc[
        bootstrap["baseline"].eq("homoscedastic")
        & bootstrap["candidate"].eq("heteroscedastic")
    ].copy()
    labels = {
        "gaussian_nll": "Gaussian NLL",
        "gaussian_crps": "Gaussian CRPS",
        "mean_interval_width": "Mean interval width",
        "winkler_score": "Winkler score",
    }
    rows = []
    order = [
        ("gaussian_nll", None),
        ("gaussian_crps", None),
        ("mean_interval_width", 0.90),
        ("mean_interval_width", 0.95),
        ("winkler_score", 0.90),
        ("winkler_score", 0.95),
    ]
    for metric, nominal in order:
        mask = values["metric"].eq(metric)
        if nominal is None:
            mask &= values["nominal_coverage"].isna()
        else:
            mask &= np.isclose(
                pd.to_numeric(values["nominal_coverage"], errors="coerce"), nominal
            )
        row = _one_row(values, mask, f"Scale ablation {metric} {nominal}")
        point = float(row["point_estimate"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        rows.append(
            {
                "metric": labels[metric],
                "nominal_coverage": pd.NA if nominal is None else nominal,
                "heteroscedastic_minus_homoscedastic": point,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "ci_excludes_zero": bool(lower > 0.0 or upper < 0.0),
                "preferred_direction": "Lower is better",
                "observed_direction": "Heteroscedastic is worse" if lower > 0.0 else "Uncertain",
                "registered_interpretation": (
                    "No heteroscedastic emphasis"
                    if analysis == SPLIT_CONFORMAL_CORRECTIVE
                    else "No heteroscedastic advantage"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_paper_tables(
    project_root: Path | str, *, analysis: str = REGISTERED_PHASE_B
) -> dict[str, pd.DataFrame]:
    """Build the five registered manuscript tables from frozen summary artifacts."""
    root = Path(project_root).resolve()
    return {
        "table_1_dataset_structure": _build_table_1(root),
        "table_2_model_matrix": _build_table_2(),
        "table_3_point_and_transfer": _build_table_3(root, analysis),
        "table_4_conformal_results": _build_table_4(root, analysis),
        "table_5_scale_ablation_bootstrap": _build_table_5(root, analysis),
    }


def _style_axes(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(INK)
    axis.tick_params(colors=INK, labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.7, alpha=0.8)
    axis.set_axisbelow(True)


def _save_figure(figure: plt.Figure, png: Path, svg: Path) -> None:
    figure.savefig(png, format="png", dpi=300, bbox_inches="tight", facecolor="white")
    figure.savefig(svg, format="svg", bbox_inches="tight", facecolor="white")
    plt.close(figure)


def _diagram_box(
    axis: plt.Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    label: str,
    *,
    facecolor: str = "#F5F7F9",
    edgecolor: str = INK,
    fontsize: float = 9.0,
    linewidth: float = 1.1,
    linestyle: str = "-",
    fontweight: str = "normal",
) -> FancyBboxPatch:
    box = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.018",
        transform=axis.transAxes,
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        linestyle=linestyle,
        clip_on=False,
    )
    axis.add_patch(box)
    axis.text(
        x + width / 2,
        y + height / 2,
        label,
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=INK,
        fontweight=fontweight,
        linespacing=1.25,
    )
    return box


def _diagram_arrow(
    axis: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linestyle: str = "-",
    connectionstyle: str = "arc3,rad=0.0",
    linewidth: float = 1.25,
) -> FancyArrowPatch:
    arrow = FancyArrowPatch(
        start,
        end,
        transform=axis.transAxes,
        arrowstyle="-|>",
        mutation_scale=13,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        connectionstyle=connectionstyle,
        shrinkA=3,
        shrinkB=3,
        clip_on=False,
    )
    axis.add_patch(arrow)
    return arrow


def _render_sgrpn_architecture(png: Path, svg: Path) -> None:
    """Render the safe gated residual mean architecture as an editable diagram."""
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(13.2, 6.8))
        axis.set_axis_off()
        figure.suptitle(
            "SGRPN safe gated residual fusion architecture",
            fontsize=15,
            fontweight="bold",
            y=0.98,
        )
        figure.text(
            0.5,
            0.925,
            "The vibration branch supplies a gated correction while the process expert remains the explicit fallback.",
            ha="center",
            fontsize=9.8,
            color="#4E5965",
        )

        _diagram_box(
            axis,
            0.025,
            0.67,
            0.17,
            0.14,
            "Process features x\n9 registered inputs",
            facecolor="#E7F0F5",
            edgecolor=BLUE,
        )
        _diagram_box(
            axis,
            0.025,
            0.41,
            0.17,
            0.14,
            "Order spectra S\n3 channels × 361 bins",
            facecolor="#FFF3E3",
            edgecolor=GOLD,
        )
        _diagram_box(
            axis,
            0.025,
            0.15,
            0.17,
            0.14,
            "Signal-quality summary q\n7 symmetric features",
            facecolor="#F2F4F6",
        )

        _diagram_box(
            axis,
            0.255,
            0.66,
            0.21,
            0.16,
            "P1 process expert\n9 → 32 → 16 → 1\nμ_process",
            facecolor="#DCEBF2",
            edgecolor=BLUE,
            fontweight="bold",
        )
        _diagram_box(
            axis,
            0.255,
            0.395,
            0.21,
            0.18,
            "Residual-CNN expert\nConv encoder + region pooling\nΔ_vibration",
            facecolor="#FDE9CF",
            edgecolor=GOLD,
            fontweight="bold",
        )
        axis.text(
            0.36,
            0.355,
            "trained on group-safe P1 OOF residuals",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.4,
            color="#6A737D",
        )
        _diagram_box(
            axis,
            0.515,
            0.17,
            0.20,
            0.18,
            "Trust gate\n[x, vibration embedding, q]\n80 → 16 → 1 + sigmoid\ng ∈ [0, 1]",
            facecolor="#F2F4F6",
            fontweight="bold",
        )

        product = Circle(
            (0.73, 0.485),
            0.027,
            transform=axis.transAxes,
            facecolor="white",
            edgecolor=INK,
            linewidth=1.2,
            clip_on=False,
        )
        axis.add_patch(product)
        axis.text(0.73, 0.485, "×", transform=axis.transAxes, ha="center", va="center", fontsize=14)

        summation = Circle(
            (0.835, 0.675),
            0.027,
            transform=axis.transAxes,
            facecolor="white",
            edgecolor=INK,
            linewidth=1.2,
            clip_on=False,
        )
        axis.add_patch(summation)
        axis.text(0.835, 0.675, "+", transform=axis.transAxes, ha="center", va="center", fontsize=14)

        _diagram_box(
            axis,
            0.885,
            0.60,
            0.10,
            0.15,
            "G1 mean\nμ̂ = μ_process\n+ gΔ_vibration",
            facecolor="#E7F0F5",
            edgecolor=BLUE,
            fontweight="bold",
        )
        _diagram_box(
            axis,
            0.54,
            0.025,
            0.39,
            0.075,
            "Safe fallback: g → 0  ⇒  μ̂ → μ_process; vibration can only supply a gated vibration correction",
            facecolor="#E7F0F5",
            edgecolor=BLUE,
            fontsize=8.8,
        )

        _diagram_arrow(axis, (0.195, 0.74), (0.255, 0.74), color=BLUE)
        _diagram_arrow(axis, (0.195, 0.48), (0.255, 0.48), color=GOLD)
        _diagram_arrow(axis, (0.465, 0.74), (0.808, 0.68), color=BLUE)
        _diagram_arrow(axis, (0.465, 0.485), (0.702, 0.485), color=GOLD)
        _diagram_arrow(axis, (0.757, 0.50), (0.813, 0.655))
        _diagram_arrow(axis, (0.862, 0.675), (0.885, 0.675), color=BLUE)

        axis.plot(
            [0.195, 0.22, 0.22],
            [0.71, 0.71, 0.31],
            transform=axis.transAxes,
            color=BLUE,
            linewidth=1.25,
            clip_on=False,
        )
        _diagram_arrow(axis, (0.22, 0.31), (0.515, 0.31), color=BLUE)
        _diagram_arrow(
            axis,
            (0.465, 0.43),
            (0.545, 0.34),
            color=GOLD,
            connectionstyle="arc3,rad=0.05",
        )
        _diagram_arrow(axis, (0.195, 0.22), (0.515, 0.24))
        _diagram_arrow(axis, (0.66, 0.35), (0.71, 0.46))

        axis.text(
            0.565,
            0.61,
            "process path",
            transform=axis.transAxes,
            fontsize=8.2,
            color=BLUE,
            ha="center",
        )
        axis.text(
            0.585,
            0.455,
            "residual path",
            transform=axis.transAxes,
            fontsize=8.2,
            color=GOLD,
            ha="center",
        )
        figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.90))
        _save_figure(figure, png, svg)


def _render_group_isolation_workflow(
    png: Path, svg: Path, *, analysis: str = REGISTERED_PHASE_B
) -> None:
    """Render the hierarchical group-isolated training and calibration workflow."""
    corrective = analysis == SPLIT_CONFORMAL_CORRECTIVE
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "text.color": INK,
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(14.2, 7.4))
        axis.set_axis_off()
        figure.suptitle(
            "Group-isolated training and group-conformal calibration",
            fontsize=15,
            fontweight="bold",
            y=0.985,
        )
        figure.text(
            0.5,
            0.94,
            (
                "Outer-train groups are divided once into proper-train and fixed calibration groups."
                if corrective
                else "All model selection, residual generation and calibration remain inside outer-train groups."
            ),
            ha="center",
            fontsize=9.8,
            color="#4E5965",
        )
        _diagram_box(
            axis,
            0.315,
            0.865,
            0.50,
            0.065,
            "No group_id crosses training, validation, calibration or outer-test boundaries",
            facecolor="#F2F4F6",
            fontsize=9.0,
            fontweight="bold",
        )

        axis.text(
            0.105,
            0.82,
            "Hierarchical observations",
            transform=axis.transAxes,
            ha="center",
            fontsize=9.4,
            fontweight="bold",
            color="#4E5965",
        )
        _diagram_box(axis, 0.02, 0.65, 0.17, 0.12, "212 machining groups\nindependent split unit", facecolor="#E7F0F5", edgecolor=BLUE)
        _diagram_box(axis, 0.02, 0.43, 0.17, 0.12, "586 surface regions\nnested within groups", facecolor="#F2F4F6")
        _diagram_box(axis, 0.02, 0.19, 0.17, 0.14, "Per region\n3 Ra readings +\nvibration windows", facecolor="#FFF3E3", edgecolor=GOLD)
        _diagram_arrow(axis, (0.105, 0.65), (0.105, 0.55), color=BLUE)
        _diagram_arrow(axis, (0.105, 0.43), (0.105, 0.33), color=GOLD)

        _diagram_box(
            axis,
            0.255,
            0.69,
            0.29,
            0.13,
            "Outer-train groups (4 folds)\nOnly source for fitting and selection",
            facecolor="#DCEBF2",
            edgecolor=BLUE,
            fontweight="bold",
        )
        _diagram_box(
            axis,
            0.675,
            0.69,
            0.29,
            0.13,
            "Outer-test groups (1 untouched fold)\nUsed once after all choices are fixed",
            facecolor="#F2F4F6",
            fontweight="bold",
        )
        axis.plot(
            [0.19, 0.22, 0.22, 0.82],
            [0.71, 0.84, 0.84, 0.84],
            transform=axis.transAxes,
            color=INK,
            linewidth=1.25,
            clip_on=False,
        )
        _diagram_arrow(axis, (0.40, 0.84), (0.40, 0.82), color=BLUE)
        _diagram_arrow(axis, (0.82, 0.84), (0.82, 0.82))
        axis.text(
            0.61,
            0.835,
            "fixed 5-fold outer GroupKFold",
            transform=axis.transAxes,
            ha="center",
            va="top",
            fontsize=8.3,
            color="#4E5965",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.5},
        )

        axis.text(0.275, 0.62, "TRAINING", transform=axis.transAxes, fontsize=8.3, fontweight="bold", color=BLUE)
        _diagram_box(
            axis,
            0.255,
            0.43,
            0.18,
            0.14,
            (
                "Fixed inner block 0 split\n126/127 proper-train groups\n+ 43 calibration groups"
                if corrective
                else "Inner group folds\nOOF P1 residual targets\n+ group-safe early stopping"
            ),
            facecolor="#E7F0F5",
            edgecolor=BLUE,
            fontsize=8.5,
        )
        _diagram_box(
            axis,
            0.47,
            0.43,
            0.18,
            0.14,
            (
                "Fit and lock mean + scale paths\nP1 → Residual-CNN → G1\non proper-train groups only"
                if corrective
                else "Refit frozen mean path\nP1 → Residual-CNN → G1\non all outer-train groups"
            ),
            facecolor="#DCEBF2",
            edgecolor=BLUE,
            fontsize=8.5,
            fontweight="bold",
        )
        _diagram_arrow(axis, (0.40, 0.69), (0.345, 0.57), color=BLUE)
        _diagram_arrow(axis, (0.435, 0.50), (0.47, 0.50), color=BLUE)

        axis.text(0.275, 0.36, "CALIBRATION", transform=axis.transAxes, fontsize=8.3, fontweight="bold", color=GOLD)
        _diagram_box(
            axis,
            0.255,
            0.18,
            0.18,
            0.13,
            (
                "Locked-predictor inference\non 43 fixed calibration groups\n(no refitting)"
                if corrective
                else "Four group-disjoint\ncalibration folds\nOOF μ and σ only"
            ),
            facecolor="#FFF3E3",
            edgecolor=GOLD,
            fontsize=8.5,
        )
        _diagram_box(
            axis,
            0.47,
            0.18,
            0.18,
            0.13,
            "Nonconformity reduction\nmaximum score per calibration group",
            facecolor="#FFF3E3",
            edgecolor=GOLD,
            fontsize=8.4,
            fontweight="bold",
        )
        _diagram_box(
            axis,
            0.685,
            0.18,
            0.12,
            0.13,
            "Group quantile\nq₁₋α\nα ∈ {0.10, 0.05}",
            facecolor="#FFF3E3",
            edgecolor=GOLD,
            fontsize=8.5,
        )
        _diagram_arrow(axis, (0.36, 0.43), (0.345, 0.31), color=GOLD)
        _diagram_arrow(axis, (0.435, 0.245), (0.47, 0.245), color=GOLD)
        _diagram_arrow(axis, (0.65, 0.245), (0.685, 0.245), color=GOLD)

        axis.text(0.70, 0.62, "OUTER TEST", transform=axis.transAxes, fontsize=8.3, fontweight="bold", color="#4E5965")
        _diagram_box(
            axis,
            0.675,
            0.43,
            0.29,
            0.14,
            "One-shot prediction\nFrozen μ̂ and σ for every outer-test region",
            facecolor="#F2F4F6",
            fontsize=8.8,
        )
        _diagram_arrow(axis, (0.82, 0.69), (0.82, 0.57))
        _diagram_box(
            axis,
            0.84,
            0.18,
            0.14,
            0.13,
            "Apply q₁₋α\nintervals for all regions\nand three readings",
            facecolor="#E7F0F5",
            edgecolor=BLUE,
            fontsize=8.3,
        )
        _diagram_arrow(axis, (0.805, 0.245), (0.84, 0.245), color=GOLD)
        _diagram_arrow(axis, (0.85, 0.43), (0.91, 0.31), connectionstyle="arc3,rad=0.08")

        _diagram_box(
            axis,
            0.255,
            0.035,
            0.725,
            0.075,
            "Repeat the frozen protocol for 5 outer folds × 3 seeds; aggregate only after every outer prediction is complete",
            facecolor="#F2F4F6",
            fontsize=8.9,
        )
        _diagram_arrow(axis, (0.91, 0.18), (0.91, 0.11), color=BLUE)
        figure.tight_layout(rect=(0.01, 0.01, 0.99, 0.91))
        _save_figure(figure, png, svg)


def _render_scale_ablation_bootstrap(table: pd.DataFrame, png: Path, svg: Path) -> None:
    """Render heteroscedastic-minus-homoscedastic paired Bootstrap intervals."""
    panels = [
        ("Gaussian NLL", "(a) Gaussian NLL", None),
        ("Gaussian CRPS", "(b) Gaussian CRPS", None),
        ("Mean interval width", "(c) Mean interval width (µm)", [0.90, 0.95]),
        ("Winkler score", "(d) Winkler score", [0.90, 0.95]),
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(2, 2, figsize=(11.2, 7.3))
        for axis, (metric, title, levels) in zip(axes.flat, panels, strict=True):
            values = table.loc[table["metric"].eq(metric)].copy()
            if levels is None:
                if len(values) != 1:
                    raise ValueError(f"Expected one aggregate Bootstrap row for {metric}")
                labels = ["Overall"]
            else:
                values["nominal_coverage"] = pd.to_numeric(
                    values["nominal_coverage"], errors="raise"
                )
                values = values.set_index("nominal_coverage").loc[levels].reset_index()
                labels = [f"{level:.0%}" for level in levels]
            points = values["heteroscedastic_minus_homoscedastic"].astype(float).to_numpy()
            lower = values["ci95_lower"].astype(float).to_numpy()
            upper = values["ci95_upper"].astype(float).to_numpy()
            y = np.arange(len(values), dtype=float)
            markers = ["o", "s"] if len(values) == 2 else ["o"]
            for index, (point, lo, hi, marker) in enumerate(
                zip(points, lower, upper, markers, strict=True)
            ):
                axis.errorbar(
                    point,
                    y[index],
                    xerr=np.array([[point - lo], [hi - point]]),
                    fmt=marker,
                    color=GOLD,
                    markerfacecolor="white" if marker == "s" else GOLD,
                    markeredgecolor=GOLD,
                    markeredgewidth=1.4,
                    markersize=7,
                    elinewidth=1.8,
                    capsize=4,
                    capthick=1.2,
                )
                point_label = f"{point:.4f}" if abs(point) < 0.01 else f"{point:.3f}"
                axis.annotate(
                    point_label,
                    (point, y[index]),
                    xytext=(0, 10),
                    textcoords="offset points",
                    ha="center",
                    va="bottom",
                    fontsize=8.4,
                    color=GOLD,
                )
            upper_limit = float(max(upper)) * 1.10
            negative_margin = max(upper_limit * 0.08, 1e-6)
            axis.set_xlim(-negative_margin, upper_limit)
            axis.axvline(0.0, color=INK, linestyle=":", linewidth=1.2)
            axis.set_yticks(y, labels)
            axis.set_ylim(-0.65, max(0.65, len(values) - 0.35))
            axis.set_xlabel("Difference (lower is better)", fontsize=9)
            axis.set_title(title, fontsize=10.5, pad=10)
            axis.grid(axis="x", color=GRID, linewidth=0.7, alpha=0.8)
            axis.grid(axis="y", visible=False)
            axis.spines[["top", "right"]].set_visible(False)
            axis.spines[["left", "bottom"]].set_color(INK)
            axis.tick_params(colors=INK, labelsize=8.7)
        figure.suptitle(
            "Paired group-Bootstrap scale-model ablation",
            fontsize=14,
            fontweight="bold",
            y=0.995,
        )
        figure.text(
            0.5,
            0.955,
            "Heteroscedastic minus homoscedastic; points show differences and bars show 95% CIs from 10,000 group resamples.",
            ha="center",
            fontsize=9.4,
            color="#4E5965",
        )
        figure.text(
            0.5,
            0.02,
            "← favors heteroscedastic    0    favors homoscedastic →",
            ha="center",
            fontsize=9.0,
            color="#4E5965",
        )
        figure.tight_layout(rect=(0.01, 0.055, 0.99, 0.92), h_pad=2.0, w_pad=2.0)
        _save_figure(figure, png, svg)


def _render_negative_transfer(table: pd.DataFrame, png: Path, svg: Path) -> None:
    phase_b = (
        "Corrective Phase B"
        if "Corrective Phase B" in set(table["phase"])
        else "Phase B"
    )
    panels = [
        ("Phase A · registered feasibility seed", "Phase A", ["F1", "R1", "G1"]),
        (
            (
                "Corrective Phase B · mean of three registered seeds"
                if phase_b == "Corrective Phase B"
                else "Phase B · mean of three registered seeds"
            ),
            phase_b,
            ["R1", "G1"],
        ),
    ]
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    ):
        figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.6), sharey=True)
        for axis, (title, phase, models) in zip(axes, panels, strict=True):
            values = (
                table.loc[
                    table["phase"].eq(phase) & table["model"].isin(models),
                    ["model", "material_negative_transfer_rate"],
                ]
                .set_index("model")
                .loc[models]
            )
            rates = values["material_negative_transfer_rate"].astype(float).to_numpy()
            colors = [BLUE if model == "G1" else GOLD for model in models]
            hatches = ["" if model == "G1" else "///" for model in models]
            bars = axis.bar(
                models,
                rates,
                color=colors,
                edgecolor=INK,
                linewidth=0.8,
                width=0.62,
            )
            for bar, hatch, rate in zip(bars, hatches, rates, strict=True):
                bar.set_hatch(hatch)
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    rate + 0.012,
                    f"{rate:.1%}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold" if bar.get_facecolor()[:3] == plt.matplotlib.colors.to_rgb(BLUE) else "normal",
                )
            axis.set_title(title, fontsize=10.5, pad=10)
            axis.set_ylim(0.0, 0.55)
            axis.set_yticks(np.arange(0.0, 0.56, 0.10), [f"{value:.0%}" for value in np.arange(0.0, 0.56, 0.10)])
            _style_axes(axis)
        axes[0].set_ylabel("Material negative-transfer rate")
        figure.suptitle("Material negative transfer relative to the P1 process expert", fontsize=13, fontweight="bold", y=1.04)
        figure.text(
            0.5,
            0.97,
            "A group is counted when its weighted MAE exceeds P1 by more than 0.01 µm; lower is safer.",
            ha="center",
            va="top",
            fontsize=9.2,
            color="#4E5965",
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        _save_figure(figure, png, svg)


def _render_corrective_prediction_scatter(
    project_root: Path, png: Path, svg: Path
) -> None:
    predictions = _read_csv(
        project_root
        / "outputs"
        / "sgrpn"
        / "phase_b_split_conformal_corrective"
        / "predictions"
        / "oof_mean_predictions.csv"
    )
    _require_columns(
        predictions,
        ["model", "seed", "target_mean", "prediction"],
        "Corrective Phase B mean predictions",
    )
    values = predictions.loc[predictions["model"].eq("G1")].copy()
    seeds = sorted(pd.to_numeric(values["seed"], errors="raise").astype(int).unique())
    if seeds != [20260723, 20260724, 20260725]:
        raise ValueError("Corrective G1 predictions must contain the three registered seeds")
    palette = {20260723: BLUE, 20260724: GOLD, 20260725: "#657A3E"}
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    ):
        figure, axis = plt.subplots(figsize=(6.8, 6.1))
        for seed in seeds:
            selected = values.loc[values["seed"].eq(seed)]
            axis.scatter(
                selected["target_mean"],
                selected["prediction"],
                s=16,
                alpha=0.42,
                color=palette[seed],
                edgecolors="none",
                label=str(seed),
            )
        lower = float(min(values["target_mean"].min(), values["prediction"].min()))
        upper = float(max(values["target_mean"].max(), values["prediction"].max()))
        padding = 0.04 * (upper - lower)
        bounds = (lower - padding, upper + padding)
        axis.plot(bounds, bounds, color=INK, linestyle="--", linewidth=1.2, label="Ideal")
        axis.set_xlim(bounds)
        axis.set_ylim(bounds)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("Measured region-mean Ra (µm)")
        axis.set_ylabel("Predicted region-mean Ra (µm)")
        axis.set_title(
            "Corrective outer-held-out G1 predictions",
            fontsize=13,
            fontweight="bold",
            pad=13,
        )
        axis.text(
            0.5,
            1.01,
            "Three registered seeds; v3/v4 remains confounded with spindle speed",
            transform=axis.transAxes,
            ha="center",
            va="bottom",
            fontsize=8.7,
            color="#4E5965",
        )
        _style_axes(axis)
        axis.grid(axis="both", color=GRID, linewidth=0.7, alpha=0.7)
        axis.legend(title="Seed", frameon=False, loc="upper left")
        figure.tight_layout()
        _save_figure(figure, png, svg)


def _render_coverage_width(table: pd.DataFrame, png: Path, svg: Path) -> None:
    order = ["Homoscedastic", "Heteroscedastic"]
    nominal = np.array([0.90, 0.95])
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "axes.labelcolor": INK,
            "text.color": INK,
            "svg.fonttype": "none",
        }
    ):
        figure, (coverage_axis, width_axis) = plt.subplots(1, 2, figsize=(11.2, 4.8))
        styles = {
            "Homoscedastic": (BLUE, "o", "-"),
            "Heteroscedastic": (GOLD, "s", "--"),
        }
        for scale_model in order:
            values = (
                table.loc[table["scale_model"].eq(scale_model)]
                .set_index("nominal_coverage")
                .loc[nominal]
            )
            color, marker, line = styles[scale_model]
            observed = values["simultaneous_group_coverage"].astype(float).to_numpy()
            coverage_axis.plot(
                nominal,
                observed,
                color=color,
                marker=marker,
                linestyle=line,
                linewidth=1.8,
                markersize=7,
                markerfacecolor="white" if scale_model == "Heteroscedastic" else color,
                markeredgewidth=1.4,
                label=scale_model,
            )
            for x, y in zip(nominal, observed, strict=True):
                coverage_axis.annotate(
                    f"{y:.1%}",
                    (x, y),
                    xytext=(0, 8 if scale_model == "Homoscedastic" else -15),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=color,
                )
        coverage_axis.plot(
            nominal,
            nominal,
            color=INK,
            linestyle=":",
            linewidth=1.4,
            marker="D",
            markersize=4.5,
            label="Nominal",
        )
        coverage_axis.set_xlim(0.89, 0.96)
        coverage_axis.set_ylim(0.88, 1.00)
        coverage_axis.set_xticks(nominal, ["90%", "95%"])
        coverage_axis.set_yticks(np.arange(0.88, 1.001, 0.02), [f"{v:.0%}" for v in np.arange(0.88, 1.001, 0.02)])
        coverage_axis.set_xlabel("Nominal coverage")
        coverage_axis.set_ylabel("Simultaneous group coverage")
        coverage_axis.set_title("(a) Group-level empirical coverage", fontsize=10.5, pad=10)
        coverage_axis.legend(frameon=False, fontsize=8.5, loc="lower right")
        _style_axes(coverage_axis)

        x = np.arange(len(nominal), dtype=float)
        bar_width = 0.34
        for offset, scale_model in zip((-bar_width / 2, bar_width / 2), order, strict=True):
            values = (
                table.loc[table["scale_model"].eq(scale_model)]
                .set_index("nominal_coverage")
                .loc[nominal, "mean_interval_width_um"]
                .astype(float)
                .to_numpy()
            )
            color = styles[scale_model][0]
            bars = width_axis.bar(
                x + offset,
                values,
                width=bar_width,
                color=color if scale_model == "Homoscedastic" else "white",
                edgecolor=color,
                linewidth=1.3,
                hatch="///" if scale_model == "Heteroscedastic" else "",
                label=scale_model,
            )
            for bar, value in zip(bars, values, strict=True):
                width_axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + 0.035,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8.5,
                    color=color,
                )
        width_axis.set_xticks(x, ["90%", "95%"])
        width_axis.set_ylim(0.0, 1.58)
        width_axis.set_xlabel("Nominal coverage")
        width_axis.set_ylabel("Mean interval width (µm)")
        width_axis.set_title("(b) Conformal interval width", fontsize=10.5, pad=10)
        width_axis.legend(frameon=False, fontsize=8.5, loc="upper left")
        _style_axes(width_axis)

        figure.suptitle("Group-conformal coverage and interval width", fontsize=13, fontweight="bold", y=1.04)
        figure.text(
            0.5,
            0.97,
            "All-seed summaries over 212 exchangeable machining groups; panel (a) uses a focused 88–100% scale.",
            ha="center",
            va="top",
            fontsize=9.2,
            color="#4E5965",
        )
        figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.91))
        _save_figure(figure, png, svg)


def _format_markdown_value(column: str, value: object) -> str:
    if pd.isna(value):
        return "—"
    if column in {
        "material_negative_transfer_rate",
        "nominal_coverage",
        "single_reading_coverage",
        "simultaneous_group_coverage",
    }:
        return f"{float(value):.2%}"
    if column == "fold_wins_vs_reference":
        return f"{int(value)}/5"
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.6f}"
    if isinstance(value, (bool, np.bool_)):
        return "Yes" if bool(value) else "No"
    return str(value)


def _frame_to_markdown(frame: pd.DataFrame) -> str:
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        cells = [
            _format_markdown_value(column, row[column]).replace("|", "\\|")
            for column in frame.columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _write_tables_markdown(tables: Mapping[str, pd.DataFrame], path: Path) -> None:
    captions = {
        "table_1_dataset_structure": "Table 1. Dataset and experimental structure.",
        "table_2_model_matrix": "Table 2. Registered model and comparator matrix.",
        "table_3_point_and_transfer": "Table 3. Point prediction and material negative transfer.",
        "table_4_conformal_results": "Table 4. All-seed group-conformal results.",
        "table_5_scale_ablation_bootstrap": "Table 5. Paired group-Bootstrap ablation of heteroscedastic versus homoscedastic scale.",
    }
    notes = {
        "table_3_point_and_transfer": "Positive MAE improvement means that the candidate has lower MAE than its reference. Material negative transfer uses a 0.01 µm group-MAE excess threshold. Phase B fold-win counts are intentionally omitted because the registered replication summary treats seeds as replicates.",
        "table_4_conformal_results": "Coverage applies to exchangeable new machining groups of the existing type. NLL and CRPS do not depend on the nominal conformal level and are repeated to keep each row self-contained.",
        "table_5_scale_ablation_bootstrap": "Differences are heteroscedastic minus homoscedastic; all listed metrics are lower-is-better. A positive interval excluding zero therefore disfavors the heteroscedastic scale model.",
    }
    blocks = ["# SGRPN manuscript tables", ""]
    for name, frame in tables.items():
        blocks.extend([f"## {captions[name]}", "", _frame_to_markdown(frame), ""])
        if name in notes:
            blocks.extend([f"Note: {notes[name]}", ""])
    path.write_text("\n".join(blocks).rstrip() + "\n", encoding="utf-8")


def _write_asset_notes(
    path: Path, tables: Mapping[str, pd.DataFrame], analysis: str
) -> None:
    table_3 = tables["table_3_point_and_transfer"]
    table_4 = tables["table_4_conformal_results"]
    phase_label = (
        "Corrective Phase B"
        if analysis == SPLIT_CONFORMAL_CORRECTIVE
        else "Phase B"
    )

    def transfer(phase: str, model: str) -> float:
        return float(
            _one_row(
                table_3,
                table_3["phase"].eq(phase) & table_3["model"].eq(model),
                f"asset note {phase} {model}",
            )["material_negative_transfer_rate"]
        )

    def conformal(model: str, nominal: float) -> pd.Series:
        return _one_row(
            table_4,
            table_4["scale_model"].eq(model)
            & np.isclose(table_4["nominal_coverage"].astype(float), nominal),
            f"asset note {model} {nominal}",
        )

    homo_90 = conformal("Homoscedastic", 0.90)
    homo_95 = conformal("Homoscedastic", 0.95)
    hetero_90 = conformal("Heteroscedastic", 0.90)
    hetero_95 = conformal("Heteroscedastic", 0.95)
    status = (
        "post-audit corrective reanalysis"
        if analysis == SPLIT_CONFORMAL_CORRECTIVE
        else "registered Phase B analysis"
    )
    phase_b_source = (
        "outputs/sgrpn/phase_b_split_conformal_corrective/evaluation"
        if analysis == SPLIT_CONFORMAL_CORRECTIVE
        else "outputs/sgrpn/phase_b/evaluation"
    )
    content = f"""# SGRPN manuscript asset notes

## Technical summary

These paper-ready assets use the **{status}**. The supported mean-model result is lower material negative transfer, not a statistically clear point-accuracy gain. Both scale variants are retained. The uncertainty claim is limited to empirical simultaneous group coverage for exchangeable new machining groups of the existing type; heteroscedasticity is not emphasized.

## Figure 1 contract and caption

**Architecture of SGRPN.** The P1 process expert supplies the fallback mean prediction. A residual CNN proposes a vibration-conditioned correction and a bounded gate controls the admitted correction. The diagram specifies computation only and does not give the gate a causal or physical interpretation.

## Figure 2 contract and caption

**Group-isolated model fitting, calibration, and testing.** Complete machining groups are assigned to proper training, fixed calibration, or outer testing. Calibration contributes one maximum standardized residual per group, and the locked predictor is applied once to outer-test groups.

## Figure 4 contract and caption

**Material negative-transfer rate relative to P1.** A group is materially harmed when candidate group MAE exceeds P1 by more than 0.01 µm. Phase A rates are {transfer('Phase A', 'F1'):.2%} for F1, {transfer('Phase A', 'R1'):.2%} for R1, and {transfer('Phase A', 'G1'):.2%} for G1. The {phase_label} rates are {transfer(phase_label, 'R1'):.2%} for R1 and {transfer(phase_label, 'G1'):.2%} for G1. Lower values indicate safer fusion; the result is not a claim of superior average accuracy.

## Figure 5 contract and caption

**Group split-conformal coverage and interval width.** At nominal 90% and 95%, the homoscedastic path attained group coverage of {float(homo_90['simultaneous_group_coverage']):.2%} and {float(homo_95['simultaneous_group_coverage']):.2%}, with widths of {float(homo_90['mean_interval_width_um']):.3f} and {float(homo_95['mean_interval_width_um']):.3f} µm. The heteroscedastic path attained {float(hetero_90['simultaneous_group_coverage']):.2%} and {float(hetero_95['simultaneous_group_coverage']):.2%}, with widths of {float(hetero_90['mean_interval_width_um']):.3f} and {float(hetero_95['mean_interval_width_um']):.3f} µm. Coverage applies only to exchangeable new groups of the observed type.

## Figure 6 contract and caption

**Paired group-Bootstrap scale ablation.** Points are heteroscedastic-minus-homoscedastic differences and bars are 95% paired group-Bootstrap intervals from 10,000 group resamples. All displayed metrics are lower-is-better; positive values therefore disfavor heteroscedasticity. The frozen corrective rule retains both variants but does not permit a heteroscedastic emphasis.

## Source inventory

- Phase A: `outputs/sgrpn/phase_a/evaluation`
- Phase B source for this asset set: `{phase_b_source}`
- Analysis status: `{status}`

The generator does not read the formal one-time claim-decision file and does not invoke training or evaluation commands.
"""
    path.write_text(content, encoding="utf-8")


def write_paper_assets(
    project_root: Path | str,
    output_dir: Path | str,
    *,
    analysis: str = REGISTERED_PHASE_B,
) -> dict[str, Path]:
    """Write five CSV tables, a Markdown table bundle, and Figures 1, 2, 4–6."""
    root = Path(project_root).resolve()
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    tables = build_paper_tables(root, analysis=analysis)
    artifacts: dict[str, Path] = {}
    for index, (name, frame) in enumerate(tables.items(), start=1):
        destination = output / TABLE_FILENAMES[name]
        frame.to_csv(destination, index=False, encoding="utf-8", float_format="%.12g")
        artifacts[f"table_{index}_csv"] = destination

    tables_markdown = output / "tables.md"
    _write_tables_markdown(tables, tables_markdown)
    artifacts["tables_markdown"] = tables_markdown

    figure_1_png = output / "fig_1_sgrpn_architecture.png"
    figure_1_svg = output / "fig_1_sgrpn_architecture.svg"
    _render_sgrpn_architecture(figure_1_png, figure_1_svg)
    artifacts["figure_1_png"] = figure_1_png
    artifacts["figure_1_svg"] = figure_1_svg

    figure_2_png = output / "fig_2_group_isolated_validation.png"
    figure_2_svg = output / "fig_2_group_isolated_validation.svg"
    _render_group_isolation_workflow(
        figure_2_png,
        figure_2_svg,
        analysis=analysis,
    )
    artifacts["figure_2_png"] = figure_2_png
    artifacts["figure_2_svg"] = figure_2_svg

    if analysis == SPLIT_CONFORMAL_CORRECTIVE:
        figure_3_png = output / "fig_3_corrective_prediction_scatter.png"
        figure_3_svg = output / "fig_3_corrective_prediction_scatter.svg"
        _render_corrective_prediction_scatter(root, figure_3_png, figure_3_svg)
        artifacts["figure_3_png"] = figure_3_png
        artifacts["figure_3_svg"] = figure_3_svg

    figure_4_png = output / "fig_4_negative_transfer.png"
    figure_4_svg = output / "fig_4_negative_transfer.svg"
    _render_negative_transfer(
        tables["table_3_point_and_transfer"], figure_4_png, figure_4_svg
    )
    artifacts["figure_4_png"] = figure_4_png
    artifacts["figure_4_svg"] = figure_4_svg

    figure_5_png = output / "fig_5_group_conformal_coverage_width.png"
    figure_5_svg = output / "fig_5_group_conformal_coverage_width.svg"
    _render_coverage_width(
        tables["table_4_conformal_results"], figure_5_png, figure_5_svg
    )
    artifacts["figure_5_png"] = figure_5_png
    artifacts["figure_5_svg"] = figure_5_svg

    figure_6_png = output / "fig_6_scale_ablation_bootstrap.png"
    figure_6_svg = output / "fig_6_scale_ablation_bootstrap.svg"
    _render_scale_ablation_bootstrap(
        tables["table_5_scale_ablation_bootstrap"], figure_6_png, figure_6_svg
    )
    artifacts["figure_6_png"] = figure_6_png
    artifacts["figure_6_svg"] = figure_6_svg

    asset_notes = output / "asset_notes.md"
    _write_asset_notes(asset_notes, tables, analysis)
    artifacts["asset_notes"] = asset_notes
    return artifacts


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate paper-ready SGRPN tables and static figures from frozen outputs."
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--output-dir", type=Path, default=Path("docs/paper/assets"),
    )
    parser.add_argument(
        "--analysis",
        choices=(REGISTERED_PHASE_B, SPLIT_CONFORMAL_CORRECTIVE),
        default=REGISTERED_PHASE_B,
    )
    args = parser.parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = args.project_root / output_dir
    artifacts = write_paper_assets(
        args.project_root,
        output_dir,
        analysis=args.analysis,
    )
    for name, path in artifacts.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
