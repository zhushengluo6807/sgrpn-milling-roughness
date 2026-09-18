from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest

from roughness.sgrpn.paper_assets import build_paper_tables, write_paper_assets


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def test_build_paper_tables_preserves_registered_paper_values() -> None:
    tables = build_paper_tables(PROJECT_ROOT)

    assert tuple(tables) == (
        "table_1_dataset_structure",
        "table_2_model_matrix",
        "table_3_point_and_transfer",
        "table_4_conformal_results",
        "table_5_scale_ablation_bootstrap",
    )

    table_1 = tables["table_1_dataset_structure"].set_index("item")
    assert table_1.loc["Independent machining groups", "value"] == "212"
    assert table_1.loc["Surface regions", "value"] == "586"
    assert table_1.loc["Vibration sampling rate", "value"] == "25.6 kHz"

    table_3 = tables["table_3_point_and_transfer"]
    phase_a_g1 = table_3.loc[
        (table_3["phase"] == "Phase A") & (table_3["model"] == "G1")
    ].iloc[0]
    assert phase_a_g1["mae_um"] == pytest.approx(0.10631405416999465)
    assert phase_a_g1["fold_wins_vs_reference"] == 2
    assert phase_a_g1["material_negative_transfer_rate"] == pytest.approx(
        0.19811320754716982
    )
    assert phase_a_g1["mae_improvement_ci95"] == "[-0.002421, 0.005166]"

    table_4 = tables["table_4_conformal_results"]
    homo_90 = table_4.loc[
        (table_4["scale_model"] == "Homoscedastic")
        & (table_4["nominal_coverage"] == 0.90)
    ].iloc[0]
    assert homo_90["simultaneous_group_coverage"] == pytest.approx(
        0.9088050314465409
    )
    assert homo_90["mean_interval_width_um"] == pytest.approx(
        0.5778113835183141
    )

    table_5 = tables["table_5_scale_ablation_bootstrap"]
    nll = table_5.loc[table_5["metric"] == "Gaussian NLL"].iloc[0]
    assert nll["heteroscedastic_minus_homoscedastic"] == pytest.approx(
        16.02695982164094
    )
    assert nll["ci_excludes_zero"]
    assert nll["registered_interpretation"] == "No heteroscedastic advantage"


def test_build_paper_tables_uses_split_conformal_corrective_results() -> None:
    tables = build_paper_tables(PROJECT_ROOT, analysis="split_conformal_corrective")

    phase_b = tables["table_3_point_and_transfer"]
    assert set(phase_b["phase"]) == {"Phase A", "Phase B", "Corrective Phase B"}
    original_g1 = phase_b.loc[
        (phase_b["phase"] == "Phase B") & (phase_b["model"] == "G1")
    ].iloc[0]
    assert original_g1["mae_um"] == pytest.approx(0.1081493024972666)
    g1 = phase_b.loc[
        (phase_b["phase"] == "Corrective Phase B") & (phase_b["model"] == "G1")
    ].iloc[0]
    assert g1["mae_um"] == pytest.approx(0.1113769446641985)
    assert g1["material_negative_transfer_rate"] == pytest.approx(
        0.1650943396226415
    )

    table_4 = tables["table_4_conformal_results"]
    homo_90 = table_4.loc[
        (table_4["scale_model"] == "Homoscedastic")
        & (table_4["nominal_coverage"] == 0.90)
    ].iloc[0]
    assert homo_90["simultaneous_group_coverage"] == pytest.approx(
        0.9245283018867924
    )
    assert homo_90["mean_interval_width_um"] == pytest.approx(
        0.6434491454134849
    )

    table_5 = tables["table_5_scale_ablation_bootstrap"]
    winkler_95 = table_5.loc[
        (table_5["metric"] == "Winkler score")
        & (table_5["nominal_coverage"] == 0.95)
    ].iloc[0]
    assert winkler_95["heteroscedastic_minus_homoscedastic"] == pytest.approx(
        1.0160079871528649
    )
    assert winkler_95["registered_interpretation"] == "No heteroscedastic emphasis"


def test_write_paper_assets_exports_tables_figures_and_manifest(tmp_path: Path) -> None:
    artifacts = write_paper_assets(PROJECT_ROOT, tmp_path)

    expected = {
        *(f"table_{index}_csv" for index in range(1, 6)),
        "tables_markdown",
        "figure_1_png",
        "figure_1_svg",
        "figure_2_png",
        "figure_2_svg",
        "figure_4_png",
        "figure_4_svg",
        "figure_5_png",
        "figure_5_svg",
        "figure_6_png",
        "figure_6_svg",
        "asset_notes",
    }
    assert set(artifacts) == expected
    assert all(path.is_file() and path.stat().st_size > 0 for path in artifacts.values())
    assert artifacts["figure_4_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert "<svg" in artifacts["figure_5_svg"].read_text(encoding="utf-8")

    figure_1_svg = artifacts["figure_1_svg"].read_text(encoding="utf-8")
    assert "Safe fallback" in figure_1_svg
    assert "gated vibration correction" in figure_1_svg

    figure_2_svg = artifacts["figure_2_svg"].read_text(encoding="utf-8")
    assert "Outer-test groups" in figure_2_svg
    assert "maximum score per calibration group" in figure_2_svg

    figure_6_svg = artifacts["figure_6_svg"].read_text(encoding="utf-8")
    assert "Heteroscedastic minus homoscedastic" in figure_6_svg
    assert "16.027" in figure_6_svg

    exported = pd.read_csv(artifacts["table_4_csv"])
    assert len(exported) == 4
    notes = artifacts["asset_notes"].read_text(encoding="utf-8")
    assert "0.01 µm" in notes
    assert "exchangeable new machining groups" in notes


def test_write_paper_assets_exports_corrective_tables_and_figures(tmp_path: Path) -> None:
    artifacts = write_paper_assets(
        PROJECT_ROOT,
        tmp_path,
        analysis="split_conformal_corrective",
    )

    table_4 = pd.read_csv(artifacts["table_4_csv"])
    homo_90 = table_4.loc[
        (table_4["scale_model"] == "Homoscedastic")
        & (table_4["nominal_coverage"] == 0.90)
    ].iloc[0]
    assert homo_90["simultaneous_group_coverage"] == pytest.approx(
        0.9245283018867924
    )
    assert artifacts["figure_3_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    assert artifacts["figure_5_png"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    figure_2 = artifacts["figure_2_svg"].read_text(encoding="utf-8")
    assert "43 fixed calibration groups" in figure_2
    assert "proper-train groups only" in figure_2
    assert "Four group-disjoint" not in figure_2
    assert "post-audit corrective" in artifacts["asset_notes"].read_text(
        encoding="utf-8"
    )


def test_cli_uses_a_writable_matplotlib_cache(tmp_path: Path) -> None:
    environment = os.environ.copy()
    environment.pop("MPLCONFIGDIR", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roughness.sgrpn.paper_assets",
            "--project-root",
            str(PROJECT_ROOT),
            "--output-dir",
            str(tmp_path / "assets"),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "font_manager cache" not in result.stderr


def test_cli_accepts_split_conformal_corrective_analysis(tmp_path: Path) -> None:
    output = tmp_path / "corrective-assets"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "roughness.sgrpn.paper_assets",
            "--project-root",
            str(PROJECT_ROOT),
            "--output-dir",
            str(output),
            "--analysis",
            "split_conformal_corrective",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    table_4 = pd.read_csv(output / "table_4_conformal_results.csv")
    assert table_4.loc[0, "simultaneous_group_coverage"] == pytest.approx(
        0.9245283018867924
    )
