from pathlib import Path, PureWindowsPath

import numpy as np
import pandas as pd

from .config import TrainingConfig


def relative_after_segments(raw: str) -> Path:
    parts = PureWindowsPath(str(raw)).parts
    lowered = [part.lower() for part in parts]
    if "segments" not in lowered:
        raise ValueError(f"Path does not contain segments: {raw}")
    index = lowered.index("segments")
    suffix = parts[index + 1 :]
    if not suffix:
        raise ValueError(f"Path has no suffix after segments: {raw}")
    return Path(*suffix)


def geometry_ra_um(fz_mm_per_tooth: float, radius_mm: float = 5.0) -> float:
    return float((fz_mm_per_tooth**2 / (32.0 * radius_mm)) * 1000.0)


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).replace("\n", "").strip() for column in frame.columns]
    return frame


def _load_process_table(path: Path, version: str) -> pd.DataFrame:
    frame = _normalize_columns(
        pd.read_excel(path, sheet_name="实验记录表", header=2, engine="openpyxl")
    )
    frame = frame.rename(
        columns={
            "执行顺序": "execution_order",
            "转速(rpm)": "n_rpm",
            "进给(mm/z)": "fz_mm_per_tooth",
            "切深(mm)": "ap_mm",
        }
    )
    required = ["execution_order", "n_rpm", "fz_mm_per_tooth", "ap_mm"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"Missing process columns in {path}: {missing}")
    frame = frame[required].copy()
    numeric_order = pd.to_numeric(frame["execution_order"], errors="coerce")
    frame = frame[numeric_order.notna()].copy()
    frame["execution_order"] = numeric_order[numeric_order.notna()].astype(int)
    frame["version"] = version
    if frame.duplicated(["version", "execution_order"]).any():
        raise ValueError(f"Duplicate process keys in {path}")
    return frame


def build_manifest(config: TrainingConfig) -> tuple[pd.DataFrame, dict]:
    labels = _normalize_columns(
        pd.read_excel(
            config.labels,
            sheet_name="粗糙度测量记录",
            header=3,
            engine="openpyxl",
        )
    )
    labels = labels.rename(
        columns={
            "版本": "version",
            "执行顺序": "execution_order",
            "原始实验组": "group_id",
            "该组切分数": "split_count",
            "区域编号": "region_index",
            "片段时长(s)": "duration_s",
            "对应信号文件": "source_signal_path",
            "Ra第1次(μm)": "ra_1",
            "Ra第2次(μm)": "ra_2",
            "Ra第3次(μm)": "ra_3",
            "Ra平均值(μm)": "ra_mean",
        }
    )
    required_label_columns = [
        "version",
        "execution_order",
        "group_id",
        "split_count",
        "region_index",
        "duration_s",
        "source_signal_path",
        "ra_1",
        "ra_2",
        "ra_3",
        "ra_mean",
    ]
    missing_label_columns = [
        column for column in required_label_columns if column not in labels
    ]
    if missing_label_columns:
        raise ValueError(f"Missing label columns: {missing_label_columns}")
    labels = labels.dropna(
        subset=["version", "execution_order", "region_index"]
    ).copy()
    labels["version"] = labels["version"].astype(str).str.lower()
    labels["execution_order"] = labels["execution_order"].astype(int)
    labels["region_index"] = labels["region_index"].astype(int)
    labels["split_count"] = labels["split_count"].astype(int)

    process = pd.concat(
        [
            _load_process_table(config.v3_records, "v3"),
            _load_process_table(config.v4_records, "v4"),
        ],
        ignore_index=True,
    )
    manifest = labels.merge(
        process,
        on=["version", "execution_order"],
        how="left",
        validate="many_to_one",
    )
    manifest["sample_id"] = manifest.apply(
        lambda row: (
            f"{row.version}_{row.execution_order}_seg{row.region_index}"
        ),
        axis=1,
    )
    manifest["signal_path"] = manifest["source_signal_path"].map(
        lambda value: config.segments_root / relative_after_segments(value)
    )
    readings = manifest[["ra_1", "ra_2", "ra_3"]].astype(float)
    manifest["ra_std"] = readings.std(axis=1, ddof=1)
    manifest["ra_range"] = readings.max(axis=1) - readings.min(axis=1)
    manifest["ra_geo_um"] = manifest["fz_mm_per_tooth"].map(geometry_ra_um)
    manifest["sample_weight"] = 1.0 / manifest["split_count"]

    required_values = [
        "sample_id",
        "group_id",
        "ra_mean",
        "n_rpm",
        "fz_mm_per_tooth",
        "ap_mm",
        "signal_path",
    ]
    if len(manifest) != 586 or manifest["group_id"].nunique() != 212:
        raise ValueError(
            "Unexpected dataset shape: "
            f"rows={len(manifest)}, groups={manifest['group_id'].nunique()}"
        )
    if manifest[required_values].isna().any().any():
        raise ValueError("Required manifest values contain missing data")
    if manifest["sample_id"].duplicated().any():
        raise ValueError("Duplicate sample_id")

    missing_files = [
        str(path) for path in manifest["signal_path"] if not Path(path).is_file()
    ]
    if missing_files:
        raise ValueError(f"Missing signal files: {missing_files[:10]}")

    group_weights = manifest.groupby("group_id")["sample_weight"].sum()
    if not np.allclose(group_weights.to_numpy(), 1.0):
        raise ValueError("Group weights do not sum to one")

    computed_mean = readings.mean(axis=1)
    if not np.allclose(computed_mean, manifest["ra_mean"].astype(float)):
        raise ValueError("Ra mean does not match the three measurements")

    audit = {
        "rows": len(manifest),
        "groups": int(manifest["group_id"].nunique()),
        "missing_signal_files": 0,
        "n_range": [float(manifest["n_rpm"].min()), float(manifest["n_rpm"].max())],
        "fz_range": [
            float(manifest["fz_mm_per_tooth"].min()),
            float(manifest["fz_mm_per_tooth"].max()),
        ],
        "ap_range": [float(manifest["ap_mm"].min()), float(manifest["ap_mm"].max())],
        "ra_range": [
            float(manifest["ra_mean"].min()),
            float(manifest["ra_mean"].max()),
        ],
    }
    return manifest.sort_values("sample_id").reset_index(drop=True), audit
