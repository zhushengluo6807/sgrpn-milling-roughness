from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from roughness.scheme1.dataset import (
    SegmentBagDataset,
    collate_segment_bags,
)
from roughness.scheme1.signals import ChannelStats
from roughness.scheme1.training import make_group_train_validation_split

from .config import Scheme1PhysicsConfig
from .physics import ScaleSelection, build_physics_outer_frames


PROCESS_COLUMNS = ("n_rpm", "fz_mm_per_tooth", "ap_mm")
SCALED_COLUMNS = (*PROCESS_COLUMNS, "base_ra")


@dataclass(frozen=True)
class InputScaler:
    mean: dict[str, float]
    scale: dict[str, float]
    source_sample_ids: tuple[str, ...]

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        for column in PROCESS_COLUMNS:
            result[column] = (
                result[column].astype(float) - self.mean[column]
            ) / self.scale[column]
        result["base_ra_scaled"] = (
            result["base_ra"].astype(float) - self.mean["base_ra"]
        ) / self.scale["base_ra"]
        return result


def fit_input_scaler(frame: pd.DataFrame) -> InputScaler:
    missing = sorted(set((*SCALED_COLUMNS, "sample_id")) - set(frame.columns))
    if missing:
        raise ValueError(f"Scaler frame missing columns: {missing}")
    values = frame[list(SCALED_COLUMNS)].astype(float)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Scaler inputs must be finite")
    mean = values.mean().to_dict()
    scale_series = values.std(ddof=0).mask(lambda value: value <= 0, 1.0)
    return InputScaler(
        mean={key: float(value) for key, value in mean.items()},
        scale={
            key: float(value) for key, value in scale_series.to_dict().items()
        },
        source_sample_ids=tuple(frame["sample_id"].astype(str)),
    )


@dataclass(frozen=True)
class PreparedPhysicsFold:
    formula: str
    outer_fold: int
    seed: int
    selection: ScaleSelection
    selection_audit: pd.DataFrame
    train_raw: pd.DataFrame
    test_raw: pd.DataFrame
    train_scaled: pd.DataFrame
    test_scaled: pd.DataFrame
    scaler: InputScaler


def prepare_physics_fold(
    config: Scheme1PhysicsConfig,
    formula: str,
    outer_fold: int,
    seed: int,
) -> PreparedPhysicsFold:
    manifest = pd.read_csv(config.source.manifest_path)
    folds = pd.read_csv(config.source.folds_path)
    train_raw, test_raw, selection, audit = build_physics_outer_frames(
        manifest,
        folds,
        outer_fold=outer_fold,
        seed=seed,
        formula=formula,
        candidates_mm=config.source.re_candidates_mm,
        inner_splits=config.source.inner_splits,
    )
    scaler = fit_input_scaler(train_raw)
    return PreparedPhysicsFold(
        formula=formula,
        outer_fold=int(outer_fold),
        seed=int(seed),
        selection=selection,
        selection_audit=audit,
        train_raw=train_raw,
        test_raw=test_raw,
        train_scaled=scaler.transform(train_raw),
        test_scaled=scaler.transform(test_raw),
        scaler=scaler,
    )


@dataclass(frozen=True)
class LoaderBundle:
    train_loader: DataLoader
    validation_loader: DataLoader
    test_loader: DataLoader
    train_groups: frozenset[str]
    validation_groups: frozenset[str]
    test_groups: frozenset[str]


def _load_channel_stats(path: Path) -> ChannelStats:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return ChannelStats(
        mean=np.asarray(raw["mean"], dtype=np.float64),
        std=np.asarray(raw["std"], dtype=np.float64),
        sample_count=int(raw["sample_count"]),
        source_segment_ids=tuple(map(str, raw["source_segment_ids"])),
    )


def build_physics_loaders(
    prepared: PreparedPhysicsFold,
    config: Scheme1PhysicsConfig,
    batch_size: int,
    validation_fraction: float,
    seed: int,
    device: str | torch.device,
) -> LoaderBundle:
    selected_device = torch.device(device)
    window_index = pd.read_csv(config.source.output_dir / "window_index.csv")
    channel_stats = _load_channel_stats(
        config.source.output_dir
        / "folds"
        / f"fold_{prepared.outer_fold}_channel_stats.json"
    )
    train_dataset = SegmentBagDataset(
        prepared.train_scaled,
        window_index,
        channel_stats,
        horizontal_mode="raw",
        physics_columns=("base_ra_scaled",),
        base_ra_column="base_ra",
        cache_signals=True,
    )
    test_dataset = SegmentBagDataset(
        prepared.test_scaled,
        window_index,
        channel_stats,
        horizontal_mode="raw",
        physics_columns=("base_ra_scaled",),
        base_ra_column="base_ra",
        cache_signals=True,
    )
    train_indices, validation_indices = make_group_train_validation_split(
        prepared.train_scaled,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    loader_args = {
        "batch_size": int(batch_size),
        "collate_fn": collate_segment_bags,
        "num_workers": 0,
        "pin_memory": selected_device.type == "cuda",
    }
    generator = torch.Generator().manual_seed(int(seed))
    train_loader = DataLoader(
        Subset(train_dataset, train_indices.tolist()),
        shuffle=True,
        generator=generator,
        **loader_args,
    )
    validation_loader = DataLoader(
        Subset(train_dataset, validation_indices.tolist()),
        shuffle=False,
        **loader_args,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_args)

    group_values = prepared.train_scaled["group_id"].astype(str)
    train_groups = frozenset(group_values.iloc[train_indices])
    validation_groups = frozenset(group_values.iloc[validation_indices])
    test_groups = frozenset(prepared.test_scaled["group_id"].astype(str))
    if (
        not train_groups.isdisjoint(validation_groups)
        or not train_groups.isdisjoint(test_groups)
        or not validation_groups.isdisjoint(test_groups)
    ):
        raise AssertionError("Group leakage across physics loaders")
    return LoaderBundle(
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        train_groups=train_groups,
        validation_groups=validation_groups,
        test_groups=test_groups,
    )
