from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np
import pandas as pd

from roughness.scheme1.crossfit import make_group_inner_splits


FORMULAS = ("word", "exact")


def _return_scalar_or_array(value: np.ndarray):
    return float(value) if value.ndim == 0 else value


def word_ra_um(fz_mm, radius_mm):
    fz = np.asarray(fz_mm, dtype=np.float64)
    radius = float(radius_mm)
    if radius <= 0 or np.any(fz < 0):
        raise ValueError(
            "radius_mm must be positive and fz_mm non-negative"
        )
    result = 1000.0 * fz**2 / (32.0 * radius)
    return _return_scalar_or_array(result)


def exact_ra_um(fz_mm, radius_mm):
    fz = np.asarray(fz_mm, dtype=np.float64)
    radius = float(radius_mm)
    if radius <= 0:
        raise ValueError("radius_mm must be positive")
    if np.any((fz < 0) | (fz > 2.0 * radius)):
        raise ValueError(
            "fz_mm must satisfy 0 <= fz_mm <= 2 * radius_mm"
        )
    half_feed_squared = (fz / 2.0) ** 2
    root = np.sqrt(np.maximum(radius**2 - half_feed_squared, 0.0))
    # Rationalized form avoids cancellation when fz / radius is small.
    result = 1000.0 * half_feed_squared / (4.0 * (radius + root))
    return _return_scalar_or_array(result)


def physical_ra_um(formula: str, fz_mm, radius_mm):
    if formula == "word":
        return word_ra_um(fz_mm, radius_mm)
    if formula == "exact":
        return exact_ra_um(fz_mm, radius_mm)
    raise ValueError(f"Unknown formula: {formula}")


@dataclass(frozen=True)
class ScaleSelection:
    formula: str
    outer_fold: int
    seed: int
    radius_mm: float
    inner_weighted_mae: float
    source_sample_ids: tuple[str, ...]


_SELECTION_COLUMNS = {
    "sample_id",
    "group_id",
    "fz_mm_per_tooth",
    "ra_mean",
    "sample_weight",
}


def select_effective_scale(
    train_frame: pd.DataFrame,
    formula: str,
    candidates_mm: Sequence[float],
    inner_splits: int,
    outer_fold: int,
    seed: int,
) -> tuple[ScaleSelection, pd.DataFrame]:
    missing = sorted(_SELECTION_COLUMNS - set(train_frame.columns))
    if missing:
        raise ValueError(f"Scale selection frame missing columns: {missing}")
    if formula not in FORMULAS:
        raise ValueError(f"Unknown formula: {formula}")
    candidates = tuple(float(value) for value in candidates_mm)
    if not candidates or any(value <= 0 for value in candidates):
        raise ValueError("candidates_mm must contain positive values")
    source_ids = tuple(sorted(train_frame["sample_id"].astype(str)))
    source_text = "|".join(source_ids)
    splits = make_group_inner_splits(train_frame, inner_splits, seed)
    target = train_frame["ra_mean"].to_numpy(dtype=np.float64)
    weights = train_frame["sample_weight"].to_numpy(dtype=np.float64)
    if np.any(weights <= 0) or not np.isfinite(target).all():
        raise ValueError("Scale selection inputs must be finite and weighted")

    rows = []
    for radius in candidates:
        prediction = np.full(len(train_frame), np.nan, dtype=np.float64)
        valid = True
        for _, validation_indices in splits:
            try:
                prediction[validation_indices] = physical_ra_um(
                    formula,
                    train_frame.iloc[validation_indices][
                        "fz_mm_per_tooth"
                    ].to_numpy(dtype=np.float64),
                    radius,
                )
            except ValueError:
                valid = False
                break
        weighted_mae = (
            float(np.average(np.abs(target - prediction), weights=weights))
            if valid and np.isfinite(prediction).all()
            else float("inf")
        )
        rows.append(
            {
                "formula": formula,
                "outer_fold": int(outer_fold),
                "seed": int(seed),
                "candidate_radius_mm": radius,
                "weighted_mae": weighted_mae,
                "valid": bool(np.isfinite(weighted_mae)),
                "source_sample_ids": source_text,
            }
        )
    audit = pd.DataFrame.from_records(rows)
    valid_audit = audit[np.isfinite(audit["weighted_mae"])].copy()
    if valid_audit.empty:
        raise ValueError("No valid effective-scale candidate")
    best = valid_audit.sort_values(
        ["weighted_mae", "candidate_radius_mm"], kind="mergesort"
    ).iloc[0]
    selection = ScaleSelection(
        formula=formula,
        outer_fold=int(outer_fold),
        seed=int(seed),
        radius_mm=float(best["candidate_radius_mm"]),
        inner_weighted_mae=float(best["weighted_mae"]),
        source_sample_ids=source_ids,
    )
    return selection, audit


def build_physics_outer_frames(
    manifest: pd.DataFrame,
    folds: pd.DataFrame,
    outer_fold: int,
    seed: int,
    formula: str,
    candidates_mm: Sequence[float],
    inner_splits: int,
) -> tuple[pd.DataFrame, pd.DataFrame, ScaleSelection, pd.DataFrame]:
    required_folds = {"sample_id", "fold"}
    missing_folds = sorted(required_folds - set(folds.columns))
    if missing_folds:
        raise ValueError(f"Fold frame missing columns: {missing_folds}")
    data = manifest.merge(
        folds[["sample_id", "fold"]],
        on="sample_id",
        validate="one_to_one",
    )
    train = data[data["fold"].astype(int) != int(outer_fold)].reset_index(
        drop=True
    )
    test = data[data["fold"].astype(int) == int(outer_fold)].reset_index(
        drop=True
    )
    if train.empty or test.empty:
        raise ValueError(f"Outer fold {outer_fold} has empty train or test")
    selection, audit = select_effective_scale(
        train,
        formula=formula,
        candidates_mm=candidates_mm,
        inner_splits=inner_splits,
        outer_fold=outer_fold,
        seed=seed,
    )
    for frame in (train, test):
        frame["base_ra"] = physical_ra_um(
            formula,
            frame["fz_mm_per_tooth"].to_numpy(dtype=np.float64),
            selection.radius_mm,
        )
    return train, test, selection, audit
