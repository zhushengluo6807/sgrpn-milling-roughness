"""Probability metrics for one Gaussian prediction shared by three readings."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import pandas as pd


_NORMAL_Z = {0.10: 1.6448536269514722, 0.05: 1.959963984540054}
_READING_COUNT = 3


@dataclass(frozen=True)
class GroupConformalResult:
    alpha: float
    group_count: int
    order_index: int
    quantile: float


def _alpha(value: Any) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("alpha must be either 0.10 or 0.05")
    try:
        alpha = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError("alpha must be either 0.10 or 0.05") from error
    if alpha not in _NORMAL_Z:
        raise ValueError("alpha must be either 0.10 or 0.05")
    return alpha


def _numeric_array(value: Any, message: str) -> np.ndarray:
    try:
        return np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error


@contextmanager
def _checked_arithmetic(message: str):
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            yield
    except FloatingPointError as error:
        raise ValueError(message) from error


def _finite_array(value: Any, message: str) -> np.ndarray:
    values = _numeric_array(value, message)
    if not np.isfinite(values).all():
        raise ValueError(message)
    return values


def _finite_metric(value: Any, message: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(message) from error
    if not math.isfinite(result):
        raise ValueError(message)
    return result


def _weighted_region_mean(values: np.ndarray, weights: np.ndarray, message: str) -> float:
    with _checked_arithmetic(message):
        result = np.dot(weights, values) / weights.sum()
    return _finite_metric(result, message)


def _regions(mu: Any, sigma: Any) -> tuple[np.ndarray, np.ndarray]:
    message = "mu and sigma must be finite equal-length regions with positive sigma"
    location = _numeric_array(mu, message)
    scale = _numeric_array(sigma, message)
    if (
        location.ndim != 1
        or scale.ndim != 1
        or len(location) == 0
        or len(location) != len(scale)
        or not np.isfinite(location).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
    ):
        raise ValueError(message)
    return location, scale


def _readings(value: Any, region_count: int) -> np.ndarray:
    readings = _numeric_array(value, "readings must be a finite [region, 3] array")
    if (
        readings.ndim != 2
        or readings.shape != (region_count, _READING_COUNT)
        or not np.isfinite(readings).all()
    ):
        raise ValueError("readings must be a finite [region, 3] array")
    return readings


def _weights(value: Any, region_count: int) -> np.ndarray:
    weights = _numeric_array(
        value, "region weights must be finite, positive, and match regions"
    )
    if (
        weights.ndim != 1
        or len(weights) != region_count
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("region weights must be finite, positive, and match regions")
    return weights


def _interval(lower: Any, upper: Any) -> tuple[np.ndarray, np.ndarray]:
    message = "interval bounds must be finite equal-length regions with lower <= upper"
    lower_values = _numeric_array(lower, message)
    upper_values = _numeric_array(upper, message)
    if (
        lower_values.ndim != 1
        or upper_values.ndim != 1
        or len(lower_values) == 0
        or len(lower_values) != len(upper_values)
        or not np.isfinite(lower_values).all()
        or not np.isfinite(upper_values).all()
        or np.any(lower_values > upper_values)
    ):
        raise ValueError(message)
    return lower_values, upper_values


def _group_ids(value: Any, region_count: int) -> np.ndarray:
    groups = np.asarray(value, dtype=object)
    if groups.ndim != 1 or len(groups) != region_count or pd.isna(groups).any():
        raise ValueError("group_ids must be non-missing and match regions")
    return groups.astype(str)


def gaussian_crps(mu: Any, sigma: Any, readings: Any, region_weights: Any) -> float:
    """Return region-weighted Gaussian CRPS, averaging three readings per region."""
    location, scale = _regions(mu, sigma)
    observed = _readings(readings, len(location))
    weights = _weights(region_weights, len(location))
    message = "Gaussian CRPS calculation must remain finite"
    with _checked_arithmetic(message):
        z = (observed - location[:, None]) / scale[:, None]
        phi = np.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
        normal_cdf = 0.5 * (1.0 + np.vectorize(math.erf)(z / math.sqrt(2.0)))
        per_reading = scale[:, None] * (
            z * (2.0 * normal_cdf - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi)
        )
        per_region = _finite_array(per_reading.mean(axis=1), message)
    return _weighted_region_mean(per_region, weights, message)


def raw_gaussian_interval(mu: Any, sigma: Any, *, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """Return the registered central Gaussian interval for each region."""
    location, scale = _regions(mu, sigma)
    z = _NORMAL_Z[_alpha(alpha)]
    message = "Gaussian interval calculation must remain finite"
    with _checked_arithmetic(message):
        lower = location - z * scale
        upper = location + z * scale
    return _finite_array(lower, message), _finite_array(upper, message)


def group_conformal_scores(
    group_ids: Any, mu: Any, sigma: Any, readings: Any
) -> pd.DataFrame:
    """Return one maximum standardized absolute-error score for each group."""
    location, scale = _regions(mu, sigma)
    groups = _group_ids(group_ids, len(location))
    observed = _readings(readings, len(location))
    message = "group conformal score calculation must remain finite"
    with _checked_arithmetic(message):
        region_scores = np.abs(observed - location[:, None]).max(axis=1) / scale
    region_scores = _finite_array(region_scores, message)
    scores = pd.DataFrame({"group_id": groups, "score": region_scores})
    grouped = scores.groupby("group_id", as_index=False, sort=True, observed=True)["score"].max()
    _finite_array(grouped["score"], message)
    return grouped


def finite_sample_group_quantile(scores: Any, *, alpha: float) -> GroupConformalResult:
    """Select the preregistered one-based finite-sample group order statistic."""
    resolved_alpha = _alpha(alpha)
    if isinstance(scores, pd.DataFrame):
        required = {"group_id", "score"}
        if not required.issubset(scores.columns) or scores["group_id"].isna().any():
            raise ValueError("group score table must include non-missing group_id and score")
        if scores["group_id"].astype(str).duplicated().any():
            raise ValueError("group score table must not contain duplicate group_id values")
        try:
            values = scores["score"].to_numpy(dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("group scores must be finite") from error
    else:
        values = _numeric_array(scores, "group scores must be a non-empty finite non-negative vector")
    if values.ndim != 1 or len(values) == 0 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("group scores must be a non-empty finite non-negative vector")
    values = np.sort(values)
    k = math.ceil((len(values) + 1) * (1.0 - resolved_alpha))
    if k > len(values):
        raise ValueError("insufficient calibration groups for requested alpha")
    return GroupConformalResult(resolved_alpha, len(values), k, float(values[k - 1]))


def conformal_interval(
    mu: Any, sigma: Any, *, quantile: float | GroupConformalResult
) -> tuple[np.ndarray, np.ndarray]:
    """Return a group-conformal interval ``mu ± q * sigma`` for every region."""
    location, scale = _regions(mu, sigma)
    q = quantile.quantile if isinstance(quantile, GroupConformalResult) else quantile
    try:
        q_value = float(q)
    except (TypeError, ValueError) as error:
        raise ValueError("conformal quantile must be finite and non-negative") from error
    if not np.isfinite(q_value) or q_value < 0.0:
        raise ValueError("conformal quantile must be finite and non-negative")
    message = "conformal interval calculation must remain finite"
    with _checked_arithmetic(message):
        lower = location - q_value * scale
        upper = location + q_value * scale
    return _finite_array(lower, message), _finite_array(upper, message)


def single_reading_coverage(
    lower: Any, upper: Any, readings: Any, region_weights: Any
) -> float:
    """Return region-weighted coverage after averaging the three indicator values."""
    low, high = _interval(lower, upper)
    observed = _readings(readings, len(low))
    weights = _weights(region_weights, len(low))
    covered = ((low[:, None] <= observed) & (observed <= high[:, None])).mean(axis=1)
    return _weighted_region_mean(
        covered, weights, "single-reading coverage calculation must remain finite"
    )


def simultaneous_group_coverage(
    group_ids: Any, lower: Any, upper: Any, readings: Any
) -> float:
    """Return the unweighted fraction of groups with all regions and repeats covered."""
    low, high = _interval(lower, upper)
    groups = _group_ids(group_ids, len(low))
    observed = _readings(readings, len(low))
    by_region = ((low[:, None] <= observed) & (observed <= high[:, None])).all(axis=1)
    grouped = pd.DataFrame({"group_id": groups, "covered": by_region}).groupby(
        "group_id", sort=True, observed=True
    )["covered"].all()
    return _finite_metric(
        grouped.mean(), "simultaneous group coverage calculation must remain finite"
    )


def mean_interval_width(lower: Any, upper: Any, region_weights: Any) -> float:
    """Return region-weighted mean interval width."""
    low, high = _interval(lower, upper)
    weights = _weights(region_weights, len(low))
    message = "mean interval width calculation must remain finite"
    with _checked_arithmetic(message):
        width = _finite_array(high - low, message)
    return _weighted_region_mean(width, weights, message)


def winkler_score(
    lower: Any, upper: Any, readings: Any, region_weights: Any, *, alpha: float
) -> float:
    """Return region-weighted Winkler score, averaging three readings per region."""
    resolved_alpha = _alpha(alpha)
    low, high = _interval(lower, upper)
    observed = _readings(readings, len(low))
    weights = _weights(region_weights, len(low))
    message = "Winkler score calculation must remain finite"
    with _checked_arithmetic(message):
        per_reading = np.broadcast_to((high - low)[:, None], observed.shape).astype(
            np.float64, copy=True
        )
        below = observed < low[:, None]
        above = observed > high[:, None]
        per_reading[below] += (2.0 / resolved_alpha) * (low[:, None] - observed)[below]
        per_reading[above] += (2.0 / resolved_alpha) * (observed - high[:, None])[above]
        per_region = _finite_array(per_reading.mean(axis=1), message)
    return _weighted_region_mean(per_region, weights, message)
