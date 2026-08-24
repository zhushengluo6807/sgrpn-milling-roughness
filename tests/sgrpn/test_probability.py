import math

import numpy as np
import pandas as pd
import pytest

from roughness.sgrpn.probability import (
    GroupConformalResult,
    conformal_interval,
    finite_sample_group_quantile,
    gaussian_crps,
    group_conformal_scores,
    mean_interval_width,
    raw_gaussian_interval,
    simultaneous_group_coverage,
    single_reading_coverage,
    winkler_score,
)


def test_gaussian_crps_at_its_mean():
    value = gaussian_crps(
        np.array([0.0]), np.array([2.0]), np.array([[0.0, 0.0, 0.0]]), np.array([1.0])
    )
    assert value == pytest.approx(2.0 * (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi))


def test_gaussian_crps_averages_three_readings_before_region_weighting():
    value = gaussian_crps(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([[0.0, 0.0, 0.0], [0.0, 1.0, -1.0]]),
        np.array([1.0, 3.0]),
    )
    at_zero = (math.sqrt(2.0) - 1.0) / math.sqrt(math.pi)
    at_one = math.erf(1.0 / math.sqrt(2.0)) + 2.0 * math.exp(-0.5) / math.sqrt(2.0 * math.pi) - 1.0 / math.sqrt(math.pi)
    assert value == pytest.approx((at_zero + 3.0 * (at_zero + 2.0 * at_one) / 3.0) / 4.0)


@pytest.mark.parametrize(
    ("alpha", "z"),
    [(0.10, 1.6448536269514722), (0.05, 1.959963984540054)],
)
def test_raw_gaussian_interval_uses_registered_normal_quantile(alpha, z):
    lower, upper = raw_gaussian_interval(np.array([1.0]), np.array([2.0]), alpha=alpha)
    np.testing.assert_allclose(lower, [1.0 - 2.0 * z])
    np.testing.assert_allclose(upper, [1.0 + 2.0 * z])


def test_each_group_contributes_one_max_score():
    scores = group_conformal_scores(
        group_ids=np.array(["g1", "g1", "g2"]),
        mu=np.array([0.0, 1.0, 0.0]),
        sigma=np.ones(3),
        readings=np.array([[1.0, 2.0, 3.0], [1.0, 1.0, 1.0], [0.5, 0.5, 0.5]]),
    )
    assert scores["group_id"].tolist() == ["g1", "g2"]
    np.testing.assert_allclose(scores["score"], [3.0, 0.5])


def test_conformal_quantile_uses_ceil_m_plus_one_without_interpolation():
    result = finite_sample_group_quantile(np.arange(1.0, 20.0), alpha=0.10)
    assert result == GroupConformalResult(alpha=0.10, group_count=19, order_index=18, quantile=18.0)


def test_conformal_quantile_rejects_unsupported_alpha_and_insufficient_groups():
    with pytest.raises(ValueError, match="alpha"):
        finite_sample_group_quantile(np.arange(1.0, 30.0), alpha=0.20)
    with pytest.raises(ValueError, match="insufficient"):
        finite_sample_group_quantile(np.arange(1.0, 9.0), alpha=0.10)


def test_conformal_interval_scales_one_quantile_by_each_region_sigma():
    lower, upper = conformal_interval(
        np.array([2.0, 4.0]), np.array([0.5, 2.0]), quantile=3.0
    )
    np.testing.assert_allclose(lower, [0.5, -2.0])
    np.testing.assert_allclose(upper, [3.5, 10.0])


def test_single_reading_coverage_averages_repeats_then_weights_regions_inclusively():
    value = single_reading_coverage(
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
        np.array([[-1.0, 0.0, 1.0], [2.0, 2.0, 0.0]]),
        np.array([1.0, 3.0]),
    )
    assert value == pytest.approx((2.0 / 3.0 + 3.0 / 3.0) / 4.0)


def test_simultaneous_coverage_requires_every_repeat_of_every_region_in_group():
    value = simultaneous_group_coverage(
        np.array(["a", "a", "b"]),
        np.array([0.0, 0.0, 0.0]),
        np.array([1.0, 1.0, 1.0]),
        np.array([[0.0, 1.0, -1.0], [0.0, 1.1, 0.0], [0.0, 1.0, 0.0]]),
    )
    assert value == pytest.approx(0.5)


def test_mean_width_and_winkler_average_repeats_then_weight_regions():
    lower = np.array([-1.0, -2.0])
    upper = np.array([1.0, 2.0])
    readings = np.array([[0.0, 0.0, 0.0], [-3.0, 0.0, 4.0]])
    weights = np.array([1.0, 3.0])
    assert mean_interval_width(lower, upper, weights) == pytest.approx(3.5)
    # Region 1 scores are 2. Region 2 scores are 24, 4, and 44 for alpha=.10.
    assert winkler_score(lower, upper, readings, weights, alpha=0.10) == pytest.approx(18.5)


@pytest.mark.parametrize(
    "call",
    [
        lambda: gaussian_crps(np.array([0.0]), np.array([0.0]), np.zeros((1, 3)), np.array([1.0])),
        lambda: raw_gaussian_interval(np.array([np.nan]), np.array([1.0]), alpha=0.10),
        lambda: group_conformal_scores(np.array(["g"]), np.array([0.0]), np.array([1.0]), np.zeros((1, 2))),
        lambda: finite_sample_group_quantile(np.array([np.nan] * 19), alpha=0.10),
        lambda: single_reading_coverage(np.array([0.0]), np.array([1.0]), np.zeros((1, 3)), np.array([0.0])),
        lambda: simultaneous_group_coverage(np.array(["g", None], dtype=object), np.zeros(2), np.ones(2), np.zeros((2, 3))),
        lambda: mean_interval_width(np.array([1.0]), np.array([0.0]), np.array([1.0])),
        lambda: winkler_score(np.array([0.0]), np.array([1.0]), np.zeros((1, 3)), np.array([1.0]), alpha=0.20),
    ],
)
def test_probability_primitives_fail_closed_on_invalid_inputs(call):
    with pytest.raises(ValueError):
        call()


def test_probability_primitives_normalize_non_numeric_structures_to_value_error():
    with pytest.raises(ValueError):
        gaussian_crps({"bad": 1}, np.array([1.0]), np.zeros((1, 3)), np.array([1.0]))


def test_group_score_tables_reject_duplicate_groups_when_quantiled():
    scores = pd.DataFrame({"group_id": ["a", "a"], "score": [1.0, 2.0]})
    with pytest.raises(ValueError, match="duplicate"):
        finite_sample_group_quantile(scores, alpha=0.10)
