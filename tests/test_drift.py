"""Tests for PSI / KS drift monitoring."""

import numpy as np
import pandas as pd

from src.monitoring.drift import (
    classify_drift,
    drift_report,
    ks_statistic,
    psi,
    psi_categorical,
)

PSI_MODERATE_THRESHOLD = 0.25


def test_psi_near_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    reference = rng.normal(50, 10, 5000)
    current = rng.normal(50, 10, 5000)
    assert psi(reference, current) < 0.02


def test_psi_increases_with_distribution_shift():
    rng = np.random.default_rng(1)
    reference = rng.normal(50, 10, 5000)
    small_shift = rng.normal(52, 10, 5000)
    large_shift = rng.normal(70, 10, 5000)

    psi_small = psi(reference, small_shift)
    psi_large = psi(reference, large_shift)
    assert psi_small < psi_large


def test_psi_classify_thresholds():
    assert classify_drift(0.05) == "stable"
    assert classify_drift(0.15) == "moderate"
    assert classify_drift(0.30) == "significant"
    # boundary values follow the "< max" convention
    assert classify_drift(0.10) == "moderate"
    assert classify_drift(0.25) == "significant"


def test_psi_categorical_near_zero_for_identical_proportions():
    rng = np.random.default_rng(2)
    reference = pd.Series(rng.choice(["A", "B", "C"], size=5000, p=[0.5, 0.3, 0.2]))
    current = pd.Series(rng.choice(["A", "B", "C"], size=5000, p=[0.5, 0.3, 0.2]))
    assert psi_categorical(reference, current) < 0.02


def test_psi_categorical_detects_proportion_shift():
    rng = np.random.default_rng(3)
    reference = pd.Series(rng.choice(["A", "B", "C"], size=5000, p=[0.5, 0.3, 0.2]))
    shifted = pd.Series(rng.choice(["A", "B", "C"], size=5000, p=[0.1, 0.2, 0.7]))
    assert psi_categorical(reference, shifted) > PSI_MODERATE_THRESHOLD


def test_psi_categorical_handles_unseen_category():
    reference = pd.Series(["A"] * 500 + ["B"] * 500)
    current = pd.Series(["A"] * 400 + ["B"] * 400 + ["C"] * 200)  # C is new
    value = psi_categorical(reference, current)
    assert value > 0
    assert np.isfinite(value)


def test_ks_statistic_zero_for_identical_arrays():
    rng = np.random.default_rng(4)
    reference = rng.normal(0, 1, 2000)
    assert ks_statistic(reference, reference) == 0.0


def test_ks_statistic_detects_shift():
    rng = np.random.default_rng(5)
    reference = rng.normal(0, 1, 3000)
    current_same = rng.normal(0, 1, 3000)
    current_shifted = rng.normal(3, 1, 3000)

    ks_same = ks_statistic(reference, current_same)
    ks_shifted = ks_statistic(reference, current_shifted)
    assert ks_same < ks_shifted
    assert ks_shifted > 0.8  # a 3-sigma mean shift should be almost fully separated


def test_ks_statistic_matches_scipy():
    from scipy.stats import ks_2samp

    rng = np.random.default_rng(6)
    reference = rng.normal(0, 1, 1000)
    current = rng.normal(0.5, 1.2, 1000)

    ours = ks_statistic(reference, current)
    scipy_stat = ks_2samp(reference, current).statistic
    assert abs(ours - scipy_stat) < 1e-9


def test_drift_report_flags_the_shifted_feature():
    rng = np.random.default_rng(7)
    n = 3000
    reference = pd.DataFrame({
        "stable_feature": rng.normal(50, 10, n),
        "shifted_feature": rng.normal(50, 10, n),
        "stable_cat": rng.choice(["X", "Y"], n, p=[0.6, 0.4]),
    })
    current = pd.DataFrame({
        "stable_feature": rng.normal(50, 10, n),
        "shifted_feature": rng.normal(90, 10, n),
        "stable_cat": rng.choice(["X", "Y"], n, p=[0.6, 0.4]),
    })

    report = drift_report(reference, current, num_cols=["stable_feature", "shifted_feature"],
                          cat_cols=["stable_cat"])

    assert report.iloc[0]["feature"] == "shifted_feature"
    assert report.iloc[0]["drift"] == "significant"
    stable_row = report[report["feature"] == "stable_feature"].iloc[0]
    assert stable_row["drift"] == "stable"
