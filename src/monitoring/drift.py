"""Data and score drift monitoring for a deployed churn model, from scratch.

A model validated on last year's data can silently degrade as the customer
population, pricing, or contract mix shifts — with no error thrown, no
exception logged, just quietly worse decisions. Drift monitoring answers:
*has the input distribution (or the model's own output distribution) moved
enough since training that the validated performance no longer applies?*

Two complementary tests, both comparing a fixed `reference` distribution
(training/baseline) against a `current` one (a recent production window):

    PSI  — Population Stability Index. Bins both distributions (using the
           reference distribution's own bin edges) and sums a symmetric,
           weighted log-ratio of bin proportions. Standard industry
           thresholds (unchanged since Siddiqi, 2006, credit-scoring
           practice): < 0.10 stable, 0.10-0.25 moderate shift, > 0.25
           significant shift requiring investigation/retraining.
    KS   — Kolmogorov-Smirnov statistic. The maximum absolute gap between
           the two empirical CDFs — more sensitive than PSI to a shift
           concentrated in one part of the distribution rather than spread
           across all bins, and doesn't depend on a binning choice.

Both apply to any numeric series — engineered features or the model's own
predicted probabilities (score drift, which catches distribution shift even
when no single input feature has moved much on its own).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PSI_STABLE_MAX = 0.10
PSI_MODERATE_MAX = 0.25

_EPS = 1e-6  # avoids log(0)/div-by-0 when a bin is empty in one distribution


def psi(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    """PSI for a numeric feature. Bin edges come from the reference
    distribution's quantiles, so bins are equal-population in `reference` by
    construction — `current` proportions in the same bins are what reveal
    the shift."""
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)

    quantiles = np.linspace(0, 1, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    edges[0], edges[-1] = -np.inf, np.inf

    ref_counts, _ = np.histogram(reference, bins=edges)
    cur_counts, _ = np.histogram(current, bins=edges)

    ref_pct = ref_counts / ref_counts.sum() + _EPS
    cur_pct = cur_counts / cur_counts.sum() + _EPS
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    """PSI for a categorical feature, using observed category proportions
    directly instead of quantile bins. Categories present in only one of the
    two series are treated as having ~0 (epsilon) share in the other."""
    ref_pct = reference.value_counts(normalize=True)
    cur_pct = current.value_counts(normalize=True)
    categories = ref_pct.index.union(cur_pct.index)

    ref_aligned = ref_pct.reindex(categories, fill_value=0.0) + _EPS
    cur_aligned = cur_pct.reindex(categories, fill_value=0.0) + _EPS
    return float(np.sum((cur_aligned - ref_aligned) * np.log(cur_aligned / ref_aligned)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample KS statistic: max gap between the empirical CDFs of
    `reference` and `current`, evaluated at every point either sample takes
    (the only points where the step-function CDFs can actually be maximally
    apart)."""
    reference = np.sort(np.asarray(reference, dtype=float))
    current = np.sort(np.asarray(current, dtype=float))
    all_values = np.concatenate([reference, current])

    cdf_ref = np.searchsorted(reference, all_values, side="right") / len(reference)
    cdf_cur = np.searchsorted(current, all_values, side="right") / len(current)
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def classify_drift(psi_value: float) -> str:
    """Standard PSI alarm bands (Siddiqi, 2006)."""
    if psi_value < PSI_STABLE_MAX:
        return "stable"
    if psi_value < PSI_MODERATE_MAX:
        return "moderate"
    return "significant"


def drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    psi_bins: int = 10,
) -> pd.DataFrame:
    """Per-feature PSI (+ KS for numeric features) between `reference` and
    `current`, sorted by PSI descending — the features most likely to be
    driving any model degradation surface at the top."""
    rows = []
    for col in num_cols:
        psi_value = psi(reference[col].values, current[col].values, bins=psi_bins)
        ks_value = ks_statistic(reference[col].values, current[col].values)
        rows.append({"feature": col, "type": "numeric", "psi": psi_value,
                     "ks": ks_value, "drift": classify_drift(psi_value)})
    for col in cat_cols:
        psi_value = psi_categorical(reference[col], current[col])
        rows.append({"feature": col, "type": "categorical", "psi": psi_value,
                     "ks": np.nan, "drift": classify_drift(psi_value)})

    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )
