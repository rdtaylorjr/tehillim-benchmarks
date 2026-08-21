import numpy as np
import pytest

from library.bootstrap_ci import bca_ci


def test_bca_ci_reduces_to_percentile_when_unbiased_and_unaccelerated() -> None:
    """z0=0 exactly (point sits at the median) and a=0 (constant jackknife) should reproduce
    the plain percentile interval exactly."""
    bootstrap_values = np.concatenate(
        [-np.arange(1, 5001, dtype=float), np.arange(1, 5001, dtype=float)]
    )
    jackknife_values = np.full(60, 7.0)

    lo, hi = bca_ci(0.0, bootstrap_values, jackknife_values)
    perc_lo, perc_hi = np.percentile(bootstrap_values, [2.5, 97.5])

    assert lo == pytest.approx(perc_lo)
    assert hi == pytest.approx(perc_hi)


def test_bca_ci_shifts_when_point_is_biased_relative_to_bootstrap_distribution() -> None:
    bootstrap_values = np.concatenate(
        [-np.arange(1, 5001, dtype=float), np.arange(1, 5001, dtype=float)]
    )
    jackknife_values = np.full(60, 7.0)

    lo, hi = bca_ci(4999.0, bootstrap_values, jackknife_values)
    perc_lo, perc_hi = np.percentile(bootstrap_values, [2.5, 97.5])

    assert (lo, hi) != pytest.approx((perc_lo, perc_hi))


def test_bca_ci_falls_back_to_the_point_for_a_zero_variance_bootstrap_distribution() -> None:
    bootstrap_values = np.full(50, 0.42)
    jackknife_values = np.full(10, 0.1)

    lo, hi = bca_ci(0.42, bootstrap_values, jackknife_values)

    assert lo == pytest.approx(0.42)
    assert hi == pytest.approx(0.42)
