import numpy as np
import pandas as pd
import pytest

from library.multiple_comparisons import (
    add_fdr_q_values,
    benjamini_hochberg,
    benjamini_yekutieli,
)


def test_benjamini_hochberg_matches_hand_worked_example() -> None:
    """Classic textbook example: 5 p-values, alpha=0.05."""
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])

    q = benjamini_hochberg(p)

    # sorted p 0.005, 0.01, 0.03, 0.04, 0.5 at ranks 1..5, n=5; q = p*n/rank is already monotone.
    expected_sorted_order_q = np.array([0.025, 0.025, 0.05, 0.05, 0.5])
    order = np.argsort(p)
    assert q[order] == pytest.approx(expected_sorted_order_q)


def test_benjamini_hochberg_never_exceeds_one() -> None:
    p = np.array([0.9, 0.95, 0.99])
    q = benjamini_hochberg(p)
    assert np.all(q <= 1.0)


def test_benjamini_hochberg_is_monotone_non_decreasing_in_sorted_p_order() -> None:
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, size=50)
    q = benjamini_hochberg(p)
    order = np.argsort(p)
    assert np.all(np.diff(q[order]) >= -1e-12)


def test_benjamini_hochberg_all_significant_p_values_stay_small() -> None:
    p = np.full(10, 0.001)
    q = benjamini_hochberg(p)
    assert np.all(q == pytest.approx(0.001))


def test_benjamini_yekutieli_matches_hand_worked_example() -> None:
    """Same 5 p-values as the BH example; BY multiplies every raw q by the harmonic number c(n)."""
    p = np.array([0.01, 0.04, 0.03, 0.005, 0.5])
    c_5 = sum(1.0 / k for k in range(1, 6))

    q = benjamini_yekutieli(p)

    # raw q = p*n*c_n/rank: 0.025*c5, 0.025*c5, 0.05*c5, 0.05*c5, 0.5*c5, clipped to 1
    expected_sorted_order_q = np.minimum(np.array([0.025, 0.025, 0.05, 0.05, 0.5]) * c_5, 1.0)
    order = np.argsort(p)
    assert q[order] == pytest.approx(expected_sorted_order_q)


def test_benjamini_yekutieli_is_more_conservative_than_benjamini_hochberg() -> None:
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 0.2, size=20)

    q_bh = benjamini_hochberg(p)
    q_by = benjamini_yekutieli(p)

    assert np.all(q_by >= q_bh - 1e-12)


def test_add_fdr_q_values_corrects_within_each_source_metric_scope_family_separately() -> None:
    long_df = pd.DataFrame(
        [
            {
                "source": "vs_baseline",
                "metric": "p_vs_baseline",
                "scope_kind": "overall",
                "value": 0.01,
            },
            {
                "source": "vs_baseline",
                "metric": "p_vs_baseline",
                "scope_kind": "overall",
                "value": 0.04,
            },
            {
                "source": "vs_baseline",
                "metric": "p_vs_baseline",
                "scope_kind": "type",
                "value": 0.01,
            },
            {
                "source": "retrieval_separation",
                "metric": "separation_p",
                "scope_kind": "overall",
                "value": 0.01,
            },
            {
                "source": "retrieval_separation",
                "metric": "separation_auc",
                "scope_kind": "overall",
                "value": 0.9,
            },
        ]
    )

    result = add_fdr_q_values(long_df)

    # non-p-value rows (separation_auc) get no q-value
    auc_row = result[result["metric"] == "separation_auc"].iloc[0]
    assert pd.isna(auc_row["q_value"])
    assert pd.isna(auc_row["q_value_by"])

    # "overall" and "type" are separate families, so the lone "type" row's q equals its own p.
    type_row = result[
        (result["source"] == "vs_baseline")
        & (result["metric"] == "p_vs_baseline")
        & (result["scope_kind"] == "type")
    ].iloc[0]
    assert type_row["q_value"] == pytest.approx(0.01)

    overall_rows = result[
        (result["source"] == "vs_baseline")
        & (result["metric"] == "p_vs_baseline")
        & (result["scope_kind"] == "overall")
    ]
    assert len(overall_rows) == 2
    assert overall_rows["q_value"].notna().all()


def test_add_fdr_q_values_only_targets_p_value_metrics() -> None:
    long_df = pd.DataFrame(
        [
            {"source": "s", "metric": "discrimination_p", "scope_kind": "overall", "value": 0.02},
            {"source": "s", "metric": "discrimination_p", "scope_kind": "overall", "value": 0.9},
            {"source": "s", "metric": "mrr_forward", "scope_kind": "overall", "value": 0.7},
        ]
    )

    result = add_fdr_q_values(long_df)

    assert result[result["metric"] == "discrimination_p"]["q_value"].notna().all()
    assert pd.isna(result[result["metric"] == "mrr_forward"].iloc[0]["q_value"])


def test_add_fdr_q_values_still_corrects_a_family_whose_scope_kind_is_missing() -> None:
    """A NaN grouping key used to drop its whole family silently, leaving q-values unset."""
    long_df = pd.DataFrame(
        {
            "source": ["a", "a", "a"],
            "metric": ["separation_p"] * 3,
            "scope_kind": [np.nan, np.nan, np.nan],
            "value": [0.01, 0.02, 0.03],
        }
    )

    result = add_fdr_q_values(long_df)

    assert result["q_value"].notna().all()
    assert result["q_value_by"].notna().all()


def test_add_fdr_q_values_is_unaffected_by_a_duplicated_input_index() -> None:
    """Concatenated frames can repeat index labels, which used to misalign the q-value writes."""
    rows = {
        "source": ["a", "a"],
        "metric": ["separation_p", "separation_p"],
        "scope_kind": ["s", "s"],
        "value": [0.01, 0.9],
    }
    unique_index = pd.DataFrame(rows)
    duplicated_index = pd.DataFrame(rows, index=[0, 0])

    expected = add_fdr_q_values(unique_index)["q_value"].tolist()

    assert add_fdr_q_values(duplicated_index)["q_value"].tolist() == expected
