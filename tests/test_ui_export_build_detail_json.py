import numpy as np
import pandas as pd

from ui_export.scripts.build_detail_json import (
    attach_genre_columns,
    choose_primary_metric,
    residualize_trajectory_metric,
    table_model_sets,
    target_models,
)


def _domain_json() -> dict:
    return {
        "parallelism_overall": [{"model": "a"}, {"model": "b"}],
        "genre_overall": [{"model": "b"}, {"model": "b_psalm"}],
        "trajectory": [{"model": "a"}, {"model": "c"}],
    }


def test_target_models_unions_every_table_a_row_could_link_from() -> None:
    assert target_models(_domain_json()) == {"a", "b", "b_psalm", "c"}


def test_table_model_sets_keeps_each_table_s_own_model_set_separate() -> None:
    """A psalm-level model excluded from parallelism_overall must not gain a parallelism section."""
    sets = table_model_sets(_domain_json())
    assert sets["parallelism"] == {"a", "b"}
    assert sets["genre"] == {"b", "b_psalm"}
    assert sets["trajectory"] == {"a", "c"}
    assert "b_psalm" not in sets["parallelism"]


def test_choose_primary_metric_picks_the_smallest_length_controlled_p() -> None:
    df = pd.DataFrame(
        [
            {"model": "m1", "metric": "content_distance", "length_controlled_p": 0.2},
            {"model": "m1", "metric": "structural_distance", "length_controlled_p": 0.01},
            {"model": "other", "metric": "content_distance", "length_controlled_p": 0.001},
        ]
    )
    assert choose_primary_metric(df, "m1") == "structural_distance"


def test_choose_primary_metric_returns_none_when_the_model_is_absent() -> None:
    df = pd.DataFrame(
        [{"model": "other", "metric": "content_distance", "length_controlled_p": 0.001}]
    )
    assert choose_primary_metric(df, "m1") is None


def test_choose_primary_metric_skips_metrics_with_a_nan_p_value() -> None:
    df = pd.DataFrame(
        [
            {"model": "m1", "metric": "content_distance", "length_controlled_p": float("nan")},
            {"model": "m1", "metric": "structural_distance", "length_controlled_p": 0.02},
        ]
    )
    assert choose_primary_metric(df, "m1") == "structural_distance"


def test_choose_primary_metric_returns_none_when_every_row_is_nan() -> None:
    df = pd.DataFrame(
        [{"model": "m1", "metric": "content_distance", "length_controlled_p": float("nan")}]
    )
    assert choose_primary_metric(df, "m1") is None


def test_choose_primary_metric_never_picks_content_distance() -> None:
    """content_distance's own content-controlled source is self-referential, always NaN."""
    df = pd.DataFrame(
        [
            {"model": "m1", "metric": "content_distance", "length_controlled_p": 0.0001},
            {"model": "m1", "metric": "structural_distance", "length_controlled_p": 0.2},
        ]
    )
    assert choose_primary_metric(df, "m1") == "structural_distance"


def test_choose_primary_metric_returns_none_when_only_content_distance_is_available() -> None:
    df = pd.DataFrame(
        [{"model": "m1", "metric": "content_distance", "length_controlled_p": 0.0001}]
    )
    assert choose_primary_metric(df, "m1") is None


def test_attach_genre_columns_derives_same_genre_from_the_two_labels() -> None:
    df = pd.DataFrame([{"psalm_a": 1, "psalm_b": 2}, {"psalm_a": 1, "psalm_b": 3}])
    genre_by_psalm = {1: "Hymn", 2: "Hymn", 3: "Lament"}
    out = attach_genre_columns(df, genre_by_psalm)
    assert out["genre_a"].tolist() == ["Hymn", "Hymn"]
    assert out["genre_b"].tolist() == ["Hymn", "Lament"]
    assert out["same_genre"].tolist() == [True, False]


def test_residualize_trajectory_metric_adds_both_controlled_columns() -> None:
    df = pd.DataFrame(
        [
            {"psalm_a": 1, "psalm_b": 2, "structural_distance": 0.5, "content_distance": 0.2},
            {"psalm_a": 1, "psalm_b": 3, "structural_distance": 0.9, "content_distance": 0.4},
            {"psalm_a": 2, "psalm_b": 3, "structural_distance": 0.3, "content_distance": 0.1},
        ]
    )
    n_cola = {1: 10, 2: 12, 3: 20}
    out = residualize_trajectory_metric(df, "structural_distance", n_cola)
    assert "length_controlled" in out.columns
    assert "length_and_content_controlled" in out.columns
    assert np.isfinite(out["length_controlled"]).all()
    assert np.isfinite(out["length_and_content_controlled"]).all()


def test_residualize_trajectory_metric_drops_rows_missing_the_metric_or_content_value() -> None:
    """A single NaN in the OLS response corrupts every row's fit, not just the missing one."""
    df = pd.DataFrame(
        [
            {"psalm_a": 1, "psalm_b": 2, "structural_distance": 0.5, "content_distance": 0.2},
            {
                "psalm_a": 1,
                "psalm_b": 3,
                "structural_distance": float("nan"),
                "content_distance": 0.4,
            },
            {
                "psalm_a": 2,
                "psalm_b": 3,
                "structural_distance": 0.3,
                "content_distance": float("nan"),
            },
        ]
    )
    n_cola = {1: 10, 2: 12, 3: 20}
    out = residualize_trajectory_metric(df, "structural_distance", n_cola)
    assert len(out) == 1
    assert np.isfinite(out["length_controlled"]).all()
    assert np.isfinite(out["length_and_content_controlled"]).all()
