import math

import pandas as pd

from trajectory.scripts.export_ui_rows import (
    trajectory_by_genre_ui_rows,
    trajectory_ui_rows,
)


def _breakdown_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "metric": "content_distance",
                "source": "raw",
                "genre": "Wisdom",
                "gap": 0.12,
                "p_perm": 0.01,
                "p_maxT": 0.05,
                "perm_q": 0.02,
                "perm_q_by": 0.04,
                "maxT_q": 0.08,
                "maxT_q_by": 0.12,
            },
            {
                "model": "miqrabert_consonantal",
                "metric": "structural_distance",
                "source": "length_controlled",
                "genre": "Lament",
                "gap": 0.03,
                "p_perm": 0.4,
                "p_maxT": 0.6,
                "perm_q": 0.45,
                "perm_q_by": 0.5,
                "maxT_q": 0.65,
                "maxT_q_by": 0.7,
            },
        ]
    )


def _validation_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "metric": "content_distance",
                "raw_gap": 0.013,
                "raw_p": 0.0001,
                "raw_effect_size": 5.2,
                "raw_q": 0.0002,
                "length_controlled_gap": 0.014,
                "length_controlled_p": 0.0001,
                "length_controlled_effect_size": 5.3,
                "length_controlled_q": 0.0002,
                "length_and_content_controlled_gap": float("nan"),
                "length_and_content_controlled_p": float("nan"),
                "length_and_content_controlled_effect_size": float("nan"),
                "length_and_content_controlled_q": float("nan"),
            },
            {
                "model": "miqrabert_consonantal",
                "metric": "structural_distance",
                "raw_gap": 0.02,
                "raw_p": 0.05,
                "raw_effect_size": 1.5,
                "raw_q": 0.06,
                "length_controlled_gap": 0.01,
                "length_controlled_p": 0.07,
                "length_controlled_effect_size": 1.4,
                "length_controlled_q": 0.07,
                "length_and_content_controlled_gap": 0.004,
                "length_and_content_controlled_p": 0.36,
                "length_and_content_controlled_effect_size": 0.33,
                "length_and_content_controlled_q": 0.52,
            },
        ]
    )


def test_trajectory_ui_rows_splits_model_into_base_and_text_variant() -> None:
    rows = trajectory_ui_rows(_validation_df())

    assert rows[0]["model"] == "bge_m3_vocalized"
    assert rows[0]["model_base"] == "bge_m3"
    assert rows[0]["text_variant"] == "vocalized"


def test_trajectory_ui_rows_splits_a_consonantal_only_model() -> None:
    rows = trajectory_ui_rows(_validation_df())

    assert rows[1]["model_base"] == "miqrabert"
    assert rows[1]["text_variant"] == "consonantal"


def test_trajectory_ui_rows_carries_metric_and_every_source_field() -> None:
    rows = trajectory_ui_rows(_validation_df())

    row = rows[1]
    assert row["metric"] == "structural_distance"
    assert row["raw_gap"] == 0.02
    assert row["raw_p"] == 0.05
    assert row["raw_effect_size"] == 1.5
    assert row["raw_q"] == 0.06
    assert row["length_controlled_gap"] == 0.01
    assert row["length_controlled_p"] == 0.07
    assert row["length_controlled_effect_size"] == 1.4
    assert row["length_controlled_q"] == 0.07
    assert row["length_and_content_controlled_gap"] == 0.004
    assert row["length_and_content_controlled_p"] == 0.36
    assert row["length_and_content_controlled_effect_size"] == 0.33
    assert row["length_and_content_controlled_q"] == 0.52


def test_trajectory_ui_rows_preserves_nan_for_the_self_covariate_case() -> None:
    rows = trajectory_ui_rows(_validation_df())

    row = rows[0]
    assert math.isnan(row["length_and_content_controlled_gap"])
    assert math.isnan(row["length_and_content_controlled_p"])
    assert math.isnan(row["length_and_content_controlled_q"])


def test_trajectory_ui_rows_returns_one_row_per_input_row() -> None:
    assert len(trajectory_ui_rows(_validation_df())) == 2


def test_trajectory_by_genre_ui_rows_splits_model_into_base_and_text_variant() -> None:
    rows = trajectory_by_genre_ui_rows(_breakdown_df())

    assert rows[0]["model"] == "bge_m3_vocalized"
    assert rows[0]["model_base"] == "bge_m3"
    assert rows[0]["text_variant"] == "vocalized"


def test_trajectory_by_genre_ui_rows_carries_every_field() -> None:
    rows = trajectory_by_genre_ui_rows(_breakdown_df())

    row = rows[1]
    assert row["metric"] == "structural_distance"
    assert row["source"] == "length_controlled"
    assert row["genre"] == "Lament"
    assert row["gap"] == 0.03
    assert row["p_perm"] == 0.4
    assert row["p_maxT"] == 0.6
    assert row["perm_q"] == 0.45
    assert row["perm_q_by"] == 0.5
    assert row["maxT_q"] == 0.65
    assert row["maxT_q_by"] == 0.7


def test_trajectory_by_genre_ui_rows_returns_one_row_per_input_row() -> None:
    assert len(trajectory_by_genre_ui_rows(_breakdown_df())) == 2
