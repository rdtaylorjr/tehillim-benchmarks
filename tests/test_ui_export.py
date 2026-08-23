import pandas as pd

from ui_export.export import build_family_data


def _parallelism_overall_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "model_base": "bge_m3",
                "text_variant": "vocalized",
                "separation_auc": 0.66,
                "average_precision": 0.33,
                "calibrated_effect_size": 0.44,
                "mrr_forward": 0.06,
                "n_true": 1110.0,
                "unused_column": "should be dropped",
            },
            {
                "model": "form_icf_posmean_psalm",
                "model_base": "form_icf_posmean_psalm",
                "text_variant": "unknown",
                "separation_auc": 0.99,
                "average_precision": 0.98,
                "calibrated_effect_size": 5.0,
                "mrr_forward": 0.9,
                "n_true": 1110.0,
                "unused_column": "should be dropped",
            },
        ]
    )


def _parallelism_by_type_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "model_base": "bge_m3",
                "text_variant": "vocalized",
                "scope": "Synonymous",
                "separation_auc": 0.7,
                "average_precision": 0.35,
                "calibrated_effect_size": 0.5,
            },
            {
                "model": "form_icf_posmean_psalm",
                "model_base": "form_icf_posmean_psalm",
                "text_variant": "unknown",
                "scope": "Synonymous",
                "separation_auc": 0.99,
                "average_precision": 0.98,
                "calibrated_effect_size": 5.0,
            },
        ]
    )


def _genre_overall_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "model_base": "bge_m3",
                "text_variant": "vocalized",
                "separation_auc": 0.6,
                "average_precision": 0.3,
                "same_genre_effect_size": 0.4,
                "n_same_genre": 500,
            }
        ]
    )


def _genre_by_genre_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "bge_m3_vocalized",
                "genre": "Wisdom",
                "separation_auc": 0.65,
                "average_precision": 0.32,
                "separation_p_perm": 0.01,
                "separation_p_maxT": 0.05,
                "perm_q": 0.02,
                "maxT_q": 0.08,
            }
        ]
    )


def _trajectory_rows() -> list[dict]:
    return [{"model": "bge_m3_vocalized", "metric": "content_distance", "raw_p": 0.001}]


def test_build_family_data_selects_only_the_uis_parallelism_overall_columns() -> None:
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["parallelism_overall"][0]
    assert set(row) == {
        "model",
        "model_base",
        "text_variant",
        "separation_auc",
        "average_precision",
        "calibrated_effect_size",
        "mrr_forward",
        "n_true",
    }
    assert "unused_column" not in row


def test_build_family_data_keeps_scope_in_parallelism_by_type() -> None:
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["parallelism_by_type"][0]
    assert row["scope"] == "Synonymous"


def test_build_family_data_drops_psalm_level_models_from_parallelism_overall() -> None:
    """Psalm-broadcast representations are architecturally degenerate for a colon-pair task."""
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    models = {row["model"] for row in data["parallelism_overall"]}
    assert models == {"bge_m3_vocalized"}


def test_build_family_data_drops_psalm_level_shuffle_control_models_from_parallelism_overall() -> (
    None
):
    """A _psalm_shuffleNN model is just as degenerate as its unshuffled _psalm base."""
    parallelism_overall = _parallelism_overall_df()
    parallelism_overall.loc[len(parallelism_overall)] = {
        "model": "morph_signature_1_2gram_psalm_shuffle03",
        "model_base": "morph_signature_1_2gram_psalm_shuffle03",
        "text_variant": "unknown",
        "separation_auc": 0.995,
        "average_precision": 0.99,
        "calibrated_effect_size": 5.0,
        "mrr_forward": 0.9,
        "n_true": 1110.0,
        "unused_column": "should be dropped",
    }
    data = build_family_data(
        parallelism_overall,
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    models = {row["model"] for row in data["parallelism_overall"]}
    assert "morph_signature_1_2gram_psalm_shuffle03" not in models


def test_build_family_data_drops_psalm_level_models_from_parallelism_by_type() -> None:
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    models = {row["model"] for row in data["parallelism_by_type"]}
    assert models == {"bge_m3_vocalized"}


def test_build_family_data_keeps_psalm_level_models_in_genre_tables() -> None:
    """Psalm-broadcast representations are exactly the right granularity for a psalm-level task."""
    genre_overall = _genre_overall_df()
    genre_overall.loc[len(genre_overall)] = {
        "model": "form_icf_posmean_psalm",
        "model_base": "form_icf_posmean_psalm",
        "text_variant": "unknown",
        "separation_auc": 0.7,
        "average_precision": 0.4,
        "same_genre_effect_size": 0.5,
        "n_same_genre": 500,
    }
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        genre_overall,
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    models = {row["model"] for row in data["genre_overall"]}
    assert "form_icf_posmean_psalm" in models


def test_build_family_data_selects_genre_overall_columns() -> None:
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["genre_overall"][0]
    assert set(row) == {
        "model",
        "model_base",
        "text_variant",
        "separation_auc",
        "average_precision",
        "same_genre_effect_size",
        "n_same_genre",
    }


def test_build_family_data_selects_genre_by_genre_columns() -> None:
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    row = data["genre_by_genre"][0]
    assert set(row) == {
        "model",
        "model_base",
        "text_variant",
        "genre",
        "separation_auc",
        "average_precision",
        "perm_q",
        "maxT_q",
    }


def test_build_family_data_derives_model_base_and_text_variant_for_genre_by_genre() -> None:
    """genre_by_genre.csv has no model_base/text_variant columns, so the Text filter needs them
    derived from `model`, or selecting a text tier silently drops every by-genre row (the bug)."""
    genre_by_genre = _genre_by_genre_df()
    genre_by_genre.loc[0, "model"] = "word_consonantal_binary"
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        genre_by_genre,
        _trajectory_rows(),
    )

    row = data["genre_by_genre"][0]
    assert row["model_base"] == "word_binary"
    assert row["text_variant"] == "consonantal"


def test_build_family_data_passes_trajectory_rows_through_unchanged() -> None:
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    assert data["trajectory"] == _trajectory_rows()


def test_build_family_data_has_exactly_the_six_ui_keys() -> None:
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    assert set(data) == {
        "parallelism_overall",
        "parallelism_by_type",
        "genre_overall",
        "genre_by_genre",
        "trajectory",
        "trajectory_by_genre",
    }


def test_build_family_data_defaults_trajectory_by_genre_to_an_empty_list() -> None:
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
    )

    assert data["trajectory_by_genre"] == []


def test_build_family_data_passes_trajectory_by_genre_rows_through_unchanged() -> None:
    by_genre_rows = [{"model": "bge_m3_vocalized", "metric": "content_distance", "genre": "Wisdom"}]
    data = build_family_data(
        _parallelism_overall_df(),
        _parallelism_by_type_df(),
        _genre_overall_df(),
        _genre_by_genre_df(),
        _trajectory_rows(),
        by_genre_rows,
    )

    assert data["trajectory_by_genre"] == by_genre_rows


def test_build_family_data_drops_shuffle_control_models_from_every_table() -> None:
    """A _shuffleNN model is a null-order control checked against its real base, not rankable."""
    shuffle_row = {
        "model": "phrase_signature_1_2gram_shuffle03",
        "model_base": "phrase_signature_1_2gram_shuffle03",
        "text_variant": "unknown",
        "separation_auc": 0.9,
        "average_precision": 0.9,
        "calibrated_effect_size": 5.0,
        "mrr_forward": 0.9,
        "n_true": 1110.0,
        "same_genre_effect_size": 0.9,
        "n_same_genre": 500,
        "scope": "Synonymous",
    }
    parallelism_overall = _parallelism_overall_df()
    parallelism_overall.loc[len(parallelism_overall)] = shuffle_row
    parallelism_by_type = _parallelism_by_type_df()
    parallelism_by_type.loc[len(parallelism_by_type)] = shuffle_row
    genre_overall = _genre_overall_df()
    genre_overall.loc[len(genre_overall)] = shuffle_row
    genre_by_genre = _genre_by_genre_df()
    genre_by_genre.loc[len(genre_by_genre)] = {
        "model": "phrase_signature_1_2gram_shuffle03",
        "genre": "Wisdom",
        "separation_auc": 0.9,
        "average_precision": 0.9,
        "separation_p_perm": 0.01,
        "separation_p_maxT": 0.05,
        "perm_q": 0.02,
        "maxT_q": 0.08,
    }
    trajectory_rows = _trajectory_rows() + [
        {"model": "phrase_signature_1_2gram_shuffle03", "metric": "content_distance", "raw_p": 0.9}
    ]
    trajectory_by_genre_rows = [
        {
            "model": "phrase_signature_1_2gram_shuffle03",
            "metric": "content_distance",
            "genre": "Wisdom",
        }
    ]

    data = build_family_data(
        parallelism_overall,
        parallelism_by_type,
        genre_overall,
        genre_by_genre,
        trajectory_rows,
        trajectory_by_genre_rows,
    )

    for table in (
        "parallelism_overall",
        "parallelism_by_type",
        "genre_overall",
        "genre_by_genre",
        "trajectory",
        "trajectory_by_genre",
    ):
        models = {row["model"] for row in data[table]}
        assert "phrase_signature_1_2gram_shuffle03" not in models, table
