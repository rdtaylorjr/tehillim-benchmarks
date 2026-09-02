from pathlib import Path

from genre.scripts.compute_bootstrap_cis import score_model

_GENRE_BY_PSALM = {1: "A", 2: "A", 3: "A", 4: "B", 5: "B", 6: "B"}
_HALF_VERSES_BY_PSALM = {psalm: [psalm] for psalm in _GENRE_BY_PSALM}
_SEPARABLE = {
    1: [1.0, 0.0],
    2: [0.95, 0.05],
    3: [0.9, 0.1],
    4: [0.0, 1.0],
    5: [0.05, 0.95],
    6: [0.1, 0.9],
}


def test_score_model_names_the_row_after_the_files_dataset_identifier(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    path = write_embeddings_parquet(tmp_path / "domain=d" / "model=mine" / "v.parquet", _SEPARABLE)

    row = score_model(path, _HALF_VERSES_BY_PSALM, _GENRE_BY_PSALM, n_resamples=50, seed=0)

    assert row is not None
    assert row["model"] == "mine"
    assert row["ap_ci_low"] <= row["point_ap"] <= row["ap_ci_high"]


def test_score_model_returns_none_when_the_population_cannot_support_a_ci(
    tmp_path: Path, write_embeddings_parquet
) -> None:
    """Two psalms of different genres leave zero same-genre pairs, so AP and AUC are undefined."""
    path = write_embeddings_parquet(
        tmp_path / "domain=d" / "model=tiny" / "v.parquet", {1: [1.0, 0.0], 2: [0.0, 1.0]}
    )

    result = score_model(path, {1: [1], 2: [2]}, {1: "A", 2: "B"}, n_resamples=20, seed=0)

    assert result is None
