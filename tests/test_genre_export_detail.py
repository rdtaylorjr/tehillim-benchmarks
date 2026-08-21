import numpy as np
import pytest

from genre.pairs import GenrePair
from genre.scripts.export_detail import build_pair_detail_rows, build_summary_rows
from library.calibration import BackgroundStats


def _pairs_and_vectors() -> tuple[list[GenrePair], dict[int, np.ndarray]]:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([0.0, 1.0]),
    }
    return pairs, psalm_vectors


def test_build_pair_detail_rows_has_one_row_per_usable_pair() -> None:
    pairs, psalm_vectors = _pairs_and_vectors()
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=3)

    rows = build_pair_detail_rows("model_a", pairs, psalm_vectors, background)

    assert len(rows) == 3
    first = next(r for r in rows if r["psalm_a"] == 1 and r["psalm_b"] == 2)
    assert first["model"] == "model_a"
    assert first["same_genre"] is True
    assert first["raw_similarity"] == pytest.approx(0.99388373, abs=1e-6)
    assert "genre_a" not in first
    assert "genre_b" not in first


def test_build_pair_detail_rows_skips_a_pair_with_a_missing_vector() -> None:
    pairs, psalm_vectors = _pairs_and_vectors()
    del psalm_vectors[3]
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=2)

    rows = build_pair_detail_rows("model_a", pairs, psalm_vectors, background)

    assert len(rows) == 1
    assert rows[0]["psalm_a"] == 1
    assert rows[0]["psalm_b"] == 2


def test_build_summary_rows_has_one_row_summarizing_the_model() -> None:
    pairs, psalm_vectors = _pairs_and_vectors()
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=3)

    rows = build_summary_rows("model_a", pairs, psalm_vectors, background)

    assert len(rows) == 1
    assert rows[0]["model"] == "model_a"
    assert rows[0]["n_same_genre"] == 1
    assert rows[0]["n_different_genre"] == 2
    assert rows[0]["gap"] == pytest.approx(
        rows[0]["same_genre_effect_size"] - rows[0]["different_genre_effect_size"]
    )
