import numpy as np

from genre.pairs import GenrePair
from genre.scripts.compare_models import compare_genre_models


def test_reports_one_row_per_model() -> None:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors_by_model = {
        "model_a": {
            1: np.array([1.0, 0.0]),
            2: np.array([0.9, 0.1]),
            3: np.array([0.0, 1.0]),
        },
        "model_b": {
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
            3: np.array([1.0, 0.0]),
        },
    }

    rows = compare_genre_models(pairs, psalm_vectors_by_model)

    assert {row["model"] for row in rows} == {"model_a", "model_b"}


def test_sorts_rows_by_average_precision_descending() -> None:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors_by_model = {
        "worse": {
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
            3: np.array([1.0, 0.0]),
        },
        "better": {
            1: np.array([1.0, 0.0]),
            2: np.array([0.9, 0.1]),
            3: np.array([0.0, 1.0]),
        },
    }

    rows = compare_genre_models(pairs, psalm_vectors_by_model)

    assert [row["model"] for row in rows] == ["better", "worse"]
