import numpy as np
import pytest

from genre.evaluate import evaluate_genre_discrimination
from genre.pairs import GenrePair


def test_reports_the_correct_same_and_different_genre_counts() -> None:
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

    report = evaluate_genre_discrimination(pairs, psalm_vectors)

    assert report.n_same_genre == 1
    assert report.n_different_genre == 2
    assert report.prevalence == pytest.approx(1 / 3)


def test_perfect_separation_gives_ap_and_auc_of_one() -> None:
    """Same-genre psalms point the same direction, different-genre psalms are orthogonal."""
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(3, 4, "Praise", "Praise", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(1, 4, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 4, "Lament", "Praise", same_genre=False),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.0, 1.0]),
    }

    report = evaluate_genre_discrimination(pairs, psalm_vectors)

    assert report.average_precision == pytest.approx(1.0)
    assert report.separation_auc == pytest.approx(1.0)


def test_random_labels_give_chance_level_auc() -> None:
    """Same-genre and different-genre pairs drawn from identical similarity distributions."""
    rng = np.random.default_rng(0)
    psalm_vectors = {i: rng.normal(size=8) for i in range(1, 41)}
    genres = ["Lament", "Praise"] * 20
    pairs = []
    for i in range(1, 41):
        for j in range(i + 1, 41):
            genre_a, genre_b = genres[i - 1], genres[j - 1]
            pairs.append(GenrePair(i, j, genre_a, genre_b, same_genre=genre_a == genre_b))

    report = evaluate_genre_discrimination(pairs, psalm_vectors)

    assert 0.35 < report.separation_auc < 0.65


def test_skips_a_pair_whose_psalm_has_no_vector() -> None:
    pairs = [
        GenrePair(1, 2, "Lament", "Lament", same_genre=True),
        GenrePair(1, 3, "Lament", "Praise", same_genre=False),
        GenrePair(1, 4, "Lament", "Praise", same_genre=False),
        GenrePair(2, 3, "Lament", "Praise", same_genre=False),
        GenrePair(2, 4, "Lament", "Praise", same_genre=False),
        GenrePair(3, 4, "Praise", "Praise", same_genre=True),
    ]
    psalm_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([0.0, 1.0]),
        # psalm 4 has no vector: every pair touching it must be excluded.
    }

    report = evaluate_genre_discrimination(pairs, psalm_vectors)

    assert report.n_same_genre == 1
    assert report.n_different_genre == 2
