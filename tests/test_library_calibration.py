import numpy as np
import pytest

from library.calibration import (
    BackgroundStats,
    background_similarity_stats,
    background_stats_from_matrix,
    calibrated_effect_size,
    calibrated_z_score,
)


def test_background_similarity_stats_excludes_self_comparisons() -> None:
    vectors = np.array([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])

    stats = background_similarity_stats(vectors)

    assert stats.mean == 1.0
    assert stats.n_vectors == 3


def test_background_similarity_stats_reflects_spread() -> None:
    vectors = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0]])

    stats = background_similarity_stats(vectors)

    assert stats.mean == pytest.approx(-1 / 3)
    assert stats.std > 0


def test_background_stats_from_matrix_matches_background_similarity_stats() -> None:
    """Proves the matrix-based shortcut gives the identical statistic as the vector-based path."""
    rng = np.random.default_rng(2)
    vectors = rng.normal(size=(6, 5))
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity_matrix = norm @ norm.T

    from_vectors = background_similarity_stats(vectors)
    from_matrix = background_stats_from_matrix(similarity_matrix)

    assert from_matrix == from_vectors


def test_calibrated_z_score_is_zero_when_true_pairs_match_the_background() -> None:
    background = BackgroundStats(mean=0.5, std=0.1, n_vectors=100)

    assert calibrated_z_score(0.5, background) == 0.0


def test_calibrated_z_score_is_positive_when_true_pairs_beat_the_background() -> None:
    background = BackgroundStats(mean=0.5, std=0.1, n_vectors=100)

    assert calibrated_z_score(0.8, background) == pytest.approx(3.0)


def test_calibrated_z_score_raises_on_zero_variance_background() -> None:
    background = BackgroundStats(mean=1.0, std=0.0, n_vectors=100)

    with pytest.raises(ValueError, match="zero variance"):
        calibrated_z_score(1.0, background)


def test_calibrated_effect_size_matches_the_z_score_formula() -> None:
    background = BackgroundStats(mean=0.5, std=0.1, n_vectors=100)

    assert calibrated_effect_size(0.8, background) == pytest.approx(3.0)
    assert calibrated_effect_size(0.8, background) == calibrated_z_score(0.8, background)
