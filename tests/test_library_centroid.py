import numpy as np

from library.centroid import psalm_centroids


def test_mean_pools_a_psalms_half_verse_vectors() -> None:
    half_verses_by_psalm = {1: [100, 101]}
    node_vectors = {100: np.array([1.0, 0.0]), 101: np.array([3.0, 2.0])}

    result = psalm_centroids(half_verses_by_psalm, node_vectors)

    assert np.allclose(result[1], [2.0, 1.0])


def test_keeps_one_centroid_per_psalm() -> None:
    half_verses_by_psalm = {1: [100], 2: [200, 201]}
    node_vectors = {
        100: np.array([1.0, 0.0]),
        200: np.array([0.0, 1.0]),
        201: np.array([0.0, 3.0]),
    }

    result = psalm_centroids(half_verses_by_psalm, node_vectors)

    assert set(result) == {1, 2}
    assert np.allclose(result[1], [1.0, 0.0])
    assert np.allclose(result[2], [0.0, 2.0])


def test_skips_a_psalm_missing_entirely_from_node_vectors() -> None:
    half_verses_by_psalm = {1: [100], 2: [200]}
    node_vectors = {100: np.array([1.0, 0.0])}

    result = psalm_centroids(half_verses_by_psalm, node_vectors)

    assert set(result) == {1}


def test_skips_a_psalm_with_a_partially_missing_node() -> None:
    """A psalm whose half-verse nodes are only partly covered is dropped, not partially pooled."""
    half_verses_by_psalm = {1: [100, 101]}
    node_vectors = {100: np.array([1.0, 0.0])}

    result = psalm_centroids(half_verses_by_psalm, node_vectors)

    assert result == {}
