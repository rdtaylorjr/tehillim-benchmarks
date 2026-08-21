import numpy as np

from trajectory.sequence import normalize_sequence, psalm_cola_sequences


def test_psalm_cola_sequences_preserves_node_order() -> None:
    half_verses_by_psalm = {1: [100, 101, 102]}
    node_vectors = {
        100: np.array([1.0, 0.0]),
        101: np.array([0.0, 1.0]),
        102: np.array([1.0, 1.0]),
    }

    result = psalm_cola_sequences(half_verses_by_psalm, node_vectors)

    assert np.allclose(result[1], [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])


def test_psalm_cola_sequences_keeps_one_sequence_per_psalm() -> None:
    half_verses_by_psalm = {1: [100], 2: [200, 201]}
    node_vectors = {
        100: np.array([1.0, 0.0]),
        200: np.array([0.0, 1.0]),
        201: np.array([0.0, 3.0]),
    }

    result = psalm_cola_sequences(half_verses_by_psalm, node_vectors)

    assert set(result) == {1, 2}
    assert result[1].shape == (1, 2)
    assert result[2].shape == (2, 2)


def test_psalm_cola_sequences_skips_a_psalm_with_a_partially_missing_node() -> None:
    half_verses_by_psalm = {1: [100, 101]}
    node_vectors = {100: np.array([1.0, 0.0])}

    result = psalm_cola_sequences(half_verses_by_psalm, node_vectors)

    assert result == {}


def test_normalize_sequence_gives_every_row_unit_norm() -> None:
    sequence = np.array([[3.0, 4.0], [1.0, 0.0], [0.0, 5.0]])

    result = normalize_sequence(sequence)

    norms = np.linalg.norm(result, axis=1)
    assert np.allclose(norms, 1.0)


def test_normalize_sequence_preserves_direction() -> None:
    sequence = np.array([[3.0, 4.0]])

    result = normalize_sequence(sequence)

    assert np.allclose(result, [[0.6, 0.8]])
