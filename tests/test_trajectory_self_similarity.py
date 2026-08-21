import numpy as np
import pytest

from trajectory.self_similarity import self_similarity_matrix


def test_self_similarity_matrix_has_unit_diagonal() -> None:
    sequence = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])

    matrix = self_similarity_matrix(sequence)

    assert matrix.shape == (3, 3)
    assert np.allclose(np.diag(matrix), 1.0)


def test_self_similarity_matrix_is_symmetric_and_reflects_orthogonality() -> None:
    sequence = np.array([[1.0, 0.0], [0.0, 1.0]])

    matrix = self_similarity_matrix(sequence)

    assert matrix[0, 1] == pytest.approx(0.0)
    assert matrix[0, 1] == matrix[1, 0]
