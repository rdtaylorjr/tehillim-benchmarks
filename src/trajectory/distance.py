"""Two independent psalm-pair distances: topical content and structural (self-similarity) shape."""

import numpy as np

from library.retrieval_metrics import cosine_similarity_matrix
from trajectory.dtw import dtw_distance


def content_distance(centroid_a: np.ndarray, centroid_b: np.ndarray) -> float:
    """1 minus cosine similarity between two psalm content centroids."""
    a_norm = centroid_a / np.linalg.norm(centroid_a)
    b_norm = centroid_b / np.linalg.norm(centroid_b)
    return float(1.0 - np.dot(a_norm, b_norm))


def structural_distance(profile_a: np.ndarray, profile_b: np.ndarray) -> float:
    """RMS elementwise difference between two same-shape arrays."""
    return float(np.sqrt(np.mean((profile_a - profile_b) ** 2)))


def dtw_curve_distance(curve_a: np.ndarray, curve_b: np.ndarray) -> float:
    """DTW-aligned distance between two variable-length 1D curves, absolute-difference cost."""
    cost = np.abs(curve_a[:, None] - curve_b[None, :])
    distance, _ = dtw_distance(cost)
    return distance


def structural_distance_dtw(
    sequence_a: np.ndarray,
    sequence_b: np.ndarray,
    self_similarity_a: np.ndarray,
    self_similarity_b: np.ndarray,
) -> float:
    """Structural distance via DTW-synchronized self-similarity matrices (Muller 2007-style)."""
    cost = 1.0 - cosine_similarity_matrix(sequence_a, sequence_b)
    _, path = dtw_distance(cost)
    idx_a = np.array([i for i, _ in path])
    idx_b = np.array([j for _, j in path])
    synchronized_a = self_similarity_a[np.ix_(idx_a, idx_a)]
    synchronized_b = self_similarity_b[np.ix_(idx_b, idx_b)]
    return structural_distance(synchronized_a, synchronized_b)
