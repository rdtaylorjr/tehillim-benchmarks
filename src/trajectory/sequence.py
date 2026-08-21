"""Builds each psalm's cola embeddings as one ordered sequence, in canonical verse order."""

import numpy as np


def psalm_cola_sequences(
    half_verses_by_psalm: dict[int, list[int]], node_vectors: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """One (n_cola, dim) array per psalm whose half-verse nodes are all present in node_vectors."""
    sequences = {}
    for psalm, nodes in half_verses_by_psalm.items():
        if not all(node in node_vectors for node in nodes):
            continue
        sequences[psalm] = np.stack([node_vectors[node] for node in nodes])
    return sequences


def normalize_sequence(sequence: np.ndarray) -> np.ndarray:
    """L2-normalizes each row so later magnitude-sensitive features aren't skewed by anisotropy."""
    norms = np.linalg.norm(sequence, axis=1, keepdims=True)
    return np.asarray(sequence / norms)
