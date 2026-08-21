"""Mean-pools a psalm's half-verse embedding vectors into one centroid per psalm."""

import numpy as np


def psalm_centroids(
    half_verses_by_psalm: dict[int, list[int]], node_vectors: dict[int, np.ndarray]
) -> dict[int, np.ndarray]:
    """One centroid per psalm whose half-verse nodes are all present in node_vectors."""
    centroids: dict[int, np.ndarray] = {}
    for psalm, nodes in half_verses_by_psalm.items():
        if not all(node in node_vectors for node in nodes):
            continue
        centroids[psalm] = np.mean([node_vectors[node] for node in nodes], axis=0)
    return centroids
