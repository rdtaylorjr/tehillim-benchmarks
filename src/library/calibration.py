"""Calibrates similarity against a model's own background; group-mean form is an effect size."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class BackgroundStats:
    mean: float
    std: float
    n_vectors: int


def background_similarity_stats(vectors: np.ndarray) -> BackgroundStats:
    """Mean/std cosine similarity across every distinct pair among vectors, self-pairs excluded."""
    norm = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = norm @ norm.T
    n = similarities.shape[0]
    off_diagonal = similarities[~np.eye(n, dtype=bool)]
    return BackgroundStats(
        mean=float(off_diagonal.mean()), std=float(off_diagonal.std()), n_vectors=n
    )


def calibrated_z_score(value: float, background: BackgroundStats) -> float:
    """How many background standard deviations above typical a single observation sits."""
    if background.std == 0:
        raise ValueError("background has zero variance; this model cannot be calibrated")
    return (value - background.mean) / background.std


def calibrated_effect_size(group_mean: float, background: BackgroundStats) -> float:
    """Cohen's-d-style standardized distance of a group's mean similarity from the background."""
    return calibrated_z_score(group_mean, background)
