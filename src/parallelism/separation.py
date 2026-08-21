"""Measures whether true parallel cola are closer in embedding space than unrelated cola."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu


@dataclass(frozen=True, slots=True)
class SeparationResult:
    auc: float
    p_value: float
    n_positive: int
    n_negative: int


def similarity_separation(
    similarity_matrix: np.ndarray, row_mask: np.ndarray | None = None
) -> SeparationResult:
    """AUC of true-pair vs. other-pair similarity; row_mask restricts positives only."""
    n = similarity_matrix.shape[0]
    rows = np.arange(n) if row_mask is None else np.flatnonzero(row_mask)
    positive = similarity_matrix[rows, rows]
    off_diagonal_by_row = similarity_matrix[~np.eye(n, dtype=bool)].reshape(n, n - 1)
    negative = off_diagonal_by_row[rows].reshape(-1)
    statistic, p_value = mannwhitneyu(positive, negative, alternative="greater")
    auc = statistic / (len(positive) * len(negative))
    return SeparationResult(
        auc=float(auc),
        p_value=float(p_value),
        n_positive=len(positive),
        n_negative=len(negative),
    )
