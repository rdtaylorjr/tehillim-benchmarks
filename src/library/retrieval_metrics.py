"""Retrieval metrics and significance tests for scoring embedding vectors against the benchmark."""

from collections.abc import Sequence
from typing import NamedTuple

import numpy as np
import scipy.sparse as sp
from scipy.stats import rankdata, wilcoxon


class DiscriminationResult(NamedTuple):
    statistic: float
    p_value: float
    rank_biserial: float


class PermutationResult(NamedTuple):
    observed_gap: float
    p_value: float
    z_score: float


class BootstrapResult(NamedTuple):
    observed_diff: float
    ci_low: float
    ci_high: float
    p_value: float


def cosine_similarity_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between every vector in a and every vector in b."""
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    if np.any(a_norm == 0) or np.any(b_norm == 0):
        raise ValueError("cannot compute cosine similarity for a zero vector")
    return np.asarray((a / a_norm) @ (b / b_norm).T)


def sparse_cosine_similarity_matrix(a: sp.csr_matrix, b: sp.csr_matrix) -> np.ndarray:
    """Same semantics as cosine_similarity_matrix, but a and b are sparse and never densified."""
    a_norm = np.sqrt(np.asarray(a.multiply(a).sum(axis=1))).ravel()
    b_norm = np.sqrt(np.asarray(b.multiply(b).sum(axis=1))).ravel()
    if np.any(a_norm == 0) or np.any(b_norm == 0):
        raise ValueError("cannot compute cosine similarity for a zero vector")
    a_normalized = sp.diags(1.0 / a_norm) @ a
    b_normalized = sp.diags(1.0 / b_norm) @ b
    return np.asarray((a_normalized @ b_normalized.T).toarray())


def paired_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity of a[i] against b[i] only, never any other row."""
    a_norm = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = b / np.linalg.norm(b, axis=1, keepdims=True)
    return np.asarray(np.sum(a_norm * b_norm, axis=1))


def ranks_from_similarity_matrix(
    similarities: np.ndarray,
    pool_ids: Sequence[str],
    true_target_ids: Sequence[str],
) -> list[float]:
    """Same ranks as retrieval_ranks, from an already-computed matrix (no matmul recomputed)."""
    pool_index = {pid: i for i, pid in enumerate(pool_ids)}
    true_columns = np.array([pool_index[tid] for tid in true_target_ids])
    rank_matrix = rankdata(-similarities, method="average", axis=1)
    rows = np.arange(similarities.shape[0])
    return [float(r) for r in rank_matrix[rows, true_columns]]


def retrieval_ranks(
    anchor_vectors: np.ndarray,
    pool_vectors: np.ndarray,
    pool_ids: Sequence[str],
    true_target_ids: Sequence[str],
) -> list[float]:
    """For each anchor, the (tie-averaged) rank of its true target among the pool, 1 = closest."""
    similarities = cosine_similarity_matrix(anchor_vectors, pool_vectors)
    return ranks_from_similarity_matrix(similarities, pool_ids, true_target_ids)


def outranking_candidates(
    similarities: np.ndarray, pool_ids: Sequence[str], true_target_ids: Sequence[str]
) -> list[list[str]]:
    """For each anchor, the other pool ids whose similarity ties or beats the true target's."""
    pool_index = {pid: i for i, pid in enumerate(pool_ids)}
    outrankers = []
    for row, true_id in zip(similarities, true_target_ids, strict=True):
        true_similarity = row[pool_index[true_id]]
        beaters = [
            pid for pid in pool_ids if pid != true_id and row[pool_index[pid]] >= true_similarity
        ]
        outrankers.append(beaters)
    return outrankers


def mean_reciprocal_rank(ranks: Sequence[float] | np.ndarray) -> float:
    """Mean of 1/rank across all queries."""
    return float(np.mean(1.0 / np.asarray(ranks, dtype=float)))


def recall_at_k(ranks: Sequence[float] | np.ndarray, k: int) -> float:
    """Fraction of queries whose true target ranked at or above k."""
    return float(np.mean(np.asarray(ranks, dtype=float) <= k))


def paired_discrimination_test(
    true_similarities: np.ndarray, null_similarities: np.ndarray
) -> DiscriminationResult:
    """Wilcoxon signed-rank test of true-pair similarity exceeding its matched null similarity."""
    true_similarities = np.asarray(true_similarities, dtype=float)
    null_similarities = np.asarray(null_similarities, dtype=float)
    diffs = true_similarities - null_similarities
    nonzero = diffs[diffs != 0]

    if nonzero.size == 0:
        return DiscriminationResult(statistic=0.0, p_value=1.0, rank_biserial=0.0)

    result = wilcoxon(nonzero, alternative="greater")
    signed_ranks = rankdata(np.abs(nonzero), method="average") * np.sign(nonzero)
    w_plus = signed_ranks[signed_ranks > 0].sum()
    w_minus = -signed_ranks[signed_ranks < 0].sum()
    n = nonzero.size
    rank_biserial = (w_plus - w_minus) / (n * (n + 1) / 2)
    return DiscriminationResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        rank_biserial=float(rank_biserial),
    )


def _per_anchor_gap(similarity_matrix: np.ndarray, true_positions: np.ndarray) -> np.ndarray:
    """true_sim minus the mean of the row's other entries, excluding the diagonal and fake-true."""
    n = similarity_matrix.shape[0]
    rows = np.arange(n)
    true_sim = similarity_matrix[rows, true_positions]
    diagonal = np.diagonal(similarity_matrix)
    is_diagonal = true_positions == rows
    excluded_sum = np.where(is_diagonal, diagonal, diagonal + true_sim)
    excluded_count = np.where(is_diagonal, 1, 2)
    # n=2 has no third column to exclude; fall back to excluding only the fake-true column.
    degenerate = (n - excluded_count) <= 0
    excluded_sum = np.where(degenerate, true_sim, excluded_sum)
    excluded_count = np.where(degenerate, 1, excluded_count)
    null_sim = (similarity_matrix.sum(axis=1) - excluded_sum) / (n - excluded_count)
    return np.asarray(true_sim - null_sim)


def _combine_by_stratum(gaps: np.ndarray, stratum: np.ndarray, weighted: bool) -> float:
    if weighted:
        return float(gaps.mean())
    stratum_means = [gaps[stratum == value].mean() for value in np.unique(stratum)]
    return float(np.mean(stratum_means))


def _per_anchor_gap_batch(
    similarity_matrix: np.ndarray, true_positions_batch: np.ndarray
) -> np.ndarray:
    """Batched _per_anchor_gap: true_positions_batch is (B, n), returns gaps of shape (B, n)."""
    n = similarity_matrix.shape[0]
    rows = np.arange(n)
    true_sim = similarity_matrix[rows[None, :], true_positions_batch]
    diagonal = np.diagonal(similarity_matrix)[None, :]
    is_diagonal = true_positions_batch == rows[None, :]
    excluded_sum = np.where(is_diagonal, diagonal, diagonal + true_sim)
    excluded_count = np.where(is_diagonal, 1, 2)
    degenerate = (n - excluded_count) <= 0
    excluded_sum = np.where(degenerate, true_sim, excluded_sum)
    excluded_count = np.where(degenerate, 1, excluded_count)
    row_sums = similarity_matrix.sum(axis=1)[None, :]
    null_sim = (row_sums - excluded_sum) / (n - excluded_count)
    return np.asarray(true_sim - null_sim)


def _combine_by_stratum_batch(
    gaps_batch: np.ndarray, stratum: np.ndarray, weighted: bool
) -> np.ndarray:
    """Batched _combine_by_stratum: gaps_batch is (B, n), returns combined values of shape (B,)."""
    if weighted:
        return np.asarray(gaps_batch.mean(axis=1))
    stratum_means = np.stack(
        [gaps_batch[:, stratum == value].mean(axis=1) for value in np.unique(stratum)], axis=1
    )
    return np.asarray(stratum_means.mean(axis=1))


def stratified_mean_gap_test(
    similarity_matrix: np.ndarray,
    anchor_stratum: np.ndarray,
    n_permutations: int = 10000,
    rng: np.random.Generator | None = None,
    weighted: bool = False,
) -> PermutationResult:
    """Permutation test of true-pair similarity; prefer z_score, p_value saturates its floor."""
    rng = rng if rng is not None else np.random.default_rng()
    similarity_matrix = np.asarray(similarity_matrix, dtype=float)
    anchor_stratum = np.asarray(anchor_stratum)
    n = similarity_matrix.shape[0]

    observed_gaps = _per_anchor_gap(similarity_matrix, np.arange(n))
    observed = _combine_by_stratum(observed_gaps, anchor_stratum, weighted)

    random_true_batch = rng.integers(0, n, size=(n_permutations, n))
    permuted_gaps_batch = _per_anchor_gap_batch(similarity_matrix, random_true_batch)
    permuted_gaps = _combine_by_stratum_batch(permuted_gaps_batch, anchor_stratum, weighted)

    p_value = (np.sum(permuted_gaps >= observed) + 1) / (n_permutations + 1)
    null_std = permuted_gaps.std(ddof=1)
    z_score = (observed - permuted_gaps.mean()) / null_std if null_std > 0 else float("nan")
    return PermutationResult(observed_gap=observed, p_value=float(p_value), z_score=float(z_score))


def paired_bootstrap_mrr_diff(
    ranks_a: Sequence[float] | np.ndarray,
    ranks_b: Sequence[float] | np.ndarray,
    n_resamples: int = 10000,
    rng: np.random.Generator | None = None,
    clusters: Sequence[object] | np.ndarray | None = None,
) -> BootstrapResult:
    """Bootstrap CI for the paired MRR difference; `clusters` switches to a block bootstrap."""
    rng = rng if rng is not None else np.random.default_rng()
    a = np.asarray(ranks_a, dtype=float)
    b = np.asarray(ranks_b, dtype=float)
    n = len(a)

    observed_diff = mean_reciprocal_rank(a) - mean_reciprocal_rank(b)
    diffs = np.empty(n_resamples)

    if clusters is None:
        for i in range(n_resamples):
            sample = rng.integers(0, n, size=n)
            diffs[i] = mean_reciprocal_rank(a[sample]) - mean_reciprocal_rank(b[sample])
    else:
        clusters_arr = np.asarray(clusters)
        groups = [np.flatnonzero(clusters_arr == value) for value in np.unique(clusters_arr)]
        n_groups = len(groups)
        for i in range(n_resamples):
            chosen = rng.integers(0, n_groups, size=n_groups)
            sample = np.concatenate([groups[c] for c in chosen])
            diffs[i] = mean_reciprocal_rank(a[sample]) - mean_reciprocal_rank(b[sample])

    ci_low, ci_high = np.percentile(diffs, [2.5, 97.5])
    below = (np.sum(diffs <= 0) + 1) / (n_resamples + 1)
    above = (np.sum(diffs >= 0) + 1) / (n_resamples + 1)
    p_value = min(2 * min(below, above), 1.0)
    return BootstrapResult(
        observed_diff=float(observed_diff),
        ci_low=float(ci_low),
        ci_high=float(ci_high),
        p_value=float(p_value),
    )
