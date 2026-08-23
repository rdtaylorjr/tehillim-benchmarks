"""Per-genre one-vs-rest permutation test for a trajectory distance metric, joint maxT."""

from dataclasses import dataclass

import numpy as np

from genre.permutation import one_vs_rest_masks


@dataclass(frozen=True, slots=True)
class GenreBreakdownResult:
    """Per-genre one-vs-rest distance gap, permutation p, and maxT-corrected p, for one source."""

    genres: tuple[str, ...]
    gap_observed: tuple[float, ...]
    p_perm: tuple[float, ...]
    p_maxT: tuple[float, ...]  # noqa: N815 -- Westfall-Young maxT term
    n_permutations: int


def _one_vs_rest_gap(distances: np.ndarray, same: np.ndarray, population: np.ndarray) -> float:
    """between-genre minus within-genre mean distance, restricted to a one-vs-rest population."""
    pop_distances = distances[population]
    pop_same = same[population]
    same_d = pop_distances[pop_same]
    diff_d = pop_distances[~pop_same]
    if len(same_d) == 0 or len(diff_d) == 0:
        return float("nan")
    return float(diff_d.mean() - same_d.mean())


def _dense_pair_matrices(
    n: int, idx_a: np.ndarray, idx_b: np.ndarray, distances: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric n x n distance and pair-validity matrices built from a sparse pair list."""
    distance_matrix = np.zeros((n, n), dtype=np.float64)
    mask_matrix = np.zeros((n, n), dtype=np.float64)
    distance_matrix[idx_a, idx_b] = distances
    distance_matrix[idx_b, idx_a] = distances
    mask_matrix[idx_a, idx_b] = 1.0
    mask_matrix[idx_b, idx_a] = 1.0
    return distance_matrix, mask_matrix


def _batched_one_vs_rest_gap_naive(
    distances: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, is_target_batch: np.ndarray
) -> np.ndarray:
    """Reference implementation: explicit (permutations x pairs) masking, kept only for testing.

    same is always a subset of population (both-target implies at-least-one-target), so the
    population sum/count minus the same sum/count gives the different-genre side directly.
    """
    same_batch = is_target_batch[:, idx_a] & is_target_batch[:, idx_b]
    population_batch = is_target_batch[:, idx_a] | is_target_batch[:, idx_b]
    d = distances.astype(np.float64, copy=False)

    same_sum = same_batch.astype(np.float64) @ d
    same_count = same_batch.sum(axis=1).astype(np.float64)
    population_sum = population_batch.astype(np.float64) @ d
    population_count = population_batch.sum(axis=1).astype(np.float64)
    diff_sum = population_sum - same_sum
    diff_count = population_count - same_count

    with np.errstate(invalid="ignore", divide="ignore"):
        gap = (diff_sum / diff_count) - (same_sum / same_count)
    invalid = (same_count == 0) | (diff_count == 0)
    return np.where(invalid, np.nan, gap)


def _batched_one_vs_rest_gap(
    is_target_batch: np.ndarray,
    distance_matrix: np.ndarray,
    mask_matrix: np.ndarray,
    distance_colsum: np.ndarray,
    mask_colsum: np.ndarray,
    total_sum: float,
    total_count: float,
) -> np.ndarray:
    """Vectorized per-permutation one-vs-rest gap via quadratic forms over the dense pair matrices.

    Reformulates the (same/population) sums as t^T D t style quadratic forms over the n x n
    psalm-pair matrix (t the 0/1 target-membership vector for one permutation draw) instead of
    materializing a (permutations x pairs) intermediate: "neither" (both sides non-target) is
    computed from (1-t), using (1-t)^T D = D_colsum - t^T D so only one (permutations x n) @
    (n x n) matmul per quantity is needed, replacing an O(permutations x pairs) cost with
    O(permutations x psalms^2). Proven exactly equivalent to `_batched_one_vs_rest_gap_naive`,
    not an approximation, in tests/test_trajectory_genre_breakdown.py.
    """
    t = is_target_batch.astype(np.float64)
    not_t = 1.0 - t

    dt = t @ distance_matrix
    mt = t @ mask_matrix
    same_sum = 0.5 * np.einsum("bi,bi->b", t, dt)
    same_count = 0.5 * np.einsum("bi,bi->b", t, mt)
    neither_sum = 0.5 * np.einsum("bi,bi->b", not_t, distance_colsum - dt)
    neither_count = 0.5 * np.einsum("bi,bi->b", not_t, mask_colsum - mt)

    population_sum = total_sum - neither_sum
    population_count = total_count - neither_count
    diff_sum = population_sum - same_sum
    diff_count = population_count - same_count

    with np.errstate(invalid="ignore", divide="ignore"):
        gap = (diff_sum / diff_count) - (same_sum / same_count)
    invalid = (same_count == 0) | (diff_count == 0)
    return np.where(invalid, np.nan, gap)


def joint_genre_breakdown_permutation_test(
    distances: np.ndarray,
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    genre_codes: np.ndarray,
    genres: tuple[str, ...],
    n_permutations: int = 2000,
    rng: np.random.Generator | None = None,
) -> GenreBreakdownResult:
    """One-sided permutation p per genre's one-vs-rest distance gap, plus a Westfall-Young maxT.

    Mirrors genre.permutation.joint_psalm_label_permutation_test's joint-null construction (one
    shared per-permutation genre-label draw so the maxT correction across genres is valid), but
    for a mean-distance gap statistic instead of an AUC, matching this module's distances-based
    (not similarity-based) inputs.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n_genres = len(genres)
    n = len(genre_codes)

    gap_observed = np.full(n_genres, np.nan)
    for g in range(n_genres):
        same_mask, population_mask = one_vs_rest_masks(genre_codes, g)
        gap_observed[g] = _one_vs_rest_gap(
            distances, same_mask[idx_a, idx_b], population_mask[idx_a, idx_b]
        )

    distance_matrix, mask_matrix = _dense_pair_matrices(n, idx_a, idx_b, distances)
    distance_colsum = distance_matrix.sum(axis=0)
    mask_colsum = mask_matrix.sum(axis=0)
    total_sum = float(distances.sum())
    total_count = float(len(distances))

    tiled_codes = np.tile(genre_codes, (n_permutations, 1))
    permuted_codes = rng.permuted(tiled_codes, axis=1)

    null_gap = np.full((n_permutations, n_genres), np.nan)
    for g in range(n_genres):
        is_target_batch = permuted_codes == g
        null_gap[:, g] = _batched_one_vs_rest_gap(
            is_target_batch,
            distance_matrix,
            mask_matrix,
            distance_colsum,
            mask_colsum,
            total_sum,
            total_count,
        )

    max_null_gap = np.nanmax(null_gap, axis=1)

    p_perm = np.full(n_genres, np.nan)
    p_maxT = np.full(n_genres, np.nan)  # noqa: N806 -- Westfall-Young maxT term
    for g in range(n_genres):
        valid = ~np.isnan(null_gap[:, g])
        p_perm[g] = (np.sum(null_gap[valid, g] >= gap_observed[g]) + 1) / (int(np.sum(valid)) + 1)
        valid_max = ~np.isnan(max_null_gap)
        p_maxT[g] = (np.sum(max_null_gap[valid_max] >= gap_observed[g]) + 1) / (
            int(np.sum(valid_max)) + 1
        )

    return GenreBreakdownResult(
        genres=genres,
        gap_observed=tuple(gap_observed.tolist()),
        p_perm=tuple(p_perm.tolist()),
        p_maxT=tuple(p_maxT.tolist()),
        n_permutations=n_permutations,
    )
