import numpy as np
import pytest
import scipy.sparse as sp

from library.errors import DegenerateVectorError, InsufficientDataError
from library.retrieval_metrics import (
    _combine_by_stratum,
    _combine_by_stratum_batch,
    _per_anchor_gap,
    _per_anchor_gap_batch,
    cosine_similarity_matrix,
    mean_reciprocal_rank,
    paired_cosine_similarity,
    paired_discrimination_test,
    recall_at_k,
    sparse_cosine_similarity_matrix,
    stratified_mean_gap_test,
)


def test_paired_cosine_similarity_is_one_for_identical_vectors() -> None:
    source = np.array([[1.0, 0.0], [0.0, 2.0]])
    target = np.array([[3.0, 0.0], [0.0, 5.0]])

    result = paired_cosine_similarity(source, target)

    assert np.allclose(result, [1.0, 1.0])


def test_paired_cosine_similarity_is_zero_for_orthogonal_vectors() -> None:
    source = np.array([[1.0, 0.0]])
    target = np.array([[0.0, 1.0]])

    result = paired_cosine_similarity(source, target)

    assert np.allclose(result, [0.0])


def test_paired_cosine_similarity_does_not_compare_across_rows() -> None:
    """Row i's similarity must only ever involve source[i] and target[i], nothing else."""
    source = np.array([[1.0, 0.0], [0.0, 1.0]])
    target = np.array([[1.0, 0.0], [1.0, 0.0]])

    result = paired_cosine_similarity(source, target)

    assert np.allclose(result, [1.0, 0.0])


def test_cosine_similarity_matrix_of_identical_unit_vectors_is_one() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    result = cosine_similarity_matrix(a, b)
    assert result[0, 0] == pytest.approx(1.0)


def test_cosine_similarity_matrix_of_orthogonal_vectors_is_zero() -> None:
    a = np.array([[1.0, 0.0]])
    b = np.array([[0.0, 1.0]])
    result = cosine_similarity_matrix(a, b)
    assert result[0, 0] == pytest.approx(0.0)


def test_cosine_similarity_matrix_is_scale_invariant() -> None:
    a = np.array([[2.0, 0.0]])
    b = np.array([[100.0, 0.0]])
    result = cosine_similarity_matrix(a, b)
    assert result[0, 0] == pytest.approx(1.0)


def test_cosine_similarity_matrix_raises_on_a_zero_vector() -> None:
    a = np.array([[0.0, 0.0]])
    b = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="zero"):
        cosine_similarity_matrix(a, b)


def test_sparse_cosine_similarity_matrix_of_identical_unit_vectors_is_one() -> None:
    a = sp.csr_matrix(np.array([[1.0, 0.0]]))
    b = sp.csr_matrix(np.array([[1.0, 0.0]]))
    result = sparse_cosine_similarity_matrix(a, b)
    assert result[0, 0] == pytest.approx(1.0)


def test_sparse_cosine_similarity_matrix_raises_on_a_zero_vector() -> None:
    a = sp.csr_matrix(np.array([[0.0, 0.0]]))
    b = sp.csr_matrix(np.array([[1.0, 0.0]]))
    with pytest.raises(ValueError, match="zero"):
        sparse_cosine_similarity_matrix(a, b)


def test_sparse_cosine_similarity_matrix_matches_the_dense_function_to_float_tolerance() -> None:
    """Proves row-normalize-then-matmul over sparse inputs gives the identical dense result."""
    rng = np.random.default_rng(0)
    dim = 500
    a_dense = np.zeros((6, dim))
    b_dense = np.zeros((4, dim))
    for row in a_dense:
        n_nonzero = rng.integers(1, 6)
        idx = rng.choice(dim, size=n_nonzero, replace=False)
        row[idx] = rng.uniform(0.1, 5.0, size=n_nonzero)
    for row in b_dense:
        n_nonzero = rng.integers(1, 6)
        idx = rng.choice(dim, size=n_nonzero, replace=False)
        row[idx] = rng.uniform(0.1, 5.0, size=n_nonzero)

    expected = cosine_similarity_matrix(a_dense, b_dense)
    actual = sparse_cosine_similarity_matrix(sp.csr_matrix(a_dense), sp.csr_matrix(b_dense))

    np.testing.assert_allclose(actual, expected, rtol=1e-6)


def test_mean_reciprocal_rank_of_all_first_place_is_one() -> None:
    assert mean_reciprocal_rank([1, 1, 1]) == 1.0


def test_mean_reciprocal_rank_averages_reciprocals() -> None:
    assert mean_reciprocal_rank([1, 2, 4]) == pytest.approx((1 + 0.5 + 0.25) / 3)


def test_recall_at_k_counts_ranks_within_k() -> None:
    assert recall_at_k([1, 2, 3, 10], k=3) == 0.75


def test_paired_discrimination_test_detects_strong_positive_signal() -> None:
    rng = np.random.default_rng(0)
    true_sims = rng.normal(0.8, 0.02, size=50)
    null_sims = rng.normal(0.2, 0.02, size=50)

    result = paired_discrimination_test(true_sims, null_sims)

    assert result.p_value < 0.001
    assert result.rank_biserial == pytest.approx(1.0)


def test_paired_discrimination_test_finds_no_signal_when_ties_cancel() -> None:
    true_sims = np.array([0.5, 0.6, 0.4, 0.5])
    null_sims = np.array([0.5, 0.4, 0.6, 0.5])

    result = paired_discrimination_test(true_sims, null_sims)

    assert result.rank_biserial == pytest.approx(0.0)


def test_per_anchor_gap_excludes_the_real_diagonal_even_under_permutation() -> None:
    """A permuted "true" column must not let the anchor's own diagonal leak into its null mean."""
    matrix = np.array([[10.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    true_positions = np.array([1, 1, 2])

    gaps = _per_anchor_gap(matrix, true_positions)

    assert gaps == pytest.approx([1.0 - 2.0, 4.0 - 4.0, 8.0 - 6.5])


def test_per_anchor_gap_falls_back_gracefully_when_n_is_too_small_to_exclude_both() -> None:
    """n=2 leaves no third column."""
    matrix = np.array([[10.0, 1.0], [2.0, 20.0]])

    gaps = _per_anchor_gap(matrix, np.array([1, 0]))

    assert not np.any(np.isnan(gaps))
    assert gaps == pytest.approx([1.0 - 10.0, 2.0 - 20.0])


def test_per_anchor_gap_matches_original_formula_when_true_position_is_the_diagonal() -> None:
    matrix = np.array([[10.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])

    gaps = _per_anchor_gap(matrix, np.array([0, 1, 2]))

    assert gaps == pytest.approx([10.0 - 1.5, 4.0 - 4.0, 8.0 - 6.5])


def test_per_anchor_gap_batch_matches_a_naive_loop_over_permutations_bit_for_bit() -> None:
    rng = np.random.default_rng(11)
    n = 12
    matrix = rng.normal(0.4, 0.1, size=(n, n))
    true_positions_batch = rng.integers(0, n, size=(50, n))

    naive = np.stack([_per_anchor_gap(matrix, row) for row in true_positions_batch])
    batched = _per_anchor_gap_batch(matrix, true_positions_batch)

    assert np.array_equal(naive, batched)


def test_combine_by_stratum_batch_matches_a_naive_loop_bit_for_bit() -> None:
    rng = np.random.default_rng(12)
    n = 12
    gaps_batch = rng.normal(0.0, 0.2, size=(50, n))
    stratum = np.repeat(["low", "mid", "high"], 4)

    for weighted in (True, False):
        naive = np.array(
            [_combine_by_stratum(row, stratum, weighted=weighted) for row in gaps_batch]
        )
        batched = _combine_by_stratum_batch(gaps_batch, stratum, weighted=weighted)
        assert np.array_equal(naive, batched)


def test_stratified_mean_gap_test_detects_signal_beyond_confound() -> None:
    """Every anchor's diagonal (true) entry gets a +0.3 bonus over the shared off-diagonal level."""
    rng = np.random.default_rng(1)
    n = 60
    strata = np.repeat(["low", "mid", "high"], 20)
    matrix = rng.normal(0.4, 0.05, size=(n, n))
    np.fill_diagonal(matrix, rng.normal(0.4, 0.05, size=n) + 0.3)

    result = stratified_mean_gap_test(matrix, strata, n_permutations=2000, rng=rng)

    assert result.observed_gap == pytest.approx(0.3, abs=0.03)
    assert result.p_value < 0.01
    assert result.z_score > 5


def test_stratified_mean_gap_test_finds_no_signal_when_diagonal_matches_baseline() -> None:
    rng = np.random.default_rng(2)
    n = 40
    strata = np.array(["only"] * n)
    matrix = rng.normal(0.5, 0.1, size=(n, n))

    result = stratified_mean_gap_test(matrix, strata, n_permutations=2000, rng=rng)

    assert result.p_value > 0.05
    assert abs(result.z_score) < 3


def test_stratified_mean_gap_test_z_score_stays_graduated_past_the_p_value_floor() -> None:
    """A much stronger effect should read as a much larger z-score, even once p hits the floor."""
    rng = np.random.default_rng(9)
    n = 30
    strata = np.array(["only"] * n)

    weak_matrix = rng.normal(0.4, 0.05, size=(n, n))
    np.fill_diagonal(weak_matrix, rng.normal(0.4, 0.05, size=n) + 0.15)
    strong_matrix = rng.normal(0.4, 0.05, size=(n, n))
    np.fill_diagonal(strong_matrix, rng.normal(0.4, 0.05, size=n) + 0.6)

    weak = stratified_mean_gap_test(weak_matrix, strata, n_permutations=500, rng=rng)
    strong = stratified_mean_gap_test(strong_matrix, strata, n_permutations=500, rng=rng)

    assert weak.p_value == pytest.approx(strong.p_value)
    assert strong.z_score > weak.z_score * 1.5


def test_stratified_mean_gap_test_is_not_fooled_by_hub_anchors() -> None:
    """Some anchors are generically similar to everything (embedding-space anisotropy/"hubness")."""
    rng = np.random.default_rng(6)
    n = 30
    anchor_level = rng.uniform(0.0, 1.0, size=n)
    matrix = rng.normal(anchor_level[:, None], 0.02, size=(n, n))
    strata = np.array(["only"] * n)

    result = stratified_mean_gap_test(matrix, strata, n_permutations=2000, rng=rng)

    assert result.p_value > 0.05


def test_stratified_mean_gap_test_weighted_and_unweighted_differ_with_unequal_strata() -> None:
    n_small, n_large = 5, 45
    n = n_small + n_large
    strata = np.array(["small"] * n_small + ["large"] * n_large)
    matrix = np.full((n, n), 0.2)
    diag = np.full(n, 0.2)
    diag[:n_small] += 0.5
    np.fill_diagonal(matrix, diag)

    unweighted = stratified_mean_gap_test(matrix, strata, n_permutations=50, weighted=False)
    weighted = stratified_mean_gap_test(matrix, strata, n_permutations=50, weighted=True)

    assert unweighted.observed_gap == pytest.approx(0.25)
    assert weighted.observed_gap == pytest.approx(0.05)


def test_paired_cosine_similarity_rejects_a_zero_vector() -> None:
    """A zero vector has no direction, so a silent NaN would flow on into AP and AUC unnoticed."""
    source = np.array([[1.0, 0.0], [0.0, 0.0]])
    target = np.array([[1.0, 0.0], [1.0, 0.0]])

    with pytest.raises(ValueError, match="zero vector"):
        paired_cosine_similarity(source, target)


def test_paired_cosine_similarity_rejects_a_zero_vector_on_the_target_side() -> None:
    source = np.array([[1.0, 0.0]])
    target = np.array([[0.0, 0.0]])

    with pytest.raises(ValueError, match="zero vector"):
        paired_cosine_similarity(source, target)


def test_stratified_mean_gap_test_rejects_a_single_anchor() -> None:
    """One anchor leaves no other column to average as the null, so the gap is undefined."""
    with pytest.raises(InsufficientDataError, match="at least 2"):
        stratified_mean_gap_test(np.array([[1.0]]), np.array(["a"]), n_permutations=10)


class TestSparsePairedCosineSimilarity:
    def test_matches_the_dense_function_on_sparse_rows(self) -> None:
        import scipy.sparse as sp

        from library.retrieval_metrics import sparse_paired_cosine_similarity

        dense_a = np.array([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
        dense_b = np.array([[1.0, 1.0, 0.0], [0.0, 1.0, 1.0]])

        result = sparse_paired_cosine_similarity(sp.csr_matrix(dense_a), sp.csr_matrix(dense_b))

        assert np.allclose(result, paired_cosine_similarity(dense_a, dense_b), rtol=0, atol=1e-9)

    def test_compares_only_row_i_against_row_i(self) -> None:
        import scipy.sparse as sp

        from library.retrieval_metrics import sparse_paired_cosine_similarity

        a = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
        b = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))

        result = sparse_paired_cosine_similarity(a, b)

        assert result.shape == (2,)
        assert np.allclose(result, [1.0, 0.0], atol=1e-9)

    def test_rejects_a_zero_row_rather_than_dividing_by_zero(self) -> None:
        import scipy.sparse as sp

        from library.retrieval_metrics import sparse_paired_cosine_similarity

        a = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, 0.0]]))
        b = sp.csr_matrix(np.array([[1.0, 0.0], [1.0, 0.0]]))

        with pytest.raises(DegenerateVectorError):
            sparse_paired_cosine_similarity(a, b)
