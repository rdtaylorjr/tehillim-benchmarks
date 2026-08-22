import numpy as np
import pytest
import scipy.sparse as sp

from library.retrieval_metrics import (
    _combine_by_stratum,
    _combine_by_stratum_batch,
    _per_anchor_gap,
    _per_anchor_gap_batch,
    cosine_similarity_matrix,
    mean_reciprocal_rank,
    outranking_candidates,
    paired_bootstrap_mrr_diff,
    paired_cosine_similarity,
    paired_discrimination_test,
    ranks_from_similarity_matrix,
    recall_at_k,
    retrieval_ranks,
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


def test_sparse_cosine_similarity_matrix_matches_the_dense_function_exactly() -> None:
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


def test_retrieval_ranks_gives_rank_one_when_true_target_is_most_similar() -> None:
    anchors = np.array([[1.0, 0.0]])
    pool = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    pool_ids = ["true", "b", "c"]
    ranks = retrieval_ranks(anchors, pool, pool_ids, true_target_ids=["true"])
    assert ranks == [1.0]


def test_retrieval_ranks_gives_worse_rank_when_true_target_is_least_similar() -> None:
    anchors = np.array([[1.0, 0.0]])
    pool = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.0]])
    pool_ids = ["b", "true", "c"]
    ranks = retrieval_ranks(anchors, pool, pool_ids, true_target_ids=["true"])
    assert ranks == [2.0]


def test_retrieval_ranks_averages_tied_similarities() -> None:
    anchors = np.array([[1.0, 0.0]])
    pool = np.array([[1.0, 0.0], [1.0, 0.0]])
    pool_ids = ["true", "tied"]
    ranks = retrieval_ranks(anchors, pool, pool_ids, true_target_ids=["true"])
    assert ranks == [1.5]


def test_retrieval_ranks_matches_a_naive_per_row_rankdata_loop_at_scale() -> None:
    from scipy.stats import rankdata

    rng = np.random.default_rng(21)
    n = 40
    anchors = rng.normal(size=(n, 5))
    pool = rng.normal(size=(n, 5))
    pool_ids = [f"p{i}" for i in range(n)]
    true_target_ids = [pool_ids[i] for i in rng.permutation(n)]

    naive = []
    similarities = cosine_similarity_matrix(anchors, pool)
    pool_index = {pid: i for i, pid in enumerate(pool_ids)}
    for row, true_id in zip(similarities, true_target_ids, strict=True):
        rank_positions = rankdata(-row, method="average")
        naive.append(float(rank_positions[pool_index[true_id]]))

    vectorized = retrieval_ranks(anchors, pool, pool_ids, true_target_ids)

    assert vectorized == naive


def test_ranks_from_similarity_matrix_matches_retrieval_ranks_exactly() -> None:
    """retrieval_ranks must be exactly reproducible by computing the matrix once and reusing it."""
    anchors = np.array([[1.0, 0.0], [0.9, 0.1], [-1.0, 0.3]])
    pool = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    pool_ids = ["a", "b", "c"]
    true_target_ids = ["b", "a", "c"]

    from_scratch = retrieval_ranks(anchors, pool, pool_ids, true_target_ids)
    similarities = cosine_similarity_matrix(anchors, pool)
    from_matrix = ranks_from_similarity_matrix(similarities, pool_ids, true_target_ids)

    assert from_matrix == from_scratch


def test_ranks_from_similarity_matrix_supports_the_transposed_backward_direction() -> None:
    """The backward direction's ranks equal ranks_from_similarity_matrix on the transpose."""
    source = np.array([[1.0, 0.0], [0.2, 0.9]])
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    ids = ["s0", "s1"]

    forward_from_scratch = retrieval_ranks(source, target, ids, true_target_ids=ids)
    backward_from_scratch = retrieval_ranks(target, source, ids, true_target_ids=ids)

    similarities = cosine_similarity_matrix(source, target)
    forward = ranks_from_similarity_matrix(similarities, ids, true_target_ids=ids)
    backward = ranks_from_similarity_matrix(similarities.T, ids, true_target_ids=ids)

    assert forward == forward_from_scratch
    assert backward == backward_from_scratch


def test_outranking_candidates_is_empty_when_true_target_ranks_first() -> None:
    similarities = np.array([[0.9, 0.1, 0.2]])
    result = outranking_candidates(similarities, pool_ids=["a", "b", "c"], true_target_ids=["a"])
    assert result == [[]]


def test_outranking_candidates_lists_ids_tied_or_ahead_of_the_true_target() -> None:
    similarities = np.array([[0.5, 0.9, 0.5, 0.1]])
    result = outranking_candidates(
        similarities, pool_ids=["true", "beats", "ties", "loses"], true_target_ids=["true"]
    )
    assert result == [["beats", "ties"]]


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
    """A permuted "true" column must not let the anchor's own diagonal leak into its null mean.

    Row 0's permuted true column is 1, not its own diagonal (0). The null average for row 0
    must exclude both column 0 (the real diagonal, similarity 10) and column 1 (this draw's
    true column, similarity 1), leaving only column 2 (similarity 2) as the null.
    """
    matrix = np.array([[10.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])
    true_positions = np.array([1, 1, 2])

    gaps = _per_anchor_gap(matrix, true_positions)

    assert gaps == pytest.approx([1.0 - 2.0, 4.0 - 4.0, 8.0 - 6.5])


def test_per_anchor_gap_falls_back_gracefully_when_n_is_too_small_to_exclude_both() -> None:
    """n=2 leaves no third column, so excluding both the diagonal and a distinct fake-true is
    impossible; this must fall back to excluding only the fake-true rather than producing NaN.
    """
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
        naive = np.array([_combine_by_stratum(row, stratum, weighted) for row in gaps_batch])
        batched = _combine_by_stratum_batch(gaps_batch, stratum, weighted)
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
    """Some anchors are generically similar to everything (embedding-space anisotropy/"hubness").

    Comparing each anchor's true similarity only against its own row's other candidates (as
    implemented) cancels this out. Pooling all anchors' candidates together before comparing
    true vs. false, as an earlier design of this test did, would not: a hub anchor's ordinary
    diagonal entry looks unusually high against the *global* pool of mostly low-similarity
    anchors, producing a spurious gap with no real parallelism signal behind it.
    """
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


def test_paired_bootstrap_mrr_diff_is_zero_for_identical_models() -> None:
    rng = np.random.default_rng(3)
    ranks = [1, 2, 3, 1, 5]

    result = paired_bootstrap_mrr_diff(ranks, ranks, n_resamples=500, rng=rng)

    assert result.observed_diff == 0.0
    assert result.p_value == 1.0


def test_paired_bootstrap_mrr_diff_detects_a_clearly_better_model() -> None:
    rng = np.random.default_rng(4)
    ranks_a = [1] * 20
    ranks_b = [20] * 20

    result = paired_bootstrap_mrr_diff(ranks_a, ranks_b, n_resamples=1000, rng=rng)

    assert result.observed_diff > 0
    assert result.ci_low > 0
    assert result.p_value < 0.01


def test_paired_bootstrap_mrr_diff_with_clusters_is_zero_for_identical_models() -> None:
    rng = np.random.default_rng(7)
    ranks = [1, 2, 3, 1, 5, 2]
    clusters = [10, 10, 11, 11, 12, 12]

    result = paired_bootstrap_mrr_diff(ranks, ranks, n_resamples=500, rng=rng, clusters=clusters)

    assert result.observed_diff == 0.0
    assert result.p_value == 1.0


def test_paired_bootstrap_mrr_diff_clustering_widens_the_confidence_interval() -> None:
    """4 clusters of 5 near-duplicate items each is really ~4 independent units, not 20.

    Plain (unclustered) resampling treats all 20 as independent and understates uncertainty;
    resampling whole clusters should produce a visibly wider interval for the same data.
    """
    ranks_a = np.array([1, 1, 1, 1, 1, 5, 5, 5, 5, 5, 10, 10, 10, 10, 10, 15, 15, 15, 15, 15])
    ranks_b = np.full(20, 20)
    clusters = np.repeat([0, 1, 2, 3], 5)

    unclustered = paired_bootstrap_mrr_diff(
        ranks_a, ranks_b, n_resamples=3000, rng=np.random.default_rng(8)
    )
    clustered = paired_bootstrap_mrr_diff(
        ranks_a, ranks_b, n_resamples=3000, rng=np.random.default_rng(8), clusters=clusters
    )

    unclustered_width = unclustered.ci_high - unclustered.ci_low
    clustered_width = clustered.ci_high - clustered.ci_low
    assert clustered_width > unclustered_width * 1.5
