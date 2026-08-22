import numpy as np
import scipy.sparse as sp

from parallelism.evaluate import (
    build_side_vectors,
    build_side_vectors_sparse,
    run_evaluation,
    run_evaluation_sparse,
)
from parallelism.pairs import RetrievalPair


def _pair(
    pair_id: str, source_nodes: tuple[int, ...], target_nodes: tuple[int, ...], ptype: str
) -> RetrievalPair:
    return RetrievalPair(
        pair_id=pair_id,
        group_range="g",
        parallelism_type=ptype,
        signature="AB",
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_indicator="A",
        target_indicator="B",
    )


def test_build_side_vectors_mean_pools_a_spanning_member() -> None:
    pairs = [_pair("p1", (1, 2), (3,), "Synonymous")]
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([3.0, 0.0]),
        3: np.array([0.0, 1.0]),
    }

    vectors = build_side_vectors(pairs, "source", node_vectors)

    assert np.allclose(vectors[0], [2.0, 0.0])


def test_build_side_vectors_matches_a_naive_per_pair_loop_with_ragged_group_sizes() -> None:
    rng = np.random.default_rng(41)
    dim = 4
    node_vectors = {n: rng.normal(size=dim) for n in range(1, 21)}
    pairs = [
        _pair("p1", (1, 2), (3,), "Synonymous"),
        _pair("p2", (4,), (5, 6, 7), "Antithetic"),
        _pair("p3", (8, 9, 10), (11,), "Synonymous"),
    ]

    naive = np.stack([np.mean([node_vectors[n] for n in p.source_nodes], axis=0) for p in pairs])
    vectorized = build_side_vectors(pairs, "source", node_vectors)

    assert np.allclose(vectorized, naive)


def test_build_side_vectors_raises_on_a_missing_node() -> None:
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    node_vectors = {1: np.array([1.0, 0.0])}

    try:
        build_side_vectors(pairs, "target", node_vectors)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_run_evaluation_reports_perfect_retrieval_for_identical_vectors() -> None:
    pairs = [
        _pair("p1", (1,), (2,), "Synonymous"),
        _pair("p2", (3,), (4,), "Synonymous"),
        _pair("p3", (5,), (6,), "Antithetic"),
    ]
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.0, 1.0]),
        5: np.array([1.0, 1.0]),
        6: np.array([1.0, 1.0]),
    }

    report = run_evaluation(pairs, node_vectors, n_permutations=100, rng=np.random.default_rng(0))

    assert report.n_pairs == 3
    assert report.separation.auc == 1.0
    assert report.mrr_forward == 1.0
    assert report.mrr_backward == 1.0
    assert report.recall_at_1_forward == 1.0
    assert {t.parallelism_type for t in report.by_type} == {"Synonymous", "Antithetic"}
    synonymous = next(t for t in report.by_type if t.parallelism_type == "Synonymous")
    assert synonymous.n_pairs == 2
    assert synonymous.separation.auc == 1.0
    assert synonymous.separation.n_positive == 2
    assert synonymous.mrr_forward == 1.0


def test_build_side_vectors_sparse_matches_the_dense_function_exactly() -> None:
    """Proves sparse pooling via matmul gives the identical dense mean-pooled result."""
    rng = np.random.default_rng(7)
    dim = 200
    node_ids = list(range(1, 21))
    dense_vectors = {}
    for n in node_ids:
        row = np.zeros(dim)
        n_nonzero = rng.integers(1, 6)
        idx = rng.choice(dim, size=n_nonzero, replace=False)
        row[idx] = rng.uniform(0.1, 5.0, size=n_nonzero)
        dense_vectors[n] = row
    pairs = [
        _pair("p1", (1, 2), (3,), "Synonymous"),
        _pair("p2", (4,), (5, 6, 7), "Antithetic"),
        _pair("p3", (8, 9, 10), (11,), "Synonymous"),
    ]
    sparse_matrix = sp.csr_matrix(np.stack([dense_vectors[n] for n in node_ids]))

    dense_result = build_side_vectors(pairs, "source", dense_vectors)
    sparse_result = build_side_vectors_sparse(pairs, "source", node_ids, sparse_matrix)

    np.testing.assert_allclose(sparse_result.toarray(), dense_result, rtol=1e-6)


def test_build_side_vectors_sparse_raises_on_a_missing_node() -> None:
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    node_ids = [1]
    matrix = sp.csr_matrix(np.array([[1.0, 0.0]]))

    try:
        build_side_vectors_sparse(pairs, "target", node_ids, matrix)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_run_evaluation_sparse_matches_run_evaluation_exactly() -> None:
    """Proves the sparse evaluation path reports identically to the dense one on the same data."""
    pairs = [
        _pair("p1", (1,), (2,), "Synonymous"),
        _pair("p2", (3,), (4,), "Synonymous"),
        _pair("p3", (5,), (6,), "Antithetic"),
    ]
    dense_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.0, 1.0]),
        5: np.array([1.0, 1.0]),
        6: np.array([1.0, 1.0]),
    }
    node_ids = sorted(dense_vectors)
    sparse_matrix = sp.csr_matrix(np.stack([dense_vectors[n] for n in node_ids]))

    dense_report = run_evaluation(
        pairs, dense_vectors, n_permutations=100, rng=np.random.default_rng(0)
    )
    sparse_report = run_evaluation_sparse(
        pairs, node_ids, sparse_matrix, n_permutations=100, rng=np.random.default_rng(0)
    )

    assert sparse_report == dense_report
