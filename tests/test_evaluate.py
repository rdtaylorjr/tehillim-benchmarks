import numpy as np

from parallelism.evaluate import build_side_vectors, run_evaluation
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
