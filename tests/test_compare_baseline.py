import numpy as np
import pytest

from library.calibration import BackgroundStats
from parallelism.scripts.compare_baseline import (
    as_node_pairs,
    compare_to_baseline,
    filter_node_pairs_with_vectors,
)


def test_filter_node_pairs_with_vectors_keeps_a_pair_whose_nodes_are_all_present() -> None:
    pairs = as_node_pairs([(1, 2), (3, 4)])
    node_vectors = {1: object(), 2: object(), 3: object(), 4: object()}

    assert filter_node_pairs_with_vectors(pairs, node_vectors) == pairs


def test_filter_node_pairs_with_vectors_drops_a_pair_missing_either_side() -> None:
    pairs = as_node_pairs([(1, 2), (3, 4)])
    node_vectors = {1: object(), 2: object(), 3: object()}  # node 4 excluded

    kept = filter_node_pairs_with_vectors(pairs, node_vectors)

    assert kept == as_node_pairs([(1, 2)])


def test_filter_node_pairs_with_vectors_handles_multi_node_spans() -> None:
    pairs = [((1, 2), (3,)), ((4,), (5, 6))]
    node_vectors = {1: object(), 2: object(), 3: object(), 4: object(), 5: object()}  # 6 excluded

    kept = filter_node_pairs_with_vectors(pairs, node_vectors)

    assert kept == [((1, 2), (3,))]


def test_compare_to_baseline_reports_higher_effect_size_when_true_pairs_are_closer() -> None:
    true_pairs = as_node_pairs([(1, 2), (3, 4)])
    baseline_pairs = as_node_pairs([(5, 6), (7, 8), (9, 10)])
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),  # identical: similarity 1.0
        3: np.array([1.0, 0.0]),
        4: np.array([0.99, 0.14107]),  # nearly identical: high similarity
        5: np.array([1.0, 0.0]),
        6: np.array([0.0, 1.0]),  # orthogonal: similarity 0.0
        7: np.array([1.0, 0.0]),
        8: np.array([0.0, 1.0]),
        9: np.array([1.0, 0.0]),
        10: np.array([0.0, 1.0]),
    }
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=10)

    result = compare_to_baseline(true_pairs, baseline_pairs, node_vectors, background)

    assert result.n_true == 2
    assert result.n_baseline == 3
    assert result.mean_true_similarity > result.mean_baseline_similarity
    assert result.true_effect_size > result.baseline_effect_size
    assert result.separation_auc == 1.0
    assert result.separation_p < 0.1  # n=2 vs n=3 caps Mann-Whitney's minimum p just above 0.05
    # perfect separation (every true pair scores above every baseline pair) gives AP = 1.0,
    # the MTEB Pair Classification convention (Muennighoff et al. 2023)
    assert result.average_precision == 1.0
    assert result.prevalence == 2 / 5


def test_compare_to_baseline_mean_pools_multi_node_spans() -> None:
    """A member spanning two half-verse nodes must be pooled, not silently dropped."""
    true_pairs = as_node_pairs([(1, 2)])
    true_pairs_multi = [((1, 10), (2,))]  # source spans nodes 1 and 10
    baseline_pairs = as_node_pairs([(5, 6)])
    node_vectors = {
        1: np.array([1.0, 0.0]),
        10: np.array([1.0, 0.0]),  # same direction as node 1, so pooling changes nothing here
        2: np.array([1.0, 0.0]),
        5: np.array([1.0, 0.0]),
        6: np.array([0.0, 1.0]),
    }
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=10)

    single = compare_to_baseline(true_pairs, baseline_pairs, node_vectors, background)
    multi = compare_to_baseline(true_pairs_multi, baseline_pairs, node_vectors, background)

    assert multi.n_true == 1
    assert multi.mean_true_similarity == single.mean_true_similarity


def test_as_node_pairs_wraps_each_side_as_a_one_tuple() -> None:
    assert as_node_pairs([(1, 2), (3, 4)]) == [((1,), (2,)), ((3,), (4,))]


def test_compare_to_baseline_average_precision_chance_level_is_prevalence_not_half() -> None:
    """With zero discrimination (every pair, true or baseline, has identical similarity), AP
    equals the positive-class prevalence, not 0.5; unlike AUC, AP's chance level depends on the
    true:baseline ratio, so raw AP is only comparable across models sharing the same ratio, not
    across scopes (e.g. types) with different amounts of imbalance.
    """
    true_pairs = as_node_pairs([(1, 2)])
    baseline_pairs = as_node_pairs([(3, 4), (5, 6), (7, 8)])
    node_vectors = {n: np.array([1.0, 0.0]) for n in range(1, 9)}
    background = BackgroundStats(mean=0.5, std=0.2, n_vectors=10)

    result = compare_to_baseline(true_pairs, baseline_pairs, node_vectors, background)

    assert result.prevalence == pytest.approx(1 / 4)
    assert result.average_precision == pytest.approx(1 / 4)
