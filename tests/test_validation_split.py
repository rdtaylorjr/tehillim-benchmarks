import numpy as np

from parallelism.scripts.compare_baseline import as_node_pairs
from parallelism.validation_split import filter_node_pairs_by_psalm, split_psalms_in_half


def test_split_psalms_in_half_partitions_every_psalm_exactly_once() -> None:
    psalms = list(range(1, 151))
    rng = np.random.default_rng(0)

    fold_a, fold_b = split_psalms_in_half(psalms, rng)

    assert fold_a | fold_b == set(psalms)
    assert fold_a & fold_b == set()
    assert abs(len(fold_a) - len(fold_b)) <= 1


def test_split_psalms_in_half_is_reproducible_with_the_same_seeded_rng() -> None:
    psalms = list(range(1, 151))

    fold_a1, fold_b1 = split_psalms_in_half(psalms, np.random.default_rng(42))
    fold_a2, fold_b2 = split_psalms_in_half(psalms, np.random.default_rng(42))

    assert fold_a1 == fold_a2
    assert fold_b1 == fold_b2


def test_filter_node_pairs_by_psalm_keeps_only_pairs_in_the_allowed_set() -> None:
    pairs = as_node_pairs([(1, 2), (3, 4), (5, 6)])
    node_to_psalm = {1: 10, 2: 10, 3: 20, 4: 20, 5: 30, 6: 30}

    kept = filter_node_pairs_by_psalm(pairs, node_to_psalm, {10, 30})

    assert kept == as_node_pairs([(1, 2), (5, 6)])
