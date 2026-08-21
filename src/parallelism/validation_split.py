"""Splits psalms into two independent random halves, to check an effect replicates out-of-sample."""

import numpy as np

from parallelism.scripts.compare_baseline import NodePairs


def split_psalms_in_half(
    psalm_numbers: list[int], rng: np.random.Generator
) -> tuple[set[int], set[int]]:
    """Randomly partitions the given psalm numbers into two disjoint, near-equal-size sets."""
    shuffled = np.array(psalm_numbers)
    rng.shuffle(shuffled)
    midpoint = len(shuffled) // 2
    return set(shuffled[:midpoint].tolist()), set(shuffled[midpoint:].tolist())


def filter_node_pairs_by_psalm(
    pairs: NodePairs, node_to_psalm: dict[int, int], allowed_psalms: set[int]
) -> NodePairs:
    """Keeps only pairs whose source side belongs to a psalm in allowed_psalms."""
    return [pair for pair in pairs if node_to_psalm[pair[0][0]] in allowed_psalms]
