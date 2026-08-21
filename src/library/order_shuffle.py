"""Domain-agnostic order-shuffle-null control: real score vs. a within-unit order-permuted null."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class OrderShuffleResult:
    delta_order: float
    p_value: float


def order_shuffle_result(real_score: float, shuffled_scores: np.ndarray) -> OrderShuffleResult:
    """Real score minus mean shuffled score, and a rank-based permutation p-value against null."""
    delta_order = real_score - float(shuffled_scores.mean())
    exceed_count = int(np.sum(shuffled_scores >= real_score))
    p_value = (exceed_count + 1) / (len(shuffled_scores) + 1)
    return OrderShuffleResult(delta_order=delta_order, p_value=p_value)
