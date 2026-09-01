"""Domain-agnostic order-shuffle-null control: real score vs. a within-unit order-permuted null."""

import math
import warnings
from dataclasses import dataclass

import numpy as np

DEFAULT_N_SHUFFLES = 1000


@dataclass(frozen=True, slots=True)
class OrderShuffleResult:
    delta_order: float
    p_value: float


def minimum_shuffles_for_fdr(n_hypotheses: int, alpha: float = 0.05) -> int:
    """Fewest shuffles whose smallest attainable p, 1/(n+1), can reach BH q <= alpha alone."""
    return math.ceil(n_hypotheses / alpha) - 1


def order_shuffle_result(
    real_score: float,
    shuffled_scores: np.ndarray,
    n_hypotheses: int = 1,
    alpha: float = 0.05,
) -> OrderShuffleResult:
    """Real score minus mean shuffled score, and a rank-based permutation p-value against null."""
    n_shuffles = len(shuffled_scores)
    required = minimum_shuffles_for_fdr(n_hypotheses, alpha)
    if n_shuffles < required:
        warnings.warn(
            f"{n_shuffles} shuffles cannot reach BH q <= {alpha} across {n_hypotheses} "
            f"hypotheses, which needs at least {required}. Any null result is resolution-limited.",
            RuntimeWarning,
            stacklevel=2,
        )
    delta_order = real_score - float(shuffled_scores.mean())
    exceed_count = int(np.sum(shuffled_scores >= real_score))
    p_value = (exceed_count + 1) / (n_shuffles + 1)
    return OrderShuffleResult(delta_order=delta_order, p_value=p_value)
