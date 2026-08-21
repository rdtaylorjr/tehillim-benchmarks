"""Raw cosine similarity of true parallel cola, with no comparison to any other candidate."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TruePairSimilarity:
    mean: float
    median: float
    std: float
    n: int


def summarize_true_pair_similarity(similarities: np.ndarray) -> TruePairSimilarity:
    return TruePairSimilarity(
        mean=float(np.mean(similarities)),
        median=float(np.median(similarities)),
        std=float(np.std(similarities)),
        n=len(similarities),
    )
