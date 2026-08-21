"""Scores same-genre vs different-genre psalm-pair similarity: AP (primary) and AUC (secondary)."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from genre.pairs import GenrePair
from library.retrieval_metrics import paired_cosine_similarity


@dataclass(frozen=True, slots=True)
class GenreEvaluationReport:
    n_same_genre: int
    n_different_genre: int
    prevalence: float
    average_precision: float
    separation_auc: float
    separation_p: float


def evaluate_genre_discrimination(
    pairs: list[GenrePair], psalm_vectors: dict[int, np.ndarray]
) -> GenreEvaluationReport:
    """MTEB Pair Classification protocol: AP ranks same-genre pairs above different-genre pairs."""
    usable = [p for p in pairs if p.psalm_a in psalm_vectors and p.psalm_b in psalm_vectors]
    a_vecs = np.stack([psalm_vectors[p.psalm_a] for p in usable])
    b_vecs = np.stack([psalm_vectors[p.psalm_b] for p in usable])
    similarities = paired_cosine_similarity(a_vecs, b_vecs)
    labels = np.array([p.same_genre for p in usable], dtype=int)

    same_sims = similarities[labels == 1]
    different_sims = similarities[labels == 0]

    ap = average_precision_score(labels, similarities)
    statistic, p_value = mannwhitneyu(same_sims, different_sims, alternative="greater")
    auc = statistic / (len(same_sims) * len(different_sims))

    return GenreEvaluationReport(
        n_same_genre=len(same_sims),
        n_different_genre=len(different_sims),
        prevalence=len(same_sims) / len(usable),
        average_precision=float(ap),
        separation_auc=float(auc),
        separation_p=float(p_value),
    )
