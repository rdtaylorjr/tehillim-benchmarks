import numpy as np

from parallelism.true_similarity import summarize_true_pair_similarity


def test_summarize_true_pair_similarity_reports_mean_median_std() -> None:
    similarities = np.array([0.2, 0.4, 0.6, 0.8])

    summary = summarize_true_pair_similarity(similarities)

    assert summary.mean == 0.5
    assert summary.median == 0.5
    assert summary.n == 4
    assert summary.std > 0
