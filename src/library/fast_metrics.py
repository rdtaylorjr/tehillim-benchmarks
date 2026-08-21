"""Vectorized AP/AUC verified equivalent to sklearn/scipy; use only inside a resampling loop."""

import numpy as np
from scipy.stats import rankdata


def fast_auc(true_sims: np.ndarray, baseline_sims: np.ndarray) -> float:
    """AUC via the Mann-Whitney U identity (Hanley & McNeil 1982): U / (n_true * n_baseline)."""
    combined = np.concatenate([true_sims, baseline_sims])
    ranks = rankdata(combined, method="average")
    n_true = len(true_sims)
    n_baseline = len(baseline_sims)
    rank_sum = ranks[:n_true].sum()
    u_statistic = rank_sum - n_true * (n_true + 1) / 2
    return float(u_statistic / (n_true * n_baseline))


def fast_average_precision(true_sims: np.ndarray, baseline_sims: np.ndarray) -> float:
    """Average precision as the area under the step PR curve, matching sklearn tie handling."""
    scores = np.concatenate([true_sims, baseline_sims])
    labels = np.concatenate([np.ones(len(true_sims)), np.zeros(len(baseline_sims))])
    order = np.argsort(-scores, kind="stable")
    scores_sorted = scores[order]
    labels_sorted = labels[order]

    cumulative_true_positives = np.cumsum(labels_sorted)
    position = np.arange(1, len(scores_sorted) + 1)
    is_last_at_threshold = np.r_[np.diff(scores_sorted) != 0, True]

    tp_at_threshold = cumulative_true_positives[is_last_at_threshold]
    n_at_threshold = position[is_last_at_threshold]
    precision = tp_at_threshold / n_at_threshold

    n_positives = labels.sum()
    tp_before = np.r_[0.0, tp_at_threshold[:-1]]
    recall_jump = (tp_at_threshold - tp_before) / n_positives
    return float(np.sum(recall_jump * precision))
