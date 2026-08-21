"""Psalm-clustered BCa bootstrap CI (Efron 1987) for AP, AUC, and the effect-size gap."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from library.bootstrap_ci import bca_ci
from library.calibration import BackgroundStats, calibrated_effect_size
from library.fast_metrics import fast_auc, fast_average_precision
from parallelism.scripts.compare_baseline import NodePairs, pair_similarities


@dataclass(frozen=True, slots=True)
class BootstrapCI:
    point_ap: float
    ap_ci_low: float
    ap_ci_high: float
    ap_ci_low_pct: float
    ap_ci_high_pct: float
    point_gap: float
    gap_ci_low: float
    gap_ci_high: float
    gap_ci_low_pct: float
    gap_ci_high_pct: float
    point_auc: float
    auc_ci_low: float
    auc_ci_high: float
    auc_ci_low_pct: float
    auc_ci_high_pct: float
    prevalence: float
    n_valid_resamples: int
    n_valid_jackknife: int


def _gap(true_sims: np.ndarray, baseline_sims: np.ndarray, background: BackgroundStats) -> float:
    return calibrated_effect_size(float(true_sims.mean()), background) - calibrated_effect_size(
        float(baseline_sims.mean()), background
    )


def _point_ap_gap_and_auc(
    true_sims: np.ndarray, baseline_sims: np.ndarray, background: BackgroundStats
) -> tuple[float, float, float]:
    """Point estimate via official sklearn/scipy; resampling and jackknife use the fast path."""
    labels = np.concatenate([np.ones(len(true_sims)), np.zeros(len(baseline_sims))])
    scores = np.concatenate([true_sims, baseline_sims])
    ap = float(average_precision_score(labels, scores))

    statistic, _ = mannwhitneyu(true_sims, baseline_sims, alternative="greater")
    auc = float(statistic / (len(true_sims) * len(baseline_sims)))
    return ap, _gap(true_sims, baseline_sims, background), auc


def _resample_ap_gap_and_auc(
    true_sims: np.ndarray, baseline_sims: np.ndarray, background: BackgroundStats
) -> tuple[float, float, float]:
    ap = fast_average_precision(true_sims, baseline_sims)
    auc = fast_auc(true_sims, baseline_sims)
    return ap, _gap(true_sims, baseline_sims, background), auc


def _psalm_indices(pair_psalms: np.ndarray, psalms: np.ndarray) -> dict[int, np.ndarray]:
    return {int(psalm): np.flatnonzero(pair_psalms == psalm) for psalm in psalms}


def _jackknife_ap_gap_and_auc(
    true_sims: np.ndarray,
    true_idx_by_psalm: dict[int, np.ndarray],
    baseline_sims: np.ndarray,
    baseline_idx_by_psalm: dict[int, np.ndarray],
    psalms: np.ndarray,
    background: BackgroundStats,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-psalm-out AP/gap/AUC; NaN for a psalm whose removal leaves too few pairs."""
    aps = np.full(len(psalms), np.nan)
    gaps = np.full(len(psalms), np.nan)
    aucs = np.full(len(psalms), np.nan)
    all_true_idx = np.arange(len(true_sims))
    all_baseline_idx = np.arange(len(baseline_sims))
    for i, psalm in enumerate(psalms):
        true_idx = np.setdiff1d(all_true_idx, true_idx_by_psalm[int(psalm)], assume_unique=True)
        baseline_idx = np.setdiff1d(
            all_baseline_idx, baseline_idx_by_psalm[int(psalm)], assume_unique=True
        )
        if len(true_idx) < 2 or len(baseline_idx) < 2:
            continue
        aps[i], gaps[i], aucs[i] = _resample_ap_gap_and_auc(
            true_sims[true_idx], baseline_sims[baseline_idx], background
        )
    return aps, gaps, aucs


def block_bootstrap_ap_gap_and_auc(
    true_pairs: NodePairs,
    baseline_pairs: NodePairs,
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
    node_to_psalm: dict[int, int],
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
) -> BootstrapCI:
    """BCa 95% CI for AP (primary), gap, and AUC, resampling whole psalms."""
    rng = rng if rng is not None else np.random.default_rng()
    true_sims = pair_similarities(true_pairs, node_vectors)
    baseline_sims = pair_similarities(baseline_pairs, node_vectors)
    true_psalms = np.array([node_to_psalm[pair[0][0]] for pair in true_pairs])
    baseline_psalms = np.array([node_to_psalm[pair[0][0]] for pair in baseline_pairs])
    prevalence = len(true_sims) / (len(true_sims) + len(baseline_sims))

    point_ap, point_gap, point_auc = _point_ap_gap_and_auc(true_sims, baseline_sims, background)

    psalms = np.array(sorted(set(true_psalms.tolist()) | set(baseline_psalms.tolist())))
    true_idx_by_psalm = _psalm_indices(true_psalms, psalms)
    baseline_idx_by_psalm = _psalm_indices(baseline_psalms, psalms)

    aps, gaps, aucs = [], [], []
    for _ in range(n_resamples):
        chosen = rng.choice(psalms, size=len(psalms), replace=True)
        true_idx = np.concatenate([true_idx_by_psalm[p] for p in chosen])
        baseline_idx = np.concatenate([baseline_idx_by_psalm[p] for p in chosen])
        if len(true_idx) < 2 or len(baseline_idx) < 2:
            continue
        ap, gap, auc = _resample_ap_gap_and_auc(
            true_sims[true_idx], baseline_sims[baseline_idx], background
        )
        aps.append(ap)
        gaps.append(gap)
        aucs.append(auc)

    if not aps:
        return BootstrapCI(
            point_ap=point_ap,
            ap_ci_low=float("nan"),
            ap_ci_high=float("nan"),
            ap_ci_low_pct=float("nan"),
            ap_ci_high_pct=float("nan"),
            point_gap=point_gap,
            gap_ci_low=float("nan"),
            gap_ci_high=float("nan"),
            gap_ci_low_pct=float("nan"),
            gap_ci_high_pct=float("nan"),
            point_auc=point_auc,
            auc_ci_low=float("nan"),
            auc_ci_high=float("nan"),
            auc_ci_low_pct=float("nan"),
            auc_ci_high_pct=float("nan"),
            prevalence=prevalence,
            n_valid_resamples=0,
            n_valid_jackknife=0,
        )

    ap_arr, gap_arr, auc_arr = np.array(aps), np.array(gaps), np.array(aucs)
    ap_low_pct, ap_high_pct = np.percentile(ap_arr, [2.5, 97.5])
    gap_low_pct, gap_high_pct = np.percentile(gap_arr, [2.5, 97.5])
    auc_low_pct, auc_high_pct = np.percentile(auc_arr, [2.5, 97.5])

    jack_aps, jack_gaps, jack_aucs = _jackknife_ap_gap_and_auc(
        true_sims, true_idx_by_psalm, baseline_sims, baseline_idx_by_psalm, psalms, background
    )
    ap_low, ap_high = bca_ci(point_ap, ap_arr, jack_aps)
    gap_low, gap_high = bca_ci(point_gap, gap_arr, jack_gaps)
    auc_low, auc_high = bca_ci(point_auc, auc_arr, jack_aucs)

    return BootstrapCI(
        point_ap=point_ap,
        ap_ci_low=ap_low,
        ap_ci_high=ap_high,
        ap_ci_low_pct=float(ap_low_pct),
        ap_ci_high_pct=float(ap_high_pct),
        point_gap=point_gap,
        gap_ci_low=gap_low,
        gap_ci_high=gap_high,
        gap_ci_low_pct=float(gap_low_pct),
        gap_ci_high_pct=float(gap_high_pct),
        point_auc=point_auc,
        auc_ci_low=auc_low,
        auc_ci_high=auc_high,
        auc_ci_low_pct=float(auc_low_pct),
        auc_ci_high_pct=float(auc_high_pct),
        prevalence=prevalence,
        n_valid_resamples=len(aps),
        n_valid_jackknife=int(np.sum(~np.isnan(jack_aps))),
    )
