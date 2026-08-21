"""Vertex-resampling BCa bootstrap CI (Efron 1987) for genre AP, gap, and AUC."""

from dataclasses import dataclass

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from library.bootstrap_ci import bca_ci
from library.calibration import BackgroundStats, calibrated_effect_size
from library.fast_metrics import fast_auc, fast_average_precision
from library.retrieval_metrics import cosine_similarity_matrix


@dataclass(frozen=True, slots=True)
class GenreBootstrapCI:
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


def build_similarity_and_genre_matrices(
    psalm_ids: list[int], psalm_vectors: dict[int, np.ndarray], genre_by_psalm: dict[int, str]
) -> tuple[np.ndarray, np.ndarray]:
    """N x N cosine similarity and genre-match matrices, ordered to match psalm_ids."""
    vectors = np.stack([psalm_vectors[p] for p in psalm_ids])
    similarity_matrix = cosine_similarity_matrix(vectors, vectors)
    genres = np.array([genre_by_psalm[p] for p in psalm_ids])
    genre_match_matrix = genres[:, None] == genres[None, :]
    return similarity_matrix, genre_match_matrix


def _upper_triangle_same_and_different(
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    population_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Splits the strict upper triangle into same/different sims, restricted to population_mask."""
    rows, cols = np.triu_indices(similarity_matrix.shape[0], k=1)
    sims = similarity_matrix[rows, cols]
    same = genre_match_matrix[rows, cols]
    if population_mask is not None:
        keep = population_mask[rows, cols]
        sims = sims[keep]
        same = same[keep]
    return sims[same], sims[~same]


def _gap(same_sims: np.ndarray, different_sims: np.ndarray, background: BackgroundStats) -> float:
    same_effect = calibrated_effect_size(float(same_sims.mean()), background)
    different_effect = calibrated_effect_size(float(different_sims.mean()), background)
    return same_effect - different_effect


def _point_ap_gap_and_auc(
    same_sims: np.ndarray, different_sims: np.ndarray, background: BackgroundStats
) -> tuple[float, float, float]:
    """Point estimate via official sklearn/scipy; resampling and jackknife use the fast path."""
    labels = np.concatenate([np.ones(len(same_sims)), np.zeros(len(different_sims))])
    scores = np.concatenate([same_sims, different_sims])
    ap = float(average_precision_score(labels, scores))
    statistic, _ = mannwhitneyu(same_sims, different_sims, alternative="greater")
    auc = float(statistic / (len(same_sims) * len(different_sims)))
    return ap, _gap(same_sims, different_sims, background), auc


def _resample_ap_gap_and_auc(
    same_sims: np.ndarray, different_sims: np.ndarray, background: BackgroundStats
) -> tuple[float, float, float]:
    ap = fast_average_precision(same_sims, different_sims)
    auc = fast_auc(same_sims, different_sims)
    return ap, _gap(same_sims, different_sims, background), auc


def _jackknife_ap_gap_and_auc(
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    background: BackgroundStats,
    population_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Leave-one-psalm-out AP/gap/AUC; NaN for a psalm whose removal leaves too few pairs."""
    n = similarity_matrix.shape[0]
    aps = np.full(n, np.nan)
    gaps = np.full(n, np.nan)
    aucs = np.full(n, np.nan)
    all_idx = np.arange(n)
    for i in range(n):
        keep = all_idx[all_idx != i]
        sim_j = similarity_matrix[np.ix_(keep, keep)]
        genre_j = genre_match_matrix[np.ix_(keep, keep)]
        population_j = population_mask[np.ix_(keep, keep)] if population_mask is not None else None
        same_j, different_j = _upper_triangle_same_and_different(sim_j, genre_j, population_j)
        if len(same_j) < 2 or len(different_j) < 2:
            continue
        aps[i], gaps[i], aucs[i] = _resample_ap_gap_and_auc(same_j, different_j, background)
    return aps, gaps, aucs


def block_bootstrap_genre_ap_gap_and_auc(
    psalm_ids: list[int],
    similarity_matrix: np.ndarray,
    genre_match_matrix: np.ndarray,
    background: BackgroundStats,
    n_resamples: int = 1000,
    rng: np.random.Generator | None = None,
    population_mask: np.ndarray | None = None,
) -> GenreBootstrapCI:
    """BCa 95% CI for AP (primary), gap, and AUC, resampling whole psalms with replacement.

    Genre pairs span every psalm on both sides, unlike parallelism's node-to-one-psalm pairs, so
    the natural generalization of the same block-bootstrap principle is a vertex bootstrap: resample
    the psalm population itself and reconstruct the pairwise similarity/genre-match structure from
    the resampled psalms, rather than resampling the derived pairs directly.

    population_mask restricts every resample/jackknife step to a one-vs-rest population (pairs
    touching a single target genre), instead of every pair, when set.
    """
    rng = rng if rng is not None else np.random.default_rng()
    n = len(psalm_ids)
    same_sims, different_sims = _upper_triangle_same_and_different(
        similarity_matrix, genre_match_matrix, population_mask
    )
    prevalence = len(same_sims) / (len(same_sims) + len(different_sims))
    point_ap, point_gap, point_auc = _point_ap_gap_and_auc(same_sims, different_sims, background)

    aps, gaps, aucs = [], [], []
    for _ in range(n_resamples):
        idx = rng.choice(n, size=n, replace=True)
        sim_r = similarity_matrix[np.ix_(idx, idx)]
        genre_r = genre_match_matrix[np.ix_(idx, idx)]
        population_r = population_mask[np.ix_(idx, idx)] if population_mask is not None else None
        same_r, different_r = _upper_triangle_same_and_different(sim_r, genre_r, population_r)
        if len(same_r) < 2 or len(different_r) < 2:
            continue
        ap, gap, auc = _resample_ap_gap_and_auc(same_r, different_r, background)
        aps.append(ap)
        gaps.append(gap)
        aucs.append(auc)

    if not aps:
        return GenreBootstrapCI(
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
        similarity_matrix, genre_match_matrix, background, population_mask
    )
    ap_low, ap_high = bca_ci(point_ap, ap_arr, jack_aps)
    gap_low, gap_high = bca_ci(point_gap, gap_arr, jack_gaps)
    auc_low, auc_high = bca_ci(point_auc, auc_arr, jack_aucs)

    return GenreBootstrapCI(
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
