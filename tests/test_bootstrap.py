import numpy as np
import pytest

from library.calibration import BackgroundStats
from parallelism.bootstrap import (
    _jackknife_ap_gap_and_auc,
    _psalm_indices,
    block_bootstrap_ap_gap_and_auc,
)
from parallelism.scripts.compare_baseline import as_node_pairs


def test_block_bootstrap_gap_and_auc_ci_contains_the_point_estimate() -> None:
    """The unresampled point estimate should fall inside its own bootstrap CI."""
    true_pairs = as_node_pairs([(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12)])
    baseline_pairs = as_node_pairs([(101, 102), (103, 104), (105, 106), (107, 108)])
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.99, 0.14]),
        3: np.array([1.0, 0.0]),
        4: np.array([0.98, 0.2]),
        5: np.array([1.0, 0.0]),
        6: np.array([0.99, 0.1]),
        7: np.array([1.0, 0.0]),
        8: np.array([0.97, 0.24]),
        9: np.array([1.0, 0.0]),
        10: np.array([0.99, 0.12]),
        11: np.array([1.0, 0.0]),
        12: np.array([0.98, 0.18]),
        101: np.array([1.0, 0.0]),
        102: np.array([0.0, 1.0]),
        103: np.array([1.0, 0.0]),
        104: np.array([0.1, 0.99]),
        105: np.array([1.0, 0.0]),
        106: np.array([-0.1, 0.99]),
        107: np.array([1.0, 0.0]),
        108: np.array([0.2, 0.98]),
    }
    node_to_psalm = {
        1: 1,
        2: 1,
        3: 2,
        4: 2,
        5: 3,
        6: 3,
        7: 4,
        8: 4,
        9: 5,
        10: 5,
        11: 6,
        12: 6,
        101: 1,
        102: 1,
        103: 2,
        104: 2,
        105: 3,
        106: 3,
        107: 4,
        108: 4,
    }
    background = BackgroundStats(mean=0.3, std=0.3, n_vectors=20)
    rng = np.random.default_rng(0)

    result = block_bootstrap_ap_gap_and_auc(
        true_pairs,
        baseline_pairs,
        node_vectors,
        background,
        node_to_psalm,
        n_resamples=200,
        rng=rng,
    )

    assert result.ap_ci_low <= result.point_ap <= result.ap_ci_high
    assert result.gap_ci_low <= result.point_gap <= result.gap_ci_high
    assert result.auc_ci_low <= result.point_auc <= result.auc_ci_high
    assert result.n_valid_resamples > 0
    assert result.prevalence == pytest.approx(6 / 10)


def test_block_bootstrap_resamples_whole_psalms_not_individual_pairs() -> None:
    """A single psalm always resamples to itself, so the CI collapses to the observed statistic."""
    true_pairs = as_node_pairs([(1, 2), (3, 4)])
    baseline_pairs = as_node_pairs([(5, 6), (7, 8)])
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([1.0, 0.0]),
        4: np.array([0.9, 0.1]),
        5: np.array([1.0, 0.0]),
        6: np.array([0.0, 1.0]),
        7: np.array([1.0, 0.0]),
        8: np.array([0.0, -1.0]),
    }
    node_to_psalm = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1}
    background = BackgroundStats(mean=0.3, std=0.3, n_vectors=10)
    rng = np.random.default_rng(0)

    result = block_bootstrap_ap_gap_and_auc(
        true_pairs,
        baseline_pairs,
        node_vectors,
        background,
        node_to_psalm,
        n_resamples=50,
        rng=rng,
    )

    assert result.gap_ci_low == pytest.approx(result.gap_ci_high, abs=1e-9)


def test_block_bootstrap_returns_nan_ci_when_every_resample_is_too_small_to_test() -> None:
    """A single baseline pair never reaches n>=2, so every resample is skipped without crashing."""
    true_pairs = as_node_pairs([(1, 2), (3, 4)])
    baseline_pairs = as_node_pairs([(5, 6)])
    node_vectors = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.9, 0.1]),
        3: np.array([1.0, 0.0]),
        4: np.array([0.9, 0.1]),
        5: np.array([1.0, 0.0]),
        6: np.array([0.0, 1.0]),
    }
    node_to_psalm = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1}
    background = BackgroundStats(mean=0.3, std=0.3, n_vectors=10)
    rng = np.random.default_rng(0)

    result = block_bootstrap_ap_gap_and_auc(
        true_pairs,
        baseline_pairs,
        node_vectors,
        background,
        node_to_psalm,
        n_resamples=20,
        rng=rng,
    )

    assert result.n_valid_resamples == 0
    assert np.isnan(result.gap_ci_low)
    assert np.isnan(result.gap_ci_high)


def test_jackknife_ap_gap_and_auc_returns_nan_when_removing_a_psalm_leaves_too_few_pairs() -> None:
    true_sims = np.array([0.9, 0.8, 0.5])
    true_psalms = np.array([1, 1, 2])
    baseline_sims = np.array([0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6])
    baseline_psalms = np.array([1, 1, 2, 2, 2, 2, 2])
    psalms = np.array([1, 2])
    true_idx_by_psalm = _psalm_indices(true_psalms, psalms)
    baseline_idx_by_psalm = _psalm_indices(baseline_psalms, psalms)
    background = BackgroundStats(mean=0.3, std=0.2, n_vectors=10)

    aps, gaps, aucs = _jackknife_ap_gap_and_auc(
        true_sims, true_idx_by_psalm, baseline_sims, baseline_idx_by_psalm, psalms, background
    )

    # removing psalm 1 leaves only 1 true pair (psalm 2's) -> NaN
    assert np.isnan(aps[0])
    assert np.isnan(gaps[0])
    assert np.isnan(aucs[0])
    # removing psalm 2 leaves 2 true + 2 baseline pairs (psalm 1's) -> a real value
    assert not np.isnan(aps[1])
    assert not np.isnan(gaps[1])
    assert not np.isnan(aucs[1])
