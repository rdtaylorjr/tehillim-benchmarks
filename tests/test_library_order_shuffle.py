import warnings

import numpy as np
import pytest

from library.order_shuffle import (
    DEFAULT_N_SHUFFLES,
    minimum_shuffles_for_fdr,
    order_shuffle_result,
)


@pytest.mark.filterwarnings("ignore::RuntimeWarning")
class TestOrderShuffleResult:
    def test_delta_order_is_real_score_minus_mean_shuffled_score(self) -> None:
        result = order_shuffle_result(real_score=0.5, shuffled_scores=np.array([0.2, 0.3, 0.4]))

        assert result.delta_order == 0.5 - 0.3

    def test_p_value_counts_shuffled_draws_at_or_above_the_real_value(self) -> None:
        # none of the 3 shuffled draws reach 0.5, so p = (0 + 1) / (3 + 1)
        result = order_shuffle_result(real_score=0.5, shuffled_scores=np.array([0.1, 0.2, 0.3]))

        assert result.p_value == 1 / 4

    def test_p_value_counts_a_tie_as_at_or_above(self) -> None:
        # one of 4 shuffled draws ties the real value, so p = (1 + 1) / (4 + 1)
        result = order_shuffle_result(
            real_score=0.5, shuffled_scores=np.array([0.1, 0.2, 0.3, 0.5])
        )

        assert result.p_value == 2 / 5

    def test_p_value_is_never_exactly_zero(self) -> None:
        result = order_shuffle_result(real_score=1.0, shuffled_scores=np.zeros(30))

        assert result.p_value == 1 / 31

    def test_matches_a_naive_python_loop(self) -> None:
        rng = np.random.default_rng(0)
        shuffled = rng.random(30)
        real_score = 0.9

        result = order_shuffle_result(real_score=real_score, shuffled_scores=shuffled)

        naive_count = sum(1 for v in shuffled if v >= real_score)
        naive_p = (naive_count + 1) / (len(shuffled) + 1)
        assert result.p_value == naive_p


class TestMinimumShufflesForFdr:
    def test_seven_hypotheses_need_139_shuffles_at_alpha_0_05(self) -> None:
        assert minimum_shuffles_for_fdr(7) == 139

    def test_a_single_hypothesis_needs_19_shuffles_at_alpha_0_05(self) -> None:
        assert minimum_shuffles_for_fdr(1) == 19

    def test_the_returned_count_is_the_smallest_one_that_clears_the_bh_threshold(self) -> None:
        for n_hypotheses in (1, 2, 3, 5, 7, 11):
            required = minimum_shuffles_for_fdr(n_hypotheses)

            assert 1 / (required + 1) <= 0.05 / n_hypotheses
            assert 1 / required > 0.05 / n_hypotheses

    def test_a_stricter_alpha_needs_more_shuffles(self) -> None:
        assert minimum_shuffles_for_fdr(7, alpha=0.01) > minimum_shuffles_for_fdr(7, alpha=0.05)


class TestResolutionGuard:
    def test_the_default_shuffle_count_clears_bh_across_seven_hypotheses(self) -> None:
        assert minimum_shuffles_for_fdr(7) <= DEFAULT_N_SHUFFLES

    def test_thirty_shuffles_across_seven_hypotheses_warn(self) -> None:
        with pytest.warns(RuntimeWarning, match="resolution-limited"):
            order_shuffle_result(real_score=1.0, shuffled_scores=np.zeros(30), n_hypotheses=7)

    def test_the_warning_names_the_required_count(self) -> None:
        with pytest.warns(RuntimeWarning, match="at least 139"):
            order_shuffle_result(real_score=1.0, shuffled_scores=np.zeros(30), n_hypotheses=7)

    def test_enough_shuffles_across_seven_hypotheses_do_not_warn(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = order_shuffle_result(
                real_score=1.0, shuffled_scores=np.zeros(139), n_hypotheses=7
            )

        assert caught == []
        assert result.p_value == 1 / 140

    def test_thirty_shuffles_do_not_warn_for_a_single_hypothesis(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            order_shuffle_result(real_score=1.0, shuffled_scores=np.zeros(30))

        assert caught == []
