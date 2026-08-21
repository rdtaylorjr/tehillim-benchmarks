import numpy as np

from library.order_shuffle import order_shuffle_result


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
