import numpy as np

from parallelism.pairs import RetrievalPair
from parallelism.scripts.compare_true_similarity import compare_true_similarity


def _pair(pair_id: str, source_nodes: tuple, target_nodes: tuple, ptype: str) -> RetrievalPair:
    return RetrievalPair(
        pair_id=pair_id,
        group_range="g",
        parallelism_type=ptype,
        signature="AB",
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_indicator="A",
        target_indicator="B",
    )


def test_compare_true_similarity_ranks_by_calibrated_effect_size_descending() -> None:
    """Equal raw similarity (1.0), but 'spread' has a more diverse background, so it ranks first."""
    pairs = [_pair("p1", (1,), (2,), "Synonymous")]
    background_ids = [1, 2, 5, 6, 7, 8]

    peaky_model = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        5: np.array([0.999, 0.045]),
        6: np.array([0.998, -0.063]),
        7: np.array([0.999, 0.032]),
        8: np.array([0.997, 0.077]),
    }
    spread_model = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        5: np.array([0.0, 1.0]),
        6: np.array([-1.0, 0.0]),
        7: np.array([0.0, -1.0]),
        8: np.array([0.7071, 0.7071]),
    }

    rows = compare_true_similarity(
        pairs, {"peaky": peaky_model, "spread": spread_model}, background_ids
    )

    assert [r["model"] for r in rows] == ["spread", "peaky"]
    assert rows[0]["mean_true_similarity"] == rows[1]["mean_true_similarity"] == 1.0
    assert rows[0]["calibrated_effect_size"] > rows[1]["calibrated_effect_size"]


def test_compare_true_similarity_flattens_per_type_effect_sizes() -> None:
    pairs = [
        _pair("p1", (1,), (2,), "Synonymous"),
        _pair("p2", (3,), (4,), "Antithetic"),
    ]
    background_ids = [1, 2, 3, 4, 5, 6]
    model = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([1.0, 0.0]),
        4: np.array([1.0, 0.0]),
        5: np.array([0.0, 1.0]),
        6: np.array([-1.0, 0.0]),
    }

    rows = compare_true_similarity(pairs, {"model": model}, background_ids)

    assert "calibrated_effect_size_Synonymous" in rows[0]
    assert "calibrated_effect_size_Antithetic" in rows[0]
