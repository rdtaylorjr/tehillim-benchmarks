import numpy as np

from parallelism.pairs import RetrievalPair
from parallelism.scripts.compare_models import compare_models


def _pair(
    pair_id: str, source_nodes: tuple[int, ...], target_nodes: tuple[int, ...]
) -> RetrievalPair:
    return RetrievalPair(
        pair_id=pair_id,
        group_range="g",
        parallelism_type="Synonymous",
        signature="AB",
        source_nodes=source_nodes,
        target_nodes=target_nodes,
        source_indicator="A",
        target_indicator="B",
    )


def test_compare_models_ranks_by_separation_auc_descending() -> None:
    pairs = [_pair("p1", (1,), (2,)), _pair("p2", (3,), (4,))]
    good_model = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.0, 1.0]),
    }
    bad_model = {
        1: np.array([1.0, 0.0]),
        2: np.array([0.0, 1.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([1.0, 0.0]),
    }

    rows = compare_models(pairs, {"bad": bad_model, "good": good_model}, n_permutations=50, seed=0)

    assert [r["model"] for r in rows] == ["good", "bad"]
    assert rows[0]["separation_auc"] == 1.0
    assert rows[1]["separation_auc"] == 0.0


def test_compare_models_flattens_per_type_metrics() -> None:
    pairs = [
        _pair("p1", (1,), (2,)),
        RetrievalPair(
            pair_id="p2",
            group_range="g2",
            parallelism_type="Antithetic",
            signature="AB",
            source_nodes=(3,),
            target_nodes=(4,),
            source_indicator="A",
            target_indicator="B",
        ),
    ]
    model = {
        1: np.array([1.0, 0.0]),
        2: np.array([1.0, 0.0]),
        3: np.array([0.0, 1.0]),
        4: np.array([0.0, 1.0]),
    }

    rows = compare_models(pairs, {"model": model}, n_permutations=50, seed=0)

    assert rows[0]["n_pairs_Synonymous"] == 1
    assert rows[0]["n_pairs_Antithetic"] == 1
    assert rows[0]["mrr_forward_Synonymous"] == 1.0
    assert rows[0]["mrr_forward_Antithetic"] == 1.0
    assert "discrimination_rank_biserial_Synonymous" in rows[0]
    assert "discrimination_rank_biserial_Antithetic" in rows[0]
