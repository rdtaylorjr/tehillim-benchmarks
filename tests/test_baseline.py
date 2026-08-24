from parallelism.baseline import build_unmarked_bicola


def test_build_unmarked_bicola_pairs_adjacent_unmarked_nodes() -> None:
    cola_by_psalm = {1: [10, 11, 12, 13]}

    pairs = build_unmarked_bicola(cola_by_psalm, marked_nodes=set())

    assert pairs == [(10, 11), (11, 12), (12, 13)]


def test_build_unmarked_bicola_excludes_a_pair_touching_a_marked_node() -> None:
    """A pair is dropped if EITHER side was ever a source or target of a true parallel pair,
    keeping the negative-control set cleanly disjoint from the real ground truth.
    """
    cola_by_psalm = {1: [10, 11, 12, 13]}

    pairs = build_unmarked_bicola(cola_by_psalm, marked_nodes={11})

    assert pairs == [(12, 13)]  # (10,11) and (11,12) both touch node 11


def test_build_unmarked_bicola_never_crosses_a_psalm_boundary() -> None:
    cola_by_psalm = {1: [10, 11], 2: [20, 21]}

    pairs = build_unmarked_bicola(cola_by_psalm, marked_nodes=set())

    assert (11, 20) not in pairs
    assert pairs == [(10, 11), (20, 21)]


def test_build_unmarked_bicola_handles_a_single_colon_psalm() -> None:
    cola_by_psalm = {1: [10]}

    pairs = build_unmarked_bicola(cola_by_psalm, marked_nodes=set())

    assert pairs == []
