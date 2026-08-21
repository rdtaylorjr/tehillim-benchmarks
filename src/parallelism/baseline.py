"""Builds a negative-control set of adjacent bicola never annotated as parallel."""


def build_unmarked_bicola(
    half_verses_by_psalm: dict[int, list[int]], marked_nodes: set[int]
) -> list[tuple[int, int]]:
    """Adjacent half-verse pairs, within one psalm, where neither node is in marked_nodes."""
    pairs = []
    for nodes in half_verses_by_psalm.values():
        for a, b in zip(nodes, nodes[1:], strict=False):
            if a not in marked_nodes and b not in marked_nodes:
                pairs.append((a, b))
    return pairs
