"""Scores a tehillim-embeddings Parquet vector file against the parallelism benchmark."""

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from library.bhsa import DEFAULT_CHECKOUT
from library.embeddings import load_embeddings
from library.retrieval_metrics import (
    DiscriminationResult,
    PermutationResult,
    cosine_similarity_matrix,
    mean_reciprocal_rank,
    paired_discrimination_test,
    recall_at_k,
    retrieval_ranks,
    stratified_mean_gap_test,
)
from parallelism.pairs import RetrievalPair, build_retrieval_pairs
from parallelism.separation import SeparationResult, similarity_separation
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


def build_side_vectors(
    pairs: list[RetrievalPair], side: str, node_vectors: dict[int, np.ndarray]
) -> np.ndarray:
    """Mean-pools each pair's source/target node tuple into one vector."""
    node_lists = [getattr(p, f"{side}_nodes") for p in pairs]
    missing = sorted({n for nodes in node_lists for n in nodes} - set(node_vectors))
    if missing:
        shown = missing[:10]
        suffix = "..." if len(missing) > 10 else ""
        raise ValueError(f"embedding file is missing {len(missing)} node id(s): {shown}{suffix}")
    group_ids = np.repeat(np.arange(len(node_lists)), [len(nodes) for nodes in node_lists])
    flat_nodes = [n for nodes in node_lists for n in nodes]
    flat_vectors = np.stack([node_vectors[n] for n in flat_nodes])
    sums = np.zeros((len(node_lists), flat_vectors.shape[1]))
    np.add.at(sums, group_ids, flat_vectors)
    counts = np.bincount(group_ids, minlength=len(node_lists))
    return sums / counts[:, None]


@dataclass
class TypeReport:
    parallelism_type: str
    n_pairs: int
    separation: SeparationResult
    discrimination: DiscriminationResult
    mrr_forward: float
    mrr_backward: float
    recall_at_1_forward: float
    recall_at_5_forward: float
    recall_at_1_backward: float
    recall_at_5_backward: float


@dataclass
class EvaluationReport:
    n_pairs: int
    separation: SeparationResult
    discrimination: DiscriminationResult
    type_gap: PermutationResult
    by_type: list[TypeReport]
    mrr_forward: float
    mrr_backward: float
    recall_at_1_forward: float
    recall_at_5_forward: float
    recall_at_10_forward: float
    recall_at_1_backward: float
    recall_at_5_backward: float
    recall_at_10_backward: float


def run_evaluation(
    pairs: list[RetrievalPair],
    node_vectors: dict[int, np.ndarray],
    n_permutations: int = 10000,
    rng: np.random.Generator | None = None,
) -> EvaluationReport:
    """Runs bidirectional retrieval, discrimination, and type-stratified tests for one model."""
    rng = rng if rng is not None else np.random.default_rng()
    pair_ids = [p.pair_id for p in pairs]
    source_vecs = build_side_vectors(pairs, "source", node_vectors)
    target_vecs = build_side_vectors(pairs, "target", node_vectors)

    ranks_forward = retrieval_ranks(source_vecs, target_vecs, pair_ids, true_target_ids=pair_ids)
    ranks_backward = retrieval_ranks(target_vecs, source_vecs, pair_ids, true_target_ids=pair_ids)

    similarities = cosine_similarity_matrix(source_vecs, target_vecs)
    n = len(pair_ids)
    true_similarities = np.diag(similarities)
    off_diagonal = ~np.eye(n, dtype=bool)
    null_similarities = np.array([similarities[i, off_diagonal[i]].mean() for i in range(n)])
    discrimination = paired_discrimination_test(true_similarities, null_similarities)
    separation = similarity_separation(similarities)

    types = np.array([p.parallelism_type for p in pairs])
    type_gap = stratified_mean_gap_test(similarities, types, n_permutations=n_permutations, rng=rng)

    ranks_forward_arr = np.asarray(ranks_forward)
    ranks_backward_arr = np.asarray(ranks_backward)
    by_type = []
    for ptype in sorted(set(types.tolist())):
        idx = np.flatnonzero(types == ptype)
        by_type.append(
            TypeReport(
                parallelism_type=ptype,
                n_pairs=int(len(idx)),
                separation=similarity_separation(similarities, row_mask=types == ptype),
                discrimination=paired_discrimination_test(
                    true_similarities[idx], null_similarities[idx]
                ),
                mrr_forward=mean_reciprocal_rank(ranks_forward_arr[idx]),
                mrr_backward=mean_reciprocal_rank(ranks_backward_arr[idx]),
                recall_at_1_forward=recall_at_k(ranks_forward_arr[idx], 1),
                recall_at_5_forward=recall_at_k(ranks_forward_arr[idx], 5),
                recall_at_1_backward=recall_at_k(ranks_backward_arr[idx], 1),
                recall_at_5_backward=recall_at_k(ranks_backward_arr[idx], 5),
            )
        )

    return EvaluationReport(
        n_pairs=n,
        separation=separation,
        discrimination=discrimination,
        type_gap=type_gap,
        by_type=by_type,
        mrr_forward=mean_reciprocal_rank(ranks_forward),
        mrr_backward=mean_reciprocal_rank(ranks_backward),
        recall_at_1_forward=recall_at_k(ranks_forward, 1),
        recall_at_5_forward=recall_at_k(ranks_forward, 5),
        recall_at_10_forward=recall_at_k(ranks_forward, 10),
        recall_at_1_backward=recall_at_k(ranks_backward, 1),
        recall_at_5_backward=recall_at_k(ranks_backward, 5),
        recall_at_10_backward=recall_at_k(ranks_backward, 10),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embedding_file", type=Path, help="a tehillim-embeddings Parquet file")
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA/module checkout spec")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    api = load_api(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    pairs = build_retrieval_pairs(groups)
    node_vectors = load_embeddings(args.embedding_file)

    rng = np.random.default_rng(args.seed)
    report = run_evaluation(pairs, node_vectors, n_permutations=args.n_permutations, rng=rng)
    print(json.dumps(asdict(report), indent=2))


if __name__ == "__main__":
    main()
