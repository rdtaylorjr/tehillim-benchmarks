"""Compares true-parallelism-pair similarity against adjacent bicola never marked as parallel."""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from library.bhsa import (
    DEFAULT_CHECKOUT,
    list_psalms_cola_by_psalm,
    list_psalms_colon_nodes,
)
from library.calibration import BackgroundStats, background_similarity_stats, calibrated_effect_size
from library.embeddings import dataset_identifier, load_embeddings
from library.incremental_cache import load_cached_rows
from library.retrieval_metrics import paired_cosine_similarity
from parallelism.baseline import build_unmarked_bicola
from parallelism.pairs import build_retrieval_pairs
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups

NodePairs = list[tuple[tuple[int, ...], tuple[int, ...]]]


def _pool_side(
    node_tuples: list[tuple[int, ...]], node_vectors: dict[int, np.ndarray]
) -> np.ndarray:
    """Stacks one vector per tuple: direct lookup when single-node, mean pool otherwise."""
    if all(len(nodes) == 1 for nodes in node_tuples):
        return np.stack([node_vectors[nodes[0]] for nodes in node_tuples])
    return np.stack([np.mean([node_vectors[n] for n in nodes], axis=0) for nodes in node_tuples])


def pair_similarities(pairs: NodePairs, node_vectors: dict[int, np.ndarray]) -> np.ndarray:
    """Row-wise cosine similarity, mean-pooling any side that spans more than one node."""
    source_vecs = _pool_side([source for source, _ in pairs], node_vectors)
    target_vecs = _pool_side([target for _, target in pairs], node_vectors)
    return paired_cosine_similarity(source_vecs, target_vecs)


def as_node_pairs(single_node_pairs: list[tuple[int, int]]) -> NodePairs:
    """Wraps plain (int, int) pairs (e.g. adjacent bicola) into the uniform NodePairs shape."""
    return [((a,), (b,)) for a, b in single_node_pairs]


def filter_node_pairs_with_vectors(
    pairs: NodePairs, node_vectors: dict[int, np.ndarray]
) -> NodePairs:
    """NodePairs whose source and target nodes are all present in node_vectors."""
    return [
        (source, target)
        for source, target in pairs
        if all(n in node_vectors for n in source + target)
    ]


@dataclass(frozen=True, slots=True)
class BaselineComparison:
    n_true: int
    n_baseline: int
    prevalence: float
    mean_true_similarity: float
    mean_baseline_similarity: float
    true_effect_size: float
    baseline_effect_size: float
    average_precision: float
    separation_auc: float
    separation_p: float


def compare_to_baseline(
    true_pairs: NodePairs,
    baseline_pairs: NodePairs,
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> BaselineComparison:
    """True-pair vs baseline similarity: Average Precision is primary, AUC/effect size secondary."""
    true_sims = pair_similarities(true_pairs, node_vectors)
    baseline_sims = pair_similarities(baseline_pairs, node_vectors)
    statistic, p_value = mannwhitneyu(true_sims, baseline_sims, alternative="greater")
    auc = statistic / (len(true_sims) * len(baseline_sims))
    mean_true = float(true_sims.mean())
    mean_baseline = float(baseline_sims.mean())

    labels = np.concatenate([np.ones(len(true_sims)), np.zeros(len(baseline_sims))])
    scores = np.concatenate([true_sims, baseline_sims])
    ap = average_precision_score(labels, scores)

    return BaselineComparison(
        n_true=len(true_sims),
        n_baseline=len(baseline_sims),
        prevalence=len(true_sims) / (len(true_sims) + len(baseline_sims)),
        mean_true_similarity=mean_true,
        mean_baseline_similarity=mean_baseline,
        true_effect_size=calibrated_effect_size(mean_true, background),
        baseline_effect_size=calibrated_effect_size(mean_baseline, background),
        average_precision=float(ap),
        separation_auc=float(auc),
        separation_p=float(p_value),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA/module checkout spec")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_api(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    retrieval_pairs = build_retrieval_pairs(groups)
    true_pairs: NodePairs = [(p.source_nodes, p.target_nodes) for p in retrieval_pairs]

    marked_nodes = {n for p in retrieval_pairs for n in p.source_nodes} | {
        n for p in retrieval_pairs for n in p.target_nodes
    }
    cola_by_psalm = list_psalms_cola_by_psalm(api)
    baseline_pairs = as_node_pairs(build_unmarked_bicola(cola_by_psalm, marked_nodes))
    background_node_ids = [n for n in list_psalms_colon_nodes(api) if n not in marked_nodes]

    rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = sorted(p for p in args.embeddings_dir.glob("**/*.parquet") if p.is_file())
    for path in model_paths:
        model = dataset_identifier(path)
        if model in cached_models:
            continue
        node_vectors = load_embeddings(path)
        background_vecs = np.stack(
            [node_vectors[n] for n in background_node_ids if n in node_vectors]
        )
        background = background_similarity_stats(background_vecs)
        model_true_pairs = filter_node_pairs_with_vectors(true_pairs, node_vectors)
        model_baseline_pairs = filter_node_pairs_with_vectors(baseline_pairs, node_vectors)
        result = compare_to_baseline(
            model_true_pairs, model_baseline_pairs, node_vectors, background
        )
        rows.append(
            {
                "model": model,
                "n_true": result.n_true,
                "n_baseline": result.n_baseline,
                "prevalence": result.prevalence,
                "average_precision": result.average_precision,
                "true_effect_size": result.true_effect_size,
                "baseline_effect_size": result.baseline_effect_size,
                "gap": result.true_effect_size - result.baseline_effect_size,
                "separation_auc": result.separation_auc,
                "separation_p": result.separation_p,
            }
        )
    rows.sort(key=lambda r: r["average_precision"], reverse=True)

    for row in rows:
        print(
            f"{row['model']:55s} AP={row['average_precision']:.3f} "
            f"(chance={row['prevalence']:.3f}) auc={row['separation_auc']:.3f} gap={row['gap']:.3f}"
        )

    if args.output:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
