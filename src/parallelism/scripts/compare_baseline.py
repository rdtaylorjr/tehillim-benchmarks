"""Compares true-pair similarity against adjacent half-verse pairs never marked as parallel."""

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from library.bhsa import (
    DEFAULT_CHECKOUT,
    list_psalms_half_verse_nodes,
    list_psalms_half_verses_by_psalm,
)
from library.calibration import (
    BackgroundStats,
    background_similarity_stats,
    background_similarity_stats_sparse,
    calibrated_effect_size,
)
from library.embeddings import (
    dataset_identifier,
    is_sparse_embeddings,
    load_embeddings,
    load_sparse_embeddings,
)
from library.incremental_cache import load_cached_rows
from library.model_files import uncached_model_paths
from library.parallel_models import DEFAULT_MAX_WORKERS, map_in_order
from library.rows_output import write_rows_csv
from parallelism.baseline import build_unmarked_half_verse_pairs
from parallelism.node_pairs import (
    NodePairs,
    as_node_pairs,
    filter_node_pairs_with_vectors,
    pair_similarities,
    pair_similarities_sparse,
    retrieval_pairs_as_node_pairs,
)
from parallelism.pairs import build_retrieval_pairs
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


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
    return compare_to_baseline_from_similarities(
        pair_similarities(true_pairs, node_vectors),
        pair_similarities(baseline_pairs, node_vectors),
        background,
    )


def compare_to_baseline_from_similarities(
    true_sims: np.ndarray, baseline_sims: np.ndarray, background: BackgroundStats
) -> BaselineComparison:
    """Same comparison as compare_to_baseline, from already-computed pair similarities."""
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


def score_model(
    path: Path,
    true_pairs: NodePairs,
    baseline_pairs: NodePairs,
    background_node_ids: list[int],
) -> dict[str, str | int | float]:
    """One model file's true-vs-baseline row; each model is scored independently of every other."""
    model = dataset_identifier(path)
    if is_sparse_embeddings(path):
        node_ids, matrix = load_sparse_embeddings(path)
        node_index = {n: i for i, n in enumerate(node_ids)}
        background_rows = [node_index[n] for n in background_node_ids if n in node_index]
        background = background_similarity_stats_sparse(matrix[background_rows])
        result = compare_to_baseline_from_similarities(
            pair_similarities_sparse(
                filter_node_pairs_with_vectors(true_pairs, node_index), node_ids, matrix
            ),
            pair_similarities_sparse(
                filter_node_pairs_with_vectors(baseline_pairs, node_index), node_ids, matrix
            ),
            background,
        )
    else:
        node_vectors = load_embeddings(path)
        background_vecs = np.stack(
            [node_vectors[n] for n in background_node_ids if n in node_vectors]
        )
        background = background_similarity_stats(background_vecs)
        result = compare_to_baseline(
            filter_node_pairs_with_vectors(true_pairs, node_vectors),
            filter_node_pairs_with_vectors(baseline_pairs, node_vectors),
            node_vectors,
            background,
        )
    return {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA/module checkout spec")
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_api(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    retrieval_pairs = build_retrieval_pairs(groups)
    true_pairs = retrieval_pairs_as_node_pairs(retrieval_pairs)

    marked_nodes = {n for p in retrieval_pairs for n in p.source_nodes} | {
        n for p in retrieval_pairs for n in p.target_nodes
    }
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    baseline_pairs = as_node_pairs(
        build_unmarked_half_verse_pairs(half_verses_by_psalm, marked_nodes)
    )
    background_node_ids = [n for n in list_psalms_half_verse_nodes(api) if n not in marked_nodes]

    rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)
    score = partial(
        score_model,
        true_pairs=true_pairs,
        baseline_pairs=baseline_pairs,
        background_node_ids=background_node_ids,
    )
    rows.extend(map_in_order(score, model_paths, args.workers))
    rows.sort(key=lambda r: r["average_precision"], reverse=True)

    for row in rows:
        print(
            f"{row['model']:55s} AP={row['average_precision']:.3f} "
            f"(chance={row['prevalence']:.3f}) auc={row['separation_auc']:.3f} gap={row['gap']:.3f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
