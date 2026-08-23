"""Psalm-clustered BCa bootstrap 95% CIs for AP (primary), gap, and AUC, every model x scope."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from library.bhsa import (
    DEFAULT_CHECKOUT,
    list_psalms_cola_by_psalm,
    list_psalms_colon_nodes,
    node_to_psalm_map,
)
from library.calibration import background_similarity_stats
from library.embeddings import dataset_identifier, load_embeddings
from library.incremental_cache import load_cached_rows
from parallelism.baseline import build_unmarked_bicola
from parallelism.bootstrap import BootstrapCI, block_bootstrap_ap_gap_and_auc
from parallelism.pairs import RetrievalPair, build_retrieval_pairs, filter_pairs_by_type
from parallelism.scripts.compare_baseline import (
    NodePairs,
    as_node_pairs,
    filter_node_pairs_with_vectors,
)
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups

_TYPES = ("Antithetic", "Emblematic", "Staircase", "Synonymous", "Synthetic")


def _as_node_pairs(pairs: list[RetrievalPair]) -> NodePairs:
    return [(p.source_nodes, p.target_nodes) for p in pairs]


def _row(model: str, scope: str, result: BootstrapCI) -> dict[str, str | int | float]:
    return {
        "model": model,
        "scope": scope,
        "prevalence": result.prevalence,
        "point_ap": result.point_ap,
        "ap_ci_low": result.ap_ci_low,
        "ap_ci_high": result.ap_ci_high,
        "ap_ci_low_pct": result.ap_ci_low_pct,
        "ap_ci_high_pct": result.ap_ci_high_pct,
        "point_gap": result.point_gap,
        "gap_ci_low": result.gap_ci_low,
        "gap_ci_high": result.gap_ci_high,
        "gap_ci_low_pct": result.gap_ci_low_pct,
        "gap_ci_high_pct": result.gap_ci_high_pct,
        "point_auc": result.point_auc,
        "auc_ci_low": result.auc_ci_low,
        "auc_ci_high": result.auc_ci_high,
        "auc_ci_low_pct": result.auc_ci_low_pct,
        "auc_ci_high_pct": result.auc_ci_high_pct,
        "n_valid_resamples": result.n_valid_resamples,
        "n_valid_jackknife": result.n_valid_jackknife,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA/module checkout spec")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_api(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    all_pairs = build_retrieval_pairs(groups)

    marked_nodes = {n for p in all_pairs for n in p.source_nodes} | {
        n for p in all_pairs for n in p.target_nodes
    }
    cola_by_psalm = list_psalms_cola_by_psalm(api)
    baseline_pairs = as_node_pairs(build_unmarked_bicola(cola_by_psalm, marked_nodes))
    background_node_ids = [n for n in list_psalms_colon_nodes(api) if n not in marked_nodes]
    node_to_psalm = node_to_psalm_map(cola_by_psalm)

    scopes: dict[str, NodePairs] = {"overall": _as_node_pairs(all_pairs)}
    for ptype in _TYPES:
        scopes[ptype] = _as_node_pairs(filter_pairs_by_type(all_pairs, frozenset({ptype})))

    rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = sorted(p for p in args.embeddings_dir.glob("**/*.parquet") if p.is_file())
    for path in model_paths:
        model = dataset_identifier(path)
        if model in cached_models:
            continue
        print(f"processing {model}")
        node_vectors = load_embeddings(path)
        background_vecs = np.stack(
            [node_vectors[n] for n in background_node_ids if n in node_vectors]
        )
        background = background_similarity_stats(background_vecs)
        rng = np.random.default_rng(args.seed)

        model_baseline_pairs = filter_node_pairs_with_vectors(baseline_pairs, node_vectors)
        for scope_name, true_pairs in scopes.items():
            model_true_pairs = filter_node_pairs_with_vectors(true_pairs, node_vectors)
            result = block_bootstrap_ap_gap_and_auc(
                model_true_pairs,
                model_baseline_pairs,
                node_vectors,
                background,
                node_to_psalm,
                n_resamples=args.n_resamples,
                rng=rng,
            )
            rows.append(_row(model, scope_name, result))

    for row in rows:
        print(
            f"{row['model']:55s} {row['scope']:12s} "
            f"AP={row['point_ap']:.3f} [{row['ap_ci_low']:.3f}, {row['ap_ci_high']:.3f}] "
            f"(chance={row['prevalence']:.3f}) "
            f"auc={row['point_auc']:.3f} gap={row['point_gap']:.3f}"
        )

    if args.output:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
