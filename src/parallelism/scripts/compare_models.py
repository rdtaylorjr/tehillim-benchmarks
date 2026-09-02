"""Runs run_evaluation across every embedding file in a directory, ranked by separation AUC."""

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import cast

import numpy as np

from library.bhsa import DEFAULT_CHECKOUT
from library.embeddings import dataset_identifier
from library.incremental_cache import load_cached_rows
from library.model_files import uncached_model_paths
from library.parallel_models import DEFAULT_MAX_WORKERS, map_in_order
from library.rows_output import write_rows_csv
from parallelism.evaluate import score_embedding_file
from parallelism.pairs import RetrievalPair, build_retrieval_pairs
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


def score_model(
    path: Path,
    pairs: list[RetrievalPair],
    n_permutations: int,
    seed: int,
) -> dict[str, str | int | float]:
    """One model file's metric row; each model is scored independently of every other."""
    model = dataset_identifier(path)
    rng = np.random.default_rng(seed)
    _, report = score_embedding_file(path, pairs, n_permutations=n_permutations, rng=rng)
    row: dict[str, str | int | float] = {
        "model": model,
        "n_pairs": report.n_pairs,
        "separation_auc": report.separation.auc,
        "separation_p": report.separation.p_value,
        "discrimination_p": report.discrimination.p_value,
        "discrimination_rank_biserial": report.discrimination.rank_biserial,
        "type_gap_z": report.type_gap.z_score,
        "type_gap_p": report.type_gap.p_value,
        "mrr_forward": report.mrr_forward,
        "mrr_backward": report.mrr_backward,
        "recall_at_1_forward": report.recall_at_1_forward,
        "recall_at_5_forward": report.recall_at_5_forward,
        "recall_at_10_forward": report.recall_at_10_forward,
        "recall_at_1_backward": report.recall_at_1_backward,
        "recall_at_5_backward": report.recall_at_5_backward,
        "recall_at_10_backward": report.recall_at_10_backward,
    }
    for type_report in report.by_type:
        suffix = type_report.parallelism_type
        row[f"n_pairs_{suffix}"] = type_report.n_pairs
        row[f"separation_auc_{suffix}"] = type_report.separation.auc
        row[f"separation_p_{suffix}"] = type_report.separation.p_value
        row[f"discrimination_p_{suffix}"] = type_report.discrimination.p_value
        row[f"discrimination_rank_biserial_{suffix}"] = type_report.discrimination.rank_biserial
        row[f"mrr_forward_{suffix}"] = type_report.mrr_forward
        row[f"mrr_backward_{suffix}"] = type_report.mrr_backward
        row[f"recall_at_1_forward_{suffix}"] = type_report.recall_at_1_forward
        row[f"recall_at_5_forward_{suffix}"] = type_report.recall_at_5_forward
        row[f"recall_at_1_backward_{suffix}"] = type_report.recall_at_1_backward
        row[f"recall_at_5_backward_{suffix}"] = type_report.recall_at_5_backward
    return row


def compare_models(
    pairs: list[RetrievalPair],
    model_paths: list[Path],
    n_permutations: int = 2000,
    seed: int = 0,
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[dict[str, str | int | float]]:
    """Scores every model file across workers, rows sorted by separation AUC descending."""
    score = partial(score_model, pairs=pairs, n_permutations=n_permutations, seed=seed)
    rows = map_in_order(score, model_paths, max_workers)
    rows.sort(key=lambda r: cast("float", r["separation_auc"]), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA/module checkout spec")
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_api(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    pairs = build_retrieval_pairs(groups)

    cached_rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)

    new_rows = compare_models(
        pairs,
        model_paths,
        n_permutations=args.n_permutations,
        seed=args.seed,
        max_workers=args.workers,
    )
    rows = sorted(cached_rows + new_rows, key=lambda r: r["separation_auc"], reverse=True)

    for row in rows:
        print(
            f"{row['model']:55s} auc={row['separation_auc']:.4f} "
            f"discrimination_r={row['discrimination_rank_biserial']:.4f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
