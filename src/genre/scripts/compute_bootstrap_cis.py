"""Psalm vertex-resampling BCa bootstrap 95% CIs for AP (primary), gap, and AUC, every model."""

import argparse
import sys
from functools import partial
from pathlib import Path

import numpy as np

from genre.bootstrap import (
    block_bootstrap_genre_ap_gap_and_auc,
    build_similarity_and_genre_matrices,
)
from genre.genre_labels import load_genre_by_psalm
from library.ap_gap_auc_bootstrap import ci_row
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.calibration import background_stats_from_matrix
from library.embeddings import dataset_identifier
from library.errors import BenchmarkDataError
from library.incremental_cache import load_cached_rows
from library.model_files import uncached_model_paths
from library.parallel_models import DEFAULT_MAX_WORKERS, map_in_order
from library.psalm_vectors import load_psalm_vectors
from library.rows_output import write_rows_csv


def score_model(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    genre_by_psalm: dict[int, str],
    n_resamples: int,
    seed: int,
) -> dict[str, str | int | float] | None:
    """One model file's CI row, or None when its psalm population cannot support a CI."""
    model = dataset_identifier(path)
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    psalm_ids = sorted(psalm_vectors)
    similarity_matrix, genre_match_matrix = build_similarity_and_genre_matrices(
        psalm_ids, psalm_vectors, genre_by_psalm
    )
    background = background_stats_from_matrix(similarity_matrix)
    try:
        result = block_bootstrap_genre_ap_gap_and_auc(
            psalm_ids,
            similarity_matrix,
            genre_match_matrix,
            background,
            n_resamples=n_resamples,
            rng=np.random.default_rng(seed),
        )
    except BenchmarkDataError as error:
        print(f"skipping {model} ({len(psalm_ids)} psalm vectors): {error}", file=sys.stderr)
        return None
    return ci_row(model, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "genre_csv",
        type=Path,
        help="third-party genre CSV, e.g. psalms-browser.csv (not in this repo)",
    )
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_bhsa_api(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)
    score = partial(
        score_model,
        half_verses_by_psalm=half_verses_by_psalm,
        genre_by_psalm=genre_by_psalm,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    rows.extend(row for row in map_in_order(score, model_paths, args.workers) if row is not None)

    for row in rows:
        print(
            f"{row['model']:55s} "
            f"AP={row['point_ap']:.3f} [{row['ap_ci_low']:.3f}, {row['ap_ci_high']:.3f}] "
            f"(chance={row['prevalence']:.3f}) "
            f"auc={row['point_auc']:.3f} gap={row['point_gap']:.3f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
