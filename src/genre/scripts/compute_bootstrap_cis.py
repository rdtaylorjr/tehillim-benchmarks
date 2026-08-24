"""Psalm vertex-resampling BCa bootstrap 95% CIs for AP (primary), gap, and AUC, every model."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from genre.bootstrap import (
    GenreBootstrapCI,
    block_bootstrap_genre_ap_gap_and_auc,
    build_similarity_and_genre_matrices,
)
from genre.genre_labels import load_genre_by_psalm
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_cola_by_psalm, load_bhsa_api
from library.calibration import background_stats_from_matrix
from library.centroid import psalm_centroids
from library.embeddings import dataset_identifier, load_embeddings
from library.incremental_cache import load_cached_rows


def _row(model: str, result: GenreBootstrapCI) -> dict[str, str | int | float]:
    return {
        "model": model,
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
    parser.add_argument(
        "genre_csv",
        type=Path,
        help="third-party genre CSV, e.g. psalms-browser.csv (not in this repo)",
    )
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_bhsa_api(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    cola_by_psalm = list_psalms_cola_by_psalm(api)

    model_paths = sorted(p for p in args.embeddings_dir.glob("**/*.parquet") if p.is_file())
    rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    for path in model_paths:
        model = dataset_identifier(path)
        if model in cached_models:
            continue
        print(f"processing {model}")
        node_vectors = load_embeddings(path)
        psalm_vectors = psalm_centroids(cola_by_psalm, node_vectors)
        psalm_ids = sorted(psalm_vectors)

        similarity_matrix, genre_match_matrix = build_similarity_and_genre_matrices(
            psalm_ids, psalm_vectors, genre_by_psalm
        )
        background = background_stats_from_matrix(similarity_matrix)

        rng = np.random.default_rng(args.seed)
        try:
            result = block_bootstrap_genre_ap_gap_and_auc(
                psalm_ids,
                similarity_matrix,
                genre_match_matrix,
                background,
                n_resamples=args.n_resamples,
                rng=rng,
            )
        except ValueError as error:
            print(
                f"skipping {model}: {error} (only {len(psalm_ids)} psalm vectors)", file=sys.stderr
            )
            continue
        rows.append(_row(model, result))

    for row in rows:
        print(
            f"{row['model']:55s} "
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
