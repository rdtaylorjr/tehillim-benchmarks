"""Runs evaluate_genre_discrimination across every embedding file in a directory."""

import argparse
import csv
import sys
from pathlib import Path
from typing import cast

import numpy as np

from genre.evaluate import evaluate_genre_discrimination
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.centroid import psalm_centroids
from library.embeddings import dataset_identifier, load_embeddings
from library.incremental_cache import load_cached_rows


def compare_genre_models(
    pairs: list[GenrePair], psalm_vectors_by_model: dict[str, dict[int, np.ndarray]]
) -> list[dict[str, str | int | float]]:
    """Evaluates every model's psalm centroids against the same genre pairs, sorted by AP."""
    rows: list[dict[str, str | int | float]] = []
    for model, psalm_vectors in psalm_vectors_by_model.items():
        report = evaluate_genre_discrimination(pairs, psalm_vectors)
        rows.append(
            {
                "model": model,
                "n_same_genre": report.n_same_genre,
                "n_different_genre": report.n_different_genre,
                "prevalence": report.prevalence,
                "average_precision": report.average_precision,
                "separation_auc": report.separation_auc,
                "separation_p": report.separation_p,
            }
        )
    rows.sort(key=lambda r: cast(float, r["average_precision"]), reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "genre_csv",
        type=Path,
        help="third-party genre CSV, e.g. psalms-browser.csv (not in this repo)",
    )
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_bhsa_api(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    cached_rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = sorted(p for p in args.embeddings_dir.glob("**/*.parquet") if p.is_file())
    psalm_vectors_by_model = {}
    for path in model_paths:
        model = dataset_identifier(path)
        if model in cached_models:
            continue
        node_vectors = load_embeddings(path)
        psalm_vectors_by_model[model] = psalm_centroids(half_verses_by_psalm, node_vectors)

    new_rows = compare_genre_models(pairs, psalm_vectors_by_model)
    rows = sorted(
        cached_rows + new_rows, key=lambda r: cast(float, r["average_precision"]), reverse=True
    )

    for row in rows:
        print(
            f"{row['model']:55s} AP={row['average_precision']:.4f} "
            f"(chance={row['prevalence']:.3f}) auc={row['separation_auc']:.4f}"
        )

    if args.output:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
