"""Runs evaluate_genre_discrimination across every embedding file in a directory."""

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import cast

from genre.evaluate import evaluate_genre_discrimination
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.embeddings import dataset_identifier
from library.incremental_cache import load_cached_rows
from library.model_files import uncached_model_paths
from library.parallel_models import DEFAULT_MAX_WORKERS, map_in_order
from library.psalm_vectors import load_psalm_vectors
from library.rows_output import write_rows_csv


def score_model(
    path: Path, half_verses_by_psalm: dict[int, list[int]], pairs: list[GenrePair]
) -> dict[str, str | int | float]:
    """One model file's genre-discrimination row, scored independently of every other model."""
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    report = evaluate_genre_discrimination(pairs, psalm_vectors)
    return {
        "model": dataset_identifier(path),
        "n_same_genre": report.n_same_genre,
        "n_different_genre": report.n_different_genre,
        "prevalence": report.prevalence,
        "average_precision": report.average_precision,
        "separation_auc": report.separation_auc,
        "separation_p": report.separation_p,
    }


def compare_genre_models(
    pairs: list[GenrePair],
    model_paths: list[Path],
    half_verses_by_psalm: dict[int, list[int]],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[dict[str, str | int | float]]:
    """Scores every model file across workers, rows sorted by Average Precision descending."""
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm, pairs=pairs)
    rows = map_in_order(score, model_paths, max_workers)
    rows.sort(key=lambda r: cast("float", r["average_precision"]), reverse=True)
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
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_bhsa_api(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    cached_rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)
    new_rows = compare_genre_models(
        pairs, model_paths, half_verses_by_psalm, max_workers=args.workers
    )
    rows = sorted(
        cached_rows + new_rows, key=lambda r: cast("float", r["average_precision"]), reverse=True
    )

    for row in rows:
        print(
            f"{row['model']:55s} AP={row['average_precision']:.4f} "
            f"(chance={row['prevalence']:.3f}) auc={row['separation_auc']:.4f}"
        )

    if args.output:
        write_rows_csv(args.output, rows)


if __name__ == "__main__":
    main()
