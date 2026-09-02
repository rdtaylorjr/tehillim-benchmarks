"""Exports row-per-pair genre-pair detail plus a per-model AP/AUC/calibration summary."""

import argparse
import sys
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs
from genre.scripts.compare_calibrated import compare_genre_calibrated
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.calibration import BackgroundStats, background_similarity_stats, calibrated_z_score
from library.embeddings import dataset_identifier
from library.errors import BenchmarkDataError
from library.incremental_cache import load_cached_parquet_set
from library.model_files import uncached_model_paths
from library.parallel_models import DEFAULT_MAX_WORKERS, map_in_order
from library.psalm_vectors import load_psalm_vectors
from library.retrieval_metrics import paired_cosine_similarity
from library.rows_output import write_dataframe_parquet

_OUTPUT_FILES = ("genre_pair_detail.parquet", "genre_summary.parquet")


def load_cached_detail(output_dir: Path) -> tuple[list[list[dict[str, Any]]], set[str]]:
    """Reads prior detail parquet files' rows and the model set already covered by both."""
    return load_cached_parquet_set(output_dir, _OUTPUT_FILES)


def build_pair_detail_rows(
    model: str,
    pairs: list[GenrePair],
    psalm_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """One row per usable pair: raw similarity, calibrated z, and a same_genre flag only."""
    usable = [p for p in pairs if p.psalm_a in psalm_vectors and p.psalm_b in psalm_vectors]
    a_vecs = np.stack([psalm_vectors[p.psalm_a] for p in usable])
    b_vecs = np.stack([psalm_vectors[p.psalm_b] for p in usable])
    similarities = paired_cosine_similarity(a_vecs, b_vecs)
    rows = []
    for pair, sim in zip(usable, similarities, strict=True):
        rows.append(
            {
                "model": model,
                "psalm_a": pair.psalm_a,
                "psalm_b": pair.psalm_b,
                "same_genre": pair.same_genre,
                "raw_similarity": float(sim),
                "calibrated_z": calibrated_z_score(float(sim), background),
            }
        )
    return rows


def build_summary_rows(
    model: str,
    pairs: list[GenrePair],
    psalm_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """Single-row-per-model AP/AUC/calibration summary, wrapping compare_genre_calibrated."""
    result = compare_genre_calibrated(pairs, psalm_vectors, background)
    return [
        {
            "model": model,
            "n_same_genre": result.n_same_genre,
            "n_different_genre": result.n_different_genre,
            "prevalence": result.prevalence,
            "average_precision": result.average_precision,
            "same_genre_effect_size": result.same_genre_effect_size,
            "different_genre_effect_size": result.different_genre_effect_size,
            "gap": result.same_genre_effect_size - result.different_genre_effect_size,
            "separation_auc": result.separation_auc,
            "separation_p": result.separation_p,
        }
    ]


def score_model(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    pairs: list[GenrePair],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One model file's (pair rows, summary rows), scored independently of every other model."""
    model = dataset_identifier(path)
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    try:
        background = background_similarity_stats(np.stack(list(psalm_vectors.values())))
        return (
            build_pair_detail_rows(model, pairs, psalm_vectors, background),
            build_summary_rows(model, pairs, psalm_vectors, background),
        )
    except BenchmarkDataError as error:
        print(f"skipping {model} ({len(psalm_vectors)} psalm vectors): {error}", file=sys.stderr)
        return [], []


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
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = load_bhsa_api(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    (cached_pair_rows, cached_summary_rows), cached_models = load_cached_detail(args.output_dir)
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output_dir}", file=sys.stderr)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)
    pair_rows: list[dict[str, Any]] = list(cached_pair_rows)
    summary_rows: list[dict[str, Any]] = list(cached_summary_rows)
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm, pairs=pairs)
    for model_pair_rows, model_summary_rows in map_in_order(score, model_paths, args.workers):
        pair_rows.extend(model_pair_rows)
        summary_rows.extend(model_summary_rows)

    write_dataframe_parquet(args.output_dir / "genre_pair_detail.parquet", pd.DataFrame(pair_rows))
    write_dataframe_parquet(args.output_dir / "genre_summary.parquet", pd.DataFrame(summary_rows))
    print(f"wrote {len(pair_rows)} pair rows, {len(summary_rows)} summary rows")


if __name__ == "__main__":
    main()
