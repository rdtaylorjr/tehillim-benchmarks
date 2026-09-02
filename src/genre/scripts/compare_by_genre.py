"""One-vs-rest genre discrimination per model: AP/AUC, jackknife CIs, and permutation p-values."""

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

from genre.bootstrap import block_bootstrap_genre_ap_gap_and_auc, psalm_similarity_matrix
from genre.evaluate import evaluate_genre_discrimination_from_matrix
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs, filter_pairs_by_genre
from genre.permutation import joint_psalm_label_permutation_test, one_vs_rest_masks
from library.ap_gap_auc_bootstrap import ApGapAucCI
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.calibration import BackgroundStats, background_stats_from_matrix
from library.embeddings import dataset_identifier
from library.errors import BenchmarkDataError, InsufficientDataError
from library.incremental_cache import load_cached_rows as _load_cached_rows
from library.model_files import uncached_model_paths
from library.multiple_comparisons import add_fdr_q_values
from library.parallel_models import DEFAULT_MAX_WORKERS, map_in_order
from library.psalm_vectors import load_psalm_vectors
from library.retrieval_metrics import sparse_cosine_similarity_matrix

_SOURCES = ("naive", "perm", "maxT")


@dataclass(frozen=True, slots=True)
class GenreRunConfig:
    """The labels and resampling budget every model in one run is scored against."""

    genre_by_psalm: dict[int, str]
    genres: tuple[str, ...]
    pairs: list[GenrePair]
    n_permutations: int
    n_resamples: int
    seed: int


_RAW_COLUMNS = (
    "model",
    "genre",
    "n_same_genre",
    "n_different_genre",
    "prevalence",
    "average_precision",
    "ap_ci_low",
    "ap_ci_high",
    "separation_auc",
    "auc_ci_low",
    "auc_ci_high",
    "separation_p_naive",
    "separation_p_perm",
    "separation_p_maxT",
    "n_permutations",
)


def _bootstrap_ci_or_none(
    psalm_ids: list[int],
    similarity_matrix: np.ndarray,
    same_mask: np.ndarray,
    population_mask: np.ndarray,
    background: BackgroundStats,
    n_resamples: int,
    seed: int,
    genre: str,
) -> ApGapAucCI | None:
    """None for a genre whose one-vs-rest population is too small to define a CI, not a crash."""
    try:
        return block_bootstrap_genre_ap_gap_and_auc(
            psalm_ids,
            similarity_matrix,
            same_mask,
            background,
            n_resamples=n_resamples,
            rng=np.random.default_rng(seed),
            population_mask=population_mask,
        )
    except InsufficientDataError as error:
        print(f"no bootstrap CI for genre {genre!r}: {error}", file=sys.stderr)
        return None


def _compare_from_similarity_matrix(
    model: str,
    psalm_ids: list[int],
    similarity_matrix: np.ndarray,
    config: GenreRunConfig,
) -> list[dict[str, str | int | float]]:
    """Shared per-genre report step for both the dense and sparse psalm-vector entry points."""
    genres = config.genres
    psalm_index = {p: i for i, p in enumerate(psalm_ids)}
    code_of = {genre: index for index, genre in enumerate(genres)}
    genre_codes = np.array([code_of[config.genre_by_psalm[p]] for p in psalm_ids])
    background = background_stats_from_matrix(similarity_matrix)

    perm_result = joint_psalm_label_permutation_test(
        similarity_matrix,
        genre_codes,
        genres,
        n_permutations=config.n_permutations,
        rng=np.random.default_rng(config.seed),
    )

    rows: list[dict[str, str | int | float]] = []
    for index, genre in enumerate(genres):
        restricted = filter_pairs_by_genre(config.pairs, genre)
        report = evaluate_genre_discrimination_from_matrix(
            restricted, similarity_matrix, psalm_index
        )

        same_mask, population_mask = one_vs_rest_masks(genre_codes, index)
        ci = _bootstrap_ci_or_none(
            psalm_ids,
            similarity_matrix,
            same_mask,
            population_mask,
            background,
            config.n_resamples,
            config.seed,
            genre,
        )

        rows.append(
            {
                "model": model,
                "genre": genre,
                "n_same_genre": report.n_same_genre,
                "n_different_genre": report.n_different_genre,
                "prevalence": report.prevalence,
                "average_precision": report.average_precision,
                "ap_ci_low": ci.ap_ci_low if ci else float("nan"),
                "ap_ci_high": ci.ap_ci_high if ci else float("nan"),
                "separation_auc": report.separation_auc,
                "auc_ci_low": ci.auc_ci_low if ci else float("nan"),
                "auc_ci_high": ci.auc_ci_high if ci else float("nan"),
                "separation_p_naive": report.separation_p,
                "separation_p_perm": perm_result.p_perm[index],
                "separation_p_maxT": perm_result.p_maxT[index],
                "n_permutations": config.n_permutations,
            }
        )
    return rows


def compare_model_across_genres(
    model: str,
    psalm_ids: list[int],
    psalm_vectors: dict[int, np.ndarray],
    config: GenreRunConfig,
) -> list[dict[str, str | int | float]]:
    """One row per genre: AP (point, unchanged), AUC, jackknife CIs, and three p-value sources."""
    similarity_matrix = psalm_similarity_matrix(psalm_ids, psalm_vectors)
    return _compare_from_similarity_matrix(model, psalm_ids, similarity_matrix, config)


def compare_model_across_genres_sparse(
    model: str,
    psalm_ids: list[int],
    psalm_vectors: sp.csr_matrix,
    config: GenreRunConfig,
) -> list[dict[str, str | int | float]]:
    """Same rows as compare_model_across_genres, comparing sparse psalm vectors, never densified."""
    similarity_matrix = sparse_cosine_similarity_matrix(psalm_vectors, psalm_vectors)
    return _compare_from_similarity_matrix(model, psalm_ids, similarity_matrix, config)


def load_cached_rows(cache_path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    """Reads a prior output CSV's raw rows (dropping stale q-values) and its covered model set."""
    return _load_cached_rows(cache_path, columns=_RAW_COLUMNS)


def add_fdr_columns(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Adds BH/BY q-values to each of the three p-value sources, scoped per genre's model family."""
    df = pd.DataFrame(rows)
    long_parts = [
        pd.DataFrame(
            {
                "model": df["model"],
                "scope_kind": df["genre"],
                "source": source,
                "metric": "separation_p",
                "value": df[f"separation_p_{source}"],
            }
        )
        for source in _SOURCES
    ]
    long_df = add_fdr_q_values(pd.concat(long_parts, ignore_index=True))

    result = df.copy()
    for source in _SOURCES:
        q_columns = long_df[long_df["source"] == source][
            ["model", "scope_kind", "q_value", "q_value_by"]
        ]
        q_columns = q_columns.rename(
            columns={
                "scope_kind": "genre",
                "q_value": f"{source}_q",
                "q_value_by": f"{source}_q_by",
            }
        )
        result = result.merge(q_columns, on=["model", "genre"], how="left")
    return result


def score_model(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    config: GenreRunConfig,
) -> list[dict[str, str | int | float]]:
    """One row per genre for one model file, scored independently of every other model."""
    model = dataset_identifier(path)
    psalm_vectors = load_psalm_vectors(path, half_verses_by_psalm)
    psalm_ids = sorted(set(config.genre_by_psalm) & set(psalm_vectors))
    try:
        return compare_model_across_genres(model, psalm_ids, psalm_vectors, config)
    except BenchmarkDataError as error:
        print(f"skipping {model} ({len(psalm_ids)} psalm vectors): {error}", file=sys.stderr)
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "genre_csv",
        type=Path,
        help="third-party genre CSV, e.g. psalms-browser.csv (not in this repo)",
    )
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--n-permutations", type=int, default=2000)
    parser.add_argument("--n-resamples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--cache",
        type=Path,
        default=None,
        help="prior output CSV to reuse; defaults to --output, so reruns cache automatically",
    )
    args = parser.parse_args()

    api = load_bhsa_api(args.checkout)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    genres = tuple(sorted(set(genre_by_psalm.values())))

    cache_path = args.cache or args.output
    rows, cached_models = load_cached_rows(cache_path) if cache_path else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {cache_path}", file=sys.stderr)

    model_paths = uncached_model_paths(args.embeddings_dir, cached_models)
    config = GenreRunConfig(
        genre_by_psalm=genre_by_psalm,
        genres=genres,
        pairs=pairs,
        n_permutations=args.n_permutations,
        n_resamples=args.n_resamples,
        seed=args.seed,
    )
    score = partial(score_model, half_verses_by_psalm=half_verses_by_psalm, config=config)
    for model_rows in map_in_order(score, model_paths, args.workers):
        rows.extend(model_rows)

    result_df = add_fdr_columns(rows)

    ordered = result_df.sort_values(["genre", "average_precision"], ascending=[True, False])
    for _, row in ordered.iterrows():
        print(
            f"{row['genre']:15s} {row['model']:55s} "
            f"AP={row['average_precision']:.4f} [{row['ap_ci_low']:.4f},{row['ap_ci_high']:.4f}] "
            f"(chance={row['prevalence']:.3f}) "
            f"p_naive={row['separation_p_naive']:.4f} p_perm={row['separation_p_perm']:.4f} "
            f"p_maxT={row['separation_p_maxT']:.4f}"
        )

    if args.output:
        result_df.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
