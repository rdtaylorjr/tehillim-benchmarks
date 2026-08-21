"""Adds calibrated same/different-genre effect size on top of the raw AP/AUC report."""

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import average_precision_score

from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.calibration import BackgroundStats, background_similarity_stats, calibrated_effect_size
from library.centroid import psalm_centroids
from library.embeddings import dataset_identifier, load_embeddings
from library.incremental_cache import load_cached_rows
from library.retrieval_metrics import paired_cosine_similarity


@dataclass(frozen=True, slots=True)
class GenreCalibratedComparison:
    n_same_genre: int
    n_different_genre: int
    prevalence: float
    mean_same_genre_similarity: float
    mean_different_genre_similarity: float
    same_genre_effect_size: float
    different_genre_effect_size: float
    average_precision: float
    separation_auc: float
    separation_p: float


def compare_genre_calibrated(
    pairs: list[GenrePair], psalm_vectors: dict[int, np.ndarray], background: BackgroundStats
) -> GenreCalibratedComparison:
    """Same-genre vs different-genre similarity: AP/AUC plus each group's calibrated effect size."""
    usable = [p for p in pairs if p.psalm_a in psalm_vectors and p.psalm_b in psalm_vectors]
    a_vecs = np.stack([psalm_vectors[p.psalm_a] for p in usable])
    b_vecs = np.stack([psalm_vectors[p.psalm_b] for p in usable])
    similarities = paired_cosine_similarity(a_vecs, b_vecs)
    labels = np.array([p.same_genre for p in usable], dtype=int)

    same_sims = similarities[labels == 1]
    different_sims = similarities[labels == 0]

    statistic, p_value = mannwhitneyu(same_sims, different_sims, alternative="greater")
    auc = statistic / (len(same_sims) * len(different_sims))
    ap = average_precision_score(labels, similarities)
    mean_same = float(same_sims.mean())
    mean_different = float(different_sims.mean())

    return GenreCalibratedComparison(
        n_same_genre=len(same_sims),
        n_different_genre=len(different_sims),
        prevalence=len(same_sims) / len(usable),
        mean_same_genre_similarity=mean_same,
        mean_different_genre_similarity=mean_different,
        same_genre_effect_size=calibrated_effect_size(mean_same, background),
        different_genre_effect_size=calibrated_effect_size(mean_different, background),
        average_precision=float(ap),
        separation_auc=float(auc),
        separation_p=float(p_value),
    )


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

    rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = sorted(p for p in args.embeddings_dir.glob("**/*.parquet") if p.is_file())
    for path in model_paths:
        model = dataset_identifier(path)
        if model in cached_models:
            continue
        node_vectors = load_embeddings(path)
        psalm_vectors = psalm_centroids(half_verses_by_psalm, node_vectors)
        # genre pairs exhaustively cover every psalm, so unlike parallelism (which excludes
        # marked nodes) the background is the full psalm-centroid population itself.
        try:
            background = background_similarity_stats(np.stack(list(psalm_vectors.values())))
            result = compare_genre_calibrated(pairs, psalm_vectors, background)
        except ValueError as error:
            print(
                f"skipping {model}: {error} (only {len(psalm_vectors)} psalm vectors)",
                file=sys.stderr,
            )
            continue
        rows.append(
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
