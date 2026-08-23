"""Order-shuffle-null control: does colon order carry more genre signal than a shuffled null."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from genre.evaluate import evaluate_genre_discrimination
from genre.genre_labels import load_genre_by_psalm
from genre.pairs import GenrePair, build_genre_pairs, filter_pairs_by_genre
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.centroid import psalm_centroids
from library.embeddings import load_embeddings
from library.order_shuffle import order_shuffle_result


def score_genre_ap(
    path: Path,
    half_verses_by_psalm: dict[int, list[int]],
    pairs: list[GenrePair],
    genres: list[str],
) -> dict[str, float]:
    """Per-genre Average Precision (no permutation testing) for one embeddings file."""
    node_vectors = load_embeddings(path)
    vectors = psalm_centroids(half_verses_by_psalm, node_vectors)
    return {
        genre: evaluate_genre_discrimination(
            filter_pairs_by_genre(pairs, genre), vectors
        ).average_precision
        for genre in genres
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "genre_csv",
        type=Path,
        help="third-party genre CSV, e.g. psalms-browser.csv (not in this repo)",
    )
    parser.add_argument("real_embeddings", type=Path)
    parser.add_argument("shuffled_embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_bhsa_api(args.checkout)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    pairs = build_genre_pairs(genre_by_psalm)
    genres = sorted(set(genre_by_psalm.values()))

    real_ap = score_genre_ap(args.real_embeddings, half_verses_by_psalm, pairs, genres)
    shuffled_paths = sorted(args.shuffled_embeddings_dir.glob("**/*.parquet"))
    shuffled_ap: dict[str, list[float]] = {genre: [] for genre in genres}
    for path in shuffled_paths:
        print(f"scoring {path.parent.name}", file=sys.stderr)
        scores = score_genre_ap(path, half_verses_by_psalm, pairs, genres)
        for genre in genres:
            shuffled_ap[genre].append(scores[genre])

    rows = []
    for genre in genres:
        result = order_shuffle_result(
            real_score=real_ap[genre], shuffled_scores=np.array(shuffled_ap[genre])
        )
        rows.append(
            {
                "genre": genre,
                "ap_real": real_ap[genre],
                "ap_shuffled_mean": float(np.mean(shuffled_ap[genre])),
                "n_shuffles": len(shuffled_ap[genre]),
                "delta_order": result.delta_order,
                "p_value": result.p_value,
            }
        )
        print(
            f"{genre:15s} ap_real={real_ap[genre]:.4f} "
            f"delta_order={result.delta_order:+.4f} p={result.p_value:.4f}"
        )

    if args.output:
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
