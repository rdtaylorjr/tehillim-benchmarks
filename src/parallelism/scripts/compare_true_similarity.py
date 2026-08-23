"""Ranks embedding files by true-pair similarity calibrated against each model's own background."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verse_nodes
from library.calibration import background_similarity_stats, calibrated_effect_size
from library.embeddings import dataset_identifier, load_embeddings
from library.incremental_cache import load_cached_rows
from library.retrieval_metrics import paired_cosine_similarity
from parallelism.evaluate import build_side_vectors
from parallelism.pairs import RetrievalPair, build_retrieval_pairs, filter_pairs_with_vectors
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups
from parallelism.true_similarity import summarize_true_pair_similarity


def compare_true_similarity(
    pairs: list[RetrievalPair],
    node_vectors_by_model: dict[str, dict[int, np.ndarray]],
    background_node_ids: list[int],
) -> list[dict[str, str | int | float]]:
    """One row per model: raw and calibrated true-pair similarity, overall and per type."""
    rows: list[dict[str, str | int | float]] = []
    for model, node_vectors in node_vectors_by_model.items():
        model_pairs = filter_pairs_with_vectors(pairs, node_vectors)
        types = np.array([p.parallelism_type for p in model_pairs])
        source_vecs = build_side_vectors(model_pairs, "source", node_vectors)
        target_vecs = build_side_vectors(model_pairs, "target", node_vectors)
        similarities = paired_cosine_similarity(source_vecs, target_vecs)
        summary = summarize_true_pair_similarity(similarities)

        background_vecs = np.stack(
            [node_vectors[n] for n in background_node_ids if n in node_vectors]
        )
        background = background_similarity_stats(background_vecs)
        effect_size = calibrated_effect_size(summary.mean, background)

        row: dict[str, str | int | float] = {
            "model": model,
            "n_pairs": summary.n,
            "mean_true_similarity": summary.mean,
            "median_true_similarity": summary.median,
            "std_true_similarity": summary.std,
            "background_mean": background.mean,
            "background_std": background.std,
            "background_n_vectors": background.n_vectors,
            "calibrated_effect_size": effect_size,
        }
        for ptype in sorted(set(types.tolist())):
            type_summary = summarize_true_pair_similarity(similarities[types == ptype])
            row[f"n_pairs_{ptype}"] = type_summary.n
            row[f"mean_true_similarity_{ptype}"] = type_summary.mean
            row[f"calibrated_effect_size_{ptype}"] = calibrated_effect_size(
                type_summary.mean, background
            )
        rows.append(row)
    rows.sort(key=lambda r: r["calibrated_effect_size"], reverse=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA/module checkout spec")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_api(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    pairs = build_retrieval_pairs(groups)
    marked_nodes = {n for p in pairs for n in p.source_nodes} | {
        n for p in pairs for n in p.target_nodes
    }
    background_node_ids = [n for n in list_psalms_half_verse_nodes(api) if n not in marked_nodes]

    cached_rows, cached_models = load_cached_rows(args.output) if args.output else ([], set())
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output}", file=sys.stderr)

    model_paths = sorted(p for p in args.embeddings_dir.glob("**/*.parquet") if p.is_file())

    new_rows: list[dict[str, str | int | float]] = []
    for path in model_paths:
        model = dataset_identifier(path)
        if model in cached_models:
            continue
        node_vectors = load_embeddings(path)
        new_rows.extend(compare_true_similarity(pairs, {model: node_vectors}, background_node_ids))

    rows = sorted(cached_rows + new_rows, key=lambda r: r["calibrated_effect_size"], reverse=True)

    for row in rows:
        print(
            f"{row['model']:55s} effect_size={row['calibrated_effect_size']:.3f} "
            f"mean_sim={row['mean_true_similarity']:.4f} "
            f"background_mean={row['background_mean']:.4f}"
        )

    if args.output:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
