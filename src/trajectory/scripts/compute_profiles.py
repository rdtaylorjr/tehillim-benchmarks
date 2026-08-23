"""Computes each psalm's content centroid and cola sequence, all models."""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.centroid import psalm_centroids
from library.embeddings import dataset_identifier, load_embeddings
from library.incremental_cache import load_cached_parquet_set
from trajectory.distance import content_distance, dtw_curve_distance, structural_distance_dtw
from trajectory.geometry import adjacent_similarity, step_magnitude, turning_angle
from trajectory.self_similarity import self_similarity_matrix
from trajectory.sequence import normalize_sequence, psalm_cola_sequences

_MIN_COLA = 4


def compute_psalm_profiles(
    sequences_by_psalm: dict[int, np.ndarray],
    centroids_by_psalm: dict[int, np.ndarray],
) -> dict[int, dict[str, np.ndarray]]:
    """One profile per psalm with a centroid and at least _MIN_COLA cola for the other curves."""
    profiles = {}
    for psalm, sequence in sequences_by_psalm.items():
        if psalm not in centroids_by_psalm or len(sequence) < _MIN_COLA:
            continue
        profiles[psalm] = {
            "centroid": centroids_by_psalm[psalm],
            "sequence": normalize_sequence(sequence),
        }
    return profiles


def profile_rows(model: str, profiles: dict[int, dict[str, np.ndarray]]) -> list[dict]:
    """Flattens each psalm's profile into one row: the real, un-resampled cola sequence."""
    rows = []
    for psalm, profile in profiles.items():
        sequence = profile["sequence"]
        rows.append(
            {
                "model": model,
                "psalm": psalm,
                "centroid": profile["centroid"].astype(np.float32),
                "sequence": sequence.flatten().astype(np.float32),
                "n_cola": sequence.shape[0],
                "dim": sequence.shape[1],
            }
        )
    return rows


def distance_rows(model: str, profiles: dict[int, dict[str, np.ndarray]]) -> list[dict]:
    """One row per unordered psalm pair: content, structural, and geometry-curve DTW distances."""
    self_similarity = {p: self_similarity_matrix(v["sequence"]) for p, v in profiles.items()}
    adjacent = {p: adjacent_similarity(v["sequence"]) for p, v in profiles.items()}
    step = {p: step_magnitude(v["sequence"]) for p, v in profiles.items()}
    turning = {p: turning_angle(v["sequence"]) for p, v in profiles.items()}

    rows = []
    for a, b in combinations(sorted(profiles), 2):
        seq_a, seq_b = profiles[a]["sequence"], profiles[b]["sequence"]
        rows.append(
            {
                "model": model,
                "psalm_a": a,
                "psalm_b": b,
                "content_distance": content_distance(
                    profiles[a]["centroid"], profiles[b]["centroid"]
                ),
                "structural_distance": structural_distance_dtw(
                    seq_a, seq_b, self_similarity[a], self_similarity[b]
                ),
                "adjacent_similarity_distance": dtw_curve_distance(adjacent[a], adjacent[b]),
                "step_magnitude_distance": dtw_curve_distance(step[a], step[b]),
                "turning_angle_distance": dtw_curve_distance(turning[a], turning[b]),
            }
        )
    return rows


def profile_shard_path(output_dir: Path, model: str) -> Path:
    """One parquet file per model, so no single shard risks GitHub's 100MB per-file limit."""
    return output_dir / f"{model}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = load_bhsa_api(args.checkout)
    half_verses_by_psalm = list_psalms_half_verses_by_psalm(api)

    (cached_distance_rows,), cached_models = load_cached_parquet_set(
        args.output_dir, ("trajectory_distances.parquet",)
    )
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output_dir}", file=sys.stderr)

    model_paths = sorted(p for p in args.embeddings_dir.glob("**/*.parquet") if p.is_file())
    all_distance_rows: list[dict] = list(cached_distance_rows)
    n_profile_rows = 0
    for path in model_paths:
        model = dataset_identifier(path)
        if model in cached_models and profile_shard_path(args.output_dir, model).exists():
            continue
        print(f"processing {model}")
        node_vectors = load_embeddings(path)
        sequences_by_psalm = psalm_cola_sequences(half_verses_by_psalm, node_vectors)
        centroids_by_psalm = psalm_centroids(half_verses_by_psalm, node_vectors)
        profiles = compute_psalm_profiles(sequences_by_psalm, centroids_by_psalm)
        rows = profile_rows(model, profiles)
        pd.DataFrame(rows).to_parquet(
            profile_shard_path(args.output_dir, model),
            index=False,
            compression="zstd",
            compression_level=19,
        )
        n_profile_rows += len(rows)
        all_distance_rows.extend(distance_rows(model, profiles))

    pd.DataFrame(all_distance_rows).to_parquet(
        args.output_dir / "trajectory_distances.parquet",
        index=False,
        compression="zstd",
        compression_level=19,
    )
    print(f"wrote {n_profile_rows} profile rows, {len(all_distance_rows)} distance rows")


if __name__ == "__main__":
    main()
