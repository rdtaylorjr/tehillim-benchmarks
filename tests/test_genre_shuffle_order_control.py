from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from genre.pairs import build_genre_pairs
from genre.scripts.shuffle_order_control import score_genre_ap


def _write_parquet(path: Path, vectors: dict[int, list[float]]) -> None:
    node_ids = sorted(vectors)
    dim = len(vectors[node_ids[0]])
    matrix = np.array([vectors[n] for n in node_ids], dtype="<f4")
    table = pa.table(
        {
            "node_id": pa.array(node_ids, type=pa.int32()),
            "vector": pa.FixedSizeListArray.from_arrays(
                pa.array(matrix.flatten(), type=pa.float32()), dim
            ),
        }
    )
    pq.write_table(table, path)


class TestScoreGenreAp:
    def test_perfect_genre_clustering_scores_ap_one(self, tmp_path: Path) -> None:
        # Psalms 1-2 share Lament, psalms 3-4 share Praise: two tight clusters, well separated.
        genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise", 4: "Praise"}
        pairs = build_genre_pairs(genre_by_psalm)
        half_verses_by_psalm = {1: [10], 2: [11], 3: [12], 4: [13]}
        path = tmp_path / "embeddings.parquet"
        _write_parquet(
            path,
            {10: [1.0, 0.0], 11: [1.0, 0.0], 12: [0.0, 1.0], 13: [0.0, 1.0]},
        )

        scores = score_genre_ap(path, half_verses_by_psalm, pairs, ["Lament", "Praise"])

        assert scores == {"Lament": 1.0, "Praise": 1.0}

    def test_returns_one_score_per_requested_genre(self, tmp_path: Path) -> None:
        genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise", 4: "Praise"}
        pairs = build_genre_pairs(genre_by_psalm)
        half_verses_by_psalm = {1: [10], 2: [11], 3: [12], 4: [13]}
        path = tmp_path / "embeddings.parquet"
        _write_parquet(
            path,
            {10: [1.0, 0.0], 11: [1.0, 0.0], 12: [0.0, 1.0], 13: [0.0, 1.0]},
        )

        scores = score_genre_ap(path, half_verses_by_psalm, pairs, ["Lament", "Praise"])

        assert set(scores) == {"Lament", "Praise"}

    def test_pools_a_psalm_s_half_verse_vectors_into_one_centroid(self, tmp_path: Path) -> None:
        # Psalm 1's two half-verses average to [1,0], matching psalm 2 exactly: still perfect AP.
        genre_by_psalm = {1: "Lament", 2: "Lament", 3: "Praise", 4: "Praise"}
        pairs = build_genre_pairs(genre_by_psalm)
        half_verses_by_psalm = {1: [10, 11], 2: [12], 3: [13], 4: [14]}
        path = tmp_path / "embeddings.parquet"
        _write_parquet(
            path,
            {
                10: [2.0, 0.0],
                11: [0.0, 0.1],
                12: [1.0, 0.05],
                13: [0.0, 1.0],
                14: [0.0, 1.0],
            },
        )

        scores = score_genre_ap(path, half_verses_by_psalm, pairs, ["Lament", "Praise"])

        assert scores["Lament"] == 1.0
