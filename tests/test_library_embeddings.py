from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from library.embeddings import dataset_identifier, load_embeddings, split_model_name


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


def test_load_embeddings_reads_a_real_parquet_file(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.parquet"
    _write_parquet(path, {7: [4.0, 5.0], 8: [1.0, -2.0]})

    vectors = load_embeddings(path)

    assert set(vectors) == {7, 8}
    np.testing.assert_array_equal(vectors[7], np.array([4.0, 5.0], dtype="<f4"))
    np.testing.assert_array_equal(vectors[8], np.array([1.0, -2.0], dtype="<f4"))


def test_load_embeddings_returns_float32_vectors(tmp_path: Path) -> None:
    path = tmp_path / "embeddings.parquet"
    _write_parquet(path, {1: [1.0, 2.0, 3.0]})

    vectors = load_embeddings(path)

    assert vectors[1].dtype == np.dtype("<f4")


def test_load_embeddings_excludes_zero_norm_vectors(tmp_path: Path) -> None:
    """A zero vector has no cosine direction, undefined for every downstream similarity metric."""
    path = tmp_path / "embeddings.parquet"
    _write_parquet(path, {7: [4.0, 5.0], 8: [0.0, 0.0]})

    vectors = load_embeddings(path)

    assert set(vectors) == {7}


def test_load_embeddings_matches_a_naive_per_row_to_pylist_conversion(tmp_path: Path) -> None:
    """Proves the batched Arrow-column read is lossless against the original per-row path."""
    rng = np.random.default_rng(0)
    n_rows, dim = 200, 5354
    node_ids = list(range(1000, 1000 + n_rows))
    matrix = rng.normal(size=(n_rows, dim)).astype("<f4")
    matrix[3] = 0.0  # one zero-norm row, must be excluded by both paths
    path = tmp_path / "embeddings.parquet"
    _write_parquet(path, {node: matrix[i].tolist() for i, node in enumerate(node_ids)})

    table = pq.read_table(path, columns=["node_id", "vector"])
    naive_node_ids = table["node_id"].to_pylist()
    naive_vectors_raw = table["vector"].to_pylist()
    naive = {
        node: np.asarray(vector, dtype="<f4")
        for node, vector in zip(naive_node_ids, naive_vectors_raw, strict=True)
        if np.any(vector)
    }

    vectorized = load_embeddings(path)

    assert set(vectorized) == set(naive)
    for node in naive:
        np.testing.assert_array_equal(vectorized[node], naive[node])


def test_dataset_identifier_reads_model_and_variation_from_the_hive_path() -> None:
    path = Path("data/type=semantic/model=bge_m3/variation=vocalized/part-0.parquet")

    assert dataset_identifier(path) == "bge_m3_vocalized"


def test_dataset_identifier_handles_a_two_level_lexical_path() -> None:
    path = Path("data/type=lexical/unit=homograph/weight=binary/part-0.parquet")

    assert dataset_identifier(path) == "homograph_binary"


def test_dataset_identifier_handles_a_three_level_path_with_an_extra_text_tier() -> None:
    path = Path("data/type=lexical/unit=word/text=consonantal/weight=binary/part-0.parquet")

    assert dataset_identifier(path) == "word_consonantal_binary"


def test_split_model_name_extracts_base_and_variant() -> None:
    assert split_model_name("semantic_gemini_embedding_2_cantillation") == (
        "gemini_embedding_2",
        "cantillation",
    )
    assert split_model_name("semantic_bge_m3_vocalized") == ("bge_m3", "vocalized")


def test_split_model_name_falls_back_to_unknown_variant() -> None:
    assert split_model_name("semantic_something_odd") == ("something_odd", "unknown")


def test_split_model_name_extracts_a_variant_embedded_in_the_middle() -> None:
    # word_consonantal_binary: unit=word, text=consonantal, weight=binary, tier is not a suffix.
    assert split_model_name("word_consonantal_binary") == ("word_binary", "consonantal")


def test_split_model_name_still_prefers_a_trailing_suffix_variant() -> None:
    assert split_model_name("bge_m3_vocalized") == ("bge_m3", "vocalized")
