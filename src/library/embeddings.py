"""Reads BHSA-node-keyed embedding vectors from tehillim-embeddings' Parquet files."""

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

TEXT_VARIANTS = ("consonantal", "vocalized", "cantillation")


def dataset_identifier(path: Path) -> str:
    """Joins every Hive partition value between the file and its `type=` root, deepest first."""
    parts = []
    node = path.parent
    while "=" in node.name and not node.name.startswith("type="):
        parts.append(node.name.split("=", 1)[1])
        node = node.parent
    return "_".join(reversed(parts))


def split_model_name(model: str, text_variants: tuple[str, ...] = TEXT_VARIANTS) -> tuple[str, str]:
    """Splits into (base_model, text_variant): a trailing suffix, else a variant token anywhere."""
    for variant in text_variants:
        suffix = f"_{variant}"
        if model.endswith(suffix):
            return model[: -len(suffix)].removeprefix("semantic_"), variant
    tokens = model.split("_")
    for variant in text_variants:
        if variant in tokens:
            base = "_".join(t for t in tokens if t != variant)
            return base, variant
    return model.removeprefix("semantic_"), "unknown"


def load_embeddings(path: Path) -> dict[int, np.ndarray]:
    """Reads a Parquet embeddings file into a {node: vector} map, excluding zero-norm vectors."""
    table = pq.read_table(path, columns=["node_id", "vector"])
    node_ids = table["node_id"].to_numpy(zero_copy_only=False)
    vector_column = table["vector"].combine_chunks()
    dim = vector_column.type.list_size
    matrix = vector_column.values.to_numpy(zero_copy_only=False).astype("<f4", copy=False)
    matrix = matrix.reshape(len(node_ids), dim)
    nonzero = np.any(matrix, axis=1)
    return {int(node_ids[i]): matrix[i] for i in np.flatnonzero(nonzero)}
