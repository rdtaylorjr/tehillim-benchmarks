from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _write_embeddings_parquet(path: Path, vectors: dict[int, list[float]]) -> Path:
    """Writes a dense tehillim-embeddings Parquet file, the shape load_embeddings expects."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


@pytest.fixture
def write_embeddings_parquet():
    """Factory writing a dense embeddings Parquet file at a given path."""
    return _write_embeddings_parquet
