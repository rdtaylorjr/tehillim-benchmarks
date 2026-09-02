"""Selects which embedding files a batch run still needs to score."""

from pathlib import Path

from library.embeddings import dataset_identifier


def uncached_model_paths(embeddings_dir: Path, cached_models: set[str]) -> list[Path]:
    """Every embeddings file, in canonical order, whose model a prior run has not already scored."""
    return [
        path
        for path in sorted(embeddings_dir.glob("**/*.parquet"))
        if path.is_file() and dataset_identifier(path) not in cached_models
    ]
