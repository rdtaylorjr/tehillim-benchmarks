from pathlib import Path

from library.model_files import uncached_model_paths


def _touch(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")
    return path


def test_returns_every_embeddings_file_when_nothing_is_cached(tmp_path: Path) -> None:
    a = _touch(tmp_path / "domain=d" / "model=a" / "part-0.parquet")
    b = _touch(tmp_path / "domain=d" / "model=b" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, set()) == [a, b]


def test_skips_a_model_already_present_in_the_cache(tmp_path: Path) -> None:
    _touch(tmp_path / "domain=d" / "model=a" / "part-0.parquet")
    b = _touch(tmp_path / "domain=d" / "model=b" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, {"a"}) == [b]


def test_orders_paths_canonically_so_reruns_submit_work_the_same_way(tmp_path: Path) -> None:
    """Row order follows submission order, so the file order must not depend on the filesystem."""
    _touch(tmp_path / "domain=d" / "model=c" / "part-0.parquet")
    _touch(tmp_path / "domain=d" / "model=a" / "part-0.parquet")
    _touch(tmp_path / "domain=d" / "model=b" / "part-0.parquet")

    found = uncached_model_paths(tmp_path, set())

    assert found == sorted(found)


def test_finds_files_nested_at_any_partition_depth(tmp_path: Path) -> None:
    deep = _touch(tmp_path / "domain=d" / "level=x" / "feature=y" / "c=z" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, set()) == [deep]


def test_ignores_a_directory_that_merely_ends_in_parquet(tmp_path: Path) -> None:
    (tmp_path / "domain=d" / "stray.parquet").mkdir(parents=True)
    real = _touch(tmp_path / "domain=d" / "model=a" / "part-0.parquet")

    assert uncached_model_paths(tmp_path, set()) == [real]


def test_returns_nothing_for_an_empty_directory(tmp_path: Path) -> None:
    assert uncached_model_paths(tmp_path, set()) == []
