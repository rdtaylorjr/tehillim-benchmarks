"""Runs an independent per-item job across worker processes, preserving submission order."""

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import TypeVar

ItemT = TypeVar("ItemT")
ResultT = TypeVar("ResultT")

DEFAULT_MAX_WORKERS = 4
# Parquet decode is already threaded by pyarrow, so throughput peaks at two processes.
IO_BOUND_MAX_WORKERS = 2


def chunksize_for(n_items: int, max_workers: int) -> int:
    """Items per task: enough chunks to balance load, few enough to stop repickling the payload."""
    return max(1, n_items // (max_workers * 4))


def map_in_order(
    fn: Callable[[ItemT], ResultT],
    items: Sequence[ItemT],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[ResultT]:
    """Applies fn to every item, returning results in submission order so reruns stay comparable."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items, chunksize=chunksize_for(len(items), max_workers)))
