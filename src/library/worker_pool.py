"""Runs an independent per-item job across worker processes, preserving submission order."""

# Named for the pool: "parallelism" here is the Hebrew poetic kind, benchmarked in src/parallelism.

from collections.abc import Callable, Sequence
from concurrent.futures import ProcessPoolExecutor

# Measured over one shuffle family of 100 draws: 55s at 2 workers, 50s at 3, 65s at 5.
DEFAULT_MAX_WORKERS = 3


def chunksize_for(n_items: int, max_workers: int) -> int:
    """Items per task: enough chunks to balance load, few enough to stop repickling the payload."""
    return max(1, n_items // (max_workers * 4))


def map_in_order[ItemT, ResultT](
    fn: Callable[[ItemT], ResultT],
    items: Sequence[ItemT],
    max_workers: int = DEFAULT_MAX_WORKERS,
) -> list[ResultT]:
    """Applies fn to every item, returning results in submission order so reruns stay comparable."""
    if max_workers <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items, chunksize=chunksize_for(len(items), max_workers)))
