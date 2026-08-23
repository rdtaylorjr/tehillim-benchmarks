"""Loads BHSA from the local clone first, falling back to Text-Fabric's use() if that fails."""

import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from tf.app import use as _real_use
from tf.fabric import Fabric as _RealFabric

_PSALMS_BOOK_NAME = "Psalmi"
# Pinned to match tehillim-representations' local BHSA clone; "latest" is an unpinned float.
DEFAULT_CHECKOUT = "v1.8.1"
DEFAULT_USE_TIMEOUT_SECONDS = 30.0
# use() re-verifies the release against the GitHub API even when cached, which can hang or back
# off for minutes under a GitHub rate limit; this local clone (same one tehillim-representations
# uses) has every BHSA feature and never touches the network.
_BHSA_CLONE_LOCATION = Path.home() / "Developer" / "hebrew" / "bhsa" / "tf" / "2021"
_MOD_CACHE_ROOT = Path.home() / "text-fabric-data" / "github"


def _call_with_timeout(
    fn: Callable[..., Any], timeout_seconds: float, /, *args: Any, **kwargs: Any
) -> Any:
    """Runs fn in a daemon thread; returns None (without waiting further) past timeout_seconds."""
    result: queue.Queue[Any] = queue.Queue(maxsize=1)

    def _run() -> None:
        try:
            result.put(fn(*args, **kwargs))
        except Exception:
            result.put(None)

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    if thread.is_alive():
        return None
    return result.get_nowait()


def _mod_cache_location(mod: str) -> Path:
    """Resolves an "org/repo/tf:version" mod spec to its cached local data directory."""
    org, repo, _tf = mod.split(":", 1)[0].split("/")
    tf_root = _MOD_CACHE_ROOT / org / repo / "tf"
    versions = sorted(p for p in tf_root.iterdir() if p.is_dir()) if tf_root.exists() else []
    if not versions:
        raise RuntimeError(f"No cached Text-Fabric data found for {org}/{repo} under {tf_root}")
    return versions[-1]


def _load_from_local_clone(mod: str | None, fabric_class: Callable[..., Any]) -> Any:
    """Loads BHSA from its full local clone, plus an optional companion module's local cache."""
    locations = [str(_BHSA_CLONE_LOCATION)]
    if mod is not None:
        locations.append(str(_mod_cache_location(mod)))
    tf = fabric_class(locations=locations, silent="deep")
    return tf.loadAll(silent="deep")


def load_bhsa_api(
    checkout: str = DEFAULT_CHECKOUT,
    mod: str | None = None,
    use_fn: Callable[..., Any] = _real_use,
    fabric_class: Callable[..., Any] = _RealFabric,
    timeout_seconds: float = DEFAULT_USE_TIMEOUT_SECONDS,
) -> Any:
    """Loads BHSA plus an optional companion module from the local clone; falls back to use()."""
    try:
        api = _load_from_local_clone(mod, fabric_class)
    except Exception:
        api = None
    if api is None:
        app = _call_with_timeout(
            use_fn, timeout_seconds, "etcbc/bhsa", checkout=checkout, mod=mod, silent="deep"
        )
        api = app.api if app is not None else None
    if api is None:
        raise RuntimeError(f"Text-Fabric failed to load BHSA (checkout={checkout!r}, mod={mod!r})")
    return api


def psalms_book_node(api: Any) -> Any:
    """The BHSA book node for Psalms."""
    F = api.F  # noqa: N806
    book_nodes = [b for b in F.otype.s("book") if F.book.v(b) == _PSALMS_BOOK_NAME]
    if not book_nodes:
        raise RuntimeError(f"Book {_PSALMS_BOOK_NAME!r} not found in loaded corpus")
    return book_nodes[0]


def list_psalms_half_verse_nodes(api: Any) -> list[int]:
    """Lists every half-verse node in Psalms, in canonical order."""
    L = api.L  # noqa: N806
    book_node = psalms_book_node(api)
    return [
        hv_node
        for chapter_node in L.d(book_node, otype="chapter")
        for hv_node in L.d(chapter_node, otype="half_verse")
    ]


def list_psalms_half_verses_by_psalm(api: Any) -> dict[int, list[int]]:
    """Lists every half-verse node, grouped and ordered by psalm number (BHSA chapter)."""
    L, T = api.L, api.T  # noqa: N806
    book_node = psalms_book_node(api)
    result: dict[int, list[int]] = {}
    for chapter_node in L.d(book_node, otype="chapter"):
        _, psalm_number = T.sectionFromNode(chapter_node)
        result[psalm_number] = list(L.d(chapter_node, otype="half_verse"))
    return result


def node_to_psalm_map(half_verses_by_psalm: dict[int, list[int]]) -> dict[int, int]:
    """Inverts a psalm-to-nodes mapping into a node-to-psalm lookup."""
    return {node: psalm for psalm, nodes in half_verses_by_psalm.items() for node in nodes}
