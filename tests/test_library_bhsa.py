import time

import pytest

from library.bhsa import (
    list_psalms_half_verse_nodes,
    list_psalms_half_verses_by_psalm,
    load_bhsa_api,
    node_to_psalm_map,
    psalms_book_node,
)


def _fake_fabric_class(api: object):
    """A fabric_class stand-in whose loadAll() always returns the given fake api."""

    def fake_fabric_class(*, locations: list[str], silent: str) -> object:
        class _FakeTF:
            def loadAll(self, silent: str) -> object:  # noqa: N802
                return api

        return _FakeTF()

    return fake_fabric_class


class _FakeOtype:
    def __init__(self, book_types: dict[int, str]) -> None:
        self._book_types = book_types

    def s(self, otype: str) -> list[int]:
        assert otype == "book"
        return list(self._book_types)


class _FakeFeature:
    def __init__(self, values: dict[int, str]) -> None:
        self._values = values

    def v(self, node: int) -> str | None:
        return self._values.get(node)


class _FakeF:
    def __init__(self, book_names: dict[int, str]) -> None:
        self.otype = _FakeOtype(book_names)
        self.book = _FakeFeature(book_names)


class _FakeL:
    def __init__(self, children: dict[tuple[int, str], list[int]]) -> None:
        self._children = children

    def d(self, node: int, otype: str) -> list[int]:
        return self._children.get((node, otype), [])


class _FakeT:
    def __init__(self, chapter_to_psalm: dict[int, int]) -> None:
        self._chapter_to_psalm = chapter_to_psalm

    def sectionFromNode(self, node: int) -> tuple[str, int]:  # noqa: N802
        return ("Psalmi", self._chapter_to_psalm[node])


class _FakeApi:
    def __init__(self, F: _FakeF, L: _FakeL, T: _FakeT) -> None:  # noqa: N803
        self.F = F
        self.L = L
        self.T = T


def _api_with_two_psalms() -> _FakeApi:
    book_names = {1: "Psalmi"}
    children = {
        (1, "chapter"): [10, 20],
        (10, "half_verse"): [100, 101],
        (20, "half_verse"): [200],
    }
    chapter_to_psalm = {10: 1, 20: 2}
    return _FakeApi(_FakeF(book_names), _FakeL(children), _FakeT(chapter_to_psalm))


class TestLoadBhsaApi:
    def test_returns_the_api_from_a_successful_use_call(self) -> None:
        class _FakeApp:
            api = object()

        result = load_bhsa_api(use_fn=lambda *a, **k: _FakeApp())

        assert result is _FakeApp.api

    def test_passes_checkout_and_mod_through_to_use(self) -> None:
        calls = []

        def fake_use(name: str, *, checkout: str, mod: str | None, silent: str) -> object:
            calls.append((name, checkout, mod))

            class _FakeApp:
                api = object()

            return _FakeApp()

        load_bhsa_api(checkout="v1.0", mod="org/repo/tf:v1.0", use_fn=fake_use)

        assert calls == [("etcbc/bhsa", "v1.0", "org/repo/tf:v1.0")]

    def test_falls_back_to_the_local_clone_when_use_returns_none(self) -> None:
        fallback_api = object()

        result = load_bhsa_api(
            use_fn=lambda *a, **k: None,
            fabric_class=_fake_fabric_class(fallback_api),
            timeout_seconds=1.0,
        )

        assert result is fallback_api

    def test_falls_back_to_the_local_clone_when_the_app_has_no_api(self) -> None:
        class _FakeApp:
            api = None

        fallback_api = object()

        result = load_bhsa_api(
            use_fn=lambda *a, **k: _FakeApp(),
            fabric_class=_fake_fabric_class(fallback_api),
            timeout_seconds=1.0,
        )

        assert result is fallback_api

    def test_falls_back_to_the_local_clone_when_use_raises(self) -> None:
        def raising_use(*a: object, **k: object) -> object:
            raise ConnectionError("no internet")

        fallback_api = object()

        result = load_bhsa_api(
            use_fn=raising_use, fabric_class=_fake_fabric_class(fallback_api), timeout_seconds=1.0
        )

        assert result is fallback_api

    def test_falls_back_to_the_local_clone_when_use_does_not_return_within_the_timeout(
        self,
    ) -> None:
        def slow_use(*a: object, **k: object) -> object:
            time.sleep(0.3)

            class _FakeApp:
                api = object()

            return _FakeApp()

        fallback_api = object()

        result = load_bhsa_api(
            use_fn=slow_use, fabric_class=_fake_fabric_class(fallback_api), timeout_seconds=0.05
        )

        assert result is fallback_api

    def test_does_not_wait_for_a_slow_use_call_past_the_timeout(self) -> None:
        started = time.monotonic()

        def slow_use(*a: object, **k: object) -> object:
            time.sleep(0.3)
            return None

        load_bhsa_api(
            use_fn=slow_use, fabric_class=_fake_fabric_class(object()), timeout_seconds=0.05
        )

        assert time.monotonic() - started < 0.2

    def test_passes_the_mod_org_and_repo_to_the_fallback_locations(self) -> None:
        captured_locations = []

        def fake_fabric_class(*, locations: list[str], silent: str) -> object:
            captured_locations.append(locations)

            class _FakeTF:
                def loadAll(self, silent: str) -> object:  # noqa: N802
                    return object()

            return _FakeTF()

        load_bhsa_api(
            mod="rdtaylorjr/tehillim-parallelism/tf:v1.0",
            use_fn=lambda *a, **k: None,
            fabric_class=fake_fabric_class,
            timeout_seconds=1.0,
        )

        assert len(captured_locations) == 1
        assert len(captured_locations[0]) == 2

    def test_raises_when_both_use_and_the_local_fallback_fail(self) -> None:
        def failing_fabric_class(*, locations: list[str], silent: str) -> object:
            class _FakeTF:
                def loadAll(self, silent: str) -> object:  # noqa: N802
                    return None

            return _FakeTF()

        with pytest.raises(RuntimeError, match="Text-Fabric failed"):
            load_bhsa_api(
                use_fn=lambda *a, **k: None,
                fabric_class=failing_fabric_class,
                timeout_seconds=1.0,
            )


class TestPsalmsBookNode:
    def test_finds_the_psalms_book_node(self) -> None:
        api = _FakeApi(_FakeF({5: "Psalmi", 6: "Genesis"}), _FakeL({}), _FakeT({}))

        assert psalms_book_node(api) == 5

    def test_raises_when_no_psalms_book_is_present(self) -> None:
        api = _FakeApi(_FakeF({6: "Genesis"}), _FakeL({}), _FakeT({}))

        with pytest.raises(RuntimeError, match="Psalmi"):
            psalms_book_node(api)


class TestListPsalmsHalfVerseNodes:
    def test_lists_every_half_verse_node_across_all_chapters(self) -> None:
        api = _api_with_two_psalms()

        assert list_psalms_half_verse_nodes(api) == [100, 101, 200]


class TestListPsalmsHalfVersesByPsalm:
    def test_groups_half_verse_nodes_by_psalm_number(self) -> None:
        api = _api_with_two_psalms()

        assert list_psalms_half_verses_by_psalm(api) == {1: [100, 101], 2: [200]}


class TestNodeToPsalmMap:
    def test_inverts_a_psalm_to_nodes_mapping(self) -> None:
        result = node_to_psalm_map({1: [100, 101], 2: [200]})

        assert result == {100: 1, 101: 1, 200: 2}
