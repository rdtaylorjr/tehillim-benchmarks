"""Holds every entry point to one CLI contract: injectable corpus, shared defaults, argv parsing."""

from __future__ import annotations

import importlib
import inspect
from types import ModuleType

import pytest

ENTRY_POINTS = [
    "genre.scripts.build_master_report",
    "genre.scripts.compare_by_genre",
    "genre.scripts.compare_calibrated",
    "genre.scripts.compare_models",
    "genre.scripts.compute_bootstrap_cis",
    "genre.scripts.export_detail",
    "genre.scripts.shuffle_order_control",
    "parallelism.evaluate",
    "parallelism.scripts.build_master_report",
    "parallelism.scripts.compare_baseline",
    "parallelism.scripts.compare_models",
    "parallelism.scripts.compare_true_similarity",
    "parallelism.scripts.compute_bootstrap_cis",
    "parallelism.scripts.export_detail",
    "parallelism.scripts.shuffle_order_control",
    "trajectory.scripts.compute_profiles",
    "trajectory.scripts.export_ui_rows",
    "trajectory.scripts.validate_against_genre",
    "ui_export.export",
    "ui_export.scripts.build_detail_json",
    "ui_export.scripts.build_ui_page",
]

#: Reading the corpus is the one dependency a test cannot supply as a file, so it is injected.
CORPUS_READING = {
    "genre.scripts.compare_by_genre",
    "genre.scripts.compare_calibrated",
    "genre.scripts.compare_models",
    "genre.scripts.compute_bootstrap_cis",
    "genre.scripts.export_detail",
    "genre.scripts.shuffle_order_control",
    "parallelism.evaluate",
    "parallelism.scripts.compare_baseline",
    "parallelism.scripts.compare_models",
    "parallelism.scripts.compare_true_similarity",
    "parallelism.scripts.compute_bootstrap_cis",
    "parallelism.scripts.export_detail",
    "parallelism.scripts.shuffle_order_control",
    "trajectory.scripts.compute_profiles",
    "trajectory.scripts.validate_against_genre",
    "ui_export.scripts.build_detail_json",
}

#: add_scoring_arguments owns these, so a second declaration is drift, not configuration.
SHARED_OPTIONS = {"--workers", "--n-resamples", "--n-permutations", "--seed"}


def _module(name: str) -> ModuleType:
    return importlib.import_module(name)


class TestSharedOptionsHaveOneDefinition:
    """The shared defaults are asserted in test_library_cli; here no entry point may restate one."""

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    @pytest.mark.parametrize("option", sorted(SHARED_OPTIONS))
    def test_no_entry_point_declares_a_shared_option_itself(self, name: str, option: str) -> None:
        source = inspect.getsource(_module(name).main)

        assert f'add_argument("{option}"' not in source


class TestEveryEntryPointAcceptsArgv:
    @pytest.mark.parametrize("name", ENTRY_POINTS)
    def test_main_takes_argv_so_a_test_never_has_to_patch_sys_argv(self, name: str) -> None:
        signature = inspect.signature(_module(name).main)

        assert "argv" in signature.parameters

    @pytest.mark.parametrize("name", ENTRY_POINTS)
    def test_main_passes_argv_through_rather_than_reading_sys_argv(self, name: str) -> None:
        source = inspect.getsource(_module(name).main)

        assert "parse_args(argv)" in source


class TestEveryCorpusReadingEntryPointInjectsItsLoader:
    @pytest.mark.parametrize("name", sorted(CORPUS_READING))
    def test_the_corpus_loader_is_a_keyword_parameter(self, name: str) -> None:
        signature = inspect.signature(_module(name).main)

        assert signature.parameters["api_factory"].kind is inspect.Parameter.KEYWORD_ONLY

    @pytest.mark.parametrize("name", sorted(CORPUS_READING))
    def test_the_injected_loader_defaults_to_the_real_one(self, name: str) -> None:
        default = inspect.signature(_module(name).main).parameters["api_factory"].default

        assert callable(default)

    @pytest.mark.parametrize("name", sorted(set(ENTRY_POINTS) - CORPUS_READING))
    def test_an_entry_point_reading_only_files_takes_no_loader(self, name: str) -> None:
        signature = inspect.signature(_module(name).main)

        assert "api_factory" not in signature.parameters
