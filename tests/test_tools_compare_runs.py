from pathlib import Path

import pandas as pd
import pytest

from tools.compare_runs import (
    EXPECTED_DIFF_COLUMNS,
    ColumnDiff,
    classify_column,
    compare_frames,
    compare_trees,
)


def _write(path: Path, frame: pd.DataFrame) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


class TestClassifyColumn:
    def test_a_ledger_column_is_expected(self) -> None:
        assert classify_column("ap_ci_low") == "expected"

    def test_a_scored_metric_column_is_unexpected(self) -> None:
        assert classify_column("average_precision") == "unexpected"

    def test_every_ledger_entry_classifies_as_expected(self) -> None:
        assert {classify_column(name) for name in EXPECTED_DIFF_COLUMNS} == {"expected"}


class TestCompareFrames:
    def test_identical_frames_report_no_differences(self) -> None:
        frame = pd.DataFrame({"model": ["a", "b"], "average_precision": [0.5, 0.6]})

        assert compare_frames(frame, frame, key_columns=["model"]) == []

    def test_reports_the_changed_column_with_its_largest_delta(self) -> None:
        old = pd.DataFrame({"model": ["a", "b"], "average_precision": [0.5, 0.6]})
        new = pd.DataFrame({"model": ["a", "b"], "average_precision": [0.5, 0.9]})

        diffs = compare_frames(old, new, key_columns=["model"])

        assert diffs == [
            ColumnDiff(
                column="average_precision",
                kind="unexpected",
                n_changed=1,
                max_abs_delta=pytest.approx(0.3),
            )
        ]

    def test_matches_rows_by_key_not_by_position(self) -> None:
        """Parallel runs may emit rows in a different order, which is not a numeric difference."""
        old = pd.DataFrame({"model": ["a", "b"], "average_precision": [0.5, 0.6]})
        reordered = pd.DataFrame({"model": ["b", "a"], "average_precision": [0.6, 0.5]})

        assert compare_frames(old, reordered, key_columns=["model"]) == []

    def test_treats_two_nans_in_the_same_cell_as_equal(self) -> None:
        """An undefined CI is NaN in both runs, which is agreement, not a difference."""
        old = pd.DataFrame({"model": ["a"], "ap_ci_low": [float("nan")]})
        new = pd.DataFrame({"model": ["a"], "ap_ci_low": [float("nan")]})

        assert compare_frames(old, new, key_columns=["model"]) == []

    def test_flags_a_cell_that_became_nan(self) -> None:
        old = pd.DataFrame({"model": ["a"], "average_precision": [0.5]})
        new = pd.DataFrame({"model": ["a"], "average_precision": [float("nan")]})

        diffs = compare_frames(old, new, key_columns=["model"])

        assert [d.column for d in diffs] == ["average_precision"]
        assert diffs[0].n_changed == 1

    def test_labels_a_ledger_column_as_expected(self) -> None:
        old = pd.DataFrame({"model": ["a"], "ap_ci_low": [0.4]})
        new = pd.DataFrame({"model": ["a"], "ap_ci_low": [0.3]})

        assert [d.kind for d in compare_frames(old, new, key_columns=["model"])] == ["expected"]

    def test_reports_rows_present_in_only_one_run(self) -> None:
        old = pd.DataFrame({"model": ["a", "b"], "average_precision": [0.5, 0.6]})
        new = pd.DataFrame({"model": ["a"], "average_precision": [0.5]})

        diffs = compare_frames(old, new, key_columns=["model"])

        assert [d.column for d in diffs] == ["<rows>"]
        assert diffs[0].kind == "unexpected"


class TestCompareTrees:
    def test_reports_a_file_present_in_only_one_tree(self, tmp_path: Path) -> None:
        old_dir, new_dir = tmp_path / "old", tmp_path / "new"
        _write(old_dir / "a.csv", pd.DataFrame({"model": ["m"], "average_precision": [0.1]}))
        new_dir.mkdir()

        report = compare_trees(old_dir, new_dir)

        assert report.missing_in_new == ["a.csv"]
        assert not report.is_clean

    def test_a_matching_tree_with_only_ledger_diffs_is_clean(self, tmp_path: Path) -> None:
        old_dir, new_dir = tmp_path / "old", tmp_path / "new"
        _write(old_dir / "a.csv", pd.DataFrame({"model": ["m"], "ap_ci_low": [0.4]}))
        _write(new_dir / "a.csv", pd.DataFrame({"model": ["m"], "ap_ci_low": [0.3]}))

        report = compare_trees(old_dir, new_dir)

        assert report.is_clean
        assert report.files["a.csv"][0].kind == "expected"

    def test_an_unexpected_diff_makes_the_report_unclean(self, tmp_path: Path) -> None:
        old_dir, new_dir = tmp_path / "old", tmp_path / "new"
        _write(old_dir / "a.csv", pd.DataFrame({"model": ["m"], "average_precision": [0.4]}))
        _write(new_dir / "a.csv", pd.DataFrame({"model": ["m"], "average_precision": [0.3]}))

        assert not compare_trees(old_dir, new_dir).is_clean
