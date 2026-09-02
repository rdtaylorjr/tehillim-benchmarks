import csv
from pathlib import Path

from library.rows_output import write_rows_csv


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def test_writes_a_header_and_one_line_per_row(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"

    write_rows_csv(path, [{"model": "a", "auc": 0.5}, {"model": "b", "auc": 0.6}])

    header, rows = _read(path)
    assert header == ["model", "auc"]
    assert [row["model"] for row in rows] == ["a", "b"]


def test_columns_follow_first_appearance_across_all_rows(tmp_path: Path) -> None:
    """Per-type metrics appear only on the models that have that type, so the header is a union."""
    path = tmp_path / "out.csv"

    write_rows_csv(path, [{"model": "a", "x": 1}, {"model": "b", "y": 2}])

    header, _ = _read(path)
    assert header == ["model", "x", "y"]


def test_a_row_missing_a_column_writes_an_empty_cell(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"

    write_rows_csv(path, [{"model": "a", "x": 1}, {"model": "b"}])

    _, rows = _read(path)
    assert rows[1]["x"] == ""


def test_writes_only_a_header_for_an_empty_row_list(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"

    write_rows_csv(path, [])

    assert path.read_text() == ""


def test_creates_the_parent_directory(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "out.csv"

    write_rows_csv(path, [{"model": "a"}])

    assert path.exists()
