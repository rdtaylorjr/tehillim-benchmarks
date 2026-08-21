from pathlib import Path

import pytest

from genre.genre_labels import load_genre_by_psalm


def _write_csv(path: Path, rows: list[tuple[str, str, str]]) -> None:
    lines = ["Psalm,Attribution,Genre,Structure,Tag,SupplementalDataId"]
    for psalm, attribution, genre in rows:
        lines.append(f'"{psalm}","{attribution}","{genre}","Chiasm","",1')
    path.write_text("\n".join(lines) + "\n")


def test_parses_the_psalm_number_out_of_the_ps_n_column(tmp_path: Path) -> None:
    path = tmp_path / "psalms-browser.csv"
    _write_csv(path, [("Ps 10", "David", "Lament")])

    result = load_genre_by_psalm(path)

    assert result == {10: "Lament"}


def test_parses_every_row(tmp_path: Path) -> None:
    path = tmp_path / "psalms-browser.csv"
    _write_csv(
        path,
        [
            ("Ps 1", "Anonymous", "Wisdom"),
            ("Ps 100", "Anonymous", "Praise"),
            ("Ps 150", "Anonymous", "Praise"),
        ],
    )

    result = load_genre_by_psalm(path)

    assert result == {1: "Wisdom", 100: "Praise", 150: "Praise"}


def test_raises_on_a_psalm_number_that_does_not_parse(tmp_path: Path) -> None:
    path = tmp_path / "psalms-browser.csv"
    path.write_text(
        "Psalm,Attribution,Genre,Structure,Tag,SupplementalDataId\n"
        '"Not A Psalm","David","Lament","Chiasm","",1\n'
    )

    with pytest.raises(ValueError, match="Not A Psalm"):
        load_genre_by_psalm(path)
