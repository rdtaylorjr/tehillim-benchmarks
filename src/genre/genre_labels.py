"""Parses a third-party psalm genre classification (psalms-browser.csv, not in this repo)."""

import csv
import re
from pathlib import Path

_PSALM_NUMBER = re.compile(r"^Ps (\d+)$")


def load_genre_by_psalm(path: Path) -> dict[int, str]:
    """Reads Psalm/Genre columns from a psalms-browser.csv export into {psalm_number: genre}."""
    result: dict[int, str] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            match = _PSALM_NUMBER.match(row["Psalm"])
            if match is None:
                raise ValueError(f"could not parse a psalm number from {row['Psalm']!r}")
            result[int(match.group(1))] = row["Genre"]
    return result
