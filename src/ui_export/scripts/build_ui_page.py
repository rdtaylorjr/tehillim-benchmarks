"""Assembles the results UI's HTML page from the built bundle, family JSON, and a template."""

import argparse
import json
import re
from pathlib import Path

FamilyData = dict[str, list[dict[str, object]]]
Families = dict[str, FamilyData]

ALL_FAMILY_IDS = ("semantic", "lexical", "phonology", "morphology", "syntax", "discourse")

_EMPTY_FAMILY_DATA: FamilyData = {
    "parallelism_overall": [],
    "parallelism_by_type": [],
    "genre_overall": [],
    "genre_by_genre": [],
    "trajectory": [],
    "trajectory_by_genre": [],
}

_DATA_PATTERN = re.compile(r"/\*UI_DATA_JSON\*/.*?/\*END_UI_DATA_JSON\*/", re.DOTALL)
_BUNDLE_MARKER = "/*UI_BUNDLE_JS*/"


def merge_family_json_files(paths: list[Path]) -> Families:
    """Merges each ui_<family>.json's single top-level family key into one dict."""
    merged: Families = {}
    for path in paths:
        merged.update(json.loads(path.read_text()))
    return merged


def build_ui_page_html(template: str, bundle_js: str, families: Families) -> str:
    """Fills any missing family with empty tables, then injects DATA and the bundle JS."""
    complete = {
        family_id: families.get(family_id, _EMPTY_FAMILY_DATA) for family_id in ALL_FAMILY_IDS
    }
    html = _DATA_PATTERN.sub(json.dumps({"families": complete}), template)
    return html.replace(_BUNDLE_MARKER, bundle_js)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family_json", type=Path, nargs="+")
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    families = merge_family_json_files(args.family_json)
    html = build_ui_page_html(args.template.read_text(), args.bundle.read_text(), families)
    args.output.write_text(html)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
