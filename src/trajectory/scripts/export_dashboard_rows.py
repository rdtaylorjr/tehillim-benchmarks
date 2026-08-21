"""Converts validate_against_genre.py's validation.csv into the dashboard's JSON row shape."""

import argparse
import json
from pathlib import Path

import pandas as pd

from library.embeddings import split_model_name

_FIELDS = (
    "raw_gap",
    "raw_p",
    "raw_effect_size",
    "raw_q",
    "length_controlled_gap",
    "length_controlled_p",
    "length_controlled_effect_size",
    "length_controlled_q",
    "length_and_content_controlled_gap",
    "length_and_content_controlled_p",
    "length_and_content_controlled_effect_size",
    "length_and_content_controlled_q",
)


def trajectory_dashboard_rows(validation_df: pd.DataFrame) -> list[dict]:
    """One dashboard row per validation.csv row, with model split into model_base/text_variant."""
    rows = []
    for _, row in validation_df.iterrows():
        model_base, text_variant = split_model_name(row["model"])
        rows.append(
            {
                "model": row["model"],
                "model_base": model_base,
                "text_variant": text_variant,
                "metric": row["metric"],
                **{field: row[field] for field in _FIELDS},
            }
        )
    return rows


_BY_GENRE_FIELDS = (
    "source",
    "genre",
    "gap",
    "p_perm",
    "p_maxT",
    "perm_q",
    "perm_q_by",
    "maxT_q",
    "maxT_q_by",
)


def trajectory_by_genre_dashboard_rows(breakdown_df: pd.DataFrame) -> list[dict]:
    """One dashboard row per validate_against_genre_by_genre.csv row, model split as above."""
    rows = []
    for _, row in breakdown_df.iterrows():
        model_base, text_variant = split_model_name(row["model"])
        rows.append(
            {
                "model": row["model"],
                "model_base": model_base,
                "text_variant": text_variant,
                "metric": row["metric"],
                **{field: row[field] for field in _BY_GENRE_FIELDS},
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("validation_csv", type=Path)
    parser.add_argument("--breakdown-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--breakdown-output", type=Path, default=None)
    args = parser.parse_args()

    validation_df = pd.read_csv(args.validation_csv)
    rows = trajectory_dashboard_rows(validation_df)
    args.output.write_text(json.dumps(rows))
    print(f"wrote {len(rows)} rows to {args.output}")

    if args.breakdown_csv and args.breakdown_output:
        breakdown_df = pd.read_csv(args.breakdown_csv)
        breakdown_rows = trajectory_by_genre_dashboard_rows(breakdown_df)
        args.breakdown_output.write_text(json.dumps(breakdown_rows))
        print(f"wrote {len(breakdown_rows)} by-genre rows to {args.breakdown_output}")


if __name__ == "__main__":
    main()
