"""Selects the results dashboard's required columns from one representation family's results."""

import argparse
import json
from pathlib import Path

import pandas as pd

from library.embeddings import split_model_name

_PARALLELISM_OVERALL_COLUMNS = [
    "model",
    "model_base",
    "text_variant",
    "separation_auc",
    "average_precision",
    "calibrated_effect_size",
    "mrr_forward",
    "n_true",
]
_PARALLELISM_BY_TYPE_COLUMNS = [
    "model",
    "model_base",
    "text_variant",
    "scope",
    "separation_auc",
    "average_precision",
    "calibrated_effect_size",
]
_GENRE_OVERALL_COLUMNS = [
    "model",
    "model_base",
    "text_variant",
    "separation_auc",
    "average_precision",
    "same_genre_effect_size",
    "n_same_genre",
]
_GENRE_BY_GENRE_COLUMNS = [
    "model",
    "model_base",
    "text_variant",
    "genre",
    "separation_auc",
    "average_precision",
    "perm_q",
    "maxT_q",
]


def _drop_psalm_level_models(df: pd.DataFrame) -> pd.DataFrame:
    """Excludes _psalm-suffixed models: degenerate for a colon-pair task, only right for genre."""
    return df[~df["model"].str.endswith("_psalm")]


def _add_model_base_and_text_variant(df: pd.DataFrame) -> pd.DataFrame:
    """Derives model_base/text_variant from `model`, for tables that don't already carry them."""
    df = df.copy()
    split = [split_model_name(model) for model in df["model"]]
    df["model_base"] = [base for base, _ in split]
    df["text_variant"] = [variant for _, variant in split]
    return df


def build_family_data(
    parallelism_overall_df: pd.DataFrame,
    parallelism_by_type_df: pd.DataFrame,
    genre_overall_df: pd.DataFrame,
    genre_by_genre_df: pd.DataFrame,
    trajectory_rows: list[dict],
    trajectory_by_genre_rows: list[dict] | None = None,
) -> dict:
    """One family's dashboard payload: the 6 tables the dashboard's tabs render."""
    parallelism_overall_df = _drop_psalm_level_models(parallelism_overall_df)
    parallelism_by_type_df = _drop_psalm_level_models(parallelism_by_type_df)
    genre_by_genre_df = _add_model_base_and_text_variant(genre_by_genre_df)
    return {
        "parallelism_overall": parallelism_overall_df[_PARALLELISM_OVERALL_COLUMNS].to_dict(
            "records"
        ),
        "parallelism_by_type": parallelism_by_type_df[_PARALLELISM_BY_TYPE_COLUMNS].to_dict(
            "records"
        ),
        "genre_overall": genre_overall_df[_GENRE_OVERALL_COLUMNS].to_dict("records"),
        "genre_by_genre": genre_by_genre_df[_GENRE_BY_GENRE_COLUMNS].to_dict("records"),
        "trajectory": trajectory_rows,
        "trajectory_by_genre": trajectory_by_genre_rows or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("family", help="representation family name, e.g. semantic, lexical")
    parser.add_argument("--parallelism-dir", type=Path, required=True)
    parser.add_argument("--genre-dir", type=Path, required=True)
    parser.add_argument("--trajectory-dashboard-rows", type=Path, required=True)
    parser.add_argument("--trajectory-by-genre-rows", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    parallelism_overall_df = pd.read_parquet(
        args.parallelism_dir / "master" / "model_metrics_overall.parquet"
    )
    parallelism_by_type_df = pd.read_parquet(
        args.parallelism_dir / "master" / "model_metrics_by_type.parquet"
    )
    genre_overall_df = pd.read_parquet(args.genre_dir / "master" / "genre_metrics_wide.parquet")
    genre_by_genre_df = pd.read_csv(args.genre_dir / "by_genre.csv")
    trajectory_rows = json.loads(args.trajectory_dashboard_rows.read_text())
    trajectory_by_genre_rows = (
        json.loads(args.trajectory_by_genre_rows.read_text())
        if args.trajectory_by_genre_rows
        else None
    )

    data = build_family_data(
        parallelism_overall_df,
        parallelism_by_type_df,
        genre_overall_df,
        genre_by_genre_df,
        trajectory_rows,
        trajectory_by_genre_rows,
    )
    args.output.write_text(json.dumps({args.family: data}))
    print(f"wrote family={args.family} to {args.output}")


if __name__ == "__main__":
    main()
