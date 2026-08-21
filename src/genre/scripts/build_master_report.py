"""Joins the AP/AUC/calibration summary and bootstrap CIs into one long table with BY-FDR q's."""

import argparse
import shutil
from pathlib import Path

import pandas as pd

from library.embeddings import split_model_name
from library.multiple_comparisons import add_fdr_q_values

_SUMMARY_METRICS = [
    "n_same_genre",
    "n_different_genre",
    "prevalence",
    "average_precision",
    "same_genre_effect_size",
    "different_genre_effect_size",
    "gap",
    "separation_auc",
    "separation_p",
]
_BOOTSTRAP_METRICS = [
    "point_ap",
    "ap_ci_low",
    "ap_ci_high",
    "ap_ci_low_pct",
    "ap_ci_high_pct",
    "point_gap",
    "gap_ci_low",
    "gap_ci_high",
    "gap_ci_low_pct",
    "gap_ci_high_pct",
    "point_auc",
    "auc_ci_low",
    "auc_ci_high",
    "auc_ci_low_pct",
    "auc_ci_high_pct",
    "n_valid_resamples",
    "n_valid_jackknife",
]


def _melt_wide(df: pd.DataFrame, metrics: list[str], source: str) -> pd.DataFrame:
    long = df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="value")
    long["scope"] = "overall"
    long["scope_kind"] = "overall"
    long["source"] = source
    return long


def build_long_metrics(summary_df: pd.DataFrame, bootstrap_df: pd.DataFrame) -> pd.DataFrame:
    """Combines the AP/AUC/calibration summary and bootstrap CIs into one tidy long table."""
    parts = [
        _melt_wide(summary_df, _SUMMARY_METRICS, "genre_discrimination"),
        _melt_wide(bootstrap_df, _BOOTSTRAP_METRICS, "bootstrap_ci"),
    ]
    long_df = pd.concat(parts, ignore_index=True)
    base_variant = long_df["model"].apply(lambda m: pd.Series(split_model_name(m)))
    long_df["model_base"] = base_variant[0]
    long_df["text_variant"] = base_variant[1]
    long_df = add_fdr_q_values(long_df)
    return long_df[
        [
            "model",
            "model_base",
            "text_variant",
            "scope",
            "scope_kind",
            "source",
            "metric",
            "value",
            "q_value",
            "q_value_by",
        ]
    ]


def _pivot_wide(long_df: pd.DataFrame) -> pd.DataFrame:
    """Pivots the long table wide on metric, adding `<metric>_q`/`_q_by` columns per p-value."""
    index_cols = ["model", "model_base", "text_variant"]
    values_wide = long_df.pivot_table(index=index_cols, columns="metric", values="value")
    for q_column, suffix in (("q_value", "_q"), ("q_value_by", "_q_by")):
        q_subset = long_df.dropna(subset=[q_column])
        if q_subset.empty:
            continue
        q_wide = q_subset.pivot_table(index=index_cols, columns="metric", values=q_column)
        q_wide.columns = [f"{col}{suffix}" for col in q_wide.columns]
        values_wide = values_wide.join(q_wide)
    return values_wide.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-csv", type=Path, required=True)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = pd.read_csv(args.summary_csv)
    bootstrap_df = pd.read_csv(args.bootstrap_csv)

    long_df = build_long_metrics(summary_df, bootstrap_df)
    long_df.to_parquet(args.output_dir / "genre_metrics_long.parquet", index=False)

    wide_df = _pivot_wide(long_df)
    wide_df.to_parquet(args.output_dir / "genre_metrics_wide.parquet", index=False)

    shutil.copyfile(
        args.detail_dir / "genre_pair_detail.parquet",
        args.output_dir / "genre_pair_detail.parquet",
    )

    print(f"genre_metrics_long: {len(long_df)} rows")
    print(f"genre_metrics_wide: {len(wide_df)} rows")


if __name__ == "__main__":
    main()
