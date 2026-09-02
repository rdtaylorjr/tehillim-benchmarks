"""Joins every metric from compare_models/compare_true_similarity/export_detail into one parquet."""

import argparse
from pathlib import Path

import pandas as pd

from library.embeddings import split_model_name
from library.multiple_comparisons import add_fdr_q_values
from library.rows_output import write_dataframe_parquet

_RETRIEVAL_OVERALL_METRICS = [
    "n_pairs",
    "separation_auc",
    "separation_p",
    "discrimination_p",
    "discrimination_rank_biserial",
    "type_gap_z",
    "type_gap_p",
    "mrr_forward",
    "mrr_backward",
    "recall_at_1_forward",
    "recall_at_5_forward",
    "recall_at_10_forward",
    "recall_at_1_backward",
    "recall_at_5_backward",
    "recall_at_10_backward",
]
_RETRIEVAL_TYPE_METRICS = [
    "n_pairs",
    "separation_auc",
    "separation_p",
    "discrimination_p",
    "discrimination_rank_biserial",
    "mrr_forward",
    "mrr_backward",
    "recall_at_1_forward",
    "recall_at_5_forward",
    "recall_at_1_backward",
    "recall_at_5_backward",
]
_CALIBRATION_OVERALL_METRICS = [
    "n_pairs",
    "mean_true_similarity",
    "median_true_similarity",
    "std_true_similarity",
    "background_mean",
    "background_std",
    "background_n_vectors",
    "calibrated_effect_size",
]
_CALIBRATION_TYPE_METRICS = ["n_pairs", "mean_true_similarity", "calibrated_effect_size"]

_TYPES = ("Antithetic", "Emblematic", "Staircase", "Synonymous", "Synthetic")


def _melt_wide_overall(df: pd.DataFrame, metrics: list[str], source: str) -> pd.DataFrame:
    long = df.melt(id_vars=["model"], value_vars=metrics, var_name="metric", value_name="value")
    long["scope"] = "overall"
    long["scope_kind"] = "overall"
    long["source"] = source
    return long


def _melt_wide_by_type(df: pd.DataFrame, metrics: list[str], source: str) -> pd.DataFrame:
    frames = []
    for ptype in _TYPES:
        rename = {f"{m}_{ptype}": m for m in metrics}
        present = [c for c in rename if c in df.columns]
        if not present:
            continue
        sub = df[["model", *present]].rename(columns=rename)
        present_metrics = [rename[c] for c in present]
        long = sub.melt(
            id_vars=["model"], value_vars=present_metrics, var_name="metric", value_name="value"
        )
        long["scope"] = ptype
        long["scope_kind"] = "type"
        long["source"] = source
        frames.append(long)
    return pd.concat(frames, ignore_index=True)


def build_long_metrics(
    retrieval_df: pd.DataFrame, calibration_df: pd.DataFrame, scope_baseline_df: pd.DataFrame
) -> pd.DataFrame:
    """Combines all three metric sources into one tidy (model, scope, metric, value) table."""
    parts = [
        _melt_wide_overall(retrieval_df, _RETRIEVAL_OVERALL_METRICS, "retrieval_separation"),
        _melt_wide_by_type(retrieval_df, _RETRIEVAL_TYPE_METRICS, "retrieval_separation"),
        _melt_wide_overall(calibration_df, _CALIBRATION_OVERALL_METRICS, "calibrated_similarity"),
        _melt_wide_by_type(calibration_df, _CALIBRATION_TYPE_METRICS, "calibrated_similarity"),
    ]
    baseline_metrics = [
        "n_true",
        "n_baseline",
        "prevalence",
        "average_precision",
        "true_effect_size",
        "baseline_effect_size",
        "gap",
        "auc_vs_baseline",
        "p_vs_baseline",
    ]
    baseline_long = scope_baseline_df.melt(
        id_vars=["model", "scope", "scope_kind"],
        value_vars=baseline_metrics,
        var_name="metric",
        value_name="value",
    )
    baseline_long["source"] = "vs_baseline"
    parts.append(baseline_long)

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


def _pivot_wide(long_df: pd.DataFrame, scope_kind: str, index_cols: list[str]) -> pd.DataFrame:
    """Pivots one scope_kind's rows wide on metric, adding `<metric>_q`/`_q_by` per p-value."""
    subset = long_df[long_df["scope_kind"] == scope_kind]
    values_wide = subset.pivot_table(index=index_cols, columns="metric", values="value")
    for q_column, suffix in (("q_value", "_q"), ("q_value_by", "_q_by")):
        q_subset = subset.dropna(subset=[q_column])
        if q_subset.empty:
            continue
        q_wide = q_subset.pivot_table(index=index_cols, columns="metric", values=q_column)
        q_wide.columns = [f"{col}{suffix}" for col in q_wide.columns]
        values_wide = values_wide.join(q_wide)
    return values_wide.reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-csv", type=Path, required=True)
    parser.add_argument("--calibration-csv", type=Path, required=True)
    parser.add_argument("--detail-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    retrieval_df = pd.read_csv(args.retrieval_csv)
    calibration_df = pd.read_csv(args.calibration_csv)
    scope_baseline_df = pd.read_parquet(args.detail_dir / "type_vs_baseline.parquet")

    long_df = build_long_metrics(retrieval_df, calibration_df, scope_baseline_df)
    write_dataframe_parquet(args.output_dir / "model_metrics_long.parquet", long_df)

    overall_wide = _pivot_wide(long_df, "overall", ["model", "model_base", "text_variant"])
    write_dataframe_parquet(args.output_dir / "model_metrics_overall.parquet", overall_wide)

    by_type_wide = _pivot_wide(long_df, "type", ["model", "model_base", "text_variant", "scope"])
    write_dataframe_parquet(args.output_dir / "model_metrics_by_type.parquet", by_type_wide)

    print(f"model_metrics_long: {len(long_df)} rows")
    print(f"model_metrics_overall: {len(overall_wide)} rows")
    print(f"model_metrics_by_type: {len(by_type_wide)} rows")


if __name__ == "__main__":
    main()
