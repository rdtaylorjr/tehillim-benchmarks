import pandas as pd

from parallelism.scripts.build_master_report import build_long_metrics


def _retrieval_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "semantic_foo_consonantal",
                "n_pairs": 10,
                "separation_auc": 0.6,
                "separation_p": 0.01,
                "discrimination_p": 0.02,
                "discrimination_rank_biserial": 0.3,
                "type_gap_z": 1.1,
                "type_gap_p": 0.05,
                "mrr_forward": 0.4,
                "mrr_backward": 0.5,
                "recall_at_1_forward": 0.2,
                "recall_at_5_forward": 0.6,
                "recall_at_10_forward": 0.7,
                "recall_at_1_backward": 0.25,
                "recall_at_5_backward": 0.65,
                "recall_at_10_backward": 0.75,
                "n_pairs_Synonymous": 5,
                "separation_auc_Synonymous": 0.65,
                "separation_p_Synonymous": 0.01,
                "discrimination_p_Synonymous": 0.02,
                "discrimination_rank_biserial_Synonymous": 0.35,
                "mrr_forward_Synonymous": 0.45,
                "mrr_backward_Synonymous": 0.55,
                "recall_at_1_forward_Synonymous": 0.3,
                "recall_at_5_forward_Synonymous": 0.7,
                "recall_at_1_backward_Synonymous": 0.35,
                "recall_at_5_backward_Synonymous": 0.75,
            }
        ]
    )


def _calibration_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "semantic_foo_consonantal",
                "n_pairs": 10,
                "mean_true_similarity": 0.5,
                "median_true_similarity": 0.5,
                "std_true_similarity": 0.1,
                "background_mean": 0.4,
                "background_std": 0.05,
                "background_n_vectors": 100,
                "calibrated_effect_size": 2.0,
                "n_pairs_Synonymous": 5,
                "mean_true_similarity_Synonymous": 0.55,
                "calibrated_effect_size_Synonymous": 3.0,
            }
        ]
    )


def _scope_baseline_df() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "model": "semantic_foo_consonantal",
                "scope": "Synonymous",
                "scope_kind": "type",
                "n_true": 5,
                "n_baseline": 20,
                "prevalence": 0.2,
                "average_precision": 0.45,
                "true_effect_size": 1.0,
                "baseline_effect_size": 0.5,
                "gap": 0.5,
                "auc_vs_baseline": 0.6,
                "p_vs_baseline": 0.03,
            },
            {
                "model": "semantic_foo_consonantal",
                "scope": "overall",
                "scope_kind": "overall",
                "n_true": 10,
                "n_baseline": 20,
                "prevalence": 0.33,
                "average_precision": 0.4,
                "true_effect_size": 0.8,
                "baseline_effect_size": 0.5,
                "gap": 0.3,
                "auc_vs_baseline": 0.55,
                "p_vs_baseline": 0.04,
            },
        ]
    )


def test_build_long_metrics_covers_every_source_and_scope_kind() -> None:
    long_df = build_long_metrics(_retrieval_df(), _calibration_df(), _scope_baseline_df())

    assert set(long_df["scope_kind"]) == {"overall", "type"}
    assert set(long_df["source"]) == {
        "retrieval_separation",
        "calibrated_similarity",
        "vs_baseline",
    }
    assert (long_df["model_base"] == "foo").all()
    assert (long_df["text_variant"] == "consonantal").all()


def test_build_long_metrics_preserves_specific_values() -> None:
    long_df = build_long_metrics(_retrieval_df(), _calibration_df(), _scope_baseline_df())

    def value(source: str, scope: str, metric: str) -> float:
        is_source = long_df["source"] == source
        is_scope = long_df["scope"] == scope
        is_metric = long_df["metric"] == metric
        row = long_df[is_source & is_scope & is_metric]
        assert len(row) == 1
        return float(row["value"].iloc[0])

    assert value("retrieval_separation", "overall", "separation_auc") == 0.6
    assert value("retrieval_separation", "Synonymous", "mrr_forward") == 0.45
    assert value("calibrated_similarity", "overall", "calibrated_effect_size") == 2.0
    assert value("calibrated_similarity", "Synonymous", "calibrated_effect_size") == 3.0
    assert value("vs_baseline", "Synonymous", "average_precision") == 0.45
    assert value("vs_baseline", "Synonymous", "gap") == 0.5


def test_build_long_metrics_adds_fdr_corrected_q_values_for_p_value_metrics() -> None:
    long_df = build_long_metrics(_retrieval_df(), _calibration_df(), _scope_baseline_df())

    p_rows = long_df[long_df["metric"].isin(["separation_p", "p_vs_baseline"])]
    assert p_rows["q_value"].notna().all()
    assert p_rows["q_value_by"].notna().all()

    non_p_rows = long_df[long_df["metric"].isin(["separation_auc", "gap"])]
    assert non_p_rows["q_value_by"].isna().all()

    non_p_rows = long_df[long_df["metric"].isin(["separation_auc", "gap"])]
    assert non_p_rows["q_value"].isna().all()
