from parallelism.bootstrap import BootstrapCI
from parallelism.scripts.compute_bootstrap_cis import _row


def _result(**overrides: float | int) -> BootstrapCI:
    defaults: dict[str, float | int] = {
        "point_ap": 0.8,
        "ap_ci_low": 0.7,
        "ap_ci_high": 0.9,
        "ap_ci_low_pct": 0.71,
        "ap_ci_high_pct": 0.89,
        "point_gap": 0.3,
        "gap_ci_low": 0.2,
        "gap_ci_high": 0.4,
        "gap_ci_low_pct": 0.21,
        "gap_ci_high_pct": 0.39,
        "point_auc": 0.75,
        "auc_ci_low": 0.65,
        "auc_ci_high": 0.85,
        "auc_ci_low_pct": 0.66,
        "auc_ci_high_pct": 0.84,
        "prevalence": 0.14,
        "n_valid_resamples": 950,
        "n_valid_jackknife": 149,
    }
    defaults.update(overrides)
    return BootstrapCI(**defaults)  # type: ignore[arg-type]


class TestRow:
    def test_carries_the_model_and_scope_alongside_every_result_field(self) -> None:
        row = _row("bge_m3", "overall", _result())

        assert row["model"] == "bge_m3"
        assert row["scope"] == "overall"
        assert row["point_ap"] == 0.8
        assert row["ap_ci_low"] == 0.7
        assert row["ap_ci_high"] == 0.9
        assert row["point_auc"] == 0.75
        assert row["prevalence"] == 0.14
        assert row["n_valid_resamples"] == 950
        assert row["n_valid_jackknife"] == 149

    def test_flattens_every_field_the_result_dataclass_carries(self) -> None:
        row = _row("bge_m3", "overall", _result())

        # "model" and "scope" plus all 18 BootstrapCI fields.
        assert len(row) == 20

    def test_different_scopes_produce_different_scope_values(self) -> None:
        overall_row = _row("bge_m3", "overall", _result())
        synonymous_row = _row("bge_m3", "Synonymous", _result())

        assert overall_row["scope"] == "overall"
        assert synonymous_row["scope"] == "Synonymous"
