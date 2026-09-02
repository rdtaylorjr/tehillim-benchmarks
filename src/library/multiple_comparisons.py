"""FDR correction (BH and BY), applied within each (source, metric, scope_kind) family."""

from collections.abc import Callable

import numpy as np
import pandas as pd

_P_VALUE_METRICS = frozenset({"separation_p", "discrimination_p", "type_gap_p", "p_vs_baseline"})


def _adjust(p_values: np.ndarray, weight_at_rank: Callable[[int], np.ndarray]) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    raw_q = ranked * weight_at_rank(n)
    monotone_q = np.minimum.accumulate(raw_q[::-1])[::-1]
    clipped = np.clip(monotone_q, 0, 1)
    result = np.empty(n)
    result[order] = clipped
    return result


def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """FDR-adjusted q-values, valid under independence or positive regression dependence."""
    return _adjust(p_values, lambda n: n / np.arange(1, n + 1))


def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """FDR-adjusted q-values, valid under arbitrary dependence; more conservative than BH."""

    def weight(n: int) -> np.ndarray:
        c_n = np.sum(1.0 / np.arange(1, n + 1))
        return n * c_n / np.arange(1, n + 1)

    return _adjust(p_values, weight)


def add_fdr_q_values(long_df: pd.DataFrame) -> pd.DataFrame:
    """Adds BH/BY q-values, corrected within each (source, metric, scope_kind) family."""
    result = long_df.reset_index(drop=True)
    result["q_value"] = np.nan
    result["q_value_by"] = np.nan
    is_p_value = result["metric"].isin(_P_VALUE_METRICS)
    group_cols = ["source", "metric", "scope_kind"]
    # dropna=False so a family with a missing key still gets q-values instead of vanishing.
    for _key, group in result[is_p_value].groupby(group_cols, dropna=False):
        values = group["value"].to_numpy()
        result.loc[group.index, "q_value"] = benjamini_hochberg(values)
        result.loc[group.index, "q_value_by"] = benjamini_yekutieli(values)
    return result
