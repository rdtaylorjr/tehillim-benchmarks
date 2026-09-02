"""Permutation test of within vs between-genre psalm distance, apart from the AP/AUC benchmark."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genre.genre_labels import load_genre_by_psalm
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from library.multiple_comparisons import add_fdr_q_values, benjamini_hochberg, benjamini_yekutieli
from trajectory.genre_breakdown import joint_genre_breakdown_permutation_test

_METRICS = (
    "content_distance",
    "structural_distance",
    "adjacent_similarity_distance",
    "step_magnitude_distance",
    "turning_angle_distance",
)
_SOURCES = ("raw", "length_controlled", "length_and_content_controlled")
# Both sides need two pairs before a gap between their means means anything.
_MIN_PAIRS_PER_SIDE = 2


def observed_gap(distances: np.ndarray, same_genre: np.ndarray) -> float:
    """mean(between-genre distance) - mean(within-genre distance); positive means closer within."""
    return float(distances[~same_genre].mean() - distances[same_genre].mean())


# Psalms are the exchangeable unit and residuals live on pairs, so the permutation acts on labels.
def residualize_on_covariates(response: np.ndarray, covariates: np.ndarray) -> np.ndarray:
    """OLS-residualizes response on one or more covariate columns, jointly (Still-White 1981)."""
    design = np.column_stack([np.ones(len(response)), covariates])
    coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
    return np.asarray(response - design @ coefficients)


def residualize_by_length(distances: np.ndarray, length_diff: np.ndarray) -> np.ndarray:
    """OLS-residualizes distances on |length difference|, a Still-White (1981) nuisance fix."""
    return residualize_on_covariates(distances, length_diff.reshape(-1, 1))


def _null_gaps(same_matrix: np.ndarray, distances: np.ndarray) -> np.ndarray:
    """observed_gap(distances, row) for every row of a (n_permutations, n_pairs) same/diff array."""
    distances = distances.astype(np.float64, copy=False)
    same_sum = same_matrix.astype(np.float64) @ distances
    same_count = same_matrix.sum(axis=1).astype(np.float64)
    total_sum = distances.sum()
    total_count = float(len(distances))
    diff_sum = total_sum - same_sum
    diff_count = total_count - same_count
    return np.asarray((diff_sum / diff_count) - (same_sum / same_count))


def same_genre_matrix(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    genre_labels: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Whether each shuffled pair shares a genre; identical for every source of one metric."""
    _, codes = np.unique(genre_labels, return_inverse=True)
    shuffled_codes = rng.permuted(np.tile(codes, (n_permutations, 1)), axis=1)
    return np.asarray(shuffled_codes[:, idx_a] == shuffled_codes[:, idx_b])


def permutation_test(
    idx_a: np.ndarray,
    idx_b: np.ndarray,
    distances: np.ndarray,
    genre_labels: np.ndarray,
    n_permutations: int = 10000,
    rng: np.random.Generator | None = None,
    same_matrix: np.ndarray | None = None,
) -> tuple[float, float, float]:
    """Observed gap, p-value, and a null-calibrated effect size (NaN if null has no variance)."""
    rng = rng if rng is not None else np.random.default_rng()
    same_genre = genre_labels[idx_a] == genre_labels[idx_b]
    observed = observed_gap(distances, same_genre)

    if same_matrix is None:
        same_matrix = same_genre_matrix(idx_a, idx_b, genre_labels, n_permutations, rng)
    null_gaps = _null_gaps(same_matrix, distances)

    p_value = (np.sum(null_gaps >= observed) + 1) / (len(null_gaps) + 1)
    null_std = null_gaps.std(ddof=1)
    effect_size = (observed - null_gaps.mean()) / null_std if null_std > 0 else float("nan")
    return observed, float(p_value), float(effect_size)


def _subset_and_length_diff(
    metric: str,
    base_subset: pd.DataFrame,
    index_of: dict[int, int],
    n_half_verses: dict[int, int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Filters to non-NaN pairs for this metric, returning subset, idx_a/idx_b, |length diff|."""
    has_content_covariate = metric != "content_distance"
    required = [metric, "content_distance"] if has_content_covariate else [metric]
    subset = base_subset[base_subset[required].notna().all(axis=1)]
    idx_a = subset["psalm_a"].map(index_of).to_numpy()
    idx_b = subset["psalm_b"].map(index_of).to_numpy()
    length_diff = (
        (subset["psalm_a"].map(n_half_verses) - subset["psalm_b"].map(n_half_verses))
        .abs()
        .to_numpy()
    )
    return subset, idx_a, idx_b, length_diff


def _values_by_source(
    metric: str, subset: pd.DataFrame, length_diff: np.ndarray
) -> dict[str, np.ndarray | None]:
    """raw/length_controlled/length_and_content_controlled value arrays for this metric's subset."""
    has_content_covariate = metric != "content_distance"
    raw = subset[metric].to_numpy()
    values_by_source: dict[str, np.ndarray | None] = {
        "raw": raw,
        "length_controlled": residualize_by_length(raw, length_diff),
        "length_and_content_controlled": None,
    }
    if has_content_covariate:
        content = subset["content_distance"].to_numpy()
        covariates = np.column_stack([length_diff, content])
        values_by_source["length_and_content_controlled"] = residualize_on_covariates(
            raw, covariates
        )
    return values_by_source


def build_validation_row(
    model: str,
    metric: str,
    base_subset: pd.DataFrame,
    index_of: dict[int, int],
    genre_labels: np.ndarray,
    n_half_verses: dict[int, int],
    n_permutations: int,
    seed: int,
) -> dict[str, str | int | float]:
    """One (model, metric) row of gap+p per source in _SOURCES, excluding NaN-valued pairs."""
    subset, idx_a, idx_b, length_diff = _subset_and_length_diff(
        metric, base_subset, index_of, n_half_verses
    )
    same_genre = genre_labels[idx_a] == genre_labels[idx_b]

    row: dict[str, str | int | float] = {
        "model": model,
        "metric": metric,
        "n_pairs_total": len(base_subset),
        "n_pairs_valid": len(subset),
    }
    if (
        int(same_genre.sum()) < _MIN_PAIRS_PER_SIDE
        or int((~same_genre).sum()) < _MIN_PAIRS_PER_SIDE
    ):
        for source in _SOURCES:
            row[f"{source}_gap"] = float("nan")
            row[f"{source}_p"] = float("nan")
            row[f"{source}_effect_size"] = float("nan")
        return row

    values_by_source = _values_by_source(metric, subset, length_diff)
    # The shuffle depends only on the pairs and the seed, so all three sources share one matrix.
    shared_matrix = same_genre_matrix(
        idx_a, idx_b, genre_labels, n_permutations, np.random.default_rng(seed)
    )

    for source in _SOURCES:
        values = values_by_source[source]
        if values is None:
            row[f"{source}_gap"] = float("nan")
            row[f"{source}_p"] = float("nan")
            row[f"{source}_effect_size"] = float("nan")
            continue
        gap, p_value, effect_size = permutation_test(
            idx_a,
            idx_b,
            values,
            genre_labels,
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed),
            same_matrix=shared_matrix,
        )
        row[f"{source}_gap"] = gap
        row[f"{source}_p"] = p_value
        row[f"{source}_effect_size"] = effect_size
    return row


def build_genre_breakdown_rows(
    model: str,
    metric: str,
    base_subset: pd.DataFrame,
    index_of: dict[int, int],
    genre_labels: np.ndarray,
    n_half_verses: dict[int, int],
    n_permutations: int,
    seed: int,
) -> list[dict[str, str | int | float]]:
    """One row per (genre, available source): one-vs-rest distance gap, perm p, and maxT p."""
    subset, idx_a, idx_b, length_diff = _subset_and_length_diff(
        metric, base_subset, index_of, n_half_verses
    )
    if len(subset) == 0:
        return []

    genres_array, genre_codes = np.unique(genre_labels, return_inverse=True)
    genres = tuple(genres_array.tolist())
    values_by_source = _values_by_source(metric, subset, length_diff)

    rows: list[dict[str, str | int | float]] = []
    for source in _SOURCES:
        values = values_by_source[source]
        if values is None:
            continue
        result = joint_genre_breakdown_permutation_test(
            values,
            idx_a,
            idx_b,
            genre_codes,
            genres,
            n_permutations=n_permutations,
            rng=np.random.default_rng(seed),
        )
        for genre, gap, p_perm, p_maxt in zip(
            result.genres, result.gap_observed, result.p_perm, result.p_maxT, strict=True
        ):
            rows.append(
                {
                    "model": model,
                    "metric": metric,
                    "source": source,
                    "genre": genre,
                    "gap": gap,
                    "p_perm": p_perm,
                    "p_maxT": p_maxt,
                }
            )
    return rows


def add_fdr_columns(rows: list[dict[str, str | int | float]]) -> pd.DataFrame:
    """Adds BH/BY q-values to every source's p-values, corrected within each metric's family."""
    df = pd.DataFrame(rows)
    long_parts = [
        pd.DataFrame(
            {
                "model": df["model"],
                "scope_kind": df["metric"],
                "source": source,
                "metric": "separation_p",
                "value": df[f"{source}_p"],
            }
        ).loc[lambda d: d["value"].notna()]
        for source in _SOURCES
    ]
    long_df = add_fdr_q_values(pd.concat(long_parts, ignore_index=True))

    result = df.copy()
    for source in _SOURCES:
        q_columns = long_df[long_df["source"] == source][
            ["model", "scope_kind", "q_value", "q_value_by"]
        ]
        q_columns = q_columns.rename(
            columns={
                "scope_kind": "metric",
                "q_value": f"{source}_q",
                "q_value_by": f"{source}_q_by",
            }
        )
        result = result.merge(q_columns, on=["model", "metric"], how="left")
    return result


def add_genre_breakdown_fdr_columns(rows: list[dict[str, str | int | float]]) -> pd.DataFrame:
    """Adds BH/BY q-values to p_perm/p_maxT, corrected within each (metric, source, genre)."""
    df = pd.DataFrame(rows)
    df["perm_q"] = np.nan
    df["perm_q_by"] = np.nan
    df["maxT_q"] = np.nan
    df["maxT_q_by"] = np.nan
    group_cols = ["metric", "source", "genre"]
    for _key, group in df.groupby(group_cols):
        df.loc[group.index, "perm_q"] = benjamini_hochberg(group["p_perm"].to_numpy())
        df.loc[group.index, "perm_q_by"] = benjamini_yekutieli(group["p_perm"].to_numpy())
        df.loc[group.index, "maxT_q"] = benjamini_hochberg(group["p_maxT"].to_numpy())
        df.loc[group.index, "maxT_q_by"] = benjamini_yekutieli(group["p_maxT"].to_numpy())
    return df


def breakdown_path_for(output: Path) -> Path:
    """The per-genre breakdown sits beside its validation CSV, since the two are always one run."""
    return output.with_name(f"{output.stem}_by_genre{output.suffix}")


def validate_one_model(
    model: str,
    group: pd.DataFrame,
    genre_by_psalm: dict[int, str],
    n_half_verses: dict[int, int],
    n_permutations: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, str | int | float]]]:
    """Every validation and breakdown row for one model, independent of every other model."""
    psalms = sorted(set(group["psalm_a"]) | set(group["psalm_b"]))
    usable_psalms = [p for p in psalms if p in genre_by_psalm]
    index_of = {psalm: i for i, psalm in enumerate(usable_psalms)}
    mask = group["psalm_a"].isin(usable_psalms) & group["psalm_b"].isin(usable_psalms)
    base_subset = group[mask]
    genre_labels = np.array([genre_by_psalm[psalm] for psalm in usable_psalms])

    rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, str | int | float]] = []
    for metric in _METRICS:
        rows.append(
            build_validation_row(
                model,
                metric,
                base_subset,
                index_of,
                genre_labels,
                n_half_verses,
                n_permutations,
                seed,
            )
        )
        breakdown_rows.extend(
            build_genre_breakdown_rows(
                model,
                metric,
                base_subset,
                index_of,
                genre_labels,
                n_half_verses,
                n_permutations,
                seed,
            )
        )
    return rows, breakdown_rows


def validate_models(
    distances_df: pd.DataFrame,
    genre_by_psalm: dict[int, str],
    n_half_verses: dict[int, int],
    n_permutations: int,
    seed: int,
    max_workers: int = 4,
) -> tuple[list[dict[str, Any]], list[dict[str, str | int | float]]]:
    """Models are independent, so they run across workers and are reassembled in submit order."""
    groups = [(model, group) for model, group in distances_df.groupby("model")]
    rows: list[dict[str, Any]] = []
    breakdown: list[dict[str, str | int | float]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [
            pool.submit(
                validate_one_model,
                model,
                group,
                genre_by_psalm,
                n_half_verses,
                n_permutations,
                seed,
            )
            for model, group in groups
        ]
        # Results are collected in submission order, so the output never depends on worker timing.
        for future in futures:
            model_rows, model_breakdown = future.result()
            rows.extend(model_rows)
            breakdown.extend(model_breakdown)
    return rows, breakdown


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "genre_csv",
        type=Path,
        help="third-party genre CSV, e.g. psalms-browser.csv (not in this repo)",
    )
    parser.add_argument("distances_parquet", type=Path, help="trajectory_distances.parquet")
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA checkout spec")
    parser.add_argument("--n-permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, default=None)
    # The breakdown is computed either way, so it is written beside --output by default.
    parser.add_argument("--breakdown-output", type=Path, default=None)
    args = parser.parse_args()

    api = load_bhsa_api(args.checkout)
    n_half_verses = {p: len(nodes) for p, nodes in list_psalms_half_verses_by_psalm(api).items()}
    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    distances_df = pd.read_parquet(args.distances_parquet)

    rows, all_breakdown_rows = validate_models(
        distances_df, genre_by_psalm, n_half_verses, args.n_permutations, args.seed, args.workers
    )

    result_df = add_fdr_columns(rows)

    for _, row in result_df.iterrows():
        header = f"{row['model']:55s} {row['metric']:28s}"
        counts = f"n_valid={row['n_pairs_valid']}/{row['n_pairs_total']}"
        parts = [f"{header} {counts}"]
        parts.extend(
            f"{source}_p={row[f'{source}_p']:.4f} q={row[f'{source}_q']:.4f} "
            f"z={row[f'{source}_effect_size']:.3f}"
            for source in _SOURCES
        )
        print(" ".join(parts))

    if args.output:
        result_df.to_csv(args.output, index=False)

    if all_breakdown_rows:
        breakdown_output = args.breakdown_output or (
            breakdown_path_for(args.output) if args.output else None
        )
        if breakdown_output:
            breakdown_df = add_genre_breakdown_fdr_columns(all_breakdown_rows)
            breakdown_df.to_csv(breakdown_output, index=False)


if __name__ == "__main__":
    main()
