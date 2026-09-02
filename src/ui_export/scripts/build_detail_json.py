"""Batch-generates one row-click detail JSON per model, for every model any ui table links to."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from genre.genre_labels import load_genre_by_psalm
from library.bhsa import DEFAULT_CHECKOUT, list_psalms_half_verses_by_psalm, load_bhsa_api
from trajectory.scripts.validate_against_genre import (
    residualize_by_length,
    residualize_on_covariates,
)
from ui_export.detail import (
    build_genre_detail,
    build_parallelism_detail,
    build_trajectory_detail,
    load_auc_ap_ci,
    load_validated_gap_stats,
)

_TRAJECTORY_SOURCES = ("length_controlled", "length_and_content_controlled")


def table_model_sets(domain_json: dict[str, Any]) -> dict[str, set[str]]:
    """Each table's own model set, kept separate so a section is built only where used."""
    return {
        "parallelism": {r["model"] for r in domain_json["parallelism_overall"]},
        "genre": {r["model"] for r in domain_json["genre_overall"]},
        "trajectory": {r["model"] for r in domain_json["trajectory"]},
    }


def choose_primary_metric(validate_df: pd.DataFrame, model: str) -> str | None:
    """Picks the metric with the strongest (smallest) length-controlled p-value for one model."""
    # content_distance excluded: its content-controlled source residualizes against itself, NaN.
    rows = validate_df[
        (validate_df.model == model) & (validate_df.metric != "content_distance")
    ].dropna(subset=["length_controlled_p"])
    if rows.empty:
        return None
    return str(rows.loc[rows.length_controlled_p.idxmin(), "metric"])


def attach_genre_columns(df: pd.DataFrame, genre_by_psalm: dict[int, str]) -> pd.DataFrame:
    """Adds genre_a/genre_b/same_genre, joined in-memory only (never persisted to a repo file)."""
    df = df.copy()
    df["genre_a"] = df.psalm_a.map(genre_by_psalm)
    df["genre_b"] = df.psalm_b.map(genre_by_psalm)
    df["same_genre"] = df.genre_a == df.genre_b
    return df


def residualize_trajectory_metric(
    traj_df: pd.DataFrame, metric: str, n_half_verses: dict[int, int]
) -> pd.DataFrame:
    """Adds length_controlled/length_and_content_controlled columns for one metric, per model."""
    df = traj_df[traj_df[[metric, "content_distance"]].notna().all(axis=1)].copy()
    length_diff = (df.psalm_a.map(n_half_verses) - df.psalm_b.map(n_half_verses)).abs().to_numpy()
    raw = df[metric].to_numpy()
    content = df["content_distance"].to_numpy()
    df["length_controlled"] = residualize_by_length(raw, length_diff)
    df["length_and_content_controlled"] = residualize_on_covariates(
        raw, np.column_stack([length_diff, content])
    )
    return df


SECTIONS = ("parallelism", "genre", "trajectory")
# A payload holding only "model" and "domain" gained no section worth writing.
_EMPTY_PAYLOAD_KEYS = 2


def split_sections(payload: dict[str, Any], output_dir: Path) -> list[Path]:
    """Writes one file per section, since the detail view renders exactly one of them at a time."""
    written: list[Path] = []
    for section in SECTIONS:
        if section not in payload:
            continue
        body = {"model": payload["model"], "domain": payload["domain"], section: payload[section]}
        output_dir.mkdir(parents=True, exist_ok=True)
        path = output_dir / f"detail_{payload['domain']}_{payload['model']}_{section}.json"
        path.write_text(json.dumps(body, allow_nan=False))
        written.append(path)
    return written


@dataclass(frozen=True, slots=True)
class ModelDetailInputs:
    """One model's per-section frames and precomputed statistics, absent where it has no data."""

    pair_detail: pd.DataFrame | None = None
    baseline_detail: pd.DataFrame | None = None
    genre_pair: pd.DataFrame | None = None
    trajectory: pd.DataFrame | None = None
    trajectory_metric: str | None = None
    parallelism_auc_ap: dict[str, Any] | None = None
    genre_auc_ap: dict[str, Any] | None = None
    gap_stats: dict[str, dict[str, float]] | None = None


def build_one_model(
    model: str,
    domain: str,
    output_dir: Path,
    genres: list[str],
    inputs: ModelDetailInputs,
) -> int:
    """Writes whichever sections have real data for one model, returning how many files it wrote."""
    payload: dict[str, Any] = {"model": model, "domain": domain}
    if (
        inputs.pair_detail is not None
        and not inputs.pair_detail.empty
        and inputs.baseline_detail is not None
    ):
        payload["parallelism"] = build_parallelism_detail(
            inputs.pair_detail, inputs.baseline_detail, inputs.parallelism_auc_ap
        )
    if inputs.genre_pair is not None and not inputs.genre_pair.empty:
        payload["genre"] = build_genre_detail(inputs.genre_pair, genres, inputs.genre_auc_ap)
    if (
        inputs.trajectory is not None
        and not inputs.trajectory.empty
        and inputs.trajectory_metric is not None
    ):
        payload["trajectory"] = build_trajectory_detail(
            inputs.trajectory, inputs.trajectory_metric, genres, inputs.gap_stats
        )
    if len(payload) <= _EMPTY_PAYLOAD_KEYS:
        return 0
    return len(split_sections(payload, output_dir))


def build_domain(
    domain: str,
    data_dir: Path,
    domain_json: dict[str, Any],
    genre_by_psalm: dict[int, str],
    n_half_verses: dict[int, int],
    output_dir: Path,
    max_workers: int,
) -> int:
    """Builds every model's detail JSON for one domain; returns the count of files written."""
    genres = sorted(set(genre_by_psalm.values()))
    table_models = table_model_sets(domain_json)
    models = table_models["parallelism"] | table_models["genre"] | table_models["trajectory"]

    pair_detail_path = (
        data_dir / f"benchmark=parallelism/domain={domain}/stage=detail/pair_detail.parquet"
    )
    baseline_detail_path = (
        data_dir / f"benchmark=parallelism/domain={domain}/stage=detail/baseline_detail.parquet"
    )
    genre_pair_path = (
        data_dir / f"benchmark=genre/domain={domain}/stage=detail/genre_pair_detail.parquet"
    )
    trajectory_path = (
        data_dir
        / f"benchmark=trajectory/domain={domain}/stage=profiles/trajectory_distances.parquet"
    )
    parallelism_ci_path = (
        data_dir / f"benchmark=parallelism/domain={domain}/stage=raw/bootstrap_cis.csv"
    )
    genre_ci_path = data_dir / f"benchmark=genre/domain={domain}/stage=raw/bootstrap_cis.csv"
    validate_path = (
        data_dir / f"benchmark=trajectory/domain={domain}/stage=raw/validate_against_genre.csv"
    )

    pair_detail_by_model = dict(tuple(pd.read_parquet(pair_detail_path).groupby("model")))
    baseline_detail_by_model = dict(tuple(pd.read_parquet(baseline_detail_path).groupby("model")))
    genre_pair_by_model = (
        dict(
            tuple(
                attach_genre_columns(pd.read_parquet(genre_pair_path), genre_by_psalm).groupby(
                    "model"
                )
            )
        )
        if genre_pair_path.exists()
        else {}
    )
    trajectory_by_model = dict(
        tuple(
            attach_genre_columns(pd.read_parquet(trajectory_path), genre_by_psalm).groupby("model")
        )
    )
    parallelism_ci_df = pd.read_csv(parallelism_ci_path) if parallelism_ci_path.exists() else None
    genre_ci_df = pd.read_csv(genre_ci_path) if genre_ci_path.exists() else None
    validate_df = pd.read_csv(validate_path)

    written = 0
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = []
        for model in sorted(models):
            trajectory_model = (
                trajectory_by_model.get(model) if model in table_models["trajectory"] else None
            )
            metric = (
                choose_primary_metric(validate_df, model) if trajectory_model is not None else None
            )
            if trajectory_model is not None and metric is not None:
                trajectory_model = residualize_trajectory_metric(
                    trajectory_model, metric, n_half_verses
                )
            pair_detail_model = (
                pair_detail_by_model.get(model) if model in table_models["parallelism"] else None
            )
            baseline_detail_model = (
                baseline_detail_by_model.get(model)
                if model in table_models["parallelism"]
                else None
            )
            genre_pair_model = (
                genre_pair_by_model.get(model) if model in table_models["genre"] else None
            )
            futures.append(
                pool.submit(
                    build_one_model,
                    model,
                    domain,
                    output_dir,
                    genres,
                    ModelDetailInputs(
                        pair_detail=pair_detail_model,
                        baseline_detail=baseline_detail_model,
                        genre_pair=genre_pair_model,
                        trajectory=trajectory_model,
                        trajectory_metric=metric,
                        parallelism_auc_ap=load_auc_ap_ci(parallelism_ci_df, model, "overall")
                        if parallelism_ci_df is not None
                        else None,
                        genre_auc_ap=load_auc_ap_ci(genre_ci_df, model, None)
                        if genre_ci_df is not None
                        else None,
                        gap_stats=load_validated_gap_stats(validate_df, model, metric)
                        if metric is not None
                        else None,
                    ),
                )
            )
        for future in futures:
            written += future.result()
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "genre_csv", type=Path, help="third-party genre CSV, e.g. psalms-browser.csv"
    )
    parser.add_argument("--data-dir", type=Path, required=True, help="tehillim-data checkout root")
    parser.add_argument(
        "--ui-dir",
        type=Path,
        required=True,
        help="directory holding ui_<domain>.json, which the site carries, not the data repo",
    )
    parser.add_argument(
        "--domains", nargs="+", default=["lexical", "morphology", "semantic", "syntax"]
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    genre_by_psalm = load_genre_by_psalm(args.genre_csv)
    api = load_bhsa_api(args.checkout)
    n_half_verses = {
        psalm: len(nodes) for psalm, nodes in list_psalms_half_verses_by_psalm(api).items()
    }

    for domain in args.domains:
        domain_json_path = args.ui_dir / f"ui_{domain}.json"
        domain_json = json.loads(domain_json_path.read_text())[domain]
        written = build_domain(
            domain,
            args.data_dir,
            domain_json,
            genre_by_psalm,
            n_half_verses,
            args.output_dir,
            args.workers,
        )
        print(f"domain={domain}: wrote {written} detail files")


if __name__ == "__main__":
    main()
