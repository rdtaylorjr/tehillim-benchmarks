"""Exports row-per-observation detail: pair/baseline similarity and per-type vs-baseline stats."""

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from library.bhsa import (
    DEFAULT_CHECKOUT,
    list_psalms_cola_by_psalm,
    list_psalms_colon_nodes,
)
from library.calibration import BackgroundStats, background_similarity_stats, calibrated_z_score
from library.embeddings import dataset_identifier, load_embeddings
from library.incremental_cache import load_cached_parquet_set
from library.retrieval_metrics import (
    cosine_similarity_matrix,
    paired_cosine_similarity,
    ranks_from_similarity_matrix,
)
from parallelism.baseline import build_unmarked_bicola
from parallelism.evaluate import build_side_vectors
from parallelism.pairs import (
    RetrievalPair,
    build_retrieval_pairs,
    filter_pairs_by_type,
    filter_pairs_with_vectors,
)
from parallelism.scripts.compare_baseline import (
    BaselineComparison,
    NodePairs,
    as_node_pairs,
    compare_to_baseline,
    filter_node_pairs_with_vectors,
)
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups

_TYPES = frozenset({"Synonymous", "Staircase", "Emblematic", "Synthetic", "Antithetic"})
_OUTPUT_FILES = ("pair_detail.parquet", "baseline_detail.parquet", "type_vs_baseline.parquet")


def load_cached_detail(output_dir: Path) -> tuple[list[list[dict[str, Any]]], set[str]]:
    """Reads prior detail parquet files' rows and the model set already covered by all three."""
    return load_cached_parquet_set(output_dir, _OUTPUT_FILES)


def _as_node_pairs(pairs: list[RetrievalPair]) -> NodePairs:
    return [(p.source_nodes, p.target_nodes) for p in pairs]


def build_pair_detail_rows(
    model: str,
    pairs: list[RetrievalPair],
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """One row per pair: raw similarity, per-pair calibrated z, and bidirectional rank."""
    source_vecs = build_side_vectors(pairs, "source", node_vectors)
    target_vecs = build_side_vectors(pairs, "target", node_vectors)
    pair_ids = [p.pair_id for p in pairs]

    similarities = cosine_similarity_matrix(source_vecs, target_vecs)
    true_similarities = np.diag(similarities)
    ranks_forward = ranks_from_similarity_matrix(similarities, pair_ids, true_target_ids=pair_ids)
    ranks_backward = ranks_from_similarity_matrix(
        similarities.T, pair_ids, true_target_ids=pair_ids
    )

    rows = []
    for pair, sim, rank_f, rank_b in zip(
        pairs, true_similarities, ranks_forward, ranks_backward, strict=True
    ):
        rows.append(
            {
                "model": model,
                "pair_id": pair.pair_id,
                "group_range": pair.group_range,
                "parallelism_type": pair.parallelism_type,
                "signature": pair.signature,
                "source_nodes": ",".join(map(str, pair.source_nodes)),
                "target_nodes": ",".join(map(str, pair.target_nodes)),
                "source_indicator": pair.source_indicator,
                "target_indicator": pair.target_indicator,
                "n_source_nodes": len(pair.source_nodes),
                "n_target_nodes": len(pair.target_nodes),
                "raw_similarity": float(sim),
                "calibrated_z": calibrated_z_score(float(sim), background),
                "rank_forward": rank_f,
                "reciprocal_rank_forward": 1.0 / rank_f,
                "rank_backward": rank_b,
                "reciprocal_rank_backward": 1.0 / rank_b,
            }
        )
    return rows


def build_baseline_detail_rows(
    model: str,
    baseline_pairs: list[tuple[int, int]],
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """One row per unmarked adjacent bicolon: raw similarity and calibrated z, no rank."""
    source_vecs = np.stack([node_vectors[a] for a, _ in baseline_pairs])
    target_vecs = np.stack([node_vectors[b] for _, b in baseline_pairs])
    sims = paired_cosine_similarity(source_vecs, target_vecs)
    rows = []
    for (a, b), sim in zip(baseline_pairs, sims, strict=True):
        rows.append(
            {
                "model": model,
                "node_a": a,
                "node_b": b,
                "raw_similarity": float(sim),
                "calibrated_z": calibrated_z_score(float(sim), background),
            }
        )
    return rows


def _scope_row(
    model: str, scope: str, scope_kind: str, result: BaselineComparison
) -> dict[str, str | int | float]:
    return {
        "model": model,
        "scope": scope,
        "scope_kind": scope_kind,
        "n_true": result.n_true,
        "n_baseline": result.n_baseline,
        "prevalence": result.prevalence,
        "average_precision": result.average_precision,
        "true_effect_size": result.true_effect_size,
        "baseline_effect_size": result.baseline_effect_size,
        "gap": result.true_effect_size - result.baseline_effect_size,
        "auc_vs_baseline": result.separation_auc,
        "p_vs_baseline": result.separation_p,
    }


def build_type_vs_baseline_rows(
    model: str,
    all_pairs: list[RetrievalPair],
    baseline_pairs: NodePairs,
    node_vectors: dict[int, np.ndarray],
    background: BackgroundStats,
) -> list[dict[str, Any]]:
    """One row per (model, scope): each of the 5 Lowth parallelism types alone, plus overall."""
    rows = []
    for ptype in sorted(_TYPES):
        type_pairs = _as_node_pairs(filter_pairs_by_type(all_pairs, frozenset({ptype})))
        if not type_pairs:
            continue
        result = compare_to_baseline(type_pairs, baseline_pairs, node_vectors, background)
        rows.append(_scope_row(model, ptype, "type", result))

    result_all = compare_to_baseline(
        _as_node_pairs(all_pairs), baseline_pairs, node_vectors, background
    )
    rows.append(_scope_row(model, "overall", "overall", result_all))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA/module checkout spec")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    api = load_api(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    all_pairs = build_retrieval_pairs(groups)

    marked_nodes = {n for p in all_pairs for n in p.source_nodes} | {
        n for p in all_pairs for n in p.target_nodes
    }
    cola_by_psalm = list_psalms_cola_by_psalm(api)
    baseline_pairs_raw = build_unmarked_bicola(cola_by_psalm, marked_nodes)
    baseline_pairs = as_node_pairs(baseline_pairs_raw)
    background_node_ids = [n for n in list_psalms_colon_nodes(api) if n not in marked_nodes]

    (cached_pair_rows, cached_baseline_rows, cached_scope_rows), cached_models = load_cached_detail(
        args.output_dir
    )
    if cached_models:
        print(f"reusing {len(cached_models)} cached models from {args.output_dir}", file=sys.stderr)

    model_paths = sorted(p for p in args.embeddings_dir.glob("**/*.parquet") if p.is_file())
    pair_rows: list[dict[str, Any]] = list(cached_pair_rows)
    baseline_rows: list[dict[str, Any]] = list(cached_baseline_rows)
    scope_rows: list[dict[str, Any]] = list(cached_scope_rows)
    for path in model_paths:
        model = dataset_identifier(path)
        if model in cached_models:
            continue
        print(f"processing {model}")
        node_vectors = load_embeddings(path)
        background_vecs = np.stack(
            [node_vectors[n] for n in background_node_ids if n in node_vectors]
        )
        background = background_similarity_stats(background_vecs)
        model_pairs = filter_pairs_with_vectors(all_pairs, node_vectors)
        model_baseline_pairs_raw = [
            (a, b) for a, b in baseline_pairs_raw if a in node_vectors and b in node_vectors
        ]
        model_baseline_pairs = filter_node_pairs_with_vectors(baseline_pairs, node_vectors)

        pair_rows.extend(build_pair_detail_rows(model, model_pairs, node_vectors, background))
        baseline_rows.extend(
            build_baseline_detail_rows(model, model_baseline_pairs_raw, node_vectors, background)
        )
        scope_rows.extend(
            build_type_vs_baseline_rows(
                model, model_pairs, model_baseline_pairs, node_vectors, background
            )
        )

    pd.DataFrame(pair_rows).to_parquet(args.output_dir / "pair_detail.parquet", index=False)
    pd.DataFrame(baseline_rows).to_parquet(args.output_dir / "baseline_detail.parquet", index=False)
    pd.DataFrame(scope_rows).to_parquet(args.output_dir / "type_vs_baseline.parquet", index=False)
    print(f"wrote {len(pair_rows)} pair rows, {len(baseline_rows)} baseline rows")
    print(f"wrote {len(scope_rows)} type scope rows")


if __name__ == "__main__":
    main()
