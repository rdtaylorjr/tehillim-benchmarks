"""Order-shuffle-null control: does colon order carry parallelism signal beyond a shuffled null."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from library.bhsa import DEFAULT_CHECKOUT
from library.embeddings import load_embeddings
from library.order_shuffle import DEFAULT_N_SHUFFLES, order_shuffle_result
from library.retrieval_metrics import cosine_similarity_matrix
from parallelism.evaluate import build_side_vectors
from parallelism.pairs import RetrievalPair, build_retrieval_pairs, filter_pairs_with_vectors
from parallelism.separation import similarity_separation
from parallelism.tf_features import load_api, read_node_feature_values, reconstruct_groups


def score_separation_auc(path: Path, all_pairs: list[RetrievalPair]) -> float:
    """Separation AUC (no permutation testing) for one embeddings file against all_pairs."""
    node_vectors = load_embeddings(path)
    pairs = filter_pairs_with_vectors(all_pairs, node_vectors)
    source_vecs = build_side_vectors(pairs, "source", node_vectors)
    target_vecs = build_side_vectors(pairs, "target", node_vectors)
    similarities = cosine_similarity_matrix(source_vecs, target_vecs)
    return similarity_separation(similarities).auc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("real_embeddings", type=Path)
    parser.add_argument("shuffled_embeddings_dir", type=Path)
    parser.add_argument("--checkout", default=DEFAULT_CHECKOUT, help="BHSA/module checkout spec")
    parser.add_argument("--n-shuffles", type=int, default=DEFAULT_N_SHUFFLES)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    api = load_api(args.checkout)
    node_values = read_node_feature_values(api)
    groups = reconstruct_groups(node_values)
    all_pairs = build_retrieval_pairs(groups)

    auc_real = score_separation_auc(args.real_embeddings, all_pairs)
    shuffled_paths = sorted(args.shuffled_embeddings_dir.glob("**/*.parquet"))[: args.n_shuffles]
    auc_shuffled = []
    for path in shuffled_paths:
        print(f"scoring {path.parent.name}", file=sys.stderr)
        auc_shuffled.append(score_separation_auc(path, all_pairs))

    result = order_shuffle_result(real_score=auc_real, shuffled_scores=np.array(auc_shuffled))
    print(
        f"auc_real={auc_real:.4f} auc_shuffled_mean={np.mean(auc_shuffled):.4f} "
        f"delta_order={result.delta_order:+.4f} p={result.p_value:.4f}"
    )

    if args.output:
        with open(args.output, "w", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "auc_real",
                    "auc_shuffled_mean",
                    "n_shuffles",
                    "delta_order",
                    "p_value",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "auc_real": auc_real,
                    "auc_shuffled_mean": float(np.mean(auc_shuffled)),
                    "n_shuffles": len(auc_shuffled),
                    "delta_order": result.delta_order,
                    "p_value": result.p_value,
                }
            )


if __name__ == "__main__":
    main()
