import { trajectoryColumns } from "./trajectoryColumns";
import { variantLabel } from "./variantLabel";
import type {
  GenreByGenreRow,
  GenreOverallRow,
  ParallelismByTypeRow,
  ParallelismOverallRow,
  TrajectoryByGenreRow,
  TrajectoryOverallRow,
} from "./types";
import type { TableColumn } from "../../../shared/ui/tableColumn";

function nameColumn<
  T extends { model_base?: string; text_variant?: string },
>(): TableColumn<T> {
  return {
    key: "model_base",
    label: "Name",
    type: "text",
    render: variantLabel,
  };
}

export function parallelismOverallColumns(): TableColumn<ParallelismOverallRow>[] {
  return [
    nameColumn<ParallelismOverallRow>(),
    { key: "separation_auc", label: "Separation AUC", type: "num", digits: 4 },
    { key: "average_precision", label: "AP", type: "num", digits: 4 },
    {
      key: "calibrated_effect_size",
      label: "Effect size",
      type: "num",
      digits: 3,
    },
    { key: "mrr_forward", label: "MRR (fwd)", type: "num", digits: 4 },
    { key: "n_true", label: "n pairs", type: "num", digits: 0 },
  ];
}

export function parallelismByTypeColumns(): TableColumn<ParallelismByTypeRow>[] {
  return [
    nameColumn<ParallelismByTypeRow>(),
    { key: "separation_auc", label: "Separation AUC", type: "num", digits: 4 },
    { key: "average_precision", label: "AP", type: "num", digits: 4 },
    {
      key: "calibrated_effect_size",
      label: "Effect size",
      type: "num",
      digits: 3,
    },
  ];
}

export function genreOverallColumns(): TableColumn<GenreOverallRow>[] {
  return [
    nameColumn<GenreOverallRow>(),
    { key: "separation_auc", label: "Separation AUC", type: "num", digits: 4 },
    { key: "average_precision", label: "AP", type: "num", digits: 4 },
    {
      key: "same_genre_effect_size",
      label: "Effect size",
      type: "num",
      digits: 3,
    },
    { key: "n_same_genre", label: "n same-genre", type: "num", digits: 0 },
  ];
}

export function genreByGenreColumns(): TableColumn<GenreByGenreRow>[] {
  return [
    { key: "model", label: "Name", type: "text" },
    { key: "separation_auc", label: "Separation AUC", type: "num", digits: 4 },
    { key: "average_precision", label: "AP", type: "num", digits: 4 },
    {
      key: "perm_q",
      label: "q (permutation, FDR)",
      type: "pill",
      pillPrefix: "q",
    },
    { key: "maxT_q", label: "q (maxT, FDR)", type: "pill", pillPrefix: "q" },
  ];
}

export function trajectoryOverallColumns(
  rows: TrajectoryOverallRow[],
): TableColumn<TrajectoryOverallRow>[] {
  return [nameColumn<TrajectoryOverallRow>(), ...trajectoryColumns(rows)];
}

export function trajectoryByGenreColumns(): TableColumn<TrajectoryByGenreRow>[] {
  return [
    nameColumn<TrajectoryByGenreRow>(),
    { key: "source", label: "Source", type: "text" },
    { key: "gap", label: "Gap", type: "num", digits: 5 },
    { key: "p_perm", label: "p (permutation)", type: "pill", pillPrefix: "p" },
    {
      key: "perm_q",
      label: "q (permutation, FDR)",
      type: "pill",
      pillPrefix: "q",
    },
    { key: "maxT_q", label: "q (maxT, FDR)", type: "pill", pillPrefix: "q" },
  ];
}
