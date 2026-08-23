export type Metric =
  | "content_distance"
  | "structural_distance"
  | "adjacent_similarity_distance"
  | "step_magnitude_distance"
  | "turning_angle_distance";

export type Source =
  "raw" | "length_controlled" | "length_and_content_controlled";

export const SOURCES: readonly Source[] = [
  "raw",
  "length_controlled",
  "length_and_content_controlled",
];

/** One (model, metric) row from validate_against_genre.py's output CSV. */
export interface ValidationRow {
  model: string;
  metric: Metric;
  n_pairs_total: number;
  n_pairs_valid: number;
  raw_gap: number;
  raw_p: number;
  raw_effect_size: number;
  raw_q: number;
  raw_q_by: number;
  length_controlled_gap: number;
  length_controlled_p: number;
  length_controlled_effect_size: number;
  length_controlled_q: number;
  length_controlled_q_by: number;
  length_and_content_controlled_gap: number;
  length_and_content_controlled_p: number;
  length_and_content_controlled_effect_size: number;
  length_and_content_controlled_q: number;
  length_and_content_controlled_q_by: number;
}

export interface ParallelismOverallRow {
  model: string;
  model_base: string;
  text_variant: string;
  separation_auc: number;
  average_precision: number;
  calibrated_effect_size: number;
  mrr_forward: number;
  n_true: number;
}

export interface ParallelismByTypeRow {
  model: string;
  model_base: string;
  text_variant: string;
  scope: string;
  separation_auc: number;
  average_precision: number;
  calibrated_effect_size: number;
}

export interface GenreOverallRow {
  model: string;
  model_base: string;
  text_variant: string;
  separation_auc: number;
  average_precision: number;
  same_genre_effect_size: number;
  n_same_genre: number;
}

export interface GenreByGenreRow {
  model: string;
  model_base: string;
  text_variant: string;
  genre: string;
  separation_auc: number;
  average_precision: number;
  perm_q: number;
  maxT_q: number;
}

export interface TrajectoryByGenreRow {
  model: string;
  model_base: string;
  text_variant: string;
  metric: Metric;
  genre: string;
  source: Source;
  gap: number;
  p_perm: number;
  p_maxT: number;
  perm_q: number;
  perm_q_by: number;
  maxT_q: number;
  maxT_q_by: number;
}

/** A ValidationRow as export_ui_rows.py enriches it: model split into model_base/text_variant. */
export interface TrajectoryOverallRow extends ValidationRow {
  model_base: string;
  text_variant: string;
}
