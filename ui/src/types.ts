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
