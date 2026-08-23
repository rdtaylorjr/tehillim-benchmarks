import { describe, expect, it } from "vitest";
import {
  genreByGenreColumns,
  genreOverallColumns,
  parallelismByTypeColumns,
  parallelismOverallColumns,
  trajectoryByGenreColumns,
  trajectoryOverallColumns,
} from "./tableColumns";
import type { TrajectoryOverallRow } from "./types";

function makeValidationRow(
  overrides: Partial<TrajectoryOverallRow> = {},
): TrajectoryOverallRow {
  return {
    model: "bge_m3_vocalized",
    model_base: "bge_m3",
    text_variant: "vocalized",
    metric: "content_distance",
    n_pairs_total: 100,
    n_pairs_valid: 100,
    raw_gap: 0.1,
    raw_p: 0.01,
    raw_effect_size: 1.0,
    raw_q: 0.02,
    raw_q_by: 0.03,
    length_controlled_gap: 0.05,
    length_controlled_p: 0.03,
    length_controlled_effect_size: 0.5,
    length_controlled_q: 0.04,
    length_controlled_q_by: 0.05,
    length_and_content_controlled_gap: NaN,
    length_and_content_controlled_p: NaN,
    length_and_content_controlled_effect_size: NaN,
    length_and_content_controlled_q: NaN,
    length_and_content_controlled_q_by: NaN,
    ...overrides,
  };
}

describe("parallelismOverallColumns", () => {
  it("matches ui_export.export's _PARALLELISM_OVERALL_COLUMNS field set", () => {
    expect(parallelismOverallColumns().map((c) => c.key)).toEqual([
      "model_base",
      "separation_auc",
      "average_precision",
      "calibrated_effect_size",
      "mrr_forward",
      "n_true",
    ]);
  });
});

describe("parallelismByTypeColumns", () => {
  it("matches ui_export.export's _PARALLELISM_BY_TYPE_COLUMNS field set", () => {
    expect(parallelismByTypeColumns().map((c) => c.key)).toEqual([
      "model_base",
      "separation_auc",
      "average_precision",
      "calibrated_effect_size",
    ]);
  });
});

describe("genreOverallColumns", () => {
  it("matches ui_export.export's _GENRE_OVERALL_COLUMNS field set", () => {
    expect(genreOverallColumns().map((c) => c.key)).toEqual([
      "model_base",
      "separation_auc",
      "average_precision",
      "same_genre_effect_size",
      "n_same_genre",
    ]);
  });
});

describe("genreByGenreColumns", () => {
  it("matches ui_export.export's _GENRE_BY_GENRE_COLUMNS field set", () => {
    expect(genreByGenreColumns().map((c) => c.key)).toEqual([
      "model",
      "separation_auc",
      "average_precision",
      "perm_q",
      "maxT_q",
    ]);
  });

  it("renders q-value columns as pills", () => {
    const columns = genreByGenreColumns();
    expect(columns.find((c) => c.key === "perm_q")?.type).toBe("pill");
    expect(columns.find((c) => c.key === "maxT_q")?.type).toBe("pill");
  });
});

describe("trajectoryOverallColumns", () => {
  it("prepends a Name column ahead of the visible-source stat columns", () => {
    const columns = trajectoryOverallColumns([makeValidationRow()]);
    expect(columns[0].key).toBe("model_base");
    expect(columns[0].label).toBe("Name");
  });

  it("omits length_and_content_controlled columns when no row in the group has them", () => {
    const columns = trajectoryOverallColumns([makeValidationRow()]);
    expect(
      columns.some((c) => c.key.startsWith("length_and_content_controlled")),
    ).toBe(false);
  });
});

describe("trajectoryByGenreColumns", () => {
  it("matches the by-genre trajectory field set, source shown as plain text", () => {
    const columns = trajectoryByGenreColumns();
    expect(columns.map((c) => c.key)).toEqual([
      "model_base",
      "source",
      "gap",
      "p_perm",
      "perm_q",
      "maxT_q",
    ]);
    expect(columns.find((c) => c.key === "source")?.type).toBe("text");
  });
});
