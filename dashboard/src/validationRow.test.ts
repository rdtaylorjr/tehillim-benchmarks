import { describe, expect, it } from "vitest";
import {
  isSourceAvailable,
  sourceField,
  visibleSources,
  visibleSourcesForMetric,
} from "./validationRow";
import type { ValidationRow } from "./types";

function makeRow(overrides: Partial<ValidationRow> = {}): ValidationRow {
  return {
    model: "bge_m3_vocalized",
    metric: "structural_distance",
    n_pairs_total: 11175,
    n_pairs_valid: 11175,
    raw_gap: 0.12,
    raw_p: 0.001,
    raw_effect_size: 3.2,
    raw_q: 0.004,
    raw_q_by: 0.01,
    length_controlled_gap: 0.08,
    length_controlled_p: 0.02,
    length_controlled_effect_size: 2.1,
    length_controlled_q: 0.05,
    length_controlled_q_by: 0.09,
    length_and_content_controlled_gap: 0.05,
    length_and_content_controlled_p: 0.15,
    length_and_content_controlled_effect_size: 1.0,
    length_and_content_controlled_q: 0.3,
    length_and_content_controlled_q_by: 0.4,
    ...overrides,
  };
}

describe("sourceField", () => {
  it("reads the p-value field for a given source", () => {
    const row = makeRow();
    expect(sourceField(row, "raw", "p")).toBe(0.001);
    expect(sourceField(row, "length_controlled", "effect_size")).toBe(2.1);
  });
});

describe("isSourceAvailable", () => {
  it("is true when the source's p-value is a finite number", () => {
    expect(isSourceAvailable(makeRow(), "raw")).toBe(true);
  });

  it("is false when the source's p-value is NaN, matching build_validation_row's insufficient-pairs case", () => {
    const row = makeRow({ length_and_content_controlled_p: NaN });
    expect(isSourceAvailable(row, "length_and_content_controlled")).toBe(false);
  });

  it("is false for length_and_content_controlled on a content_distance row, matching has_content_covariate=False", () => {
    const row = makeRow({
      metric: "content_distance",
      length_and_content_controlled_p: NaN,
      length_and_content_controlled_gap: NaN,
      length_and_content_controlled_effect_size: NaN,
    });
    expect(isSourceAvailable(row, "length_and_content_controlled")).toBe(false);
  });
});

describe("visibleSources", () => {
  it("returns all three sources when every p-value is finite", () => {
    expect(visibleSources(makeRow())).toEqual([
      "raw",
      "length_controlled",
      "length_and_content_controlled",
    ]);
  });

  it("omits length_and_content_controlled for a content_distance row, its self-covariate case", () => {
    const row = makeRow({
      metric: "content_distance",
      length_and_content_controlled_p: NaN,
      length_and_content_controlled_gap: NaN,
      length_and_content_controlled_effect_size: NaN,
    });
    expect(visibleSources(row)).toEqual(["raw", "length_controlled"]);
  });

  it("omits every controlled source when the model has too few same/different genre pairs", () => {
    const row = makeRow({
      length_controlled_p: NaN,
      length_controlled_gap: NaN,
      length_controlled_effect_size: NaN,
      length_and_content_controlled_p: NaN,
      length_and_content_controlled_gap: NaN,
      length_and_content_controlled_effect_size: NaN,
    });
    expect(visibleSources(row)).toEqual(["raw"]);
  });

  it("returns an empty array when the row has no usable data at all", () => {
    const row = makeRow({
      raw_p: NaN,
      raw_gap: NaN,
      raw_effect_size: NaN,
      length_controlled_p: NaN,
      length_controlled_gap: NaN,
      length_controlled_effect_size: NaN,
      length_and_content_controlled_p: NaN,
      length_and_content_controlled_gap: NaN,
      length_and_content_controlled_effect_size: NaN,
    });
    expect(visibleSources(row)).toEqual([]);
  });
});

describe("visibleSourcesForMetric", () => {
  it("returns all three sources when every row in the group has them", () => {
    const rows = [makeRow({ model: "a" }), makeRow({ model: "b" })];
    expect(visibleSourcesForMetric(rows)).toEqual([
      "raw",
      "length_controlled",
      "length_and_content_controlled",
    ]);
  });

  it("omits length_and_content_controlled for the whole content_distance group, its self-covariate case", () => {
    const rows = [
      makeRow({
        model: "a",
        metric: "content_distance",
        length_and_content_controlled_p: NaN,
        length_and_content_controlled_gap: NaN,
        length_and_content_controlled_effect_size: NaN,
      }),
      makeRow({
        model: "b",
        metric: "content_distance",
        length_and_content_controlled_p: NaN,
        length_and_content_controlled_gap: NaN,
        length_and_content_controlled_effect_size: NaN,
      }),
    ];
    expect(visibleSourcesForMetric(rows)).toEqual(["raw", "length_controlled"]);
  });

  it("keeps a source visible if even one row in the group has it, so one model's too-few-pairs case doesn't hide the column for everyone", () => {
    const rows = [
      makeRow({ model: "a" }),
      makeRow({
        model: "b",
        length_and_content_controlled_p: NaN,
        length_and_content_controlled_gap: NaN,
        length_and_content_controlled_effect_size: NaN,
      }),
    ];
    expect(visibleSourcesForMetric(rows)).toEqual([
      "raw",
      "length_controlled",
      "length_and_content_controlled",
    ]);
  });
});
