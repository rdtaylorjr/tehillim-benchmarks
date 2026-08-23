import { describe, expect, it } from "vitest";
import { trajectoryColumns } from "./trajectoryColumns";
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

describe("trajectoryColumns", () => {
  it("includes raw, length-controlled, and content-controlled columns when all three sources are available", () => {
    const keys = trajectoryColumns([makeRow()]).map((c) => c.key);
    expect(keys).toEqual([
      "raw_effect_size",
      "raw_gap",
      "raw_p",
      "raw_q",
      "length_controlled_gap",
      "length_controlled_p",
      "length_controlled_q",
      "length_and_content_controlled_gap",
      "length_and_content_controlled_p",
      "length_and_content_controlled_q",
    ]);
  });

  it("omits the content-controlled columns for a content_distance group, the self-covariate case", () => {
    const rows = [
      makeRow({
        metric: "content_distance",
        length_and_content_controlled_gap: NaN,
        length_and_content_controlled_p: NaN,
        length_and_content_controlled_effect_size: NaN,
        length_and_content_controlled_q: NaN,
      }),
    ];
    const keys = trajectoryColumns(rows).map((c) => c.key);
    expect(keys).toEqual([
      "raw_effect_size",
      "raw_gap",
      "raw_p",
      "raw_q",
      "length_controlled_gap",
      "length_controlled_p",
      "length_controlled_q",
    ]);
  });

  it("marks the p and q columns as pill-typed for significance styling", () => {
    const columns = trajectoryColumns([makeRow()]);
    const pillKeys = columns.filter((c) => c.type === "pill").map((c) => c.key);
    expect(pillKeys).toEqual([
      "raw_p",
      "raw_q",
      "length_controlled_p",
      "length_controlled_q",
      "length_and_content_controlled_p",
      "length_and_content_controlled_q",
    ]);
  });

  it("marks p columns with a p prefix and q columns with a q prefix, for pill display text", () => {
    const columns = trajectoryColumns([makeRow()]);
    const byKey = Object.fromEntries(columns.map((c) => [c.key, c.pillPrefix]));
    expect(byKey["raw_p"]).toBe("p");
    expect(byKey["raw_q"]).toBe("q");
    expect(byKey["length_controlled_p"]).toBe("p");
    expect(byKey["length_controlled_q"]).toBe("q");
  });

  it("gives every column a human-readable label", () => {
    const columns = trajectoryColumns([makeRow()]);
    const byKey = Object.fromEntries(columns.map((c) => [c.key, c.label]));
    expect(byKey["raw_effect_size"]).toBe("Effect size (calibrated)");
    expect(byKey["raw_gap"]).toBe("Raw gap");
    expect(byKey["raw_p"]).toBe("Raw p");
    expect(byKey["raw_q"]).toBe("Raw q (FDR)");
    expect(byKey["length_controlled_gap"]).toBe("Length-controlled gap");
    expect(byKey["length_and_content_controlled_q"]).toBe(
      "Content-controlled q (FDR)",
    );
  });
});
