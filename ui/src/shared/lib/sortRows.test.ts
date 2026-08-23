import { describe, expect, it } from "vitest";
import { sortRows } from "./sortRows";

describe("sortRows", () => {
  it("sorts numeric rows ascending", () => {
    const rows = [{ auc: 0.7 }, { auc: 0.3 }, { auc: 0.5 }];
    expect(sortRows(rows, "auc", "asc").map((r) => r.auc)).toEqual([
      0.3, 0.5, 0.7,
    ]);
  });

  it("sorts numeric rows descending", () => {
    const rows = [{ auc: 0.7 }, { auc: 0.3 }, { auc: 0.5 }];
    expect(sortRows(rows, "auc", "desc").map((r) => r.auc)).toEqual([
      0.7, 0.5, 0.3,
    ]);
  });

  it("treats a NaN or missing numeric value as -Infinity, sorting it to the bottom descending", () => {
    const rows = [{ auc: 0.7 }, { auc: NaN }, { auc: 0.5 }];
    expect(sortRows(rows, "auc", "desc").map((r) => r.auc)).toEqual([
      0.7,
      0.5,
      NaN,
    ]);
  });

  it("sorts string fields with localeCompare ascending", () => {
    const rows = [{ name: "beta" }, { name: "alpha" }, { name: "gamma" }];
    expect(sortRows(rows, "name", "asc").map((r) => r.name)).toEqual([
      "alpha",
      "beta",
      "gamma",
    ]);
  });

  it("sorts string fields with localeCompare descending", () => {
    const rows = [{ name: "beta" }, { name: "alpha" }, { name: "gamma" }];
    expect(sortRows(rows, "name", "desc").map((r) => r.name)).toEqual([
      "gamma",
      "beta",
      "alpha",
    ]);
  });

  it("does not mutate the input array", () => {
    const rows = [{ auc: 0.7 }, { auc: 0.3 }];
    sortRows(rows, "auc", "asc");
    expect(rows.map((r) => r.auc)).toEqual([0.7, 0.3]);
  });
});
