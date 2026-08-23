import { describe, expect, it } from "vitest";
import {
  applyFacetFilter,
  applyNameFilter,
  applyTextFilter,
  familyHasTextVariant,
} from "./rowFilters";

describe("applyFacetFilter", () => {
  const rows = [
    { model: "phrase_typ_1gram", model_base: "phrase_typ_1gram" },
    { model: "clause_typ_1gram", model_base: "clause_typ_1gram" },
    { model: "bge_m3", model_base: "bge_m3" },
  ];

  it("returns every row unfiltered when unit is all", () => {
    expect(applyFacetFilter(rows, "syntax", "all")).toEqual(rows);
  });

  it("keeps only rows matching the chosen facet value for a faceted family", () => {
    expect(applyFacetFilter(rows, "syntax", "phrase")).toEqual([rows[0]]);
  });

  it("returns every row unfiltered for a family with no facet at all", () => {
    expect(applyFacetFilter(rows, "semantic", "phrase")).toEqual(rows);
  });
});

describe("applyTextFilter", () => {
  const rows = [{ text_variant: "vocalized" }, { text_variant: "consonantal" }];

  it("returns every row when text is all", () => {
    expect(applyTextFilter(rows, "all")).toEqual(rows);
  });

  it("keeps only rows matching the chosen text variant", () => {
    expect(applyTextFilter(rows, "vocalized")).toEqual([rows[0]]);
  });
});

describe("applyNameFilter", () => {
  const rows = [{ model: "BGE_M3_Vocalized" }, { model: "homograph_binary" }];

  it("returns every row when the filter is empty", () => {
    expect(applyNameFilter(rows, "")).toEqual(rows);
  });

  it("matches case-insensitively against the model name", () => {
    expect(applyNameFilter(rows, "bge")).toEqual([rows[0]]);
  });

  it("returns an empty array when nothing matches", () => {
    expect(applyNameFilter(rows, "nonexistent")).toEqual([]);
  });
});

describe("familyHasTextVariant", () => {
  it("is true when at least one row has a real text variant", () => {
    expect(
      familyHasTextVariant([
        { text_variant: "unknown" },
        { text_variant: "vocalized" },
      ]),
    ).toBe(true);
  });

  it("is false when every row's text variant is unknown", () => {
    expect(familyHasTextVariant([{ text_variant: "unknown" }])).toBe(false);
  });

  it("is false when no row carries a text_variant field at all", () => {
    expect(familyHasTextVariant([{}])).toBe(false);
  });
});
