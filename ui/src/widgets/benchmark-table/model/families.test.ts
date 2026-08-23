import { describe, expect, it } from "vitest";
import { FACETS, FAMILIES, facetOf } from "./families";

describe("FAMILIES", () => {
  it("lists all six families in linguistic-hierarchy order, semantic pinned first", () => {
    expect(FAMILIES.map((f) => f.id)).toEqual([
      "semantic",
      "lexical",
      "phonology",
      "morphology",
      "syntax",
      "discourse",
    ]);
  });

  it("gives morphology and syntax their noun labels, not the old adjective/bare-noun ones", () => {
    const label = (id: string) => FAMILIES.find((f) => f.id === id)?.label;
    expect(label("morphology")).toBe("Morphology");
    expect(label("syntax")).toBe("Syntax");
  });
});

describe("FACETS", () => {
  it("only lexical and syntax carry a facet dropdown", () => {
    expect(Object.keys(FACETS).sort()).toEqual(["lexical", "syntax"]);
  });

  it("labels lexical's facet Unit and syntax's facet Level", () => {
    expect(FACETS.lexical.label).toBe("Unit");
    expect(FACETS.syntax.label).toBe("Level");
  });

  it("lists syntax's two rank values", () => {
    expect(FACETS.syntax.values).toEqual(["clause", "phrase"]);
  });
});

describe("facetOf", () => {
  it("matches a model name equal to a facet value", () => {
    expect(facetOf("phrase", ["clause", "phrase"])).toBe("phrase");
  });

  it("matches a model name prefixed by a facet value plus underscore", () => {
    expect(facetOf("phrase_signature_1gram", ["clause", "phrase"])).toBe(
      "phrase",
    );
  });

  it("does not match a value that only shares a prefix without the underscore boundary", () => {
    expect(facetOf("phraseology", ["phrase"])).toBeNull();
  });

  it("returns null when no facet value matches", () => {
    expect(facetOf("bge_m3_vocalized", ["clause", "phrase"])).toBeNull();
  });
});
