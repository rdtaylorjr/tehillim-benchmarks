export interface Family {
  id: string;
  label: string;
}

/** The results UI's family tabs, linguistic-hierarchy order (lexical < phonology < morphology < syntax < discourse). */
export const FAMILIES: readonly Family[] = [
  { id: "semantic", label: "Semantic" },
  { id: "lexical", label: "Lexical" },
  { id: "phonology", label: "Phonology" },
  { id: "morphology", label: "Morphology" },
  { id: "syntax", label: "Syntax" },
  { id: "discourse", label: "Discourse" },
];

export interface Facet {
  label: string;
  values: readonly string[];
}

/** Families whose models split into a rank/unit sub-dropdown, beyond the plain family tabs. */
export const FACETS: Readonly<Record<string, Facet>> = {
  lexical: { label: "Unit", values: ["homograph", "lexeme", "word"] },
  syntax: { label: "Level", values: ["clause", "phrase"] },
};

/** The facet value a model name belongs to, matching an exact name or a `${value}_` prefix. */
export function facetOf(
  modelBase: string,
  values: readonly string[],
): string | null {
  for (const v of values) {
    if (modelBase === v || modelBase.startsWith(`${v}_`)) return v;
  }
  return null;
}
