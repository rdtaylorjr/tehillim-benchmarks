import { FACETS, facetOf } from "./families";

export interface FacetableRow {
  model?: string;
  model_base?: string;
}

/** Rows whose model falls in the family's chosen facet bucket; passthrough for "all" or a facet-less family. */
export function applyFacetFilter<T extends FacetableRow>(
  rows: T[],
  familyId: string,
  unit: string,
): T[] {
  if (unit === "all") return rows;
  const facet = FACETS[familyId];
  if (!facet) return rows;
  return rows.filter(
    (r) => facetOf(r.model_base ?? r.model ?? "", facet.values) === unit,
  );
}

export interface TextVariantRow {
  text_variant?: string;
}

/** Rows matching the chosen text variant; passthrough for "all". */
export function applyTextFilter<T extends TextVariantRow>(
  rows: T[],
  text: string,
): T[] {
  if (text === "all") return rows;
  return rows.filter((r) => r.text_variant === text);
}

export interface NamedRow {
  model?: string;
}

/** Rows whose model name contains the filter text, case-insensitively; passthrough when the filter is empty. */
export function applyNameFilter<T extends NamedRow>(
  rows: T[],
  filter: string,
): T[] {
  if (!filter) return rows;
  const needle = filter.toLowerCase();
  return rows.filter((r) => (r.model ?? "").toLowerCase().includes(needle));
}

/** True when at least one row carries a real (non-"unknown") text variant, gating the Text dropdown's visibility. */
export function familyHasTextVariant<T extends TextVariantRow>(
  rows: T[],
): boolean {
  return rows.some(
    (r) => r.text_variant !== undefined && r.text_variant !== "unknown",
  );
}
