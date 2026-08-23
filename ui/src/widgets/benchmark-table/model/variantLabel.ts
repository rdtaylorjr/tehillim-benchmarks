export interface VariantLabelRow {
  model_base?: string;
  text_variant?: string;
}

/** "model_base" alone, or "model_base <tag>variant</tag>" when a real (non-"unknown") text variant is present. */
export function variantLabel(row: VariantLabelRow): string {
  const base = row.model_base ?? "";
  return row.text_variant && row.text_variant !== "unknown"
    ? `${base} <span class="variant-tag">${row.text_variant}</span>`
    : base;
}
