const UNAVAILABLE = "—";

/** Fixed-decimal number for a table cell, 4 digits by default; em dash for a missing value. */
export function formatNumber(value: number, digits = 4): string {
  return Number.isFinite(value) ? value.toFixed(digits) : UNAVAILABLE;
}
