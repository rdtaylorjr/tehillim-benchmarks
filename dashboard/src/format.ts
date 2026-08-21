const UNAVAILABLE = "—";

/** 4-decimal p-value, matching validate_against_genre.py's CLI printer. */
export function formatPValue(p: number): string {
  return Number.isFinite(p) ? p.toFixed(4) : UNAVAILABLE;
}

/** 4-decimal FDR q-value, matching validate_against_genre.py's CLI printer. */
export function formatQValue(q: number): string {
  return Number.isFinite(q) ? q.toFixed(4) : UNAVAILABLE;
}

/** 3-decimal null-calibrated effect size with an explicit sign, matching the CLI printer. */
export function formatEffectSize(z: number): string {
  if (!Number.isFinite(z)) return UNAVAILABLE;
  return `${z >= 0 ? "+" : "-"}${Math.abs(z).toFixed(3)}`;
}
