import { SOURCES } from "./types";
import type { Source, ValidationRow } from "./types";

/** Reads `${source}_${field}` off a validation row. */
export function sourceField(
  row: ValidationRow,
  source: Source,
  field: "gap" | "p" | "effect_size" | "q" | "q_by",
): number {
  return row[`${source}_${field}`];
}

/** True when a source has a real (non-NaN) p-value, i.e. build_validation_row didn't null it out. */
export function isSourceAvailable(row: ValidationRow, source: Source): boolean {
  return Number.isFinite(sourceField(row, source, "p"));
}

/** The sources with real data for this row, covering both the content-covariate and too-few-pairs cases. */
export function visibleSources(row: ValidationRow): Source[] {
  return SOURCES.filter((source) => isSourceAvailable(row, source));
}

/** The sources with real data in at least one row of a metric's group, for building that metric's columns. */
export function visibleSourcesForMetric(rows: ValidationRow[]): Source[] {
  return SOURCES.filter((source) =>
    rows.some((row) => isSourceAvailable(row, source)),
  );
}
