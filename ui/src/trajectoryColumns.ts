import { visibleSourcesForMetric } from "./validationRow";
import type { Source, ValidationRow } from "./types";

export interface Column {
  key: string;
  label: string;
  type: "num" | "pill";
  digits?: number;
  pillPrefix?: "p" | "q";
}

const SOURCE_LABEL: Record<Source, string> = {
  raw: "Raw",
  length_controlled: "Length-controlled",
  length_and_content_controlled: "Content-controlled",
};

function statColumns(source: Source): Column[] {
  const label = SOURCE_LABEL[source];
  const columns: Column[] = [];
  if (source === "raw") {
    columns.push({
      key: "raw_effect_size",
      label: "Effect size (calibrated)",
      type: "num",
      digits: 3,
    });
  }
  columns.push(
    { key: `${source}_gap`, label: `${label} gap`, type: "num", digits: 5 },
    { key: `${source}_p`, label: `${label} p`, type: "pill", pillPrefix: "p" },
    {
      key: `${source}_q`,
      label: `${label} q (FDR)`,
      type: "pill",
      pillPrefix: "q",
    },
  );
  return columns;
}

/** Column set for one trajectory metric's table, hiding a source entirely when no row in the group has it. */
export function trajectoryColumns(rows: ValidationRow[]): Column[] {
  return visibleSourcesForMetric(rows).flatMap(statColumns);
}
