import { formatPValue, formatQValue } from "../lib/format";

export type PillPrefix = "p" | "q";

function severityClass(value: number): "good" | "warn" | "bad" {
  if (value < 0.01) return "good";
  if (value < 0.05) return "warn";
  return "bad";
}

/** Color-coded significance pill HTML; "" for a non-finite value, "<0.001" below that floor to avoid a misleading 0.0000. */
export function pill(value: number, prefix: PillPrefix = "p"): string {
  if (!Number.isFinite(value)) return "";
  const text =
    value < 0.001
      ? "<0.001"
      : prefix === "q"
        ? formatQValue(value)
        : formatPValue(value);
  return `<span class="pill ${severityClass(value)}">${prefix}=${text}</span>`;
}
