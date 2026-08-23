import { formatNumber } from "../lib/numberFormat";
import { pill } from "./pills";
import { sortRows } from "../lib/sortRows";
import type { SortDir } from "../lib/sortRows";
import type { TableColumn } from "./tableColumn";

function headerCell<T>(
  column: TableColumn<T>,
  sortKey: string,
  sortDir: SortDir,
): string {
  const classes =
    column.type === "num" || column.type === "pill" ? ["num"] : [];
  if (column.key === sortKey)
    classes.push(sortDir === "asc" ? "sorted-asc" : "sorted-desc");
  return `<th data-key="${column.key}" class="${classes.join(" ")}">${column.label}</th>`;
}

function bodyCell<T>(row: T, column: TableColumn<T>): string {
  const raw = (row as Record<string, unknown>)[column.key];
  const content = column.render
    ? column.render(row)
    : column.type === "pill"
      ? pill(raw as number, column.pillPrefix)
      : column.type === "num"
        ? formatNumber(raw as number, column.digits)
        : String(raw ?? "");
  const isModelColumn = column.key === "model" || column.key === "model_base";
  const cls =
    column.type === "num" || column.type === "pill"
      ? "num"
      : isModelColumn
        ? "model-cell"
        : "";
  return `<td class="${cls}">${content}</td>`;
}

/** The thead/tbody HTML for one table: sorted rows, sort-indicator classes, and each column's cell rule. */
export function renderTableHTML<T extends object>(
  rows: T[],
  columns: TableColumn<T>[],
  sortKey: string,
  sortDir: SortDir,
): string {
  const sorted = sortRows(rows, sortKey, sortDir);
  const thead = `<thead><tr>${columns.map((c) => headerCell(c, sortKey, sortDir)).join("")}</tr></thead>`;
  const tbody = `<tbody>${sorted
    .map((row, i) => {
      const cells = columns.map((c) => bodyCell(row, c)).join("");
      return `<tr class="${i === 0 ? "rank-1" : ""}">${cells}</tr>`;
    })
    .join("")}</tbody>`;
  return thead + tbody;
}
