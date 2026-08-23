import { describe, expect, it } from "vitest";
import { renderTableHTML } from "./renderTableHTML";
import type { TableColumn } from "./tableColumn";

interface Row {
  model_base: string;
  auc: number;
  p: number;
}

const columns: TableColumn<Row>[] = [
  { key: "model_base", label: "Name", type: "text" },
  { key: "auc", label: "AUC", type: "num", digits: 3 },
  { key: "p", label: "P", type: "pill", pillPrefix: "p" },
];

const rows: Row[] = [
  { model_base: "a", auc: 0.5, p: 0.2 },
  { model_base: "b", auc: 0.9, p: 0.01 },
];

describe("renderTableHTML", () => {
  it("sorts rows by the given key and direction before rendering", () => {
    const html = renderTableHTML(rows, columns, "auc", "desc");
    const firstRowIndex = html.indexOf(">b<");
    const secondRowIndex = html.indexOf(">a<");
    expect(firstRowIndex).toBeGreaterThan(-1);
    expect(firstRowIndex).toBeLessThan(secondRowIndex);
  });

  it("marks the sorted column's header with a sorted-asc or sorted-desc class", () => {
    const asc = renderTableHTML(rows, columns, "auc", "asc");
    expect(asc).toContain('class="num sorted-asc"');
    const desc = renderTableHTML(rows, columns, "auc", "desc");
    expect(desc).toContain('class="num sorted-desc"');
  });

  it("formats a num column with the column's digit count", () => {
    const html = renderTableHTML(rows, columns, "model_base", "asc");
    expect(html).toContain(">0.500<");
  });

  it("formats a pill column via the shared pill helper", () => {
    const html = renderTableHTML(rows, columns, "model_base", "asc");
    expect(html).toContain('class="pill bad"');
    expect(html).toContain('class="pill warn"');
  });

  it("uses a column's custom render function when present", () => {
    const withRender: TableColumn<Row>[] = [
      {
        key: "model_base",
        label: "Name",
        type: "text",
        render: (r) => `<b>${r.model_base}</b>`,
      },
    ];
    const html = renderTableHTML(rows, withRender, "model_base", "asc");
    expect(html).toContain("<b>a</b>");
  });

  it("marks the first rendered row rank-1", () => {
    const html = renderTableHTML(rows, columns, "auc", "desc");
    expect(html).toMatch(/<tr class="rank-1">[^<]*<td[^>]*>b/);
  });
});
