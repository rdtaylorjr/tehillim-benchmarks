/** A generic table column descriptor: how to render one field of any row shape as a cell. */
export interface TableColumn<T> {
  key: string;
  label: string;
  type: "text" | "num" | "pill";
  digits?: number;
  pillPrefix?: "p" | "q";
  render?: (row: T) => string;
}
