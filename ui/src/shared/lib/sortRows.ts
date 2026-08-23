export type SortDir = "asc" | "desc";

/** Sorts a copy of rows by one field: localeCompare for strings, numeric otherwise with NaN/missing as -Infinity. */
export function sortRows<T extends object>(
  rows: T[],
  key: string,
  dir: SortDir,
): T[] {
  return [...rows].sort((a, b) => {
    const av = (a as Record<string, unknown>)[key];
    const bv = (b as Record<string, unknown>)[key];
    if (typeof av === "string") {
      const left = av;
      const right = (bv as string) ?? "";
      return dir === "asc"
        ? left.localeCompare(right)
        : right.localeCompare(left);
    }
    const leftNum =
      av === null || av === undefined || Number.isNaN(av as number)
        ? -Infinity
        : (av as number);
    const rightNum =
      bv === null || bv === undefined || Number.isNaN(bv as number)
        ? -Infinity
        : (bv as number);
    return dir === "asc" ? leftNum - rightNum : rightNum - leftNum;
  });
}
