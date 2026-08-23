import { describe, expect, it } from "vitest";
import { formatNumber } from "./numberFormat";

describe("formatNumber", () => {
  it("defaults to 4 decimal places", () => {
    expect(formatNumber(0.669312)).toBe("0.6693");
  });

  it("honors an explicit digit count", () => {
    expect(formatNumber(0.02894, 5)).toBe("0.02894");
  });

  it("renders an unavailable value as an em dash, never the string NaN", () => {
    expect(formatNumber(NaN)).toBe("—");
  });
});
