import { describe, expect, it } from "vitest";
import { formatEffectSize, formatPValue, formatQValue } from "./format";

describe("formatPValue", () => {
  it("matches validate_against_genre.py's CLI printer, 4 decimal places", () => {
    expect(formatPValue(0.0314159)).toBe("0.0314");
  });

  it("shows the smallest attainable permutation p-value at full precision", () => {
    expect(formatPValue(1 / 10001)).toBe("0.0001");
  });

  it("renders an unavailable value as an em dash, never the string NaN", () => {
    expect(formatPValue(NaN)).toBe("—");
  });
});

describe("formatQValue", () => {
  it("matches validate_against_genre.py's CLI printer, 4 decimal places", () => {
    expect(formatQValue(0.0567)).toBe("0.0567");
  });

  it("renders an unavailable value as an em dash, never the string NaN", () => {
    expect(formatQValue(NaN)).toBe("—");
  });
});

describe("formatEffectSize", () => {
  it("matches validate_against_genre.py's CLI printer, 3 decimal places with an explicit sign", () => {
    expect(formatEffectSize(3.2)).toBe("+3.200");
  });

  it("keeps the negative sign for a below-null effect size", () => {
    expect(formatEffectSize(-1.5)).toBe("-1.500");
  });

  it("renders an unavailable value as an em dash, never the string NaN", () => {
    expect(formatEffectSize(NaN)).toBe("—");
  });
});
