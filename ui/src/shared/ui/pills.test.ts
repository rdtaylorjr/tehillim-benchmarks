import { describe, expect, it } from "vitest";
import { pill } from "./pills";

describe("pill", () => {
  it("returns an empty string for a non-finite value", () => {
    expect(pill(NaN)).toBe("");
  });

  it("marks a value below 0.01 as good", () => {
    expect(pill(0.005)).toContain('class="pill good"');
  });

  it("marks a value between 0.01 and 0.05 as warn", () => {
    expect(pill(0.02)).toContain('class="pill warn"');
  });

  it("marks a value at or above 0.05 as bad", () => {
    expect(pill(0.2)).toContain('class="pill bad"');
  });

  it("floors the displayed text at <0.001 instead of a misleading 0.0000", () => {
    expect(pill(0.00003)).toContain("p=<0.001");
  });

  it("uses format.ts's 4-decimal p-value precision above the floor, matching the CLI printer", () => {
    expect(pill(0.0234)).toContain(">p=0.0234<");
  });

  it("uses format.ts's 4-decimal q-value precision for the q prefix", () => {
    expect(pill(0.0234, "q")).toContain(">q=0.0234<");
  });
});
