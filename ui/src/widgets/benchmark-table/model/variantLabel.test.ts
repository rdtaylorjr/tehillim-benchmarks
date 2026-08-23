import { describe, expect, it } from "vitest";
import { variantLabel } from "./variantLabel";

describe("variantLabel", () => {
  it("appends a variant tag when a real text variant is present", () => {
    expect(
      variantLabel({ model_base: "bge_m3", text_variant: "vocalized" }),
    ).toBe('bge_m3 <span class="variant-tag">vocalized</span>');
  });

  it("returns the bare model_base when the text variant is unknown", () => {
    expect(
      variantLabel({ model_base: "homograph_binary", text_variant: "unknown" }),
    ).toBe("homograph_binary");
  });

  it("returns the bare model_base when there is no text variant field at all", () => {
    expect(variantLabel({ model_base: "phrase_typ_1gram" })).toBe(
      "phrase_typ_1gram",
    );
  });
});
