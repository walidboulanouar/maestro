import { describe, expect, it } from "vitest";
import { costOf, roundUsd } from "../src/core/cost.js";
import { contentToText, lastUserMessage, toRawTranscript, totalChars } from "../src/core/transcript.js";
import type { ModelSpec } from "../src/types.js";

const model: ModelSpec = {
  slot: "x", id: "x", provider: "openrouter", tier: "mid", strength: 80, caps: [],
  price: { in: 1, out: 2, updated: "x" }, contextWindow: 1000,
};

describe("cost", () => {
  it("computes USD from per-1M prices", () => {
    // 1M in @ $1 + 0.5M out @ $2 = 1 + 1 = 2
    expect(costOf(model, { in: 1_000_000, out: 500_000 })).toBeCloseTo(2);
    expect(costOf(model, { in: 0, out: 0 })).toBe(0);
  });
  it("roundUsd keeps sane precision", () => {
    expect(roundUsd(0)).toBe(0);
    expect(roundUsd(0.0001234)).toBe(0.000123);
  });
});

describe("transcript helpers", () => {
  it("contentToText handles string, null, and multimodal arrays", () => {
    expect(contentToText("hi")).toBe("hi");
    expect(contentToText(null)).toBe("");
    expect(contentToText(undefined)).toBe("");
    expect(contentToText([{ type: "text", text: "a" }, { type: "text", text: "b" }])).toBe("a b");
  });
  it("lastUserMessage returns the last user content as text", () => {
    expect(
      lastUserMessage([
        { role: "system", content: "sys" },
        { role: "user", content: "first" },
        { role: "assistant", content: "reply" },
        { role: "user", content: [{ type: "text", text: "second" }] },
      ]),
    ).toBe("second");
  });
  it("toRawTranscript uses the raw role: content form", () => {
    expect(toRawTranscript([{ role: "user", content: "hi" }])).toBe("user: hi");
  });
  it("totalChars counts flattened content", () => {
    expect(totalChars([{ role: "user", content: "abc" }, { role: "assistant", content: null }])).toBe(3);
  });
});
