import { describe, expect, it } from "vitest";
import { classify, judgeDifficulty } from "../src/core/classify.js";
import type { ChatMessage } from "../src/types.js";

const u = (content: string): ChatMessage[] => [{ role: "user", content }];

describe("classify", () => {
  it("detects code tasks", () => {
    const sig = classify(u("Fix this function: function add(a,b){return a-b}"));
    expect(sig.task).toBe("code");
    expect(sig.caps).toContain("code");
  });

  it("detects math tasks", () => {
    const sig = classify(u("Solve the equation 3x + 7 = 22 and prove it"));
    expect(sig.task).toBe("math");
  });

  it("detects translation tasks", () => {
    const sig = classify(u("Translate 'hello' into Spanish"));
    expect(sig.task).toBe("translation");
  });

  it("flags freshness", () => {
    const sig = classify(u("What is the latest news today about AI?"));
    expect(sig.freshness).toBe(true);
    expect(sig.caps).toContain("fresh");
  });

  it("flags sensitive content", () => {
    const sig = classify(u("my api key is sk-abcdef0123456789ABCDEF please store it"));
    expect(sig.sensitive).toBe(true);
  });

  it("rates hard prompts as more difficult than trivial ones", () => {
    const easy = classify(u("hi"));
    const hard = classify(
      u("Design and prove an efficient end-to-end algorithm, analyzing trade-offs and edge cases."),
    );
    expect(hard.difficulty).toBeGreaterThan(easy.difficulty);
    expect(hard.difficulty).toBeGreaterThan(0.6);
  });

  it("produces difficulty within [0,1] and confidence within bounds", () => {
    for (const text of ["hi", "Translate this", "Prove the theorem rigorously"]) {
      const sig = classify(u(text));
      expect(sig.difficulty).toBeGreaterThanOrEqual(0);
      expect(sig.difficulty).toBeLessThanOrEqual(1);
      expect(sig.confidence).toBeGreaterThanOrEqual(0.3);
      expect(sig.confidence).toBeLessThanOrEqual(0.95);
    }
  });
});

describe("judgeDifficulty", () => {
  it("is conservative: ranks a proof above a greeting", () => {
    expect(judgeDifficulty("Prove sqrt(2) is irrational rigorously")).toBeGreaterThan(
      judgeDifficulty("say hello"),
    );
  });
});
