import { describe, expect, it } from "vitest";
import { guardrailScore, route, type RouteContext } from "../src/core/route.js";
import type { ModelSpec, TaskSignature } from "../src/types.js";

const M = (over: Partial<ModelSpec> & Pick<ModelSpec, "id" | "tier" | "strength">): ModelSpec => ({
  slot: over.id,
  provider: "openrouter",
  caps: ["code", "math", "reasoning", "factual", "translation"],
  price: { in: 1, out: 1, updated: "2026-06-22" },
  contextWindow: 128_000,
  ...over,
});

const pool: ModelSpec[] = [
  M({ id: "cheap", tier: "cheap", strength: 60, price: { in: 0.1, out: 0.3, updated: "x" } }),
  M({ id: "mid", tier: "mid", strength: 82, tags: { "strong:code": 1 }, price: { in: 0.5, out: 2, updated: "x" } }),
  M({ id: "frontier", tier: "frontier", strength: 97, price: { in: 5, out: 25, updated: "x" } }),
];

const sig = (over: Partial<TaskSignature>): TaskSignature => ({
  task: "code",
  difficulty: 0.5,
  caps: [],
  freshness: false,
  sensitive: false,
  confidence: 0.8,
  reason: "",
  ...over,
});

const ctx: RouteContext = {
  thresholds: { low: 0.33, high: 0.7 },
  estInputTokens: 1000,
  estOutputTokens: 600,
};

describe("guardrailScore", () => {
  it("rewards strong-tag matches for the task", () => {
    const withTag = guardrailScore(pool[1]!, sig({ task: "code" }));
    const withoutTag = guardrailScore(pool[1]!, sig({ task: "math" }));
    expect(withTag).toBeGreaterThan(withoutTag);
  });
});

describe("route", () => {
  it("starts cheap for easy tasks", () => {
    const d = route(pool, sig({ difficulty: 0.1 }), ctx);
    expect(d.ladder[0]!.model.tier).toBe("cheap");
  });

  it("starts frontier for hard tasks", () => {
    const d = route(pool, sig({ difficulty: 0.95 }), ctx);
    expect(d.ladder[0]!.model.tier).toBe("frontier");
  });

  it("builds an escalation ladder up to frontier", () => {
    const d = route(pool, sig({ difficulty: 0.1 }), ctx);
    const tiers = d.ladder.map((r) => r.model.tier);
    expect(tiers).toEqual(["cheap", "mid", "frontier"]);
  });

  it("filters by required capability", () => {
    const restricted: ModelSpec[] = [
      M({ id: "novision", tier: "cheap", strength: 60, caps: ["code"] }),
      M({ id: "vision", tier: "frontier", strength: 95, caps: ["code", "vision"] }),
    ];
    const d = route(restricted, sig({ caps: ["vision"] }), ctx);
    expect(d.ladder.every((r) => r.model.caps.includes("vision"))).toBe(true);
  });

  it("never strands a request (returns at least one rung)", () => {
    const d = route(pool, sig({ caps: ["vision"] }), ctx); // none have vision
    expect(d.ladder.length).toBeGreaterThanOrEqual(1);
  });

  it("estimates the frontier-only cost from the strongest model", () => {
    const d = route(pool, sig({ difficulty: 0.1 }), ctx);
    expect(d.frontierOnlyEstimateUsd).toBeGreaterThan(0);
  });
});
