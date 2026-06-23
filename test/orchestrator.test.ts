import { describe, expect, it } from "vitest";
import { loadConfig } from "../src/config.js";
import { orchestrate, resolveMode } from "../src/core/orchestrator.js";
import { ProviderSet } from "../src/providers/index.js";
import { ModelRegistry } from "../src/registry/registry.js";

function deps(env: NodeJS.ProcessEnv = {}) {
  const config = loadConfig(env);
  return { config, registry: ModelRegistry.default(), providers: new ProviderSet(config) };
}

describe("resolveMode", () => {
  it("maps virtual model ids to modes", () => {
    expect(resolveMode("maestro-auto", "fugu").mode).toBe("auto");
    expect(resolveMode("maestro-fugu", "fugu").mode).toBe("fugu");
    expect(resolveMode("maestro-ultra", "fugu").mode).toBe("ultra");
  });
  it("treats a concrete model id as passthrough", () => {
    const r = resolveMode("anthropic/claude-opus-4.8", "fugu");
    expect(r.mode).toBe("passthrough");
    expect(r.pin).toBe("anthropic/claude-opus-4.8");
  });
});

describe("orchestrate (mock provider, offline)", () => {
  it("answers a simple prompt with the cheap tier and stops", async () => {
    const result = await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: "say hello" }] },
      deps(),
    );
    expect(result.answer.length).toBeGreaterThan(0);
    expect(result.trace[0]!.model).toContain("cheap");
    expect(result.trace.at(-1)!.verdict).toBe("ACCEPT");
  });

  it("escalates hard prompts toward the frontier tier", async () => {
    const result = await orchestrate(
      {
        model: "maestro-auto",
        messages: [
          {
            role: "user",
            content:
              "Design and prove an efficient end-to-end algorithm, rigorously analyzing trade-offs and edge cases.",
          },
        ],
      },
      deps(),
    );
    expect(result.trace.at(-1)!.model).toContain("frontier");
    expect(result.trace.at(-1)!.verdict).toBe("ACCEPT");
  });

  it("passthrough mode does not classify or verify", async () => {
    const result = await orchestrate(
      { model: "mock-frontier", messages: [{ role: "user", content: "anything" }] },
      deps(),
    );
    expect(result.mode).toBe("passthrough");
    expect(result.turns).toBe(1);
    expect(result.trace[0]!.verdict).toBeUndefined();
  });

  it("reports cost and a frontier-only comparison", async () => {
    const result = await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: "hello" }] },
      deps({ OPENROUTER_API_KEY: "x", MAESTRO_FORCE_MOCK: "true" }),
    );
    expect(result.costVsFrontierOnlyUsd).toBeGreaterThanOrEqual(result.costUsd);
  });

  it("maestro-ultra runs the decomposition path and returns an answer", async () => {
    const result = await orchestrate(
      {
        model: "maestro-ultra",
        messages: [{ role: "user", content: "Design and implement a rate limiter, then write tests for it." }],
      },
      deps(),
    );
    expect(result.mode).toBe("ultra");
    expect(result.answer.length).toBeGreaterThan(0);
    expect(result.turns).toBeGreaterThanOrEqual(1);
  });

  it("respects maestro.verify=false (single shot)", async () => {
    const result = await orchestrate(
      {
        model: "maestro-fugu",
        messages: [{ role: "user", content: "prove a hard theorem rigorously step by step" }],
        maestro: { verify: false },
      },
      deps(),
    );
    expect(result.turns).toBe(1);
  });
});
