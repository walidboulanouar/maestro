import { describe, expect, it } from "vitest";
import { loadConfig } from "../src/config.js";
import { orchestrate } from "../src/core/orchestrator.js";
import { ModelRegistry } from "../src/registry/registry.js";
import type { ChatParams, ChatResult, ProviderAdapter, StreamChunk } from "../src/types.js";
import type { ProviderSet } from "../src/providers/index.js";

/** First call (planner) returns a JSON subtask list; later calls return step output. */
class PlannerThenSteps implements ProviderAdapter {
  readonly name = "mock";
  calls = 0;
  async chat(_p: ChatParams): Promise<ChatResult> {
    this.calls++;
    if (this.calls === 1) {
      return { text: '["analyze the problem", "write the solution", "summarize it"]', usage: { in: 5, out: 5 } };
    }
    return { text: `step ${this.calls} output`, usage: { in: 3, out: 4 }, finishReason: "stop" };
  }
  isConfigured() {
    return true;
  }
  async *stream(): AsyncIterable<StreamChunk> {
    yield { delta: "x", done: false };
    yield { delta: "", done: true, usage: { in: 1, out: 1 } };
  }
}

function depsWith(adapter: ProviderAdapter) {
  const providers = {
    get: () => adapter,
    configuredNames: () => new Set(["openrouter"]),
    hasRealProvider: () => true,
  } as unknown as ProviderSet;
  return { config: loadConfig({}), registry: ModelRegistry.default(), providers };
}

describe("maestro-ultra decomposition (with a planner)", () => {
  it("decomposes into multiple steps, executes each, last step is the answer", async () => {
    const adapter = new PlannerThenSteps();
    const result = await orchestrate(
      { model: "maestro-ultra", messages: [{ role: "user", content: "Build and document a feature." }] },
      depsWith(adapter),
    );
    expect(result.mode).toBe("ultra");
    expect(result.turns).toBe(3); // three planned subtasks
    expect(result.trace).toHaveLength(3);
    expect(result.trace[0]!.verifyReason).toContain("ultra step 1/3");
    expect(result.answer).toBe("step 4 output"); // last step (call #4: 1 planner + 3 steps)
    // every step + the planner consumed usage / cost
    expect(result.costUsd).toBeGreaterThanOrEqual(0);
  });

  it("falls back to a single step when the planner returns no list", async () => {
    const adapter: ProviderAdapter = {
      name: "mock",
      isConfigured: () => true,
      chat: async () => ({ text: "not a json list", usage: { in: 1, out: 1 } }),
      async *stream() {
        yield { delta: "", done: true, usage: { in: 1, out: 1 } };
      },
    };
    const result = await orchestrate(
      { model: "maestro-ultra", messages: [{ role: "user", content: "simple" }] },
      depsWith(adapter),
    );
    expect(result.mode).toBe("ultra");
    expect(result.turns).toBe(1);
  });
});
