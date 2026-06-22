import { describe, expect, it } from "vitest";
import { learnedRoute } from "../src/core/learned-router.js";
import { loadConfig } from "../src/config.js";
import { orchestrate } from "../src/core/orchestrator.js";
import { ProviderSet } from "../src/providers/index.js";
import { ModelRegistry } from "../src/registry/registry.js";

describe("learned router seam (safe by default)", () => {
  it("returns null on an unreachable sidecar (never throws)", async () => {
    const r = await learnedRoute("user: hi", "http://127.0.0.1:1/none", 200);
    expect(r).toBeNull();
  });

  it("is unconfigured by default", () => {
    expect(loadConfig({}).routerUrl).toBeUndefined();
  });

  it("orchestration still works when a bad router URL is set (falls back to heuristic)", async () => {
    const config = loadConfig({ MAESTRO_ROUTER_URL: "http://127.0.0.1:1/none" });
    const deps = { config, registry: ModelRegistry.default(), providers: new ProviderSet(config) };
    const result = await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: "hello" }] },
      deps,
    );
    expect(result.answer.length).toBeGreaterThan(0);
  });
});
