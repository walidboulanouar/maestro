import { describe, expect, it } from "vitest";
import { loadConfig } from "../src/config.js";
import { ProviderSet } from "../src/providers/index.js";
import { ModelRegistry } from "../src/registry/registry.js";
import { TraceStore } from "../src/transparency/trace.js";
import { orchestrate } from "../src/core/orchestrator.js";
import { buildDeps, createApp } from "../src/server.js";
import type { OrchestrationResult } from "../src/types.js";

const json = (body: unknown, headers: Record<string, string> = {}) => ({
  method: "POST",
  headers: { "content-type": "application/json", ...headers },
  body: JSON.stringify(body),
});

describe("orchestration profiles", () => {
  it("cheap profile stays on cheaper tiers longer, quality escalates sooner", () => {
    const cheap = loadConfig({ MAESTRO_PROFILE: "cheap" });
    const quality = loadConfig({ MAESTRO_PROFILE: "quality" });
    expect(cheap.thresholds.high).toBeGreaterThan(quality.thresholds.high);
    expect(quality.maxTurns).toBeGreaterThan(cheap.maxTurns);
  });
  it("explicit env overrides the profile preset", () => {
    const c = loadConfig({ MAESTRO_PROFILE: "cheap", MAESTRO_MAX_TURNS: "5" });
    expect(c.maxTurns).toBe(5);
  });
  it("defaults to balanced", () => {
    expect(loadConfig({}).profile).toBe("balanced");
  });
});

describe("multi-provider config", () => {
  it("knows the common OpenAI-compatible providers", () => {
    const p = loadConfig({}).providers;
    for (const name of ["openrouter", "vercel-gateway", "openai", "groq", "together", "fireworks", "deepinfra", "local-openai"]) {
      expect(p[name]).toBeDefined();
    }
  });
  it("marks a provider configured when its key is set", () => {
    const config = loadConfig({ GROQ_API_KEY: "gsk_test" });
    const names = new ProviderSet(config).configuredNames();
    expect(names.has("groq")).toBe(true);
  });
  it("supports custom providers via MAESTRO_PROVIDERS", () => {
    const config = loadConfig({
      MAESTRO_PROVIDERS: JSON.stringify([{ name: "myrouter", baseUrl: "https://x/v1", apiKey: "k" }]),
    });
    expect(config.providers["myrouter"]?.baseUrl).toBe("https://x/v1");
    expect(new ProviderSet(config).configuredNames().has("myrouter")).toBe(true);
  });
  it("falls back to mock when nothing is configured", () => {
    const names = new ProviderSet(loadConfig({})).configuredNames();
    expect([...names]).toEqual(["mock"]);
  });
});

describe("config: retries + timeout", () => {
  it("defaults and overrides", () => {
    expect(loadConfig({}).maxRetries).toBe(2);
    expect(loadConfig({ MAESTRO_MAX_RETRIES: "5" }).maxRetries).toBe(5);
    expect(loadConfig({ MAESTRO_REQUEST_TIMEOUT_MS: "9000" }).requestTimeoutMs).toBe(9000);
  });
});

describe("trace redaction", () => {
  it("scrubs tool args + secrets and drops upstreamRaw", () => {
    const store = new TraceStore(10, undefined, true);
    const result = {
      id: "chatcmpl-1", answer: "your key is sk-abcdef0123456789ABCDEF ok", mode: "auto",
      signature: { task: "general", difficulty: 0, caps: [], freshness: false, sensitive: false, confidence: 1, reason: "" },
      trace: [], turns: 1, usageByModel: {}, costUsd: 0, costVsFrontierOnlyUsd: 0, createdAt: 1,
      finishReason: "tool_calls",
      toolCalls: [{ id: "c1", type: "function", function: { name: "f", arguments: '{"secret":"x"}' } }],
      upstreamRaw: { huge: true },
    } satisfies OrchestrationResult;
    store.record(result);
    const got = store.get("chatcmpl-1")!;
    expect(got.answer).toContain("[redacted-key]");
    expect((got.toolCalls![0] as { function: { arguments: string } }).function.arguments).toBe("[redacted]");
    expect((got as { upstreamRaw?: unknown }).upstreamRaw).toBeUndefined();
  });
});

describe("skip-verify when confident", () => {
  it("does a single turn when classifier confidence is high enough", async () => {
    const config = loadConfig({ MAESTRO_SKIP_VERIFY_ABOVE_CONFIDENCE: "0.8" });
    const deps = { config, registry: ModelRegistry.default(), providers: new ProviderSet(config) };
    // a clear single-signal code task -> classifier confidence ~0.85
    const result = await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: "Translate this sentence into Spanish." }] },
      deps,
    );
    expect(result.turns).toBe(1);
    expect(result.trace[0]!.verdict).toBeUndefined();
  });
});

describe("dedupe cache + trace viewer", () => {
  function app(env: Record<string, string>) {
    const config = loadConfig(env);
    return createApp(buildDeps(config, ModelRegistry.default(), new ProviderSet(config)));
  }
  it("serves an identical second request from cache", async () => {
    const a = app({ MAESTRO_CACHE: "true" });
    const body = { model: "maestro-auto", messages: [{ role: "user", content: "cache me please" }] };
    const r1 = await a.request("/v1/chat/completions", json(body));
    const b1 = (await r1.json()) as { maestro: { cached: boolean } };
    const r2 = await a.request("/v1/chat/completions", json(body));
    const b2 = (await r2.json()) as { maestro: { cached: boolean } };
    expect(b1.maestro.cached).toBe(false);
    expect(b2.maestro.cached).toBe(true);
  });
  it("lists recent traces and serves the UI", async () => {
    const a = app({});
    await a.request("/v1/chat/completions", json({ model: "maestro-auto", messages: [{ role: "user", content: "hi" }] }));
    const list = (await (await a.request("/v1/traces")).json()) as { data: unknown[] };
    expect(list.data.length).toBeGreaterThanOrEqual(1);
    const ui = await a.request("/ui");
    expect(ui.status).toBe(200);
    expect((await ui.text())).toContain("Maestro");
  });
});

describe("prompt registry hint", () => {
  it("is a safe no-op when the named prompt is unknown", async () => {
    const config = loadConfig({});
    const a = createApp(buildDeps(config, ModelRegistry.default(), new ProviderSet(config)));
    const res = await a.request(
      "/v1/chat/completions",
      json({ model: "maestro-auto", messages: [{ role: "user", content: "hi" }], maestro: { prompt: "nope" } }),
    );
    expect(res.status).toBe(200);
  });
});

describe("auth + rate limit", () => {
  function app(env: Record<string, string>) {
    const config = loadConfig(env);
    return createApp(buildDeps(config, ModelRegistry.default(), new ProviderSet(config)));
  }
  it("401 without a key when keys are configured", async () => {
    const res = await app({ MAESTRO_API_KEYS: "secret1" }).request("/v1/models");
    expect(res.status).toBe(401);
  });
  it("200 with a valid key", async () => {
    const res = await app({ MAESTRO_API_KEYS: "secret1" }).request("/v1/models", {
      headers: { authorization: "Bearer secret1" },
    });
    expect(res.status).toBe(200);
  });
  it("429 when over the per-minute rate limit", async () => {
    const a = app({ MAESTRO_API_KEYS: "k", MAESTRO_RATE_LIMIT_PER_MIN: "1" });
    const h = { authorization: "Bearer k" };
    const r1 = await a.request("/v1/models", { headers: h });
    const r2 = await a.request("/v1/models", { headers: h });
    expect(r1.status).toBe(200);
    expect(r2.status).toBe(429);
  });
  it("stays open when no keys configured", async () => {
    const res = await app({}).request("/v1/models");
    expect(res.status).toBe(200);
  });
});
