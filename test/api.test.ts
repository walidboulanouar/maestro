import { describe, expect, it } from "vitest";
import { loadConfig } from "../src/config.js";
import { ProviderSet } from "../src/providers/index.js";
import { ModelRegistry } from "../src/registry/registry.js";
import { buildDeps, createApp } from "../src/server.js";

function app() {
  const config = loadConfig({});
  const registry = ModelRegistry.default();
  const providers = new ProviderSet(config);
  return createApp(buildDeps(config, registry, providers));
}

const json = (body: unknown) => ({
  method: "POST",
  headers: { "content-type": "application/json" },
  body: JSON.stringify(body),
});

describe("HTTP API", () => {
  it("GET /healthz reports providers", async () => {
    const res = await app().request("/healthz");
    expect(res.status).toBe(200);
    const body = (await res.json()) as { status: string; providers_configured: string[] };
    expect(body.status).toBe("ok");
    expect(body.providers_configured).toContain("mock");
  });

  it("GET /v1/models lists virtual + registry models", async () => {
    const res = await app().request("/v1/models");
    const body = (await res.json()) as { data: { id: string }[] };
    const ids = body.data.map((m) => m.id);
    expect(ids).toContain("maestro-auto");
    expect(ids).toContain("mock-frontier");
  });

  it("POST /v1/chat/completions returns OpenAI shape + maestro block", async () => {
    const res = await app().request(
      "/v1/chat/completions",
      json({ model: "maestro-auto", messages: [{ role: "user", content: "hello" }] }),
    );
    expect(res.status).toBe(200);
    expect(res.headers.get("x-maestro-mode")).toBeTruthy();
    const body = (await res.json()) as {
      choices: { message: { content: string } }[];
      maestro: { turns: number; cost_usd: number; route: unknown[] };
    };
    expect(body.choices[0]!.message.content.length).toBeGreaterThan(0);
    expect(body.maestro.route.length).toBeGreaterThanOrEqual(1);
  });

  it("POST /v1/route dry-runs without executing", async () => {
    const res = await app().request(
      "/v1/route",
      json({ model: "maestro-auto", messages: [{ role: "user", content: "write a regex" }] }),
    );
    expect(res.status).toBe(200);
    const body = (await res.json()) as { ladder: unknown[]; classify: { task: string } };
    expect(body.ladder.length).toBeGreaterThanOrEqual(1);
    expect(body.classify.task).toBe("code");
  });

  it("rejects malformed requests with 400", async () => {
    const res = await app().request("/v1/chat/completions", json({ messages: [] }));
    expect(res.status).toBe(400);
  });

  it("supports streaming (SSE)", async () => {
    const res = await app().request(
      "/v1/chat/completions",
      json({ model: "maestro-auto", stream: true, messages: [{ role: "user", content: "hi" }] }),
    );
    expect(res.status).toBe(200);
    const text = await res.text();
    expect(text).toContain("data:");
    expect(text).toContain("[DONE]");
  });
});
