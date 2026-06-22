/**
 * The Maestro HTTP server — an OpenAI-compatible endpoint with routing built in.
 *
 *   POST /v1/chat/completions   full OpenAI semantics (+ stream), model selects mode
 *   GET  /v1/models             maestro-* virtual models + every registry model
 *   POST /v1/route              dry-run: the route decision, no execution
 *   GET  /v1/traces/:id         inspect a past request's full trace
 *   GET  /healthz               liveness + which providers are configured
 */
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import type { MaestroConfig } from "./config.js";
import {
  anthropicMessageDelta,
  anthropicStreamStart,
  anthropicToChat,
  toAnthropicResponse,
  type AnthropicRequest,
} from "./api/anthropic.js";
import { maestroBlock, maestroHeaders, toOpenAIResponse } from "./api/shape.js";
import { classify } from "./core/classify.js";
import { orchestrate, resolveMode, type OrchestratorDeps } from "./core/orchestrator.js";
import { route, type RouteContext } from "./core/route.js";
import { totalChars } from "./core/transcript.js";
import { ChatCompletionRequestSchema } from "./types.js";
import { TraceStore } from "./transparency/trace.js";

export interface AppDeps extends OrchestratorDeps {
  trace: TraceStore;
}

export function createApp(deps: AppDeps): Hono {
  const app = new Hono();
  const { config, registry, providers, trace } = deps;

  app.get("/", (c) =>
    c.json({
      name: "maestro",
      description: "Open-source LLM orchestration brain. OpenAI-compatible.",
      endpoints: ["/v1/chat/completions", "/v1/messages", "/v1/models", "/v1/route", "/v1/traces/:id", "/healthz"],
      modes: ["maestro-auto", "maestro-fugu", "maestro-ultra", "<model-id> (passthrough)"],
    }),
  );

  app.get("/healthz", (c) =>
    c.json({
      status: "ok",
      providers_configured: [...providers.configuredNames()],
      has_real_provider: providers.hasRealProvider(),
      registry_models: registry.all().length,
      registry_updated: registry.updated,
      registry_age_days: registry.ageInDays(),
    }),
  );

  app.get("/v1/models", (c) => {
    const virtual = ["maestro-auto", "maestro-fugu", "maestro-ultra"].map((id) => ({
      id,
      object: "model",
      owned_by: "maestro",
    }));
    const real = registry.all().map((m) => ({
      id: m.id,
      object: "model",
      owned_by: m.provider,
      maestro: { slot: m.slot, tier: m.tier, strength: m.strength, caps: m.caps },
    }));
    return c.json({ object: "list", data: [...virtual, ...real] });
  });

  app.post("/v1/route", async (c) => {
    const parsed = ChatCompletionRequestSchema.safeParse(await c.req.json().catch(() => ({})));
    if (!parsed.success) {
      return c.json({ error: { message: parsed.error.message, type: "invalid_request_error" } }, 400);
    }
    const req = parsed.data;
    const sig = classify(req.messages);
    const configured = providers.configuredNames();
    const pool = registry.available(configured);
    const ctx: RouteContext = {
      thresholds: config.thresholds,
      hint: req.maestro,
      estInputTokens: Math.max(1, Math.ceil(totalChars(req.messages) / 4)),
      estOutputTokens: 600,
    };
    const decision = route(pool.length ? pool : registry.all(), sig, ctx);
    return c.json({
      mode: resolveMode(req.model, config.defaultMode).mode,
      classify: sig,
      ladder: decision.ladder.map((r) => ({
        slot: r.model.slot,
        model: r.model.id,
        provider: r.model.provider,
        tier: r.model.tier,
        effort: r.effort,
        strength: r.model.strength,
      })),
      frontier_only_estimate_usd: decision.frontierOnlyEstimateUsd,
      reason: decision.reason,
    });
  });

  app.get("/v1/traces/:id", (c) => {
    const t = trace.get(c.req.param("id"));
    if (!t) return c.json({ error: { message: "trace not found", type: "not_found" } }, 404);
    return c.json({ ...t, maestro: maestroBlock(t) });
  });

  // Anthropic Messages API — lets Claude Code use Maestro (ANTHROPIC_BASE_URL).
  app.post("/v1/messages", async (c) => {
    const raw = (await c.req.json().catch(() => ({}))) as AnthropicRequest;
    if (!Array.isArray(raw.messages) || raw.messages.length === 0) {
      return c.json({ type: "error", error: { type: "invalid_request_error", message: "messages required" } }, 400);
    }
    const echoModel = typeof raw.model === "string" ? raw.model : "maestro-auto";
    let result;
    try {
      result = await orchestrate(anthropicToChat(raw), { config, registry, providers });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return c.json({ type: "error", error: { type: "api_error", message } }, 502);
    }
    trace.record(result);

    if (!raw.stream) {
      for (const [k, v] of Object.entries(maestroHeaders(result))) c.header(k, v);
      return c.json(toAnthropicResponse(result, echoModel));
    }

    return streamSSE(c, async (stream) => {
      await stream.writeSSE({ event: "message_start", data: JSON.stringify(anthropicStreamStart(result, echoModel)) });
      await stream.writeSSE({
        event: "content_block_start",
        data: JSON.stringify({ type: "content_block_start", index: 0, content_block: { type: "text", text: "" } }),
      });
      for (const piece of chunkText(result.answer)) {
        await stream.writeSSE({
          event: "content_block_delta",
          data: JSON.stringify({ type: "content_block_delta", index: 0, delta: { type: "text_delta", text: piece } }),
        });
      }
      await stream.writeSSE({ event: "content_block_stop", data: JSON.stringify({ type: "content_block_stop", index: 0 }) });
      await stream.writeSSE({ event: "message_delta", data: JSON.stringify(anthropicMessageDelta(result)) });
      await stream.writeSSE({ event: "message_stop", data: JSON.stringify({ type: "message_stop" }) });
    });
  });

  app.post("/v1/chat/completions", async (c) => {
    const parsed = ChatCompletionRequestSchema.safeParse(await c.req.json().catch(() => ({})));
    if (!parsed.success) {
      return c.json({ error: { message: parsed.error.message, type: "invalid_request_error" } }, 400);
    }
    const req = parsed.data;

    let result;
    try {
      result = await orchestrate(req, { config, registry, providers });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return c.json({ error: { message, type: "upstream_error" } }, 502);
    }
    trace.record(result);

    if (!req.stream) {
      const headers = maestroHeaders(result);
      for (const [k, v] of Object.entries(headers)) c.header(k, v);
      return c.json(toOpenAIResponse(result, req.model));
    }

    // Streaming: emit the final answer as OpenAI-style SSE deltas, then a finish
    // chunk carrying the maestro block, then [DONE].
    return streamSSE(c, async (stream) => {
      const base = {
        id: result.id,
        object: "chat.completion.chunk",
        created: Math.floor(result.createdAt / 1000),
        model: req.model,
      };
      await stream.writeSSE({
        data: JSON.stringify({ ...base, choices: [{ index: 0, delta: { role: "assistant" } }] }),
      });
      for (const piece of chunkText(result.answer)) {
        await stream.writeSSE({
          data: JSON.stringify({ ...base, choices: [{ index: 0, delta: { content: piece } }] }),
        });
      }
      await stream.writeSSE({
        data: JSON.stringify({
          ...base,
          choices: [{ index: 0, delta: {}, finish_reason: "stop" }],
          maestro: maestroBlock(result),
        }),
      });
      await stream.writeSSE({ data: "[DONE]" });
    });
  });

  return app;
}

function chunkText(text: string, size = 24): string[] {
  const out: string[] = [];
  for (let i = 0; i < text.length; i += size) out.push(text.slice(i, i + size));
  return out.length ? out : [""];
}

export function buildDeps(config: MaestroConfig, registry: AppDeps["registry"], providers: AppDeps["providers"]): AppDeps {
  return { config, registry, providers, trace: new TraceStore(200, config.traceFile) };
}
