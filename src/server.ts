/**
 * The Maestro HTTP server — an OpenAI-compatible endpoint with routing built in.
 *
 *   POST /v1/chat/completions   full OpenAI semantics (+ stream), model selects mode
 *   GET  /v1/models             maestro-* virtual models + every registry model
 *   POST /v1/route              dry-run: the route decision, no execution
 *   GET  /v1/traces/:id         inspect a past request's full trace
 *   GET  /healthz               liveness + which providers are configured
 */
import { readFileSync } from "node:fs";
import { Hono } from "hono";
import { streamSSE } from "hono/streaming";
import type { MaestroConfig } from "./config.js";
import type { ChatCompletionRequest } from "./types.js";
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

  // --- auth + rate limit + budget (all opt-in via config) ---
  const rl = new Map<string, { n: number; t: number }>();
  const spend = new Map<string, number>();
  const auth = config.auth;
  app.use("/v1/*", async (c, next) => {
    if (auth.apiKeys.length === 0) return next(); // open by default
    const key = (c.req.header("authorization") ?? "").replace(/^Bearer\s+/i, "");
    if (!auth.apiKeys.includes(key)) {
      return c.json({ error: { message: "invalid api key", type: "auth_error" } }, 401);
    }
    if (auth.rateLimitPerMin > 0) {
      const now = Date.now();
      const w = rl.get(key);
      if (!w || now - w.t > 60_000) rl.set(key, { n: 1, t: now });
      else if (++w.n > auth.rateLimitPerMin) {
        return c.json({ error: { message: "rate limit exceeded", type: "rate_limit_error" } }, 429);
      }
    }
    if (auth.budgetUsd > 0 && (spend.get(key) ?? 0) >= auth.budgetUsd) {
      return c.json({ error: { message: "budget exceeded", type: "budget_error" } }, 402);
    }
    c.set("apiKey" as never, key as never);
    await next();
  });
  const charge = (c: { get: (k: never) => unknown }, usd: number) => {
    const key = c.get("apiKey" as never) as string | undefined;
    if (key && auth.budgetUsd > 0) spend.set(key, (spend.get(key) ?? 0) + usd);
  };

  // --- prompt registry (opt-in): maestro.prompt prepends a named system prompt ---
  let prompts: Record<string, string> = {};
  if (config.promptsFile) {
    try {
      prompts = JSON.parse(readFileSync(config.promptsFile, "utf8")) as Record<string, string>;
    } catch {
      /* ignore missing/invalid prompts file */
    }
  }
  const applyPrompt = (r: ChatCompletionRequest): ChatCompletionRequest => {
    const name = r.maestro?.prompt;
    if (name && prompts[name]) r.messages = [{ role: "system", content: prompts[name] }, ...r.messages];
    return r;
  };

  // --- dedupe cache (opt-in): identical requests skip the model call ---
  const cache = new Map<string, import("./types.js").OrchestrationResult>();
  const cacheKey = (req: { model: string; messages: unknown; response_format?: unknown; tools?: unknown }) =>
    JSON.stringify({ m: req.model, msgs: req.messages, rf: req.response_format, tools: req.tools });

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

  // recent traces (for the UI)
  app.get("/v1/traces", (c) => {
    const n = Math.min(100, Number(c.req.query("limit") ?? 30) || 30);
    return c.json({ data: trace.recent(n).map((t) => ({ id: t.id, ...maestroBlock(t), created: t.createdAt })) });
  });

  // minimal trace-viewer UI
  app.get("/ui", (c) => c.html(TRACE_UI));

  // Anthropic Messages API — lets Claude Code use Maestro (ANTHROPIC_BASE_URL).
  app.post("/v1/messages", async (c) => {
    const raw = (await c.req.json().catch(() => ({}))) as AnthropicRequest;
    if (!Array.isArray(raw.messages) || raw.messages.length === 0) {
      return c.json({ type: "error", error: { type: "invalid_request_error", message: "messages required" } }, 400);
    }
    const echoModel = typeof raw.model === "string" ? raw.model : "maestro-auto";
    let result;
    try {
      result = await orchestrate(applyPrompt(anthropicToChat(raw)), { config, registry, providers });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      return c.json({ type: "error", error: { type: "api_error", message } }, 502);
    }
    trace.record(result);
    charge(c, result.costUsd);

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
    const req = applyPrompt(parsed.data);

    const ckey = config.cacheEnabled && !req.stream ? cacheKey(req) : null;
    if (ckey) {
      const hit = cache.get(ckey);
      if (hit) {
        const served = { ...hit, cached: true };
        for (const [k, v] of Object.entries(maestroHeaders(served))) c.header(k, v);
        c.header("x-maestro-cached", "true");
        return c.json(toOpenAIResponse(served, req.model));
      }
    }

    let result;
    try {
      result = await orchestrate(req, { config, registry, providers });
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      const timedOut = /timed out/i.test(message);
      return c.json(
        { error: { message, type: timedOut ? "timeout_error" : "upstream_error" } },
        timedOut ? 504 : 502,
      );
    }
    trace.record(result);
    charge(c, result.costUsd);
    if (ckey) cache.set(ckey, result);

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
      if (result.toolCalls) {
        await stream.writeSSE({
          data: JSON.stringify({ ...base, choices: [{ index: 0, delta: { tool_calls: result.toolCalls } }] }),
        });
      }
      await stream.writeSSE({
        data: JSON.stringify({
          ...base,
          choices: [{ index: 0, delta: {}, finish_reason: result.finishReason ?? "stop" }],
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
  return { config, registry, providers, trace: new TraceStore(200, config.traceFile, config.redactTraces) };
}

const TRACE_UI = `<!doctype html><html><head><meta charset="utf-8"/>
<title>Maestro traces</title><style>
*{box-sizing:border-box;margin:0;border-radius:0}
body{background:#f3f2ec;color:#0a0a0a;font-family:ui-monospace,Menlo,monospace;padding:24px}
h1{text-transform:uppercase;font-size:22px;border-bottom:4px solid #0a0a0a;padding-bottom:8px}
.sub{color:#666;font-size:12px;margin:6px 0 18px}
table{width:100%;border-collapse:collapse;font-size:12.5px}
th,td{border:2px solid #0a0a0a;padding:8px 10px;text-align:left}
th{background:#0a0a0a;color:#f3f2ec;text-transform:uppercase}
.acc{font-weight:700}.rev{color:#666}
.empty{border:3px dashed #0a0a0a;padding:20px;text-align:center;color:#666;margin-top:12px}
</style></head><body>
<h1>Maestro &middot; traces</h1>
<div class="sub">live routing decisions (auto-refresh 3s) &middot; redacted by default</div>
<div id="out"><div class="empty">no requests yet, send one to /v1/chat/completions</div></div>
<script>
async function load(){
  try{
    const r=await fetch('/v1/traces?limit=40');const d=(await r.json()).data||[];
    if(!d.length)return;
    let h='<table><tr><th>when</th><th>mode</th><th>task</th><th>route</th><th>turns</th><th>cost $</th><th>saved</th><th>cached</th></tr>';
    for(const t of d){
      const route=(t.route||[]).map(x=>x.model.split('/').pop()+(x.verdict?(':'+x.verdict):'')).join(' → ');
      h+='<tr><td>'+new Date(t.created).toLocaleTimeString()+'</td><td>'+t.mode+'</td><td>'+(t.classify?t.classify.task:'')+'</td><td>'+route+'</td><td>'+t.turns+'</td><td>'+t.cost_usd+'</td><td>'+t.savings_pct+'%</td><td>'+(t.cached?'yes':'')+'</td></tr>';
    }
    document.getElementById('out').innerHTML=h+'</table>';
  }catch(e){}
}
load();setInterval(load,3000);
</script></body></html>`;
