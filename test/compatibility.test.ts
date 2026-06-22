/**
 * OpenRouter / agent-harness compatibility tests (per Codex's verdict).
 * Proves Maestro behaves as a drop-in OpenAI/OpenRouter-compatible endpoint:
 * it forwards every request field verbatim, preserves upstream response fields,
 * runs two clean paths (passthrough vs routed), and never executes tools.
 */
import { describe, expect, it } from "vitest";
import { loadConfig } from "../src/config.js";
import { orchestrate } from "../src/core/orchestrator.js";
import { ModelRegistry } from "../src/registry/registry.js";
import { toOpenAIResponse } from "../src/api/shape.js";
import type {
  ChatParams,
  ChatResult,
  OrchestrationResult,
  ProviderAdapter,
  StreamChunk,
} from "../src/types.js";
import type { ProviderSet } from "../src/providers/index.js";

class CaptureAdapter implements ProviderAdapter {
  readonly name = "mock";
  last?: ChatParams;
  toReturn: ChatResult = { text: "ok", usage: { in: 5, out: 5 }, finishReason: "stop" };
  isConfigured() {
    return true;
  }
  async chat(params: ChatParams): Promise<ChatResult> {
    this.last = params;
    return this.toReturn;
  }
  async *stream(): AsyncIterable<StreamChunk> {
    yield { delta: "ok", done: false };
    yield { delta: "", done: true, usage: { in: 5, out: 5 } };
  }
}

function depsWith(cap: CaptureAdapter) {
  const providers = {
    get: () => cap,
    configuredNames: () => new Set(["mock"]),
    hasRealProvider: () => false,
  } as unknown as ProviderSet;
  return { config: loadConfig({}), registry: ModelRegistry.default(), providers };
}

describe("compatibility: request field pass-through (routed mode)", () => {
  it("forwards response_format and seed (OpenAI SDK style)", async () => {
    const cap = new CaptureAdapter();
    await orchestrate(
      {
        model: "maestro-auto",
        messages: [{ role: "user", content: "Return JSON" }],
        response_format: { type: "json_object" },
        temperature: 0,
        seed: 123,
      },
      depsWith(cap),
    );
    expect(cap.last?.extra?.response_format).toEqual({ type: "json_object" });
    expect(cap.last?.extra?.seed).toBe(123);
    expect(cap.last?.extra?.temperature).toBe(0);
  });

  it("forwards Vercel-AI-SDK-style json_schema structured output", async () => {
    const cap = new CaptureAdapter();
    const rf = { type: "json_schema", json_schema: { name: "x", schema: { type: "object" } } };
    await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: "hi" }], response_format: rf },
      depsWith(cap),
    );
    expect(cap.last?.extra?.response_format).toEqual(rf);
  });

  it("forwards reasoning, parallel_tool_calls, plugins, max_completion_tokens", async () => {
    const cap = new CaptureAdapter();
    await orchestrate(
      {
        model: "maestro-auto",
        messages: [{ role: "user", content: "hi" }],
        reasoning: { effort: "high" },
        parallel_tool_calls: false,
        plugins: [{ id: "web" }],
        max_completion_tokens: 1024,
      },
      depsWith(cap),
    );
    expect(cap.last?.extra?.reasoning).toEqual({ effort: "high" });
    expect(cap.last?.extra?.parallel_tool_calls).toBe(false);
    expect(cap.last?.extra?.plugins).toEqual([{ id: "web" }]);
    expect(cap.last?.extra?.max_completion_tokens).toBe(1024);
  });

  it("never sends model/messages/maestro inside extra", async () => {
    const cap = new CaptureAdapter();
    await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: "hi" }], maestro: { maxTurns: 2 } },
      depsWith(cap),
    );
    expect(cap.last?.extra && "model" in cap.last.extra).toBe(false);
    expect(cap.last?.extra && "messages" in cap.last.extra).toBe(false);
    expect(cap.last?.extra && "maestro" in cap.last.extra).toBe(false);
  });
});

describe("compatibility: concrete-model passthrough (Path A)", () => {
  it("does not classify/verify and forwards provider/session_id/metadata/trace", async () => {
    const cap = new CaptureAdapter();
    const result = await orchestrate(
      {
        model: "anthropic/claude-opus-4.8",
        messages: [{ role: "user", content: "hello" }],
        provider: { require_parameters: true, sort: "latency" },
        session_id: "agent-session-123",
        metadata: { app: "test-agent" },
        trace: { trace_id: "trace_123" },
      },
      depsWith(cap),
    );
    expect(result.mode).toBe("passthrough");
    expect(result.turns).toBe(1);
    expect(result.trace[0]!.verdict).toBeUndefined(); // no verify in passthrough
    expect(cap.last?.extra?.provider).toEqual({ require_parameters: true, sort: "latency" });
    expect(cap.last?.extra?.session_id).toBe("agent-session-123");
    expect(cap.last?.extra?.metadata).toEqual({ app: "test-agent" });
    expect(cap.last?.extra?.trace).toEqual({ trace_id: "trace_123" });
    expect(cap.last?.model).toBe("anthropic/claude-opus-4.8"); // not mutated
  });

  it("does not inject reasoning_effort in passthrough (preserves caller params)", async () => {
    const cap = new CaptureAdapter();
    await orchestrate(
      { model: "openai/gpt-5.5", messages: [{ role: "user", content: "hi" }] },
      depsWith(cap),
    );
    expect(cap.last?.effort).toBeUndefined();
  });
});

describe("compatibility: upstream response fields are preserved", () => {
  it("toOpenAIResponse keeps native_finish_reason, system_fingerprint, usage.cost, openrouter_metadata", () => {
    const upstream = {
      id: "gen-abc",
      object: "chat.completion",
      model: "anthropic/claude-opus-4.8",
      system_fingerprint: "fp_123",
      choices: [
        {
          index: 0,
          message: { role: "assistant", content: "hi" },
          finish_reason: "stop",
          native_finish_reason: "end_turn",
        },
      ],
      usage: { prompt_tokens: 5, completion_tokens: 3, cost: 0.0004, cost_details: { upstream: 0.0004 } },
      openrouter_metadata: { provider: "anthropic" },
    };
    const result = {
      id: "chatcmpl-x",
      answer: "hi",
      mode: "passthrough",
      signature: { task: "general", difficulty: 0, caps: [], freshness: false, sensitive: false, confidence: 1, reason: "" },
      trace: [],
      turns: 1,
      usageByModel: {},
      costUsd: 0,
      costVsFrontierOnlyUsd: 0,
      createdAt: 1_700_000_000_000,
      finishReason: "stop",
      upstreamRaw: upstream,
    } satisfies OrchestrationResult;

    const res = toOpenAIResponse(result, "anthropic/claude-opus-4.8") as Record<string, any>;
    expect(res.system_fingerprint).toBe("fp_123");
    expect(res.choices[0].native_finish_reason).toBe("end_turn");
    expect(res.usage.cost).toBe(0.0004);
    expect(res.usage.cost_details).toEqual({ upstream: 0.0004 });
    expect(res.openrouter_metadata).toEqual({ provider: "anthropic" });
    expect(res.maestro).toBeDefined(); // added without erasing upstream
  });
});
