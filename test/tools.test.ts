import { describe, expect, it } from "vitest";
import { loadConfig } from "../src/config.js";
import { orchestrate } from "../src/core/orchestrator.js";
import { ModelRegistry } from "../src/registry/registry.js";
import { toOpenAIResponse } from "../src/api/shape.js";
import { ChatCompletionRequestSchema } from "../src/types.js";
import type { ChatParams, ChatResult, OrchestrationResult, ProviderAdapter, StreamChunk } from "../src/types.js";
import type { ProviderSet } from "../src/providers/index.js";

/** Adapter that records the params it received and can return tool_calls. */
class CaptureAdapter implements ProviderAdapter {
  readonly name = "mock";
  last?: ChatParams;
  toolCallsToReturn?: unknown[];
  isConfigured() {
    return true;
  }
  async chat(params: ChatParams): Promise<ChatResult> {
    this.last = params;
    if (this.toolCallsToReturn) {
      return { text: "", usage: { in: 5, out: 5 }, toolCalls: this.toolCallsToReturn, finishReason: "tool_calls" };
    }
    return { text: "hello", usage: { in: 5, out: 5 }, finishReason: "stop" };
  }
  async *stream(): AsyncIterable<StreamChunk> {
    yield { delta: "hello", done: false };
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

const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_weather",
      parameters: { type: "object", properties: { city: { type: "string" } } },
    },
  },
];

describe("tool-calling: schema accepts real agent-loop messages", () => {
  it("accepts an assistant message with content:null + tool_calls", () => {
    const parsed = ChatCompletionRequestSchema.safeParse({
      model: "maestro-auto",
      messages: [
        { role: "user", content: "weather in Paris?" },
        { role: "assistant", content: null, tool_calls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }] },
      ],
    });
    expect(parsed.success).toBe(true);
  });

  it("accepts a tool result message with tool_call_id", () => {
    const parsed = ChatCompletionRequestSchema.safeParse({
      model: "maestro-auto",
      messages: [
        { role: "user", content: "weather?" },
        { role: "tool", tool_call_id: "call_1", content: "18°C, sunny" },
      ],
    });
    expect(parsed.success).toBe(true);
  });

  it("accepts multimodal content arrays", () => {
    const parsed = ChatCompletionRequestSchema.safeParse({
      model: "maestro-auto",
      messages: [{ role: "user", content: [{ type: "text", text: "hi" }] }],
    });
    expect(parsed.success).toBe(true);
  });
});

describe("tool-calling: transparent pass-through", () => {
  it("forwards tools and tool_choice to the provider unchanged (via extra)", async () => {
    const cap = new CaptureAdapter();
    await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: "weather?" }], tools: TOOLS, tool_choice: "auto" },
      depsWith(cap),
    );
    expect(cap.last?.extra?.tools).toEqual(TOOLS);
    expect(cap.last?.extra?.tool_choice).toBe("auto");
  });

  it("returns tool_calls and does NOT verify/escalate (single hand-over)", async () => {
    const cap = new CaptureAdapter();
    cap.toolCallsToReturn = [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: "{}" } }];
    const result = await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: "weather?" }], tools: TOOLS },
      depsWith(cap),
    );
    expect(result.turns).toBe(1);
    expect(result.toolCalls).toEqual(cap.toolCallsToReturn);
    expect(result.finishReason).toBe("tool_calls");
    expect(result.trace[0]!.verdict).toBeUndefined(); // verifier never ran
  });

  it("shapes tool_calls into an OpenAI response (content null + finish_reason)", () => {
    const result = {
      id: "chatcmpl-x",
      answer: "",
      mode: "auto",
      signature: { task: "general", difficulty: 0, caps: [], freshness: false, sensitive: false, confidence: 1, reason: "" },
      trace: [],
      turns: 1,
      usageByModel: { "m": { in: 5, out: 5 } },
      costUsd: 0,
      costVsFrontierOnlyUsd: 0,
      createdAt: 1_700_000_000_000,
      toolCalls: [{ id: "call_1", type: "function", function: { name: "get_weather", arguments: "{}" } }],
      finishReason: "tool_calls",
    } satisfies OrchestrationResult;
    const res = toOpenAIResponse(result, "maestro-auto") as {
      choices: { message: { content: string | null; tool_calls?: unknown[] }; finish_reason: string }[];
    };
    expect(res.choices[0]!.finish_reason).toBe("tool_calls");
    expect(res.choices[0]!.message.content).toBeNull();
    expect(res.choices[0]!.message.tool_calls).toEqual(result.toolCalls);
  });
});
