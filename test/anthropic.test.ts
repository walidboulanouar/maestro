import { describe, expect, it } from "vitest";
import { anthropicToChat, toAnthropicResponse } from "../src/api/anthropic.js";
import type { OrchestrationResult } from "../src/types.js";

describe("anthropic <-> openai tool mapping", () => {
  it("converts Anthropic tools + tool_choice to OpenAI form", () => {
    const out = anthropicToChat({
      model: "claude-opus-4.8",
      messages: [{ role: "user", content: "weather?" }],
      tools: [{ name: "get_weather", description: "w", input_schema: { type: "object" } }],
      tool_choice: { type: "any" },
    });
    expect(out.tools).toEqual([
      { type: "function", function: { name: "get_weather", description: "w", parameters: { type: "object" } } },
    ]);
    expect(out.tool_choice).toBe("required");
  });

  it("converts an assistant tool_use block to OpenAI tool_calls", () => {
    const out = anthropicToChat({
      messages: [
        { role: "user", content: "weather in Paris" },
        {
          role: "assistant",
          content: [{ type: "tool_use", id: "tu_1", name: "get_weather", input: { city: "Paris" } }],
        },
      ],
    });
    const asst = out.messages.find((m) => m.role === "assistant")!;
    expect(asst.content).toBeNull();
    expect((asst.tool_calls as { id: string; function: { name: string; arguments: string } }[])[0]).toMatchObject({
      id: "tu_1",
      function: { name: "get_weather", arguments: '{"city":"Paris"}' },
    });
  });

  it("converts a user tool_result block to an OpenAI tool message", () => {
    const out = anthropicToChat({
      messages: [
        { role: "user", content: [{ type: "tool_result", tool_use_id: "tu_1", content: "18C sunny" }] },
      ],
    });
    const toolMsg = out.messages.find((m) => m.role === "tool")!;
    expect(toolMsg.tool_call_id).toBe("tu_1");
    expect(toolMsg.content).toBe("18C sunny");
  });

  it("emits Anthropic tool_use blocks + stop_reason tool_use in the response", () => {
    const result = {
      id: "chatcmpl-1", answer: "", mode: "auto",
      signature: { task: "general", difficulty: 0, caps: [], freshness: false, sensitive: false, confidence: 1, reason: "" },
      trace: [], turns: 1, usageByModel: { m: { in: 1, out: 1 } }, costUsd: 0, costVsFrontierOnlyUsd: 0, createdAt: 1,
      finishReason: "tool_calls",
      toolCalls: [{ id: "c1", type: "function", function: { name: "get_weather", arguments: '{"city":"Paris"}' } }],
    } satisfies OrchestrationResult;
    const res = toAnthropicResponse(result, "claude-opus-4.8") as {
      content: { type: string; name?: string; input?: unknown }[];
      stop_reason: string;
    };
    expect(res.stop_reason).toBe("tool_use");
    const block = res.content.find((b) => b.type === "tool_use")!;
    expect(block.name).toBe("get_weather");
    expect(block.input).toEqual({ city: "Paris" });
  });
});
