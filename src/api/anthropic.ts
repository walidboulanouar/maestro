/**
 * Anthropic Messages API compatibility (`POST /v1/messages`).
 *
 * Lets Claude Code (and any Anthropic-SDK client) use Maestro: point
 * ANTHROPIC_BASE_URL at the server and every request is routed across the pool.
 * We translate Anthropic <-> OpenAI (including tools / tool_use / tool_result)
 * and reuse the orchestrator. Tool execution stays with the caller.
 */
import type { ChatCompletionRequest, ChatMessage, OrchestrationResult } from "../types.js";

type Block =
  | string
  | {
      type?: string;
      text?: string;
      // tool_use (assistant) / tool_result (user)
      id?: string;
      name?: string;
      input?: unknown;
      tool_use_id?: string;
      content?: unknown;
    };

interface AnthMessage {
  role: "user" | "assistant";
  content: Block | Block[];
}

export interface AnthropicRequest {
  model?: string;
  system?: Block | Block[];
  messages?: AnthMessage[];
  tools?: Array<{ name: string; description?: string; input_schema?: unknown }>;
  tool_choice?: { type?: string; name?: string };
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  maestro?: ChatCompletionRequest["maestro"];
}

function asArray(content: Block | Block[] | undefined): Block[] {
  if (content === undefined) return [];
  return Array.isArray(content) ? content : [content];
}

function textOf(content: Block | Block[] | undefined): string {
  return asArray(content)
    .map((b) => (typeof b === "string" ? b : b.type === "text" ? (b.text ?? "") : ""))
    .filter(Boolean)
    .join("\n");
}

function anthToolsToOpenAI(tools: AnthropicRequest["tools"]): unknown[] | undefined {
  if (!tools?.length) return undefined;
  return tools.map((t) => ({
    type: "function",
    function: { name: t.name, description: t.description, parameters: t.input_schema ?? { type: "object" } },
  }));
}

function anthToolChoiceToOpenAI(tc: AnthropicRequest["tool_choice"]): unknown | undefined {
  if (!tc) return undefined;
  if (tc.type === "any") return "required";
  if (tc.type === "tool" && tc.name) return { type: "function", function: { name: tc.name } };
  return "auto";
}

/** Convert an Anthropic request into Maestro's internal (OpenAI-shaped) request. */
export function anthropicToChat(req: AnthropicRequest): ChatCompletionRequest {
  const messages: ChatMessage[] = [];
  const sys = textOf(req.system);
  if (sys) messages.push({ role: "system", content: sys });

  for (const m of req.messages ?? []) {
    const blocks = asArray(m.content);
    if (m.role === "assistant") {
      const text = textOf(m.content);
      const toolUses = blocks.filter((b) => typeof b !== "string" && b.type === "tool_use") as Exclude<Block, string>[];
      const msg: ChatMessage = { role: "assistant", content: text || null };
      if (toolUses.length) {
        msg.tool_calls = toolUses.map((b) => ({
          id: b.id,
          type: "function",
          function: { name: b.name, arguments: JSON.stringify(b.input ?? {}) },
        }));
      }
      messages.push(msg);
    } else {
      // user: tool_result blocks become OpenAI tool-role messages; text stays a user message
      const toolResults = blocks.filter((b) => typeof b !== "string" && b.type === "tool_result") as Exclude<Block, string>[];
      for (const b of toolResults) {
        const c = b.content;
        const content = typeof c === "string" ? c : textOf(c as Block[]) || JSON.stringify(c ?? "");
        messages.push({ role: "tool", tool_call_id: b.tool_use_id ?? b.id ?? "", content });
      }
      const text = textOf(m.content);
      if (text) messages.push({ role: "user", content: text });
    }
  }

  const model = typeof req.model === "string" && req.model.startsWith("maestro") ? req.model : "maestro-auto";
  const out: ChatCompletionRequest = { model, messages };
  if (req.max_tokens !== undefined) out.max_tokens = req.max_tokens;
  if (req.temperature !== undefined) out.temperature = req.temperature;
  if (req.stream !== undefined) out.stream = req.stream;
  if (req.maestro !== undefined) out.maestro = req.maestro;
  const tools = anthToolsToOpenAI(req.tools);
  if (tools) out.tools = tools;
  const tc = anthToolChoiceToOpenAI(req.tool_choice);
  if (tc !== undefined) out.tool_choice = tc;
  return out;
}

function totals(result: OrchestrationResult): { input_tokens: number; output_tokens: number } {
  let input = 0;
  let output = 0;
  for (const u of Object.values(result.usageByModel)) {
    input += u.in;
    output += u.out;
  }
  return { input_tokens: input, output_tokens: output };
}

/** OpenAI tool_calls -> Anthropic tool_use content blocks. */
function toolUseBlocks(result: OrchestrationResult): unknown[] {
  return (result.toolCalls ?? []).map((tc) => {
    const call = tc as { id?: string; function?: { name?: string; arguments?: string } };
    let input: unknown = {};
    try {
      input = JSON.parse(call.function?.arguments ?? "{}");
    } catch {
      input = { _raw: call.function?.arguments };
    }
    return { type: "tool_use", id: call.id, name: call.function?.name, input };
  });
}

function hasTools(result: OrchestrationResult): boolean {
  return Boolean(result.toolCalls?.length);
}

export function toAnthropicResponse(result: OrchestrationResult, model: string) {
  const content: unknown[] = [];
  if (result.answer) content.push({ type: "text", text: result.answer });
  if (hasTools(result)) content.push(...toolUseBlocks(result));
  if (content.length === 0) content.push({ type: "text", text: "" });
  return {
    id: result.id.replace("chatcmpl-", "msg_"),
    type: "message",
    role: "assistant",
    model,
    content,
    stop_reason: hasTools(result) ? "tool_use" : "end_turn",
    stop_sequence: null,
    usage: totals(result),
  };
}

export function anthropicStreamStart(result: OrchestrationResult, model: string) {
  return {
    type: "message_start",
    message: {
      id: result.id.replace("chatcmpl-", "msg_"),
      type: "message",
      role: "assistant",
      model,
      content: [],
      stop_reason: null,
      stop_sequence: null,
      usage: { input_tokens: totals(result).input_tokens, output_tokens: 0 },
    },
  };
}

export function anthropicMessageDelta(result: OrchestrationResult) {
  return {
    type: "message_delta",
    delta: { stop_reason: hasTools(result) ? "tool_use" : "end_turn", stop_sequence: null },
    usage: { output_tokens: totals(result).output_tokens },
  };
}
