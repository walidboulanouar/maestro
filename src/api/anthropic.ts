/**
 * Anthropic Messages API compatibility (`POST /v1/messages`).
 *
 * This is what lets **Claude Code** (and any Anthropic-SDK client) use Maestro:
 * point ANTHROPIC_BASE_URL at the server and every request gets routed across
 * your model pool. We translate Anthropic ⇄ internal, then reuse the orchestrator.
 */
import type { ChatCompletionRequest, ChatMessage, OrchestrationResult } from "../types.js";

type Block = string | { type?: string; text?: string };

export interface AnthropicRequest {
  model?: string;
  system?: Block | Block[];
  messages?: { role: "user" | "assistant"; content: Block | Block[] }[];
  max_tokens?: number;
  temperature?: number;
  stream?: boolean;
  maestro?: ChatCompletionRequest["maestro"];
}

function blocksToText(content: Block | Block[] | undefined): string {
  if (content === undefined) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map((b) => (typeof b === "string" ? b : (b.text ?? ""))).join("\n");
  }
  return content.text ?? "";
}

/** Convert an Anthropic request into Maestro's internal chat request. */
export function anthropicToChat(req: AnthropicRequest): ChatCompletionRequest {
  const messages: ChatMessage[] = [];
  const sys = blocksToText(req.system);
  if (sys) messages.push({ role: "system", content: sys });
  for (const m of req.messages ?? []) {
    messages.push({ role: m.role, content: blocksToText(m.content) });
  }
  // A concrete Anthropic model id (e.g. "claude-...") → route via maestro-auto.
  const model = typeof req.model === "string" && req.model.startsWith("maestro") ? req.model : "maestro-auto";
  const out: ChatCompletionRequest = { model, messages };
  if (req.max_tokens !== undefined) out.max_tokens = req.max_tokens;
  if (req.temperature !== undefined) out.temperature = req.temperature;
  if (req.stream !== undefined) out.stream = req.stream;
  if (req.maestro !== undefined) out.maestro = req.maestro;
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

export function toAnthropicResponse(result: OrchestrationResult, model: string) {
  return {
    id: result.id.replace("chatcmpl-", "msg_"),
    type: "message",
    role: "assistant",
    model,
    content: [{ type: "text", text: result.answer }],
    stop_reason: "end_turn",
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
    delta: { stop_reason: "end_turn", stop_sequence: null },
    usage: { output_tokens: totals(result).output_tokens },
  };
}
