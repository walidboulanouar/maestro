/**
 * Deterministic mock provider — lets Maestro run, demo, test and benchmark with
 * ZERO API keys and ZERO cost. No randomness (so results are reproducible).
 *
 * It returns plausible, model-flavoured answers. Answer *quality* is not encoded
 * here; the verifier judges mock answers from the answering model's `strength`
 * vs the task difficulty (see `core/verify.ts`), which makes the cheap→frontier
 * escalation deterministic and visible in the demo.
 */
import { contentToText } from "../core/transcript.js";
import type {
  ChatParams,
  ChatResult,
  ProviderAdapter,
  StreamChunk,
  TokenUsage,
} from "../types.js";

function lastUser(params: ChatParams): string {
  for (let i = params.messages.length - 1; i >= 0; i--) {
    const m = params.messages[i];
    if (m && m.role === "user") return contentToText(m.content);
  }
  return contentToText(params.messages[params.messages.length - 1]?.content);
}

function tierLabel(model: string): string {
  if (/frontier|opus|gpt-5|gemini-3|grok/i.test(model)) return "a frontier model";
  if (/mid|deepseek|kimi/i.test(model)) return "a mid-tier model";
  return "a fast model";
}

export function mockAnswer(params: ChatParams): string {
  const q = lastUser(params).replace(/\s+/g, " ").trim().slice(0, 160);
  const flavour = tierLabel(params.model);
  const effort = params.effort ? ` (effort: ${params.effort})` : "";
  return (
    `**[mock answer from \`${params.model}\` — ${flavour}${effort}]**\n\n` +
    `You asked: "${q}".\n\n` +
    `This is a deterministic mock response so Maestro runs without API keys. ` +
    `Set a provider key (e.g. OPENROUTER_API_KEY) to route to real models.`
  );
}

function usageFor(params: ChatParams, text: string): TokenUsage {
  const inTokens = Math.max(
    1,
    Math.ceil(params.messages.map((m) => m.content).join("\n").length / 4),
  );
  return { in: inTokens, out: Math.max(1, Math.ceil(text.length / 4)) };
}

export class MockAdapter implements ProviderAdapter {
  readonly name = "mock" as const;

  isConfigured(): boolean {
    return true;
  }

  async chat(params: ChatParams): Promise<ChatResult> {
    const text = mockAnswer(params);
    return { text, usage: usageFor(params, text) };
  }

  async *stream(params: ChatParams): AsyncIterable<StreamChunk> {
    const text = mockAnswer(params);
    // Stream word-by-word for a realistic feel.
    const words = text.split(/(\s+)/);
    for (const w of words) {
      if (w) yield { delta: w, done: false };
    }
    yield { delta: "", done: true, usage: usageFor(params, text) };
  }
}
