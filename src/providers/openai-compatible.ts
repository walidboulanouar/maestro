/**
 * A single adapter that speaks the OpenAI Chat Completions wire format.
 *
 * OpenRouter, the Vercel AI Gateway, Ollama, vLLM, llama.cpp and LM Studio are
 * all OpenAI-compatible — so they differ only by `baseUrl` + key. One adapter,
 * many backends.
 */
import type {
  ChatParams,
  ChatResult,
  ProviderAdapter,
  ProviderName,
  StreamChunk,
  TokenUsage,
} from "../types.js";

export interface OpenAICompatibleOptions {
  baseUrl?: string;
  apiKey?: string;
  /** Whether a key is required for this backend to be considered configured. */
  requireKey: boolean;
  /** Extra headers (e.g. OpenRouter ranking headers). */
  headers?: Record<string, string>;
}

/** Rough token estimate when a backend omits usage (≈4 chars/token). */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

export class OpenAICompatibleAdapter implements ProviderAdapter {
  constructor(
    readonly name: ProviderName,
    private readonly opts: OpenAICompatibleOptions,
  ) {}

  isConfigured(): boolean {
    if (!this.opts.baseUrl) return false;
    return this.opts.requireKey ? Boolean(this.opts.apiKey) : true;
  }

  private headers(): Record<string, string> {
    const h: Record<string, string> = {
      "content-type": "application/json",
      ...this.opts.headers,
    };
    if (this.opts.apiKey) h["authorization"] = `Bearer ${this.opts.apiKey}`;
    return h;
  }

  private body(params: ChatParams, stream: boolean): string {
    const body: Record<string, unknown> = {
      model: params.model,
      messages: params.messages,
      stream,
    };
    if (params.temperature !== undefined) body["temperature"] = params.temperature;
    if (params.maxTokens !== undefined) body["max_tokens"] = params.maxTokens;
    if (params.effort) body["reasoning_effort"] = params.effort;
    if (stream) body["stream_options"] = { include_usage: true };
    return JSON.stringify(body);
  }

  private url(): string {
    return `${this.opts.baseUrl!.replace(/\/$/, "")}/chat/completions`;
  }

  async chat(params: ChatParams): Promise<ChatResult> {
    const res = await fetch(this.url(), {
      method: "POST",
      headers: this.headers(),
      body: this.body(params, false),
      signal: params.signal ?? null,
    });
    if (!res.ok) {
      const detail = await res.text().catch(() => "");
      throw new Error(`${this.name} ${res.status}: ${detail.slice(0, 500)}`);
    }
    const json = (await res.json()) as {
      choices?: { message?: { content?: string } }[];
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    const text = json.choices?.[0]?.message?.content ?? "";
    const usage: TokenUsage = {
      in: json.usage?.prompt_tokens ?? estimateInput(params),
      out: json.usage?.completion_tokens ?? estimateTokens(text),
    };
    return { text, usage };
  }

  async *stream(params: ChatParams): AsyncIterable<StreamChunk> {
    const res = await fetch(this.url(), {
      method: "POST",
      headers: this.headers(),
      body: this.body(params, true),
      signal: params.signal ?? null,
    });
    if (!res.ok || !res.body) {
      const detail = res.ok ? "no body" : await res.text().catch(() => "");
      throw new Error(`${this.name} ${res.status}: ${detail.slice(0, 500)}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let outText = "";
    let usage: TokenUsage | undefined;

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";
      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed.startsWith("data:")) continue;
        const payload = trimmed.slice(5).trim();
        if (payload === "[DONE]") continue;
        try {
          const chunk = JSON.parse(payload) as {
            choices?: { delta?: { content?: string } }[];
            usage?: { prompt_tokens?: number; completion_tokens?: number };
          };
          const delta = chunk.choices?.[0]?.delta?.content ?? "";
          if (chunk.usage) {
            usage = {
              in: chunk.usage.prompt_tokens ?? estimateInput(params),
              out: chunk.usage.completion_tokens ?? estimateTokens(outText),
            };
          }
          if (delta) {
            outText += delta;
            yield { delta, done: false };
          }
        } catch {
          /* ignore malformed keep-alive lines */
        }
      }
    }
    yield {
      delta: "",
      done: true,
      usage: usage ?? { in: estimateInput(params), out: estimateTokens(outText) },
    };
  }
}

function estimateInput(params: ChatParams): number {
  return estimateTokens(params.messages.map((m) => m.content).join("\n"));
}
