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
  /** Per-request timeout in ms (default 120s) so a hung provider can't hang us. */
  timeoutMs?: number;
  /** Retries on 429/5xx/timeout (default 2). */
  maxRetries?: number;
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
    // Start from ALL forwarded request fields, then override only what Maestro
    // owns. This keeps response_format, provider, seed, tools, tool_choice,
    // session_id, metadata, trace, plugins, reasoning, etc. fully intact.
    const body: Record<string, unknown> = { ...(params.extra ?? {}) };
    body["model"] = params.model;
    body["messages"] = params.messages;
    body["stream"] = stream;
    // Only inject effort for routed modes, and never clobber a caller's reasoning.
    if (params.effort && body["reasoning_effort"] === undefined && body["reasoning"] === undefined) {
      body["reasoning_effort"] = params.effort;
    }
    if (stream && body["stream_options"] === undefined) {
      body["stream_options"] = { include_usage: true };
    }
    return JSON.stringify(body);
  }

  private url(): string {
    return `${this.opts.baseUrl!.replace(/\/$/, "")}/chat/completions`;
  }

  /** Caller's signal combined with a timeout, so requests never hang forever. */
  private signalFor(params: ChatParams): AbortSignal {
    const timeout = AbortSignal.timeout(this.opts.timeoutMs ?? 120_000);
    return params.signal ? AbortSignal.any([params.signal, timeout]) : timeout;
  }

  private isTimeout(err: unknown): boolean {
    return err instanceof Error && (err.name === "TimeoutError" || err.name === "AbortError");
  }

  /** POST with retries on timeout / 429 / 5xx (exponential backoff). Returns an OK Response. */
  private async fetchWithRetry(params: ChatParams): Promise<Response> {
    const retries = this.opts.maxRetries ?? 2;
    let lastErr = "";
    for (let attempt = 0; attempt <= retries; attempt++) {
      if (attempt > 0) await new Promise((r) => setTimeout(r, 250 * 2 ** (attempt - 1)));
      let res: Response;
      try {
        res = await fetch(this.url(), {
          method: "POST",
          headers: this.headers(),
          body: this.body(params, false),
          signal: this.signalFor(params),
        });
      } catch (err) {
        if (this.isTimeout(err)) {
          lastErr = `${this.name} request timed out after ${this.opts.timeoutMs ?? 120_000}ms`;
          continue; // retry timeouts
        }
        throw err;
      }
      if (res.ok) return res;
      const detail = await res.text().catch(() => "");
      lastErr = `${this.name} ${res.status}: ${detail.slice(0, 300)}`;
      // retry only transient statuses
      if (res.status === 429 || res.status >= 500) continue;
      throw new Error(lastErr);
    }
    throw new Error(`${lastErr} (after ${retries} retries)`);
  }

  async chat(params: ChatParams): Promise<ChatResult> {
    const res = await this.fetchWithRetry(params);
    const json = (await res.json()) as {
      choices?: { message?: { content?: string; tool_calls?: unknown[] }; finish_reason?: string }[];
      usage?: { prompt_tokens?: number; completion_tokens?: number };
    };
    const choice = json.choices?.[0];
    const text = choice?.message?.content ?? "";
    const usage: TokenUsage = {
      in: json.usage?.prompt_tokens ?? estimateInput(params),
      out: json.usage?.completion_tokens ?? estimateTokens(text),
    };
    const result: ChatResult = { text, usage, raw: json };
    if (choice?.message?.tool_calls?.length) result.toolCalls = choice.message.tool_calls;
    if (choice?.finish_reason) result.finishReason = choice.finish_reason;
    return result;
  }

  async *stream(params: ChatParams): AsyncIterable<StreamChunk> {
    let res: Response;
    try {
      res = await fetch(this.url(), {
        method: "POST",
        headers: this.headers(),
        body: this.body(params, true),
        signal: this.signalFor(params),
      });
    } catch (err) {
      if (this.isTimeout(err)) {
        throw new Error(`${this.name} request timed out after ${this.opts.timeoutMs ?? 120_000}ms`);
      }
      throw err;
    }
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
