/**
 * Trace store: keeps the last N orchestration results in memory and (optionally)
 * appends each as a line of JSONL. This is the transparency wedge: every routing
 * decision is inspectable. When `redact` is on, tool-call arguments and obvious
 * secrets are scrubbed and the raw upstream payload is dropped before storing.
 */
import { appendFileSync } from "node:fs";
import type { OrchestrationResult } from "../types.js";

function scrubSecrets(s: string): string {
  return s
    .replace(/sk-[a-zA-Z0-9-]{16,}/g, "[redacted-key]")
    .replace(/AKIA[0-9A-Z]{16}/g, "[redacted-key]")
    .replace(/-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g, "[redacted-key]");
}

function redactResult(r: OrchestrationResult): OrchestrationResult {
  const clone: OrchestrationResult = { ...r };
  // drop the raw upstream payload (large; may contain sensitive content/args)
  delete (clone as { upstreamRaw?: unknown }).upstreamRaw;
  clone.answer = typeof clone.answer === "string" ? scrubSecrets(clone.answer) : clone.answer;
  if (clone.toolCalls) {
    clone.toolCalls = clone.toolCalls.map((tc) => {
      const call = tc as { function?: { arguments?: unknown } };
      if (call && call.function && "arguments" in call.function) {
        return { ...call, function: { ...call.function, arguments: "[redacted]" } };
      }
      return tc;
    });
  }
  return clone;
}

export class TraceStore {
  private readonly ring: OrchestrationResult[] = [];
  constructor(
    private readonly capacity = 200,
    private readonly file?: string,
    private readonly redact = false,
  ) {}

  record(result: OrchestrationResult): void {
    const stored = this.redact ? redactResult(result) : result;
    this.ring.push(stored);
    if (this.ring.length > this.capacity) this.ring.shift();
    if (this.file) {
      try {
        appendFileSync(this.file, JSON.stringify(stored) + "\n");
      } catch {
        /* tracing must never break a request */
      }
    }
  }

  get(id: string): OrchestrationResult | undefined {
    return this.ring.find((r) => r.id === id);
  }

  recent(n = 20): OrchestrationResult[] {
    return this.ring.slice(-n).reverse();
  }
}
