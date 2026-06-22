/**
 * Trace store: keeps the last N orchestration results in memory and (optionally)
 * appends each as a line of JSONL. This is the transparency wedge — every routing
 * decision is inspectable.
 */
import { appendFileSync } from "node:fs";
import type { OrchestrationResult } from "../types.js";

export class TraceStore {
  private readonly ring: OrchestrationResult[] = [];
  constructor(
    private readonly capacity = 200,
    private readonly file?: string,
  ) {}

  record(result: OrchestrationResult): void {
    this.ring.push(result);
    if (this.ring.length > this.capacity) this.ring.shift();
    if (this.file) {
      try {
        appendFileSync(this.file, JSON.stringify(result) + "\n");
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
