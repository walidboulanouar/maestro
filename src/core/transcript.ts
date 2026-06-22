/**
 * Transcript helpers.
 *
 * NOTE the raw `"role: content\n"` format. OpenFugu's docs show the learned
 * TRINITY router needs this raw form (95% vs 11% accuracy with a chat template).
 * v0 only uses heuristics, but we keep the format so the learned-router seam is
 * clean later.
 */
import type { ChatMessage } from "../types.js";

export function toRawTranscript(messages: ChatMessage[]): string {
  return messages.map((m) => `${m.role}: ${m.content}`).join("\n");
}

export function lastUserMessage(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    const m = messages[i];
    if (m && m.role === "user") return m.content;
  }
  return messages[messages.length - 1]?.content ?? "";
}

export function totalChars(messages: ChatMessage[]): number {
  return messages.reduce((sum, m) => sum + m.content.length, 0);
}
