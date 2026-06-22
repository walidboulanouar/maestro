/**
 * Cost accounting. Prices are USD per 1M tokens (see registry).
 */
import type { ModelSpec, TokenUsage } from "../types.js";

export function costOf(model: ModelSpec, usage: TokenUsage): number {
  return (usage.in / 1_000_000) * model.price.in + (usage.out / 1_000_000) * model.price.out;
}

/** Round to a sane number of significant digits for display. */
export function roundUsd(n: number): number {
  if (n === 0) return 0;
  return Number(n.toPrecision(3));
}
