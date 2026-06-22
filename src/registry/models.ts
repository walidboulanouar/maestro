/**
 * Default model registry — a DATED snapshot. Numbers are indicative and meant to
 * be overridden by your own `maestro.config.json`. Run `maestro registry check`
 * to see how stale this is.
 *
 * Slot labels are abstract and remappable (OpenFugu's key lesson): the router
 * reasons over slots/tiers; the `id`/`provider` map them to a concrete backend.
 *
 * The three `mock` models are always present so Maestro runs with zero API keys.
 */
import type { Registry } from "../types.js";

export const DEFAULT_REGISTRY: Registry = {
  updated: "2026-06-22",
  models: [
    /* ---- mock tier: always available, no keys, deterministic ---- */
    {
      slot: "mock-cheap",
      id: "mock-cheap",
      provider: "mock",
      tier: "cheap",
      strength: 55,
      caps: ["code", "math", "reasoning", "translation", "factual"],
      price: { in: 0, out: 0, updated: "2026-06-22" },
      contextWindow: 128_000,
      tags: { "positive:general": 1 },
    },
    {
      slot: "mock-mid",
      id: "mock-mid",
      provider: "mock",
      tier: "mid",
      strength: 75,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context"],
      price: { in: 0, out: 0, updated: "2026-06-22" },
      contextWindow: 256_000,
      tags: { "strong:code": 1, "positive:reasoning": 1 },
    },
    {
      slot: "mock-frontier",
      id: "mock-frontier",
      provider: "mock",
      tier: "frontier",
      strength: 95,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context", "vision"],
      price: { in: 0, out: 0, updated: "2026-06-22" },
      contextWindow: 1_000_000,
      tags: { "strong:code": 1, "strong:reasoning": 1, "strong:math": 1 },
    },

    /* ---- cheap tier (open models via OpenRouter) ---- */
    {
      slot: "cheap-generalist",
      id: "z-ai/glm-5.2-air",
      provider: "openrouter",
      tier: "cheap",
      strength: 70,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context"],
      price: { in: 0.2, out: 0.6, updated: "2026-06-22" },
      contextWindow: 1_000_000,
      tags: { "positive:general": 1, "positive:code": 1 },
    },
    {
      slot: "cheap-fast",
      id: "qwen/qwen3.6",
      provider: "openrouter",
      tier: "cheap",
      strength: 68,
      caps: ["code", "math", "reasoning", "translation", "factual"],
      price: { in: 0.15, out: 0.5, updated: "2026-06-22" },
      contextWindow: 256_000,
      tags: { "positive:translation": 1, "positive:general": 1 },
    },

    /* ---- mid tier (strong open models) ---- */
    {
      slot: "mid-coder",
      id: "moonshotai/kimi-k2.7-code",
      provider: "openrouter",
      tier: "mid",
      strength: 84,
      caps: ["code", "reasoning", "long-context"],
      price: { in: 0.6, out: 2.5, updated: "2026-06-22" },
      contextWindow: 256_000,
      tags: { "strong:code": 1, "positive:reasoning": 1 },
    },
    {
      slot: "mid-reasoner",
      id: "deepseek/deepseek-v4",
      provider: "openrouter",
      tier: "mid",
      strength: 86,
      caps: ["code", "math", "reasoning", "long-context", "factual"],
      price: { in: 0.5, out: 2.2, updated: "2026-06-22" },
      contextWindow: 256_000,
      tags: { "strong:reasoning": 1, "strong:math": 1, "positive:code": 1 },
    },

    /* ---- frontier tier (closed) ---- */
    {
      slot: "frontier-coder",
      id: "anthropic/claude-opus-4.8",
      provider: "openrouter",
      tier: "frontier",
      strength: 97,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context", "vision"],
      efforts: ["low", "medium", "high"],
      price: { in: 5, out: 25, updated: "2026-06-22" },
      contextWindow: 1_000_000,
      tags: { "strong:code": 1, "strong:reasoning": 1 },
    },
    {
      slot: "frontier-generalist",
      id: "openai/gpt-5.5",
      provider: "openrouter",
      tier: "frontier",
      strength: 96,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context", "vision"],
      efforts: ["low", "medium", "high", "xhigh"],
      price: { in: 4, out: 20, updated: "2026-06-22" },
      contextWindow: 400_000,
      tags: { "strong:reasoning": 1, "strong:math": 1, "positive:code": 1 },
    },
    {
      slot: "frontier-fresh",
      id: "google/gemini-3-pro",
      provider: "openrouter",
      tier: "frontier",
      strength: 95,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context", "vision", "fresh"],
      price: { in: 3, out: 15, updated: "2026-06-22" },
      contextWindow: 2_000_000,
      tags: { "strong:fresh": 1, "positive:reasoning": 1, "positive:factual": 1 },
    },
  ],
};
