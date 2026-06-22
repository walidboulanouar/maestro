/**
 * Default model registry — a DATED snapshot. Run `maestro registry check` to see
 * how stale it is, or point `MAESTRO_REGISTRY` at your own JSON to override.
 *
 * Slot labels are abstract and remappable (OpenFugu's key lesson): the router
 * reasons over slots/tiers; the `id`/`provider` map them to a concrete backend.
 *
 * Every real model id + price lives in ONE place — the OPENROUTER dictionary
 * below — so you can read and swap models at a glance. Ids and prices were
 * verified against the live OpenRouter catalog on 2026-06-22.
 *
 * The three `mock` models are always present so Maestro runs with zero API keys.
 */
import type { ModelSpec, Registry } from "../types.js";

const PRICES_CHECKED = "2026-06-22";

/**
 * Dictionary of real models. `in`/`out` are USD per 1M tokens; `ctx` is the
 * context window. To swap a model, change it here (or override via your own
 * registry JSON) — nothing else needs to change.
 */
export const OPENROUTER = {
  // cheap tier — fast, inexpensive open models
  GLM_4_7_FLASH: { id: "z-ai/glm-4.7-flash", in: 0.06, out: 0.4, ctx: 202_752 },
  QWEN_3_5_FLASH: { id: "qwen/qwen3.5-flash-02-23", in: 0.07, out: 0.26, ctx: 1_000_000 },
  // mid tier — strong, good-value open models
  KIMI_K2_7_CODE: { id: "moonshotai/kimi-k2.7-code", in: 0.61, out: 3.07, ctx: 262_144 },
  DEEPSEEK_V4_PRO: { id: "deepseek/deepseek-v4-pro", in: 0.43, out: 0.87, ctx: 163_840 },
  // frontier tier — top closed models
  CLAUDE_OPUS_4_8: { id: "anthropic/claude-opus-4.8", in: 5.0, out: 25.0, ctx: 1_000_000 },
  GPT_5_5: { id: "openai/gpt-5.5", in: 5.0, out: 30.0, ctx: 1_050_000 },
  GEMINI_3_1_PRO: { id: "google/gemini-3.1-pro-preview", in: 2.0, out: 12.0, ctx: 1_048_576 },
} as const;

type ModelEntry = { id: string; in: number; out: number; ctx: number };

/** Build a ModelSpec from a dictionary entry + its routing attributes. */
function spec(
  m: ModelEntry,
  rest: Omit<ModelSpec, "id" | "provider" | "price" | "contextWindow">,
): ModelSpec {
  return {
    id: m.id,
    provider: "openrouter",
    price: { in: m.in, out: m.out, updated: PRICES_CHECKED },
    contextWindow: m.ctx,
    ...rest,
  };
}

export const DEFAULT_REGISTRY: Registry = {
  updated: PRICES_CHECKED,
  models: [
    /* ---- mock tier: always available, no keys, deterministic ---- */
    {
      slot: "mock-cheap",
      id: "mock-cheap",
      provider: "mock",
      tier: "cheap",
      strength: 55,
      caps: ["code", "math", "reasoning", "translation", "factual"],
      price: { in: 0, out: 0, updated: PRICES_CHECKED },
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
      price: { in: 0, out: 0, updated: PRICES_CHECKED },
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
      price: { in: 0, out: 0, updated: PRICES_CHECKED },
      contextWindow: 1_000_000,
      tags: { "strong:code": 1, "strong:reasoning": 1, "strong:math": 1 },
    },

    /* ---- cheap tier (open models via OpenRouter) ---- */
    spec(OPENROUTER.GLM_4_7_FLASH, {
      slot: "cheap-generalist",
      tier: "cheap",
      strength: 70,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context"],
      tags: { "positive:general": 1, "positive:code": 1 },
    }),
    spec(OPENROUTER.QWEN_3_5_FLASH, {
      slot: "cheap-fast",
      tier: "cheap",
      strength: 68,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context"],
      tags: { "positive:translation": 1, "positive:general": 1 },
    }),

    /* ---- mid tier (strong open models) ---- */
    spec(OPENROUTER.KIMI_K2_7_CODE, {
      slot: "mid-coder",
      tier: "mid",
      strength: 84,
      caps: ["code", "reasoning", "long-context"],
      tags: { "strong:code": 1, "positive:reasoning": 1 },
    }),
    spec(OPENROUTER.DEEPSEEK_V4_PRO, {
      slot: "mid-reasoner",
      tier: "mid",
      strength: 87,
      caps: ["code", "math", "reasoning", "long-context", "factual"],
      tags: { "strong:reasoning": 1, "strong:math": 1, "positive:code": 1 },
    }),

    /* ---- frontier tier (closed) ---- */
    spec(OPENROUTER.CLAUDE_OPUS_4_8, {
      slot: "frontier-coder",
      tier: "frontier",
      strength: 97,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context", "vision"],
      efforts: ["low", "medium", "high"],
      tags: { "strong:code": 1, "strong:reasoning": 1 },
    }),
    spec(OPENROUTER.GPT_5_5, {
      slot: "frontier-generalist",
      tier: "frontier",
      strength: 96,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context", "vision"],
      efforts: ["low", "medium", "high", "xhigh"],
      tags: { "strong:reasoning": 1, "strong:math": 1, "positive:code": 1 },
    }),
    spec(OPENROUTER.GEMINI_3_1_PRO, {
      slot: "frontier-fresh",
      tier: "frontier",
      strength: 95,
      caps: ["code", "math", "reasoning", "translation", "factual", "long-context", "vision", "fresh"],
      tags: { "strong:fresh": 1, "positive:reasoning": 1, "positive:factual": 1 },
    }),
  ],
};
