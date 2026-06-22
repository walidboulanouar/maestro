/**
 * Shared contracts for Maestro.
 *
 * Everything that crosses a module boundary is defined here, validated with zod
 * where it comes from the outside world (HTTP, config, registry files).
 */
import { z } from "zod";

/* ------------------------------------------------------------------ *
 * OpenAI-compatible wire types (the subset we use)
 * ------------------------------------------------------------------ */

export const RoleSchema = z.enum(["system", "user", "assistant", "tool"]);
export type Role = z.infer<typeof RoleSchema>;

/**
 * OpenAI-compatible message. Tool loops require more than `content: string`:
 * assistant turns can have `content: null` + `tool_calls`, tool results carry
 * `tool_call_id`, and content may be a multimodal array. We accept all of these
 * and pass them through to the model unchanged (Maestro is a transparent proxy).
 */
export const ChatMessageSchema = z
  .object({
    role: RoleSchema,
    content: z.union([z.string(), z.null(), z.array(z.any())]).optional(),
    name: z.string().optional(),
    tool_call_id: z.string().optional(),
    tool_calls: z.array(z.any()).optional(),
  })
  .passthrough();
export type ChatMessage = z.infer<typeof ChatMessageSchema>;

/** Maestro's optional routing hint, accepted as an extra field on the request. */
export const MaestroHintSchema = z
  .object({
    /** Soft budget ceiling in USD for this request. */
    budget: z.number().positive().optional(),
    /** Named policy from config (e.g. "eu-only", "no-closed"). */
    policy: z.string().optional(),
    /** Region constraint, e.g. "eu". */
    region: z.string().optional(),
    /** Max orchestration turns (overrides config default). */
    maxTurns: z.number().int().min(1).max(8).optional(),
    /** Disable the verify/escalate loop (single shot). */
    verify: z.boolean().optional(),
    /** Pin to a specific registry model id or slot (bypasses routing). */
    pin: z.string().optional(),
  })
  .strict();
export type MaestroHint = z.infer<typeof MaestroHintSchema>;

export const ChatCompletionRequestSchema = z
  .object({
    model: z.string(),
    messages: z.array(ChatMessageSchema).min(1),
    stream: z.boolean().optional(),
    temperature: z.number().min(0).max(2).optional(),
    max_tokens: z.number().int().positive().optional(),
    tools: z.array(z.any()).optional(),
    tool_choice: z.any().optional(),
    maestro: MaestroHintSchema.optional(),
  })
  .passthrough();
export type ChatCompletionRequest = z.infer<typeof ChatCompletionRequestSchema>;

/* ------------------------------------------------------------------ *
 * Model registry
 * ------------------------------------------------------------------ */

export const TierSchema = z.enum(["cheap", "mid", "frontier"]);
export type Tier = z.infer<typeof TierSchema>;

export const CapabilitySchema = z.enum([
  "code",
  "math",
  "reasoning",
  "translation",
  "factual",
  "long-context",
  "vision",
  "fresh",
]);
export type Capability = z.infer<typeof CapabilitySchema>;

/**
 * Provider name. A free-form string so Maestro works with ANY OpenAI-compatible
 * router/provider. Built-ins: openrouter, vercel-gateway, openai, groq, together,
 * fireworks, deepinfra, local-openai, mock — plus any you define in config
 * (`MAESTRO_PROVIDERS`). The string maps to a base URL + key at runtime.
 */
export const ProviderNameSchema = z.string();
export type ProviderName = string;

export const ModelSpecSchema = z.object({
  /** Abstract, remappable slot label (OpenFugu: slots are metadata). */
  slot: z.string(),
  /** Provider-native model id, e.g. "anthropic/claude-opus-4.8". */
  id: z.string(),
  provider: ProviderNameSchema,
  tier: TierSchema,
  /** 0..100 overall strength (from public leaderboards; keep `updated` honest). */
  strength: z.number().min(0).max(100),
  caps: z.array(CapabilitySchema).default([]),
  efforts: z.array(z.string()).optional(),
  price: z.object({
    /** USD per 1M input tokens. */
    in: z.number().min(0),
    /** USD per 1M output tokens. */
    out: z.number().min(0),
    /** ISO date the price was last checked. */
    updated: z.string(),
  }),
  contextWindow: z.number().int().positive(),
  regions: z.array(z.string()).optional(),
  privacyOk: z.boolean().optional(),
  /** Guardrail scoring tags, e.g. { "strong:code": 1, "positive:math": 1 }. */
  tags: z.record(z.string(), z.number()).optional(),
});
export type ModelSpec = z.infer<typeof ModelSpecSchema>;

export const RegistrySchema = z.object({
  updated: z.string(),
  models: z.array(ModelSpecSchema).min(1),
});
export type Registry = z.infer<typeof RegistrySchema>;

/* ------------------------------------------------------------------ *
 * Routing & orchestration
 * ------------------------------------------------------------------ */

export type Task =
  | "code"
  | "math"
  | "reasoning"
  | "translation"
  | "factual"
  | "general";

export interface TaskSignature {
  task: Task;
  /** 0..1 estimated difficulty. */
  difficulty: number;
  /** Capabilities the route must satisfy. */
  caps: Capability[];
  /** Needs recent/world-knowledge or tools. */
  freshness: boolean;
  /** Contains PII/secrets → respect privacy policy. */
  sensitive: boolean;
  /** 0..1 classifier confidence (feeds calibration metrics). */
  confidence: number;
  /** Human-readable explanation. */
  reason: string;
}

export interface Rung {
  model: ModelSpec;
  effort: string;
}

export type Verdict = "ACCEPT" | "REVISE";

export interface TurnTrace {
  turn: number;
  slot: string;
  model: string;
  provider: ProviderName;
  effort: string;
  role: "Worker";
  verdict?: Verdict;
  verifyReason?: string;
  verifyConfidence?: number;
  usage: TokenUsage;
  costUsd: number;
  ms: number;
}

export interface TokenUsage {
  in: number;
  out: number;
}

export interface RouteDecision {
  signature: TaskSignature;
  ladder: Rung[];
  /** Cost if we used the strongest rung for everything (the comparison baseline). */
  frontierOnlyEstimateUsd: number;
  reason: string;
}

export interface OrchestrationResult {
  id: string;
  answer: string;
  mode: Mode;
  signature: TaskSignature;
  trace: TurnTrace[];
  turns: number;
  usageByModel: Record<string, TokenUsage>;
  costUsd: number;
  costVsFrontierOnlyUsd: number;
  createdAt: number;
  /** Present when the final turn returned tool calls instead of a text answer. */
  toolCalls?: unknown[];
  finishReason: string;
  /** Raw upstream response of the final turn — preserved so provider fields
   * (native_finish_reason, usage.cost, system_fingerprint, openrouter_metadata,
   * per-choice errors, …) aren't erased. Undefined for the mock provider. */
  upstreamRaw?: unknown;
  /** True when this result was served from the dedupe cache (no model call). */
  cached?: boolean;
}

export type Mode = "auto" | "fugu" | "ultra" | "passthrough";

/* ------------------------------------------------------------------ *
 * Provider adapter contract
 * ------------------------------------------------------------------ */

export interface ChatParams {
  model: string;
  messages: ChatMessage[];
  effort?: string;
  signal?: AbortSignal;
  /**
   * Every other OpenAI/OpenRouter request field (response_format, provider,
   * seed, top_p, tools, tool_choice, session_id, metadata, trace, plugins,
   * parallel_tool_calls, reasoning, …) — forwarded to the model VERBATIM.
   * Maestro only overrides `model`/`messages` (and `reasoning_effort` for routed
   * modes); everything else passes through untouched.
   */
  extra?: Record<string, unknown>;
}

export interface ChatResult {
  text: string;
  usage: TokenUsage;
  /** OpenAI tool_calls, when the model decided to call a tool. */
  toolCalls?: unknown[];
  finishReason?: string;
  /** The raw upstream response JSON, so we can preserve provider-specific fields. */
  raw?: unknown;
}

export interface StreamChunk {
  delta: string;
  done: boolean;
  usage?: TokenUsage;
}

export interface ProviderAdapter {
  readonly name: ProviderName;
  /** Whether this adapter is usable (has a key / base url). */
  isConfigured(): boolean;
  chat(params: ChatParams): Promise<ChatResult>;
  stream(params: ChatParams): AsyncIterable<StreamChunk>;
}
