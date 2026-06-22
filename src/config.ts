/**
 * Runtime configuration, resolved from environment variables with safe defaults.
 *
 * Maestro runs with ZERO configuration: if no provider keys are present it falls
 * back to the built-in `mock` provider, so `maestro serve` + curl works offline.
 */
import type { Mode } from "./types.js";

export type Profile = "cheap" | "balanced" | "quality";

export interface ProviderConfig {
  baseUrl?: string;
  apiKey?: string;
  /** Whether a key is required for this provider to count as configured. */
  requireKey: boolean;
  headers?: Record<string, string>;
}

export interface AuthConfig {
  /** If non-empty, requests must send Authorization: Bearer <one of these>. */
  apiKeys: string[];
  /** Per-key requests per minute (0 = unlimited). */
  rateLimitPerMin: number;
  /** Per-key cumulative USD budget for this process (0 = unlimited). */
  budgetUsd: number;
}

export interface MaestroConfig {
  port: number;
  host: string;
  defaultMode: Mode;
  profile: Profile;
  maxTurns: number;
  thresholds: { low: number; high: number };
  verifyByDefault: boolean;
  /** Skip the verify/escalate round-trip when classifier confidence is at least this (0 disables). */
  skipVerifyAboveConfidence: number;
  requestTimeoutMs: number;
  /** Retries on 429/5xx/timeout per upstream call. */
  maxRetries: number;
  verifierModel: string;
  /** Any OpenAI-compatible provider, keyed by name. */
  providers: Record<string, ProviderConfig>;
  traceFile?: string;
  /** Redact tool-call arguments / obvious secrets before storing traces. */
  redactTraces: boolean;
  /** Semantic/dedupe cache for identical-ish requests (off by default). */
  cacheEnabled: boolean;
  forceMock: boolean;
  auth: AuthConfig;
}

const PROFILES: Record<Profile, { low: number; high: number; maxTurns: number; verify: boolean }> = {
  // stay on cheaper tiers longer, fewer escalations
  cheap: { low: 0.45, high: 0.85, maxTurns: 2, verify: true },
  balanced: { low: 0.33, high: 0.7, maxTurns: 3, verify: true },
  // escalate to strong models sooner, more turns
  quality: { low: 0.2, high: 0.45, maxTurns: 4, verify: true },
};

function num(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function bool(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}

function list(value: string | undefined): string[] {
  return (value ?? "")
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);
}

function buildProviders(env: NodeJS.ProcessEnv): Record<string, ProviderConfig> {
  const p: Record<string, ProviderConfig> = {
    openrouter: {
      baseUrl: env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1",
      apiKey: env.OPENROUTER_API_KEY,
      requireKey: true,
      headers: {
        "HTTP-Referer": "https://maestro.ayautomate.com",
        "X-Title": "Maestro",
        "X-OpenRouter-Title": "Maestro",
        "X-OpenRouter-Metadata": "enabled",
      },
    },
    "vercel-gateway": {
      baseUrl: env.AI_GATEWAY_BASE_URL ?? "https://ai-gateway.vercel.sh/v1",
      apiKey: env.AI_GATEWAY_API_KEY,
      requireKey: true,
    },
    openai: { baseUrl: env.OPENAI_BASE_URL ?? "https://api.openai.com/v1", apiKey: env.OPENAI_API_KEY, requireKey: true },
    groq: { baseUrl: "https://api.groq.com/openai/v1", apiKey: env.GROQ_API_KEY, requireKey: true },
    together: { baseUrl: "https://api.together.xyz/v1", apiKey: env.TOGETHER_API_KEY, requireKey: true },
    fireworks: { baseUrl: "https://api.fireworks.ai/inference/v1", apiKey: env.FIREWORKS_API_KEY, requireKey: true },
    deepinfra: { baseUrl: "https://api.deepinfra.com/v1/openai", apiKey: env.DEEPINFRA_API_KEY, requireKey: true },
    "local-openai": { baseUrl: env.LOCAL_OPENAI_BASE_URL, apiKey: env.LOCAL_OPENAI_API_KEY, requireKey: false },
  };
  // Custom providers: MAESTRO_PROVIDERS=[{"name","baseUrl","apiKeyEnv"|"apiKey","requireKey"?}]
  if (env.MAESTRO_PROVIDERS) {
    try {
      const custom = JSON.parse(env.MAESTRO_PROVIDERS) as Array<{
        name: string;
        baseUrl: string;
        apiKey?: string;
        apiKeyEnv?: string;
        requireKey?: boolean;
        headers?: Record<string, string>;
      }>;
      for (const c of custom) {
        p[c.name] = {
          baseUrl: c.baseUrl,
          apiKey: c.apiKeyEnv ? env[c.apiKeyEnv] : c.apiKey,
          requireKey: c.requireKey ?? true,
          headers: c.headers,
        };
      }
    } catch {
      /* ignore malformed MAESTRO_PROVIDERS */
    }
  }
  return p;
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): MaestroConfig {
  const profile = (["cheap", "balanced", "quality"].includes(env.MAESTRO_PROFILE ?? "")
    ? env.MAESTRO_PROFILE
    : "balanced") as Profile;
  const preset = PROFILES[profile];

  return {
    port: num(env.MAESTRO_PORT ?? env.PORT, 8080),
    host: env.MAESTRO_HOST ?? "0.0.0.0",
    defaultMode: (env.MAESTRO_DEFAULT_MODE as Mode) ?? "fugu",
    profile,
    maxTurns: num(env.MAESTRO_MAX_TURNS, preset.maxTurns),
    thresholds: {
      low: num(env.MAESTRO_DIFFICULTY_LOW, preset.low),
      high: num(env.MAESTRO_DIFFICULTY_HIGH, preset.high),
    },
    verifyByDefault: bool(env.MAESTRO_VERIFY, preset.verify),
    skipVerifyAboveConfidence: num(env.MAESTRO_SKIP_VERIFY_ABOVE_CONFIDENCE, 0),
    requestTimeoutMs: num(env.MAESTRO_REQUEST_TIMEOUT_MS, 120_000),
    maxRetries: num(env.MAESTRO_MAX_RETRIES, 2),
    verifierModel: env.MAESTRO_VERIFIER_MODEL ?? "auto",
    providers: buildProviders(env),
    traceFile: env.MAESTRO_TRACE_FILE,
    redactTraces: bool(env.MAESTRO_REDACT_TRACES, true),
    cacheEnabled: bool(env.MAESTRO_CACHE, false),
    forceMock: bool(env.MAESTRO_FORCE_MOCK, false),
    auth: {
      apiKeys: list(env.MAESTRO_API_KEYS),
      rateLimitPerMin: num(env.MAESTRO_RATE_LIMIT_PER_MIN, 0),
      budgetUsd: num(env.MAESTRO_BUDGET_USD, 0),
    },
  };
}
