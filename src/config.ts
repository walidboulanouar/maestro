/**
 * Runtime configuration, resolved from environment variables with safe defaults.
 *
 * Maestro runs with ZERO configuration: if no provider keys are present it falls
 * back to the built-in `mock` provider, so `maestro serve` + curl works offline.
 */
import type { Mode } from "./types.js";

export interface MaestroConfig {
  port: number;
  host: string;
  defaultMode: Mode;
  maxTurns: number;
  /** Difficulty thresholds for tier selection. */
  thresholds: { low: number; high: number };
  verifyByDefault: boolean;
  /** Model used for the optional LLM classifier / verifier (a registry slot or id). */
  verifierModel: string;
  providers: {
    openrouter: { apiKey?: string; baseUrl: string };
    vercelGateway: { apiKey?: string; baseUrl: string };
    localOpenai: { apiKey?: string; baseUrl?: string };
  };
  /** Append every request trace to this JSONL file (optional). */
  traceFile?: string;
  /** Force the mock provider even if keys exist (handy for demos/tests). */
  forceMock: boolean;
}

function num(value: string | undefined, fallback: number): number {
  if (value === undefined) return fallback;
  const n = Number(value);
  return Number.isFinite(n) ? n : fallback;
}

function bool(value: string | undefined, fallback: boolean): boolean {
  if (value === undefined) return fallback;
  return ["1", "true", "yes", "on"].includes(value.toLowerCase());
}

export function loadConfig(env: NodeJS.ProcessEnv = process.env): MaestroConfig {
  return {
    port: num(env.MAESTRO_PORT ?? env.PORT, 8080),
    host: env.MAESTRO_HOST ?? "0.0.0.0",
    defaultMode: (env.MAESTRO_DEFAULT_MODE as Mode) ?? "fugu",
    maxTurns: num(env.MAESTRO_MAX_TURNS, 3),
    thresholds: {
      low: num(env.MAESTRO_DIFFICULTY_LOW, 0.33),
      high: num(env.MAESTRO_DIFFICULTY_HIGH, 0.7),
    },
    verifyByDefault: bool(env.MAESTRO_VERIFY, true),
    verifierModel: env.MAESTRO_VERIFIER_MODEL ?? "auto",
    providers: {
      openrouter: {
        apiKey: env.OPENROUTER_API_KEY,
        baseUrl: env.OPENROUTER_BASE_URL ?? "https://openrouter.ai/api/v1",
      },
      vercelGateway: {
        apiKey: env.AI_GATEWAY_API_KEY,
        baseUrl: env.AI_GATEWAY_BASE_URL ?? "https://ai-gateway.vercel.sh/v1",
      },
      localOpenai: {
        apiKey: env.LOCAL_OPENAI_API_KEY,
        baseUrl: env.LOCAL_OPENAI_BASE_URL,
      },
    },
    traceFile: env.MAESTRO_TRACE_FILE,
    forceMock: bool(env.MAESTRO_FORCE_MOCK, false),
  };
}
