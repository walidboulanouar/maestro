/**
 * The orchestration loop — Maestro's brain.
 *
 *   classify → route → execute(worker) → verify → (escalate | accept) → result
 *
 * Honest by design: we don't claim "we never lose to a single model". We try the
 * cheap rung first, verify, and escalate only when needed — and report exactly
 * what happened (route, turns, tokens, cost) on every response.
 */
import { randomUUID } from "node:crypto";
import type { MaestroConfig } from "../config.js";
import type { ProviderSet } from "../providers/index.js";
import type { ModelRegistry } from "../registry/registry.js";
import type {
  ChatCompletionRequest,
  ChatMessage,
  Mode,
  ModelSpec,
  OrchestrationResult,
  Rung,
  TaskSignature,
  TokenUsage,
  TurnTrace,
} from "../types.js";
import { classify } from "./classify.js";
import { costOf } from "./cost.js";
import { route, type RouteContext } from "./route.js";
import { totalChars } from "./transcript.js";
import { chooseVerifier, verify } from "./verify.js";

export interface OrchestratorDeps {
  config: MaestroConfig;
  registry: ModelRegistry;
  providers: ProviderSet;
}

const NOMINAL_OUTPUT_TOKENS = 600;

export function resolveMode(model: string, defaultMode: Mode): { mode: Mode; pin?: string } {
  const m = model.toLowerCase();
  if (m === "maestro-auto" || m === "auto" || m === "maestro") return { mode: "auto" };
  if (m === "maestro-fugu" || m === "fugu") return { mode: "fugu" };
  if (m === "maestro-ultra" || m === "ultra") return { mode: "ultra" };
  // A concrete model id → passthrough to exactly that model.
  return { mode: "passthrough", pin: model };
}

function estInput(messages: ChatMessage[]): number {
  return Math.max(1, Math.ceil(totalChars(messages) / 4));
}

function reviseHint(reason: string): ChatMessage {
  return {
    role: "system",
    content:
      `A previous answer was rejected by the verifier (reason: ${reason}). ` +
      `Produce a more correct, complete answer.`,
  };
}

/** Resolve a passthrough/pinned model id to a usable ModelSpec. */
function resolvePinned(
  id: string,
  registry: ModelRegistry,
  providers: ProviderSet,
): ModelSpec {
  const found = registry.byId(id);
  if (found) return found;
  // Unknown id: guess a provider, price 0 (unknown).
  const configured = providers.configuredNames();
  const provider = configured.has("openrouter")
    ? "openrouter"
    : configured.has("vercel-gateway")
      ? "vercel-gateway"
      : configured.has("local-openai")
        ? "local-openai"
        : "mock";
  return {
    slot: id,
    id: provider === "mock" ? `mock-${id}` : id,
    provider,
    tier: "frontier",
    strength: 90,
    caps: [],
    price: { in: 0, out: 0, updated: "unknown" },
    contextWindow: 128_000,
  };
}

export async function orchestrate(
  req: ChatCompletionRequest,
  deps: OrchestratorDeps,
): Promise<OrchestrationResult> {
  const { config, registry, providers } = deps;
  const { mode: rawMode, pin: modelPin } = resolveMode(req.model, config.defaultMode);
  const mode: Mode = rawMode === "ultra" ? "fugu" : rawMode; // ultra falls back to fugu in v0
  const hint = req.maestro;
  const pin = hint?.pin ?? modelPin;
  const maxTurns = hint?.maxTurns ?? config.maxTurns;
  const verifyEnabled = mode !== "passthrough" && (hint?.verify ?? config.verifyByDefault);

  const configured = providers.configuredNames();
  const pool = registry.available(configured);
  const inTokens = estInput(req.messages);

  // Build the escalation ladder.
  let ladder: Rung[];
  let signature: TaskSignature;
  if (mode === "passthrough" && pin) {
    const spec = resolvePinned(pin, registry, providers);
    ladder = [{ model: spec, effort: "medium" }];
    signature = {
      task: "general",
      difficulty: 0,
      caps: [],
      freshness: false,
      sensitive: false,
      confidence: 1,
      reason: "passthrough (pinned model, no routing)",
    };
  } else if (pin) {
    const spec = resolvePinned(pin, registry, providers);
    ladder = [{ model: spec, effort: "medium" }];
    signature = classify(req.messages);
  } else {
    signature = classify(req.messages);
    const ctx: RouteContext = {
      thresholds: config.thresholds,
      hint,
      estInputTokens: inTokens,
      estOutputTokens: NOMINAL_OUTPUT_TOKENS,
    };
    ladder = route(pool.length ? pool : registry.all(), signature, ctx).ladder;
  }

  const verifierModel = chooseVerifier(
    pool.length ? pool : registry.all(),
    configured,
    config.verifierModel,
  );

  // Run the loop.
  const trace: TurnTrace[] = [];
  const usageByModel: Record<string, TokenUsage> = {};
  let answer = "";
  let costUsd = 0;

  const addUsage = (model: ModelSpec, usage: TokenUsage) => {
    const prev = usageByModel[model.id] ?? { in: 0, out: 0 };
    usageByModel[model.id] = { in: prev.in + usage.in, out: prev.out + usage.out };
    costUsd += costOf(model, usage);
  };

  for (let turn = 1; turn <= maxTurns && turn <= ladder.length; turn++) {
    const rung = ladder[turn - 1]!;
    const adapter = providers.get(rung.model.provider);
    const messages = turn === 1 ? req.messages : [...req.messages, reviseHint(trace[turn - 2]?.verifyReason ?? "")];

    const started = Date.now();
    const result = await adapter.chat({
      model: rung.model.id,
      messages,
      effort: rung.effort,
      temperature: req.temperature ?? 0,
      maxTokens: req.max_tokens,
    });
    const ms = Date.now() - started;
    answer = result.text;
    addUsage(rung.model, result.usage);

    const turnTrace: TurnTrace = {
      turn,
      slot: rung.model.slot,
      model: rung.model.id,
      provider: rung.model.provider,
      effort: rung.effort,
      role: "Worker",
      usage: result.usage,
      costUsd: costOf(rung.model, result.usage),
      ms,
    };

    if (!verifyEnabled) {
      trace.push(turnTrace);
      break;
    }

    const v = await verify(
      lastUserOf(req.messages),
      answer,
      rung,
      signature,
      verifierModel,
      providers,
      config.forceMock,
    );
    if (v.usage.in + v.usage.out > 0) addUsage(verifierModel, v.usage);
    turnTrace.verdict = v.verdict;
    turnTrace.verifyReason = v.reason;
    turnTrace.verifyConfidence = v.confidence;
    trace.push(turnTrace);

    if (v.verdict === "ACCEPT") break;
    if (turn >= ladder.length) break; // no stronger rung available
  }

  // "What if you'd sent everything to the strongest model" comparison.
  const strongest = (pool.length ? pool : registry.all()).reduce(
    (a, b) => (b.strength > a.strength ? b : a),
  );
  const outTokens = Math.max(1, Math.ceil(answer.length / 4));
  const costVsFrontierOnlyUsd = costOf(strongest, { in: inTokens, out: outTokens });

  return {
    id: `chatcmpl-${randomUUID()}`,
    answer,
    mode,
    signature,
    trace,
    turns: trace.length,
    usageByModel,
    costUsd,
    costVsFrontierOnlyUsd,
    createdAt: Date.now(),
  };
}

function lastUserOf(messages: ChatMessage[]): string {
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i]?.role === "user") return messages[i]!.content;
  }
  return messages[messages.length - 1]?.content ?? "";
}
