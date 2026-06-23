/**
 * maestro-ultra: Conductor-style multi-step decomposition (prompt-based, no
 * training). A planner model breaks a hard task into <=5 ordered subtasks; each
 * subtask is routed independently and sees prior step outputs (shared memory);
 * the last step's output is the answer. Falls back to a single step when the
 * planner returns one subtask (e.g. the mock provider), so it is always safe.
 *
 * This is the v3 roadmap item. It is gated: callers ask for it with
 * model "maestro-ultra". Use it only for genuinely hard, multi-part work.
 */
import { randomUUID } from "node:crypto";
import type { OrchestratorDeps } from "./orchestrator.js";
import { classify } from "./classify.js";
import { costOf } from "./cost.js";
import { route, type RouteContext } from "./route.js";
import { contentToText, lastUserMessage, totalChars } from "./transcript.js";
import type {
  ChatCompletionRequest,
  ModelSpec,
  OrchestrationResult,
  TokenUsage,
  TurnTrace,
} from "../types.js";

const MAX_STEPS = 5;
const PLANNER_SYSTEM =
  "You are a task planner. Break the user's task into 2 to 5 ordered, concrete " +
  "subtasks that build on each other (later steps may use earlier results). " +
  "Reply with ONLY a JSON array of strings. If the task is simple, reply with a " +
  "single-element array.";

function extractList(text: string): string[] {
  const m = text.match(/\[[\s\S]*\]/);
  if (!m) return [];
  try {
    const arr = JSON.parse(m[0]) as unknown[];
    return arr.filter((x): x is string => typeof x === "string" && x.trim().length > 0).slice(0, MAX_STEPS);
  } catch {
    return [];
  }
}

export async function orchestrateUltra(
  req: ChatCompletionRequest,
  deps: OrchestratorDeps,
): Promise<OrchestrationResult> {
  const { config, registry, providers } = deps;
  const configured = providers.configuredNames();
  const pool = registry.available(configured);
  const usable = pool.length ? pool : registry.all();
  const query = lastUserMessage(req.messages);

  const reqRecord = req as unknown as Record<string, unknown>;
  const { model: _m, messages: _msgs, stream: _s, maestro: _mae, ...extra } = reqRecord;

  const usageByModel: Record<string, TokenUsage> = {};
  let costUsd = 0;
  const addUsage = (model: ModelSpec, usage: TokenUsage) => {
    const prev = usageByModel[model.id] ?? { in: 0, out: 0 };
    usageByModel[model.id] = { in: prev.in + usage.in, out: prev.out + usage.out };
    costUsd += costOf(model, usage);
  };

  // 1. Plan with the strongest available model.
  const planner = usable.reduce((a, b) => (b.strength > a.strength ? b : a));
  let subtasks: string[] = [];
  try {
    const plan = await providers.get(planner.provider).chat({
      model: planner.id,
      messages: [
        { role: "system", content: PLANNER_SYSTEM },
        { role: "user", content: query },
      ],
      extra: { temperature: 0 },
    });
    addUsage(planner, plan.usage);
    subtasks = extractList(plan.text);
  } catch {
    /* planner failed; fall back to a single step */
  }
  if (subtasks.length === 0) subtasks = [query];

  // 2. Execute steps sequentially with shared memory.
  const trace: TurnTrace[] = [];
  const context: string[] = [];
  let answer = "";
  let finishReason = "stop";
  let upstreamRaw: unknown;

  for (let i = 0; i < subtasks.length; i++) {
    const sub = subtasks[i]!;
    const sig = classify([{ role: "user", content: sub }]);
    const ctx: RouteContext = {
      thresholds: config.thresholds,
      estInputTokens: Math.max(1, Math.ceil((sub.length + context.join("").length) / 4)),
      estOutputTokens: 600,
    };
    const rung = route(usable, sig, ctx).ladder[0]!;
    const messages = [
      ...(context.length ? [{ role: "system" as const, content: "Results so far:\n" + context.join("\n\n") }] : []),
      { role: "user" as const, content: sub },
    ];
    const started = Date.now();
    const res = await providers.get(rung.model.provider).chat({
      model: rung.model.id,
      messages,
      effort: rung.effort,
      extra,
    });
    const ms = Date.now() - started;
    answer = res.text;
    if (res.raw !== undefined) upstreamRaw = res.raw;
    if (res.finishReason) finishReason = res.finishReason;
    addUsage(rung.model, res.usage);
    context.push(`Step ${i + 1}: ${res.text}`);
    trace.push({
      turn: i + 1,
      slot: rung.model.slot,
      model: rung.model.id,
      provider: rung.model.provider,
      effort: rung.effort,
      role: "Worker",
      verifyReason: `ultra step ${i + 1}/${subtasks.length}: ${sub.slice(0, 60)}`,
      usage: res.usage,
      costUsd: costOf(rung.model, res.usage),
      ms,
    });
  }

  const inTokens = Math.max(1, Math.ceil(totalChars(req.messages) / 4));
  const strongest = usable.reduce((a, b) => (b.strength > a.strength ? b : a));
  const outTokens = Math.max(1, Math.ceil(answer.length / 4));
  const costVsFrontierOnlyUsd = costOf(strongest, { in: inTokens, out: outTokens });

  return {
    id: `chatcmpl-${randomUUID()}`,
    answer,
    mode: "ultra",
    signature: {
      task: "reasoning",
      difficulty: 1,
      caps: [],
      freshness: false,
      sensitive: false,
      confidence: 1,
      reason: `ultra: ${subtasks.length} step(s)`,
    },
    trace,
    turns: trace.length,
    usageByModel,
    costUsd,
    costVsFrontierOnlyUsd,
    createdAt: Date.now(),
    finishReason,
    ...(upstreamRaw !== undefined ? { upstreamRaw } : {}),
  };
}

// silence unused-content helper import lints in some setups
void contentToText;
