/**
 * Routing policy (v0): capability/policy filter → tier by difficulty →
 * guardrail score → an ordered escalation ladder of (model, effort).
 *
 * The guardrail score adapts MoL-Harness: priority + strong*100 + positive*10 −
 * negative*250, where priority is the model's overall strength and the tag bonus
 * is matched against the task.
 */
import type {
  MaestroHint,
  ModelSpec,
  Rung,
  RouteDecision,
  TaskSignature,
  Tier,
} from "../types.js";
import { costOf } from "./cost.js";

export interface RouteContext {
  thresholds: { low: number; high: number };
  hint?: MaestroHint | undefined;
  /** Estimated tokens for cost comparison. */
  estInputTokens: number;
  estOutputTokens: number;
}

const TIER_ORDER: Tier[] = ["cheap", "mid", "frontier"];

export function guardrailScore(model: ModelSpec, sig: TaskSignature): number {
  let score = model.strength; // priority
  const tags = model.tags ?? {};
  for (const [tag, weight] of Object.entries(tags)) {
    const [kind, subject] = tag.split(":");
    const matches = subject === sig.task || subject === "general";
    if (!matches) continue;
    if (kind === "strong") score += 100 * weight;
    else if (kind === "positive") score += 10 * weight;
    else if (kind === "negative") score -= 250 * weight;
  }
  return score;
}

function effortFor(tier: Tier, sig: TaskSignature, model: ModelSpec): string {
  const efforts = model.efforts;
  if (!efforts || efforts.length === 0) return "medium";
  if (tier === "frontier" && sig.difficulty >= 0.85) {
    return efforts[efforts.length - 1]!; // strongest effort for the hardest tasks
  }
  if (sig.difficulty < 0.4 && efforts.includes("low")) return "low";
  return efforts.includes("medium") ? "medium" : efforts[0]!;
}

function passesPolicy(model: ModelSpec, sig: TaskSignature, hint?: MaestroHint): boolean {
  if (!sig.caps.every((c) => model.caps.includes(c))) return false;
  if (sig.sensitive && model.privacyOk === false) return false;
  if (hint?.region) {
    if (model.regions && !model.regions.includes(hint.region)) return false;
  }
  if (hint?.policy === "no-closed" && model.tier === "frontier" && model.provider !== "local-openai") {
    // crude example policy: avoid closed frontier models
    if (!/glm|qwen|deepseek|kimi|llama|mistral|mock/i.test(model.id)) return false;
  }
  return true;
}

export function route(
  pool: ModelSpec[],
  sig: TaskSignature,
  ctx: RouteContext,
): RouteDecision {
  const eligible = pool.filter((m) => passesPolicy(m, sig, ctx.hint));
  const usable = eligible.length > 0 ? eligible : pool; // never strand a request

  const startTier: Tier =
    sig.difficulty < ctx.thresholds.low
      ? "cheap"
      : sig.difficulty < ctx.thresholds.high
        ? "mid"
        : "frontier";

  const ladder: Rung[] = [];
  const seen = new Set<string>();
  for (const tier of TIER_ORDER.slice(TIER_ORDER.indexOf(startTier))) {
    const best = usable
      .filter((m) => m.tier === tier)
      .sort((a, b) => guardrailScore(b, sig) - guardrailScore(a, sig))[0];
    if (best && !seen.has(best.id)) {
      ladder.push({ model: best, effort: effortFor(tier, sig, best) });
      seen.add(best.id);
    }
  }
  if (ladder.length === 0) {
    const best = usable.sort((a, b) => guardrailScore(b, sig) - guardrailScore(a, sig))[0]!;
    ladder.push({ model: best, effort: effortFor(best.tier, sig, best) });
  }

  // Strongest available model → the "frontier-only" comparison baseline.
  const strongest = usable.reduce((a, b) => (b.strength > a.strength ? b : a), usable[0]!);
  const frontierOnlyEstimateUsd = costOf(strongest, {
    in: ctx.estInputTokens,
    out: ctx.estOutputTokens,
  });

  const reason =
    `start tier=${startTier} (difficulty ${sig.difficulty.toFixed(2)} vs ` +
    `${ctx.thresholds.low}/${ctx.thresholds.high}); ladder=` +
    ladder.map((r) => `${r.model.id}@${r.effort}`).join(" → ");

  return { signature: sig, ladder, frontierOnlyEstimateUsd, reason };
}
