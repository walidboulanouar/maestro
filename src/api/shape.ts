/**
 * Shapes an OrchestrationResult into an OpenAI-compatible response plus Maestro's
 * non-breaking `maestro` transparency block and `x-maestro-*` headers.
 */
import { roundUsd } from "../core/cost.js";
import type { OrchestrationResult, TokenUsage } from "../types.js";

function totalUsage(result: OrchestrationResult): TokenUsage {
  let inTok = 0;
  let outTok = 0;
  for (const u of Object.values(result.usageByModel)) {
    inTok += u.in;
    outTok += u.out;
  }
  return { in: inTok, out: outTok };
}

export function maestroBlock(result: OrchestrationResult) {
  const savings =
    result.costVsFrontierOnlyUsd > 0
      ? Math.max(0, 1 - result.costUsd / result.costVsFrontierOnlyUsd)
      : 0;
  return {
    mode: result.mode,
    route: result.trace.map((t) => ({
      turn: t.turn,
      slot: t.slot,
      model: t.model,
      provider: t.provider,
      effort: t.effort,
      verdict: t.verdict,
      verify_reason: t.verifyReason,
      verify_confidence: t.verifyConfidence,
      ms: t.ms,
    })),
    turns: result.turns,
    classify: {
      task: result.signature.task,
      difficulty: Number(result.signature.difficulty.toFixed(2)),
      caps: result.signature.caps,
      confidence: Number(result.signature.confidence.toFixed(2)),
      reason: result.signature.reason,
    },
    usage_by_model: result.usageByModel,
    cost_usd: roundUsd(result.costUsd),
    cost_vs_frontier_only_usd: roundUsd(result.costVsFrontierOnlyUsd),
    savings_pct: Math.round(savings * 100),
  };
}

export function toOpenAIResponse(result: OrchestrationResult, requestedModel: string) {
  // Preserve the upstream provider response verbatim (native_finish_reason,
  // usage.cost, system_fingerprint, openrouter_metadata, per-choice errors,
  // tool_calls, …) and only add the non-breaking `maestro` block. We fall back
  // to a reconstructed response only when there is no upstream (e.g. mock).
  if (
    result.upstreamRaw &&
    typeof result.upstreamRaw === "object" &&
    "choices" in (result.upstreamRaw as Record<string, unknown>)
  ) {
    return { ...(result.upstreamRaw as Record<string, unknown>), maestro: maestroBlock(result) };
  }

  const usage = totalUsage(result);
  return {
    id: result.id,
    object: "chat.completion",
    created: Math.floor(result.createdAt / 1000),
    model: requestedModel,
    choices: [
      {
        index: 0,
        message: {
          role: "assistant",
          // OpenAI convention: content is null when the model returns tool_calls.
          content: result.toolCalls ? (result.answer || null) : result.answer,
          ...(result.toolCalls ? { tool_calls: result.toolCalls } : {}),
        },
        finish_reason: result.finishReason ?? "stop",
      },
    ],
    usage: {
      prompt_tokens: usage.in,
      completion_tokens: usage.out,
      total_tokens: usage.in + usage.out,
    },
    maestro: maestroBlock(result),
  };
}

export function maestroHeaders(result: OrchestrationResult): Record<string, string> {
  const models = result.trace.map((t) => t.model).join(",");
  return {
    "x-maestro-mode": result.mode,
    "x-maestro-models": models,
    "x-maestro-turns": String(result.turns),
    "x-maestro-cost-usd": String(roundUsd(result.costUsd)),
    "x-maestro-task": result.signature.task,
  };
}
