/**
 * Eval metrics. Honest by construction: routing decisions are graded against the
 * fixture's GROUND-TRUTH difficulty, not against the classifier's own estimate —
 * so a mis-estimate shows up as a real failure / regret, never hidden.
 */
import type { ModelSpec, TokenUsage } from "../src/types.js";
import { costOf } from "../src/core/cost.js";

/** Mirror of the verifier's strength requirement: 0.0→50, 1.0→95. */
export function requiredStrength(difficulty: number): number {
  return 50 + difficulty * 45;
}

export function passes(model: ModelSpec, difficulty: number): boolean {
  return model.strength >= requiredStrength(difficulty);
}

/** Cheapest model that actually passes (the oracle); else the strongest. */
export function oracleModel(pool: ModelSpec[], difficulty: number, usage: TokenUsage): ModelSpec {
  const passing = pool.filter((m) => passes(m, difficulty));
  if (passing.length === 0) {
    return pool.reduce((a, b) => (b.strength > a.strength ? b : a));
  }
  return passing.reduce((a, b) => (costOf(b, usage) < costOf(a, usage) ? b : a));
}

export interface StrategyRow {
  name: string;
  passRate: number;
  meanCostUsd: number;
  passesPerDollar: number;
  meanRegretUsd: number;
  fails: number;
  n: number;
}

export function summarize(
  name: string,
  records: { pass: boolean; cost: number; oracleCost: number }[],
): StrategyRow {
  const n = records.length;
  const passCount = records.filter((r) => r.pass).length;
  const totalCost = records.reduce((s, r) => s + r.cost, 0);
  const regrets = records.filter((r) => r.pass).map((r) => Math.max(0, r.cost - r.oracleCost));
  const meanRegret = regrets.length ? regrets.reduce((s, x) => s + x, 0) / regrets.length : 0;
  return {
    name,
    passRate: passCount / n,
    meanCostUsd: totalCost / n,
    passesPerDollar: totalCost > 0 ? passCount / totalCost : Number.POSITIVE_INFINITY,
    meanRegretUsd: meanRegret,
    fails: n - passCount,
    n,
  };
}

/** Brier score: mean squared error of predicted prob vs binary outcome. */
export function brier(predictions: { p: number; outcome: boolean }[]): number {
  if (predictions.length === 0) return 0;
  return (
    predictions.reduce((s, { p, outcome }) => s + (p - (outcome ? 1 : 0)) ** 2, 0) /
    predictions.length
  );
}

/** Expected Calibration Error over equal-width bins. */
export function ece(predictions: { p: number; outcome: boolean }[], bins = 5): number {
  if (predictions.length === 0) return 0;
  let total = 0;
  for (let b = 0; b < bins; b++) {
    const lo = b / bins;
    const hi = (b + 1) / bins;
    const inBin = predictions.filter(({ p }) => p >= lo && (b === bins - 1 ? p <= hi : p < hi));
    if (inBin.length === 0) continue;
    const avgP = inBin.reduce((s, x) => s + x.p, 0) / inBin.length;
    const acc = inBin.filter((x) => x.outcome).length / inBin.length;
    total += (inBin.length / predictions.length) * Math.abs(avgP - acc);
  }
  return total;
}
