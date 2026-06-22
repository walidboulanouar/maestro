/**
 * Maestro eval harness — runs fully offline (mock execution) over the *priced*
 * model registry, so it measures ROUTING quality and cost, deterministically and
 * for free. Regenerate the README table with `npm run eval`.
 *
 * Strategies compared:
 *   maestro          classify → route → verify/escalate (the real code path)
 *   best-single      always the strongest model
 *   cheapest-single  always the cheapest model
 *   random-route     deterministic pseudo-random pick (a sanity floor)
 *
 * Grading uses each fixture's GROUND-TRUTH difficulty (not the classifier's
 * estimate), and oracle-route regret vs the cheapest model that would pass.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, join } from "node:path";
import { loadConfig } from "../src/config.js";
import { costOf, roundUsd } from "../src/core/cost.js";
import { orchestrate } from "../src/core/orchestrator.js";
import { ProviderSet } from "../src/providers/index.js";
import { DEFAULT_REGISTRY } from "../src/registry/models.js";
import { ModelRegistry } from "../src/registry/registry.js";
import type { ModelSpec, TokenUsage } from "../src/types.js";
import {
  brier,
  ece,
  oracleModel,
  passes,
  requiredStrength,
  summarize,
  type StrategyRow,
} from "./metrics.js";

interface Fixture {
  id: string;
  prompt: string;
  task: string;
  difficulty: number;
}

const here = dirname(fileURLToPath(import.meta.url));

function loadFixtures(): Fixture[] {
  const text = readFileSync(join(here, "fixtures", "tasks.jsonl"), "utf8");
  return text
    .split("\n")
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l) as Fixture);
}

function estUsage(prompt: string): TokenUsage {
  return { in: Math.max(1, Math.ceil(prompt.length / 4)), out: 600 };
}

function pad(s: string, n: number): string {
  return s.length >= n ? s : s + " ".repeat(n - s.length);
}
function padL(s: string, n: number): string {
  return s.length >= n ? s : " ".repeat(n - s.length) + s;
}

function printTable(rows: StrategyRow[]): void {
  console.log(
    "\n" +
      pad("strategy", 18) +
      padL("pass%", 8) +
      padL("mean $", 12) +
      padL("pass/$", 12) +
      padL("regret $", 12) +
      padL("fails", 8),
  );
  console.log("-".repeat(70));
  for (const r of rows) {
    console.log(
      pad(r.name, 18) +
        padL((r.passRate * 100).toFixed(0) + "%", 8) +
        padL(r.meanCostUsd.toFixed(5), 12) +
        padL(Number.isFinite(r.passesPerDollar) ? r.passesPerDollar.toFixed(1) : "∞", 12) +
        padL(r.meanRegretUsd.toFixed(5), 12) +
        padL(String(r.fails), 8),
    );
  }
}

async function main(): Promise<void> {
  const fixtures = loadFixtures();
  const config = loadConfig({
    ...process.env,
    OPENROUTER_API_KEY: "eval-mock-key",
    MAESTRO_FORCE_MOCK: "true",
    MAESTRO_VERIFY: "true",
  });
  const registry = new ModelRegistry({
    updated: DEFAULT_REGISTRY.updated,
    models: DEFAULT_REGISTRY.models.filter((m) => m.provider !== "mock"),
  });
  const providers = new ProviderSet(config);
  const deps = { config, registry, providers };
  const pool = registry.all();

  const strongest = pool.reduce((a, b) => (b.strength > a.strength ? b : a));
  const cheapest = pool.reduce((a, b) =>
    costOf(b, { in: 1000, out: 600 }) < costOf(a, { in: 1000, out: 600 }) ? b : a,
  );

  const maestroRecs: { pass: boolean; cost: number; oracleCost: number }[] = [];
  const bestRecs: { pass: boolean; cost: number; oracleCost: number }[] = [];
  const cheapRecs: { pass: boolean; cost: number; oracleCost: number }[] = [];
  const randRecs: { pass: boolean; cost: number; oracleCost: number }[] = [];
  const calib: { p: number; outcome: boolean }[] = [];
  const perFixture: string[] = [];

  for (let i = 0; i < fixtures.length; i++) {
    const fx = fixtures[i]!;
    const usage = estUsage(fx.prompt);
    const oracle = oracleModel(pool, fx.difficulty, usage);
    const oracleCost = costOf(oracle, usage);

    // --- maestro (real code path) ---
    const result = await orchestrate(
      { model: "maestro-auto", messages: [{ role: "user", content: fx.prompt }] },
      deps,
    );
    const finalModel = registry.byId(result.trace.at(-1)?.model ?? "") ?? strongest;
    const mPass = passes(finalModel, fx.difficulty);
    maestroRecs.push({ pass: mPass, cost: result.costUsd, oracleCost });

    // calibration: did the FIRST rung actually pass (ground truth)?
    const firstModel = registry.byId(result.trace[0]?.model ?? "") ?? finalModel;
    calib.push({ p: result.signature.confidence, outcome: passes(firstModel, fx.difficulty) });

    // --- baselines (analytical, same usage estimate) ---
    bestRecs.push({ pass: passes(strongest, fx.difficulty), cost: costOf(strongest, usage), oracleCost });
    cheapRecs.push({ pass: passes(cheapest, fx.difficulty), cost: costOf(cheapest, usage), oracleCost });
    const rnd = pickDeterministic(pool, i);
    randRecs.push({ pass: passes(rnd, fx.difficulty), cost: costOf(rnd, usage), oracleCost });

    perFixture.push(
      `${pad(fx.id, 16)} d=${fx.difficulty.toFixed(2)} ` +
        `→ ${pad(finalModel.id, 28)} turns=${result.turns} ` +
        `${mPass ? "PASS" : "FAIL"} $${roundUsd(result.costUsd)}`,
    );
  }

  const rows = [
    summarize("maestro", maestroRecs),
    summarize("best-single", bestRecs),
    summarize("cheapest-single", cheapRecs),
    summarize("random-route", randRecs),
  ];

  console.log(`Maestro eval — ${fixtures.length} fixtures, ${pool.length} models (offline mock execution)`);
  printTable(rows);

  console.log(`\ncalibration (classifier confidence vs first-rung-correct):`);
  console.log(`  Brier = ${brier(calib).toFixed(3)}   ECE = ${ece(calib).toFixed(3)}  (lower is better)`);

  if (process.argv.includes("--verbose")) {
    console.log(`\nper-fixture:`);
    for (const line of perFixture) console.log("  " + line);
  }

  const maestro = rows[0]!;
  const best = rows[1]!;
  const savings = best.meanCostUsd > 0 ? 1 - maestro.meanCostUsd / best.meanCostUsd : 0;
  console.log(
    `\nsummary: maestro pass ${(maestro.passRate * 100).toFixed(0)}% ` +
      `(best-single ${(best.passRate * 100).toFixed(0)}%) at ` +
      `${(savings * 100).toFixed(0)}% lower mean cost than best-single.\n`,
  );
}

function pickDeterministic(pool: ModelSpec[], i: number): ModelSpec {
  // deterministic stand-in for "random" (no Math.random → reproducible)
  return pool[(i * 7 + 3) % pool.length]!;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
