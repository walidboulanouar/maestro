#!/usr/bin/env node
/**
 * Maestro CLI.
 *
 *   maestro serve [--port N]        start the OpenAI-compatible server
 *   maestro route "<prompt>"        dry-run: show the route decision, no call
 *   maestro registry check          report registry staleness
 *   maestro version                 print version
 */
import { serve } from "@hono/node-server";
import { loadConfig } from "./config.js";
import { classify } from "./core/classify.js";
import { route, type RouteContext } from "./core/route.js";
import { totalChars } from "./core/transcript.js";
import { ProviderSet } from "./providers/index.js";
import { ModelRegistry } from "./registry/registry.js";
import { buildDeps, createApp } from "./server.js";

const VERSION = "0.1.0";

function loadRegistry(): ModelRegistry {
  const path = process.env.MAESTRO_REGISTRY;
  if (path) {
    try {
      return ModelRegistry.fromFile(path);
    } catch (err) {
      console.error(`Failed to load registry ${path}: ${(err as Error).message}`);
      process.exit(1);
    }
  }
  return ModelRegistry.default();
}

function flag(args: string[], name: string): string | undefined {
  const i = args.indexOf(name);
  return i >= 0 ? args[i + 1] : undefined;
}

async function cmdServe(args: string[]): Promise<void> {
  const config = loadConfig();
  const portArg = flag(args, "--port");
  if (portArg) config.port = Number(portArg);
  const registry = loadRegistry();
  const providers = new ProviderSet(config);
  const app = createApp(buildDeps(config, registry, providers));

  const real = providers.hasRealProvider();
  serve({ fetch: app.fetch, port: config.port, hostname: config.host }, (info) => {
    console.log(`\n  maestro v${VERSION}  →  http://${config.host}:${info.port}`);
    console.log(`  providers: ${[...providers.configuredNames()].join(", ")}`);
    if (!real) {
      console.log(
        `  ⚠  no provider key found — running on the built-in MOCK provider.\n` +
          `     set OPENROUTER_API_KEY (or AI_GATEWAY_API_KEY / LOCAL_OPENAI_BASE_URL) to route to real models.`,
      );
    }
    console.log(`\n  try:`);
    console.log(
      `    curl -s localhost:${info.port}/v1/chat/completions -H 'content-type: application/json' \\`,
    );
    console.log(
      `      -d '{"model":"maestro-auto","messages":[{"role":"user","content":"hello"}]}' | jq .maestro\n`,
    );
  });
}

function cmdRoute(args: string[]): void {
  const prompt = args.find((a) => !a.startsWith("--")) ?? "";
  if (!prompt) {
    console.error('usage: maestro route "<prompt>"');
    process.exit(1);
  }
  const config = loadConfig();
  const registry = loadRegistry();
  const providers = new ProviderSet(config);
  const sig = classify([{ role: "user", content: prompt }]);
  const pool = registry.available(providers.configuredNames());
  const ctx: RouteContext = {
    thresholds: config.thresholds,
    estInputTokens: Math.max(1, Math.ceil(totalChars([{ role: "user", content: prompt }]) / 4)),
    estOutputTokens: 600,
  };
  const decision = route(pool.length ? pool : registry.all(), sig, ctx);
  console.log(JSON.stringify({ classify: sig, decision }, null, 2));
}

function cmdRegistryCheck(): void {
  const registry = loadRegistry();
  const age = registry.ageInDays();
  console.log(`registry updated: ${registry.updated} (${age} days ago)`);
  console.log(`models: ${registry.all().length}`);
  if (age > 30) {
    console.log(`⚠  registry is ${age} days old — prices/leaderboards may be stale. Refresh before relying on routing.`);
  }
  const stale = registry.all().filter((m) => {
    const a = Math.floor((Date.now() - Date.parse(m.price.updated)) / 86_400_000);
    return a > 30;
  });
  if (stale.length) console.log(`⚠  ${stale.length} model price(s) older than 30 days.`);
}

async function main(): Promise<void> {
  const [cmd, ...args] = process.argv.slice(2);
  switch (cmd) {
    case "serve":
      await cmdServe(args);
      break;
    case "route":
      cmdRoute(args);
      break;
    case "registry":
      if (args[0] === "check") cmdRegistryCheck();
      else console.error("usage: maestro registry check");
      break;
    case "version":
    case "--version":
    case "-v":
      console.log(VERSION);
      break;
    default:
      console.log(
        `maestro v${VERSION}\n\n` +
          `usage:\n` +
          `  maestro serve [--port N]     start the OpenAI-compatible server\n` +
          `  maestro route "<prompt>"     dry-run the router (no model call)\n` +
          `  maestro registry check       report registry staleness\n` +
          `  maestro version              print version\n`,
      );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
