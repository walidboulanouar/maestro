/**
 * Builds the set of provider adapters from config and reports which are usable.
 */
import type { MaestroConfig } from "../config.js";
import type { ProviderAdapter, ProviderName } from "../types.js";
import { MockAdapter } from "./mock.js";
import { OpenAICompatibleAdapter } from "./openai-compatible.js";

export class ProviderSet {
  private readonly map = new Map<ProviderName, ProviderAdapter>();
  private readonly forceMock: boolean;

  constructor(config: MaestroConfig) {
    this.forceMock = config.forceMock;
    const mock = new MockAdapter();
    this.map.set("mock", mock);
    this.map.set(
      "openrouter",
      new OpenAICompatibleAdapter("openrouter", {
        baseUrl: config.providers.openrouter.baseUrl,
        apiKey: config.providers.openrouter.apiKey,
        requireKey: true,
        timeoutMs: config.requestTimeoutMs,
        headers: {
          "HTTP-Referer": "https://maestro.ayautomate.com",
          "X-Title": "Maestro",
          "X-OpenRouter-Title": "Maestro",
          // surface routing metadata in `openrouter_metadata` on responses
          "X-OpenRouter-Metadata": "enabled",
        },
      }),
    );
    this.map.set(
      "vercel-gateway",
      new OpenAICompatibleAdapter("vercel-gateway", {
        baseUrl: config.providers.vercelGateway.baseUrl,
        apiKey: config.providers.vercelGateway.apiKey,
        requireKey: true,
        timeoutMs: config.requestTimeoutMs,
      }),
    );
    this.map.set(
      "local-openai",
      new OpenAICompatibleAdapter("local-openai", {
        baseUrl: config.providers.localOpenai.baseUrl,
        apiKey: config.providers.localOpenai.apiKey,
        requireKey: false,
        timeoutMs: config.requestTimeoutMs,
      }),
    );
  }

  get(name: ProviderName): ProviderAdapter {
    if (this.forceMock) return this.map.get("mock")!;
    return this.map.get(name) ?? this.map.get("mock")!;
  }

  /** Provider names that are usable right now (mock is always available). */
  configuredNames(): Set<string> {
    // `forceMock` = demo mode: route over the full PRICED registry but execute
    // on the mock provider (no spend). Otherwise use providers that have a key;
    // fall back to the always-on mock models when none are configured.
    if (this.forceMock) {
      return new Set(["openrouter", "vercel-gateway", "local-openai"]);
    }
    const names = new Set<string>();
    for (const [name, adapter] of this.map) {
      if (name !== "mock" && adapter.isConfigured()) names.add(name);
    }
    if (names.size === 0) names.add("mock");
    return names;
  }

  /** True if a real (non-mock) provider actually has credentials. */
  hasRealProvider(): boolean {
    return (["openrouter", "vercel-gateway", "local-openai"] as const).some((n) =>
      this.map.get(n)?.isConfigured(),
    );
  }
}
