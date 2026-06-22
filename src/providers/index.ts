/**
 * Builds the set of provider adapters from config and reports which are usable.
 * Any OpenAI-compatible provider works (built-ins + custom from MAESTRO_PROVIDERS).
 */
import type { MaestroConfig } from "../config.js";
import type { ProviderAdapter, ProviderName } from "../types.js";
import { MockAdapter } from "./mock.js";
import { OpenAICompatibleAdapter } from "./openai-compatible.js";

export class ProviderSet {
  private readonly map = new Map<string, ProviderAdapter>();
  private readonly realNames: string[];
  private readonly forceMock: boolean;

  constructor(config: MaestroConfig) {
    this.forceMock = config.forceMock;
    this.map.set("mock", new MockAdapter());
    for (const [name, pc] of Object.entries(config.providers)) {
      this.map.set(
        name,
        new OpenAICompatibleAdapter(name, {
          baseUrl: pc.baseUrl,
          apiKey: pc.apiKey,
          requireKey: pc.requireKey,
          headers: pc.headers,
          timeoutMs: config.requestTimeoutMs,
          maxRetries: config.maxRetries,
        }),
      );
    }
    this.realNames = Object.keys(config.providers);
  }

  get(name: ProviderName): ProviderAdapter {
    if (this.forceMock) return this.map.get("mock")!;
    return this.map.get(name) ?? this.map.get("mock")!;
  }

  /**
   * `forceMock` = demo mode: route over the full PRICED registry but execute on
   * the mock provider. Otherwise use providers that have a key; fall back to the
   * always-on mock models when none are configured.
   */
  configuredNames(): Set<string> {
    if (this.forceMock) return new Set(this.realNames);
    const names = new Set<string>();
    for (const name of this.realNames) {
      if (this.map.get(name)?.isConfigured()) names.add(name);
    }
    if (names.size === 0) names.add("mock");
    return names;
  }

  hasRealProvider(): boolean {
    return this.realNames.some((n) => this.map.get(n)?.isConfigured());
  }
}
