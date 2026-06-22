/**
 * Registry loading, validation, and queries.
 */
import { readFileSync } from "node:fs";
import { RegistrySchema, type ModelSpec, type Registry } from "../types.js";
import { DEFAULT_REGISTRY } from "./models.js";

export class ModelRegistry {
  private readonly models: ModelSpec[];
  readonly updated: string;

  constructor(registry: Registry) {
    this.models = registry.models;
    this.updated = registry.updated;
  }

  static default(): ModelRegistry {
    return new ModelRegistry(DEFAULT_REGISTRY);
  }

  /** Load and validate a registry JSON file, falling back to the default. */
  static fromFile(path: string): ModelRegistry {
    const raw = JSON.parse(readFileSync(path, "utf8"));
    const parsed = RegistrySchema.parse(raw);
    return new ModelRegistry(parsed);
  }

  all(): ModelSpec[] {
    return this.models;
  }

  byId(id: string): ModelSpec | undefined {
    return this.models.find((m) => m.id === id || m.slot === id);
  }

  /**
   * Models reachable given the configured providers. Real models take priority;
   * the built-in `mock` models are a last-resort fallback used only when no real
   * provider is configured (so a real key never routes to a free mock model).
   */
  available(configured: Set<string>): ModelSpec[] {
    const real = this.models.filter((m) => m.provider !== "mock" && configured.has(m.provider));
    if (real.length > 0) return real;
    return this.models.filter((m) => m.provider === "mock");
  }

  /** Age of the registry snapshot in days, relative to `now`. */
  ageInDays(now: number = Date.now()): number {
    const t = Date.parse(this.updated);
    if (Number.isNaN(t)) return Number.POSITIVE_INFINITY;
    return Math.floor((now - t) / 86_400_000);
  }
}
