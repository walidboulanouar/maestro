import { describe, expect, it } from "vitest";
import { extractCode, runJs } from "../src/core/code-verify.js";
import { verify } from "../src/core/verify.js";
import type { ModelSpec, Rung, TaskSignature } from "../src/types.js";
import type { ProviderSet } from "../src/providers/index.js";

describe("code-verify", () => {
  it("extracts a fenced code block", () => {
    expect(extractCode("text\n```js\nconsole.log(1)\n```\nmore")).toBe("console.log(1)");
    expect(extractCode("no code here")).toBeNull();
  });

  it("accepts code that runs and passes assertions", () => {
    const r = runJs("const add=(a,b)=>a+b; assert(add(2,3)===5); console.log('ok')");
    expect(r.ok).toBe(true);
    expect(r.logs).toContain("ok");
  });

  it("rejects code that throws", () => {
    expect(runJs("assert(1===2, 'nope')").ok).toBe(false);
    expect(runJs("throw new Error('boom')").ok).toBe(false);
  });

  it("enforces a timeout (no infinite hangs)", () => {
    const r = runJs("while(true){}", 150);
    expect(r.ok).toBe(false);
  });

  it("does not expose require/process", () => {
    expect(runJs("require('fs')").ok).toBe(false);
    expect(runJs("process.exit(1)").ok).toBe(false);
  });
});

describe("verify uses the executable verifier for code", () => {
  const sig: TaskSignature = { task: "code", difficulty: 0.5, caps: ["code"], freshness: false, sensitive: false, confidence: 0.8, reason: "" };
  const model: ModelSpec = { slot: "mock-mid", id: "mock-mid", provider: "mock", tier: "mid", strength: 75, caps: ["code"], price: { in: 0, out: 0, updated: "x" }, contextWindow: 1000 };
  const rung: Rung = { model, effort: "medium" };
  const providers = {} as ProviderSet;

  it("ACCEPTs working code", async () => {
    const out = await verify("write add", "```js\nassert((1+1)===2)\n```", rung, sig, model, providers, true, true);
    expect(out.verdict).toBe("ACCEPT");
  });
  it("REVISEs broken code", async () => {
    const out = await verify("write add", "```js\nassert(false)\n```", rung, sig, model, providers, true, true);
    expect(out.verdict).toBe("REVISE");
  });
});
