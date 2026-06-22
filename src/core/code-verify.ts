/**
 * Executable verifier for code answers (v1, EXPERIMENTAL, opt-in).
 *
 * Instead of asking an LLM "is this answer good?", run the code and any
 * assertions it contains. Deterministic, free, and far more reliable for code.
 *
 * SECURITY: this runs MODEL-GENERATED CODE in a node:vm context. `vm` is NOT a
 * hardened sandbox (escapes exist). It is off by default and should only be
 * enabled (MAESTRO_CODE_VERIFY=true) when you accept running model code locally,
 * ideally inside an already-isolated container. We expose no require / process /
 * fs / network and enforce a timeout, but treat this as best-effort isolation.
 */
import vm from "node:vm";

/** Pull the first fenced JavaScript/TypeScript block out of an answer. */
export function extractCode(text: string): string | null {
  const m = text.match(/```(?:js|javascript|ts|typescript|node)?\s*([\s\S]*?)```/i);
  const code = m?.[1]?.trim();
  return code && code.length > 0 ? code : null;
}

export interface RunResult {
  ok: boolean;
  error?: string;
  logs: string[];
}

/** Run a snippet in a minimal frozen context with a hard timeout. */
export function runJs(code: string, timeoutMs = 1500): RunResult {
  const logs: string[] = [];
  const sandbox = {
    console: {
      log: (...a: unknown[]) => logs.push(a.map(String).join(" ")),
      error: (...a: unknown[]) => logs.push(a.map(String).join(" ")),
    },
    assert: (cond: unknown, msg?: string) => {
      if (!cond) throw new Error(msg ?? "assertion failed");
    },
  };
  try {
    vm.runInNewContext(code, sandbox, { timeout: timeoutMs });
    return { ok: true, logs };
  } catch (e) {
    return { ok: false, error: e instanceof Error ? e.message : String(e), logs };
  }
}
