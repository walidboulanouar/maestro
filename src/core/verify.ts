/**
 * The Verifier role. Decides ACCEPT (good enough → stop) or REVISE (escalate).
 *
 * - Mock provider: deterministic verdict from the answering model's strength vs
 *   the task difficulty. This makes the cheap→frontier escalation reproducible
 *   and visible in the keyless demo.
 * - Real provider: a strict rubric prompt returning JSON {verdict,reason,confidence}.
 */
import type { ProviderSet } from "../providers/index.js";
import type {
  ModelSpec,
  Rung,
  TaskSignature,
  TokenUsage,
  Verdict,
} from "../types.js";
import { judgeDifficulty } from "./classify.js";

export interface VerifyOutcome {
  verdict: Verdict;
  reason: string;
  confidence: number;
  usage: TokenUsage;
}

/** Pick a verifier model: explicit config, else a capable available model, else mock. */
export function chooseVerifier(
  pool: ModelSpec[],
  configured: Set<string>,
  configuredModel: string,
): ModelSpec {
  const usable = pool.filter((m) => configured.has(m.provider));
  if (configuredModel !== "auto") {
    const found = usable.find((m) => m.id === configuredModel || m.slot === configuredModel);
    if (found) return found;
  }
  // Prefer a mid-tier reasoner; fall back to the strongest cheap; else anything.
  const mid = usable
    .filter((m) => m.tier === "mid" && m.caps.includes("reasoning"))
    .sort((a, b) => b.strength - a.strength)[0];
  if (mid) return mid;
  const anyUsable = usable.sort((a, b) => b.strength - a.strength);
  return anyUsable[0] ?? pool[0]!;
}

function mockVerdict(rung: Rung, judgedDifficulty: number): VerifyOutcome {
  // Required strength climbs with difficulty: 0.0→50, 1.0→95.
  const required = 50 + judgedDifficulty * 45;
  const strength = rung.model.strength;
  const verdict: Verdict = strength >= required ? "ACCEPT" : "REVISE";
  const margin = Math.abs(strength - required);
  const confidence = Math.max(0.5, Math.min(0.98, 0.5 + margin / 60));
  const reason =
    verdict === "ACCEPT"
      ? `strength ${strength} ≥ required ${required.toFixed(0)} (judged difficulty ${judgedDifficulty.toFixed(2)})`
      : `strength ${strength} < required ${required.toFixed(0)} (judged difficulty ${judgedDifficulty.toFixed(2)}) — escalate`;
  return { verdict, reason, confidence, usage: { in: 0, out: 0 } };
}

const VERIFIER_SYSTEM =
  "You are a strict answer verifier. Given a TASK and an ANSWER, decide if the " +
  "answer fully and correctly satisfies the task. Reply with ONLY a JSON object: " +
  '{"verdict":"ACCEPT"|"REVISE","reason":"<short>","confidence":0..1}. ' +
  'Use ACCEPT only if the answer is correct and complete; otherwise REVISE.';

function parseVerifierJson(text: string): { verdict: Verdict; reason: string; confidence: number } {
  const match = text.match(/\{[\s\S]*\}/);
  if (match) {
    try {
      const obj = JSON.parse(match[0]) as Partial<{
        verdict: string;
        reason: string;
        confidence: number;
      }>;
      const verdict: Verdict = obj.verdict === "REVISE" ? "REVISE" : "ACCEPT";
      return {
        verdict,
        reason: typeof obj.reason === "string" ? obj.reason : "",
        confidence: typeof obj.confidence === "number" ? obj.confidence : 0.6,
      };
    } catch {
      /* fall through */
    }
  }
  // Fallback: keyword scan; default ACCEPT to avoid runaway escalation/cost.
  const verdict: Verdict = /\brevise\b/i.test(text) && !/\baccept\b/i.test(text) ? "REVISE" : "ACCEPT";
  return { verdict, reason: "unparseable verifier output; keyword fallback", confidence: 0.4 };
}

export async function verify(
  query: string,
  answer: string,
  rung: Rung,
  sig: TaskSignature,
  verifierModel: ModelSpec,
  providers: ProviderSet,
  deterministic: boolean,
): Promise<VerifyOutcome> {
  if (deterministic || verifierModel.provider === "mock") {
    return mockVerdict(rung, judgeDifficulty(query));
  }
  const adapter = providers.get(verifierModel.provider);
  const res = await adapter.chat({
    model: verifierModel.id,
    messages: [
      { role: "system", content: VERIFIER_SYSTEM },
      {
        role: "user",
        content:
          `DIFFICULTY=${sig.difficulty.toFixed(2)}\n\nTASK:\n${query}\n\nANSWER:\n${answer}`,
      },
    ],
    extra: { temperature: 0 },
  });
  const parsed = parseVerifierJson(res.text);
  return { ...parsed, usage: res.usage };
}
