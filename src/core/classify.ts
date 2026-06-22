/**
 * Heuristic task classifier (v0). Fast, deterministic, zero-cost.
 *
 * Produces a TaskSignature: task type, difficulty, required capabilities, and
 * freshness/sensitivity flags — plus a confidence the eval harness can calibrate
 * (ECE/Brier). The learned router (v1) is a drop-in replacement for this.
 */
import type { Capability, ChatMessage, Task, TaskSignature } from "../types.js";
import { lastUserMessage, totalChars } from "./transcript.js";

const CODE_RE =
  /```|\b(function|class|def |import |const |let |var |async |await|return|npm |pip |regex|stack ?trace|exception|compile|null pointer|segfault|typescript|python|javascript|rust|golang)\b|\.(ts|js|py|rs|go|java|cpp|sql)\b/i;
const MATH_RE =
  /\b(integral|derivative|equation|theorem|prove|proof|matrix|probability|calculus|algebra|factorial|modulo|summation)\b|[0-9]\s*[+\-*/^]\s*[0-9]|\\frac|\\sum|\\int/i;
const TRANSLATE_RE =
  /\b(translate|translation|in (french|spanish|german|arabic|chinese|japanese|portuguese|italian|russian|korean)|traduis|traduce)\b/i;
const FACTUAL_RE =
  /\b(who|what|when|where|which|capital of|how many|define|definition of|meaning of)\b/i;
const FRESH_RE =
  /\b(today|todays|latest|current|currently|right now|this (week|month|year)|breaking|news|recent|as of|202[6-9]|live)\b/i;
const HARD_RE =
  /\b(prove|derive|design|architect|optimi[sz]e|refactor|analy[sz]e|explain why|step by step|trade-?offs?|complex|end-to-end|production-grade|edge cases?|formal)\b/i;

const SENSITIVE_RE =
  /\b(password|passwd|secret|api[_-]?key|private key|ssn|credit card)\b|sk-[a-zA-Z0-9]{16,}|AKIA[0-9A-Z]{16}|-----BEGIN [A-Z ]*PRIVATE KEY-----|[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}/;

function countMatches(re: RegExp, text: string): number {
  const g = new RegExp(re.source, re.flags.includes("g") ? re.flags : re.flags + "g");
  return (text.match(g) ?? []).length;
}

const MEDIUM_RE =
  /\b(explain|solve|implement|compute|describe|compare|difference between|how (do|does|to)|write a|walk through|step|fix|debug|error|stack ?trace|bug|review)\b/i;

/**
 * The verifier's INDEPENDENT difficulty judgment — leans conservative (high), as
 * a careful strong model would. Used only by the mock verifier so the escalation
 * loop is driven by a different, more reliable signal than the router's classify.
 * The real verifier is an actual strong LLM inspecting the answer.
 */
export function judgeDifficulty(query: string): number {
  const text = query;
  const lower = text.toLowerCase();
  const isCode = CODE_RE.test(text);
  const isMath = MATH_RE.test(text);
  const isReason = /\b(analy[sz]e|reason|argue|evaluate|trade-?offs?|decide|whether|effective)\b/i.test(lower);
  const hard = countMatches(HARD_RE, text);
  const medium = countMatches(MEDIUM_RE, text);
  const multi = Math.min(1, (countMatches(/\?/g, text) + countMatches(/\b(and|also|then)\b/gi, lower)) / 4);
  const len = Math.min(1, text.length / 1200);
  let d = 0.22;
  if (isCode) d += 0.22;
  if (isMath) d += 0.28;
  if (isReason) d += 0.22;
  d += Math.min(0.5, hard * 0.2);
  d += Math.min(0.28, medium * 0.12);
  d += 0.2 * multi;
  d += 0.15 * len;
  return Math.max(0, Math.min(1, d));
}

export function classify(messages: ChatMessage[]): TaskSignature {
  const text = lastUserMessage(messages);
  const lower = text.toLowerCase();
  const chars = totalChars(messages);

  const isCode = CODE_RE.test(text);
  const isMath = MATH_RE.test(text);
  const isTranslate = TRANSLATE_RE.test(text);
  const isFactual = FACTUAL_RE.test(lower);
  const freshness = FRESH_RE.test(lower);
  const sensitive = SENSITIVE_RE.test(text);

  // Task priority: translation > code > math > factual > general.
  let task: Task = "general";
  if (isTranslate) task = "translation";
  else if (isCode) task = "code";
  else if (isMath) task = "math";
  else if (isFactual && chars < 400) task = "factual";

  // Difficulty: blend of length, multi-part structure, and "hard" cues.
  const hardHits = countMatches(HARD_RE, text);
  const questionMarks = countMatches(/\?/g, text);
  const conjunctions = countMatches(/\b(and|also|then|as well as|plus)\b/gi, lower);
  const lengthScore = Math.min(1, chars / 1500);
  const multiPart = Math.min(1, (questionMarks + conjunctions) / 5);
  const hardScore = Math.min(1, hardHits / 3);
  let difficulty = 0.2 + 0.3 * lengthScore + 0.25 * multiPart + 0.35 * hardScore;
  if (task === "code" || task === "math") difficulty += 0.1;
  if (task === "factual" && hardHits === 0) difficulty -= 0.1;
  difficulty = Math.max(0, Math.min(1, difficulty));

  // Capabilities the route must satisfy.
  const caps = new Set<Capability>();
  if (task === "code") caps.add("code");
  if (task === "math") {
    caps.add("math");
    caps.add("reasoning");
  }
  if (task === "translation") caps.add("translation");
  if (task === "factual") caps.add("factual");
  if (difficulty >= 0.6) caps.add("reasoning");
  if (chars > 8000) caps.add("long-context");
  if (freshness) caps.add("fresh");

  // Confidence: strong when exactly one task signal fired; weak when ambiguous.
  const signals = [isCode, isMath, isTranslate, isFactual].filter(Boolean).length;
  let confidence = signals === 1 ? 0.85 : signals === 0 ? 0.5 : 0.6;
  if (chars < 30) confidence -= 0.1;
  confidence = Math.max(0.3, Math.min(0.95, confidence));

  const reason =
    `task=${task} (signals: ` +
    [
      isCode && "code",
      isMath && "math",
      isTranslate && "translation",
      isFactual && "factual",
    ]
      .filter(Boolean)
      .join(",") +
    `); difficulty=${difficulty.toFixed(2)} ` +
    `[len=${lengthScore.toFixed(2)} multi=${multiPart.toFixed(2)} hard=${hardScore.toFixed(2)}]` +
    (freshness ? "; fresh" : "") +
    (sensitive ? "; sensitive" : "");

  return {
    task,
    difficulty,
    caps: [...caps],
    freshness,
    sensitive,
    confidence,
    reason,
  };
}
