# Maestro — definitive implementation plan (build this)

> The one doc to implement from. Synthesizes: the two papers (TRINITY 2512.04695 / Conductor 2512.04388), the official Sakana report, OpenFugu's reverse-eng docs, the external feasibility critique, and the locked scope (`MAESTRO-SCOPE.md`).
> Status: plan, ready to implement. Goal: a repo good enough to earn real attention on launch day.

---

## 0. The pitch (this is what earns stars)

**Maestro — the open-source orchestration brain for LLMs. Route any models (open *and* closed, your keys) behind one OpenAI-compatible endpoint, and get frontier-level answers with full cost/route transparency. Self-hostable. MIT.**

One sentence: *"Sakana Fugu, but open, transparent, EU-clean, and not locked to 3 models."*

Why people star it on day one (the honest version):
- **Drop-in.** Change one `base_url`; your existing OpenAI code routes across models with zero rewrites.
- **Any model.** Open (GLM-5.2, DeepSeek V4, Kimi K2.7) *and* closed (Opus 4.8, GPT-5.5, Gemini) through one gateway, your keys (BYOK). Fugu gives you 3 closed models and blocks the EU.
- **Honest transparency.** Every response tells you which model answered, why, how many turns, tokens, and **cost** — the exact thing Fugu's report hides.
- **It refuses to overspend.** A cheap model tries first; a verifier escalates to a frontier model only when needed. You see the savings per request.
- **One command to try.** `docker run` or `npx`, set one key, done.
- **Reproducible honesty.** A bundled eval (`maestro eval`) shows *when routing helps and when it doesn't* — engineers reward that over hype.

> Reality check: 4K stars day-one is a top-1% outcome. The code below is necessary but not sufficient — **§13 (launch) is half the work.** A 20-second demo GIF + drop-in compatibility + an honest benchmark table + posting in the right threads is what converts a good repo into a viral one.

---

## 1. Scope (locked — see MAESTRO-SCOPE.md)
**Maestro is JUST the orchestration/routing brain.** No GPU, no model hosting, no inference. Models are consumed through an AI gateway (OpenRouter / Vercel AI Gateway, BYOK). We do not rebuild what's solved (gateways, serving, inference). We build the router/coordinator + an OpenAI-compatible surface + transparency + an honest eval. Everything else (multi-agent debate, tools/sandbox, MoL serving, learned-router *training*, dashboard) is post-v0.

---

## 2. Architecture

```mermaid
flowchart TD
  C[Client: any OpenAI SDK] -->|POST /v1/chat/completions<br/>model: maestro-auto| S[Maestro server (Hono)]
  S --> P[Parse: messages, hints, budget, policy]
  P --> CL{Pinned model?}
  CL -- yes --> EX
  CL -- no --> CLS[Classify: task, difficulty, caps, freshness, sensitivity]
  CLS --> RT[Route: filter registry -> ordered (model,effort) ladder]
  RT --> EX[Execute via Provider Adapter -> Gateway]
  EX --> V{Verify enabled?}
  V -- no --> OUT
  V -- yes --> VR[Verifier role: ACCEPT or REVISE]
  VR -- ACCEPT / max-turns / budget --> OUT
  VR -- REVISE + budget left --> ESC[Escalate to stronger model] --> EX
  OUT[Return answer + transparency] --> C
  RT -. reads .-> REG[(Model Registry: dated, slots->gateway ids)]
  EX -. logs .-> TR[(Trace + cost accounting: SQLite/JSONL)]
```

Modules (all TypeScript, one process):
- **server** — Hono app exposing the OpenAI-compatible API.
- **core/orchestrator** — the ≤K-turn loop (classify → route → execute → verify → escalate).
- **core/classify** — task signature (heuristic fast-path + optional cheap-LLM classifier).
- **core/route** — the routing policy (capability/policy filter → tier by difficulty → guardrail score → escalation ladder).
- **core/verify** — Thinker/Worker/Verifier roles as prompts; ACCEPT/REVISE; `<suggested_role>` override.
- **core/transcript** — raw `"role: content\n"` formatting (the OpenFugu gotcha; matters for the learned router later, harmless now).
- **registry** — load/validate a dated model registry; slot → gateway model id mapping (the key decoupling).
- **providers** — `ProviderAdapter` interface + `openrouter` + `vercel-gateway` adapters (BYOK, streaming).
- **transparency** — trace store + cost accounting; response metadata + headers.
- **eval** — fixtures, baselines, metrics (regret, cost-normalized quality, calibration). The proof.

---

## 3. Request lifecycle (detailed)

1. **Parse.** Read OpenAI request. The `model` field selects behavior:
   - `maestro-auto` → full routing (classify + route + verify).
   - `maestro-fugu` → single best worker + verify loop (low latency; v0 default).
   - `maestro-ultra` → Conductor-style decompose (post-v0; falls back to `fugu` until built).
   - any real model id (e.g. `anthropic/claude-opus-4.8`) → **pure passthrough** (no routing) — so Maestro is always a safe drop-in even when you don't want routing.
   - Optional `maestro` hint object in request body: `{budget, policy, region, maxTurns, verify, pin}`.
2. **Classify** (skipped if pinned). Produce a `TaskSignature` (§5).
3. **Route.** Filter the registry by required capabilities + policy/region/budget → pick a tier by difficulty → order an **escalation ladder** of `(model, effort)`.
4. **Execute.** Call the chosen model via its provider adapter (streaming supported). Stream the *final* turn to the client; intermediate verify turns are internal.
5. **Verify loop.** Verifier judges the answer → `ACCEPT` ends it; `REVISE` + remaining budget escalates to the next rung. Bounded by `maxTurns` (default 3).
6. **Return.** OpenAI-shaped response + `maestro` transparency block + `x-maestro-*` headers.

---

## 4. OpenAI-compatible API surface
- `POST /v1/chat/completions` — full OpenAI semantics incl. `stream: true` (SSE). Adds optional `maestro` hint object; adds `maestro` block to the response.
- `GET /v1/models` — lists `maestro-auto|fugu|ultra` + every registry model.
- `POST /v1/route` — **dry-run**: returns the route decision (chosen ladder, reasons, estimated cost) **without executing**. (The critique's "route dry-run endpoint" — also a great demo.)
- `GET /healthz`, `GET /metrics` (Prometheus-style, optional).

Response extension (non-breaking — clients ignore unknown fields):
```jsonc
{
  "id": "...", "choices": [...], "usage": {...},
  "maestro": {
    "mode": "fugu",
    "route": [{"slot":"cheap-coder","model":"z-ai/glm-5.2-air","effort":"medium","turn":1,"verdict":"REVISE"},
              {"slot":"frontier-coder","model":"anthropic/claude-opus-4.8","effort":"high","turn":2,"verdict":"ACCEPT"}],
    "turns": 2,
    "classify": {"task":"code","difficulty":0.78,"caps":["code","reasoning"]},
    "usage_by_model": {"z-ai/glm-5.2-air":{"in":1200,"out":300},"anthropic/claude-opus-4.8":{"in":1500,"out":900}},
    "cost_usd": 0.0182,
    "cost_vs_frontier_only_usd": 0.0241,
    "reason": "high difficulty + code caps; cheap rung failed verify, escalated"
  }
}
```
Headers: `x-maestro-models`, `x-maestro-turns`, `x-maestro-cost-usd`, `x-maestro-mode`.

---

## 5. The router (v0) — the actual algorithm

`TaskSignature`:
```ts
interface TaskSignature {
  task: "code" | "math" | "reasoning" | "translation" | "factual" | "general";
  difficulty: number;        // 0..1
  caps: string[];            // required capabilities, e.g. ["code","long-context"]
  freshness: boolean;        // needs recent/web info (route to a model w/ search, or flag)
  sensitive: boolean;        // PII/secret → respect provider/region policy
  confidence: number;        // classifier confidence (feeds calibration eval)
}
```

**Classifier (v0, two modes, configurable):**
- *Heuristic fast-path* (zero extra cost/latency): code-fence/keyword/stacktrace detection → `task`; length + nesting + "prove/derive/why" cues → `difficulty`; "today/latest/news" → `freshness`; regex for secrets/PII → `sensitive`.
- *Cheap-LLM classifier* (optional, one small gateway call): a tiny prompt to a cheap model returning the `TaskSignature` JSON. Used when heuristics are low-confidence.
- The router records `confidence` so the eval can measure **calibration (ECE/Brier)** — the critique's point.

**Routing policy:**
```ts
function route(sig: TaskSignature, registry: ModelSpec[], policy: Policy): Rung[] {
  let pool = registry
    .filter(m => sig.caps.every(c => m.caps.includes(c)))     // capability filter
    .filter(m => policy.allow(m) && regionOk(m, policy) && priceOk(m, policy)); // guardrails
  if (sig.sensitive) pool = pool.filter(m => m.privacyOk);
  // tier by difficulty
  const startTier = sig.difficulty < 0.33 ? "cheap" : sig.difficulty < 0.7 ? "mid" : "frontier";
  // build escalation ladder: start at tier, climb to frontier
  const order = ["cheap","mid","frontier"];
  const rungs: Rung[] = [];
  for (const tier of order.slice(order.indexOf(startTier))) {
    const best = pool.filter(m => m.tier === tier).sort(byGuardrailScore(sig))[0];
    if (best) rungs.push({ model: best, effort: effortFor(tier, sig) });
  }
  return rungs.length ? rungs : [{ model: bestOverall(pool), effort: "high" }];
}
```
**Guardrail score** (from MoL, adapted): `priority + strong*100 + positive*10 − negative*250`, where `strong/positive/negative` come from the model's `tags` matched against `sig.task` (e.g. a model tagged `strong:code` wins code routes). Deterministic, explainable.

**The loop:**
```ts
async function orchestrate(req, sig, rungs, opts): Promise<Result> {
  let answer, trace = [];
  for (let turn = 1; turn <= opts.maxTurns && rungs.length; turn++) {
    const rung = rungs.shift();
    answer = await execute(rung, req, role="Worker");
    if (!opts.verify) { trace.push({...rung, turn}); break; }
    const v = await verify(req.query, answer, rung);   // Verifier role
    trace.push({...rung, turn, verdict: v.verdict, confidence: v.confidence});
    if (v.verdict === "ACCEPT") break;
    if (!rungs.length || budgetExhausted(opts)) break;  // no stronger rung / out of budget -> return best so far
    // optional: Thinker emits <suggested_role> to override next rung
  }
  return { answer, trace };
}
```
**Honesty:** this loop *replaces* the false "we never lose to a single model" claim with a measured fallback. We don't promise we always win; we promise we try cheap first, verify, and escalate — and we *show* the result.

---

## 6. Model registry (the decoupling that makes it work)

OpenFugu's key lesson: **slot labels are metadata, remappable to any provider without retraining.** So we route over abstract slots and map them to gateway model ids in config.

```ts
interface ModelSpec {
  slot: string;                 // abstract, e.g. "frontier-coder"
  id: string;                   // gateway id, e.g. "anthropic/claude-opus-4.8"
  provider: "openrouter" | "vercel-gateway";
  tier: "cheap" | "mid" | "frontier";
  strength: number;             // 0..100 (from leaderboards; DATED)
  caps: string[];               // ["code","math","reasoning","long-context","vision","fresh"]
  efforts?: string[];           // supported effort/reasoning levels
  price: { in: number; out: number; updated: string }; // $/Mtok + date
  contextWindow: number;
  regions?: string[];           // allowed regions (EU filtering = a feature, per critique)
  privacyOk?: boolean;          // ok for sensitive content
  tags?: Record<string, number>;// guardrail scoring: {"strong:code":1,"positive:math":1}
}
```
Ship a **default dated registry** (`models.config.ts`) with a handful of strong picks (e.g. cheap: GLM-5.2-Air/Qwen3.6; mid: DeepSeek V4 / Kimi K2.7-Code; frontier: Opus 4.8 / GPT-5.5 / Gemini 3.x). Each entry carries an `updated` date; `maestro registry check` warns when prices/leaderboards are stale (critique's "time-sensitive evidence" fix). Users override via their own `maestro.config.{ts,json}`.

---

## 7. Provider adapters (gateways)
```ts
interface ProviderAdapter {
  name: string;
  chat(opts: {
    model: string; messages: Msg[]; effort?: string;
    stream?: boolean; signal?: AbortSignal; maxTokens?: number; temperature?: number;
  }): Promise<ChatResult> | AsyncIterable<Chunk>;
  // returns text + token usage (for cost accounting)
}
```
- **openrouter** — OpenAI-compatible REST; pass-through; reads usage. ~5.5% fee (their side).
- **vercel-gateway** — OpenAI-compatible; 0% markup incl. BYOK; also serves `sakana/fugu-ultra` → lets us include **Fugu itself as a worker AND as an eval baseline** (great for the README comparison).
- **local-openai** — point at any local OpenAI-compatible server (Ollama / vLLM / llama.cpp / LM Studio) via `baseUrl`. Same adapter shape → **fully offline routing for anyone with a GPU, no inference code on our side.** This is how Maestro stays "run it 100% locally" without hosting models.
- Both behind the registry; selecting a slot picks the adapter. Keys via env (`OPENROUTER_API_KEY`, `AI_GATEWAY_API_KEY`), never logged. Cost computed from `usage × registry.price`.

---

## 8. Transparency & traces
- Every request writes a `Trace` (SQLite via `better-sqlite3`, or JSONL if no native deps wanted): request id, signature, route ladder, per-model usage, cost, latency per turn, verdicts.
- Exposed live in the response `maestro` block + `x-maestro-*` headers.
- `maestro trace <id>` CLI prints it; optional tiny `/trace/:id` HTML view later.
- This is the wedge — cheap to build, and it's the demo that makes people trust it.

---

## 9. Roles & verify (prompts, no ML in v0)
- **Worker** — answers the task (the routed model).
- **Verifier** — a strict prompt: "Given the task and the answer, return `{verdict: ACCEPT|REVISE, reason, confidence}`. ACCEPT only if it fully and correctly satisfies the task." For `task=code` later: run unit tests (executable check) instead of an LLM judge — far more reliable (critique: "executable verifier for code/tool tasks").
- **Thinker (optional)** — may emit `<suggested_role>`/`<suggested_model>` to override the next rung (OpenFugu's override mechanism).
- Verifier runs on a cheap-but-capable model by default (configurable); we measure verifier false-accept in the eval.

---

## 10. Config & self-host
- Single Node process. `maestro.config.ts` (or JSON/env) for: registry, gateway keys, default mode, `maxTurns`, difficulty thresholds, budget caps, region policy, classifier mode.
- Ship: `docker run -e OPENROUTER_API_KEY=... -p 8080:8080 maestro` and `npx openmaestro@<pinned> serve`.
- Zero external infra for v0 (no Postgres/Redis/queue — a single self-hosted router doesn't need them; SQLite/JSONL is enough). Concurrency via Node async; add a queue only if someone needs batch.

---

## 11. The eval harness (non-negotiable — it's the proof, and the README's credibility)
- **Fixtures** (`eval/fixtures/*.jsonl`): coding-with-tests, math-with-exact-answers, factual-QA, long-context retrieval, and the critique's **mixed-capability traps** (e.g. *"Fix a React hydration bug AND explain its GDPR impact"*, *"Translate this stack trace into a fix"*) + a paraphrase set for brittleness.
- **Baselines:** best-single, cheapest-single, random-route, rule-only, Maestro(`fugu`). (Add Fugu-Ultra via Vercel gateway as a headline baseline.)
- **Metrics:** exact pass/unit-test pass; **oracle-route regret** (vs the best model per item from an offline sweep); **cost-normalized quality**; p50/p95 latency; fallback precision/recall; verifier false-accept; **calibration (ECE/Brier)** of `difficulty`/`confidence`.
- **Reproducibility:** frozen prompts, pinned model versions, temperatures, seeds, dataset hashes, a price snapshot, full trace export. `maestro eval` regenerates the README table. **No benchmark claim without ablations.**
- **Success bar (state it in the README):** *"On a fixed public eval, Maestro matches best-single-model quality within a confidence interval at materially lower median cost, with bounded p95 latency."* Not "beats Fugu."

---

## 12. Project structure & stack

```
maestro/
  package.json            # type: module, bin: maestro
  tsconfig.json
  src/
    cli.ts                # serve | eval | trace | registry check
    server.ts            # Hono app
    api/{chat,models,route,health}.ts
    core/{orchestrator,classify,route,verify,transcript}.ts
    registry/{registry,models.config}.ts
    providers/{adapter,openrouter,vercel-gateway}.ts
    transparency/{trace,cost}.ts
    config.ts
    types.ts             # zod schemas + inferred types
  eval/{run,baselines,metrics}.ts  eval/fixtures/*.jsonl
  docker/Dockerfile
  README.md  LICENSE(MIT)
```
**Stack decision (changed from the old Next.js plan):** for a self-hostable router the product *is* an API service, so use **Hono** (tiny, runs on Node/Bun/Deno/edge) + **TypeScript** + **Zod** (request/registry validation) + native `fetch`/`undici` for gateway calls + **better-sqlite3** (or JSONL) for traces + **Vitest** (tests/eval) + **tsx** (dev). A Next.js dashboard is an *optional later* add, not v0. Optional dep: **Vercel AI SDK** for streaming helpers (or hand-roll SSE to stay dep-light).

> Security pre-flight (required before first install): run `python3 ~/.claude/skills/pin-guard/scripts/scan.py .`, **pin every dependency to an exact version**, and `verify.py <pkg>` each new one. Never `npx pkg@latest`.

### Key contracts (write these first)
```ts
// types.ts (zod-backed)
export type Mode = "auto" | "fugu" | "ultra" | "passthrough";
export interface MaestroHint { budget?: number; policy?: string; region?: string;
  maxTurns?: number; verify?: boolean; pin?: string; }
export interface Rung { model: ModelSpec; effort?: string; }
export interface TurnTrace { slot: string; model: string; effort?: string; turn: number;
  verdict?: "ACCEPT"|"REVISE"; confidence?: number; usage: {in:number;out:number}; ms: number; }
export interface Result { answer: string; trace: TurnTrace[]; costUsd: number;
  signature: TaskSignature; mode: Mode; }
```

---

## 13. Learned router (v1 — deferred, but design the seam now)
The `classify`/`route` steps are a clean interface. v1 swaps the heuristic classifier for the **TRINITY-style learned head** (frozen Qwen3-0.6B early-exit + 19,456-float vector → `(model,effort)` logits), runnable on CPU. From OpenFugu's docs, when we get there:
- Feed the **raw `"role: content\n"` transcript, NOT a chat template** (95% vs 11% role accuracy).
- Read the **penultimate-token hidden state**; monkeypatch an **early exit** (no LM-head, no decode).
- Train with **plain isotropic ES** (sep-CMA collapses to it at this scale) on labeled routes; optional soft-KL SFT first. Use OpenFugu's training scaffold as reference.
Keep this OUT of v0; just don't paint over the seam.

---

## 14. Milestones (so we can start)
- **M0 — Walking skeleton (drop-in passthrough).** Hono server, `/v1/chat/completions` + `/v1/models`, one adapter (OpenRouter), passthrough mode, streaming, cost accounting, trace. *Demo: point OpenAI SDK at Maestro, get an answer + cost.* ← ship-able alone.
- **M1 — The router + verify loop.** `classify` (heuristic), `route` (tiers + guardrail + ladder), the ≤K-turn verify/escalate loop, `maestro-auto`/`-fugu`, `/v1/route` dry-run, transparency block. Add Vercel-gateway adapter (+ Fugu as a worker).
- **M2 — The eval harness.** Fixtures (incl. mixed-capability traps), baselines, metrics, `maestro eval` → the README table. This is the credibility.
- **M3 — Polish for launch.** README + demo GIF, `docker run`/`npx`, config docs, registry-staleness check, examples, MIT, "not affiliated with Sakana" note. Delete the dead pre-pivot Python scaffold.
- **M4 — Launch (§15).**
- **Post-v0:** `ultra`/Conductor decompose, executable code verifier, learned router (v1), optional dashboard, semantic cache.

---

## 15. Launch plan for day-one traction
- **README (top to bottom):** hero tagline → 20-sec **GIF** (same code, swap base_url → routed answer with cost shown) → "Why Maestro" (Fugu = closed/EU-blocked/opaque/3-models; Maestro = open/any-model/transparent/self-host) → **Quickstart** (one `docker run`, one key, point your SDK) → **honest benchmark table** with `reproduce: maestro eval` → architecture diagram → roadmap → disclaimer.
- **The demo is the product.** A terminal GIF: cheap model tries, verifier rejects, escalates to frontier, ACCEPT — with the cost line "$0.018 (vs $0.024 frontier-only)". That single GIF is what gets shared.
- **Distribution (the user's "share in comment"):** Show HN (*"Show HN: Maestro — open-source Fugu, route any LLMs behind one endpoint"*), r/LocalLLaMA, X thread (reply into Sakana/Fugu/Fusion discussions), Hacker News + dev.to writeup. Post where Fugu/Fusion are *already being discussed*.
- **Star-earning hooks:** drop-in OpenAI compat (zero rewrite), open AND closed models, full cost transparency, one-command try, and **radical honesty** ("here's when routing helps and when a single model wins — measured"). Engineers star honesty.
- **Name check first:** confirm `maestro` is free on npm/GitHub (it's common — have a scoped fallback like `@<org>/maestro` or a distinct name ready) before announcing.

---

## 16. Risks & honest caveats
- **"It's just a wrapper."** → Rebuttal built in: the verify/escalate loop, the eval/regret proof, full transparency, and the learned-router roadmap. Lead with the eval.
- **Day-one 4K stars is ambitious.** Code is necessary, not sufficient; demo + timing + distribution dominate. Plan §15 as seriously as the code.
- **Gateway ToS / reselling.** We don't resell; users BYOK. Keep keys client/self-hosted; document it.
- **Benchmark backlash.** Only publish numbers `maestro eval` reproduces; show the losses too (the critique's point — honesty is the moat).
- **Name collision / EU claim.** Make region-filtering a *feature flag*, not a blanket "EU-clean" claim (critique). Verify the name.
- **Leaderboard staleness.** Registry entries are dated; `registry check` nags. Re-pull before each release.
```
