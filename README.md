<div align="center">

# 🎼 Maestro

### The open-source orchestration brain for LLMs.

**Route any models (open and closed) behind one OpenAI-compatible endpoint.
Cheap-first, verify, escalate. Full cost and route transparency. Self-hostable. MIT.**

### 🐡 Maestro is the open-source version of [Sakana's Fugu](https://sakana.ai/fugu-release/), built from the two papers: [TRINITY](papers/) (2512.04695) and [Conductor](papers/) (2512.04388).

*Open-source Fugu: open, honest, EU-clean, runs anywhere, and not locked to three closed models.*

> **The one thing no other open-source project does:** Maestro is the only open-source LLM router you can run with **zero setup, no GPU, no keys** (`npx openmaestro serve`) that is **both OpenAI- and Anthropic-compatible** (works in Claude Code and opencode) with **per-request cost transparency** and a **reproducible benchmark**. OpenFugu needs a GPU and ships a *mock* benchmark; LoRA-Harness ships none. See [COMPARISON.md](COMPARISON.md).

> ⚠️ **Status: early. This is an honest ~5-hour build (v0.1).** The core works and is tested live on real models (routing, verify/escalate, tool/agent loops, OpenAI + Anthropic APIs, streaming, transparency). It is NOT production-hardened yet, and the learned router is not built. Treat it as a strong, runnable foundation, not a finished product. See the [Roadmap](#roadmap) for what is next and [VERIFICATION.md](VERIFICATION.md) for exactly what is proven vs assumed.

[![license](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![node](https://img.shields.io/badge/node-%E2%89%A520-3c873a.svg)](package.json)
[![tests](https://img.shields.io/badge/tests-43%20passing-brightgreen.svg)](test)
[![status](https://img.shields.io/badge/status-v0.1%20early-orange.svg)](#roadmap)
[![site](https://img.shields.io/badge/site-maestro.ayautomate.com-2b6fff.svg)](https://maestro.ayautomate.com)
[![npm](https://img.shields.io/badge/npm-openmaestro-cb3837.svg)](https://www.npmjs.com/package/openmaestro)

</div>

---

Modern LLM stacks have a problem: the best model for a one-line translation is not the best model for a distributed-systems design, but you pay frontier prices for both. Routers exist (Sakana's Fugu is the famous one), but they're closed, opaque about cost, region-locked, and limited to a fixed pool.

**Maestro is just the routing brain.** It doesn't host models or run a GPU. It sits in front of an AI gateway (OpenRouter, the Vercel AI Gateway, or your local Ollama/vLLM) and, for every request, decides *which model, at what effort, in how many turns*: it tries a cheap model first, **verifies** the answer, and **escalates** to a stronger one only when needed. Then it tells you exactly what it did.

```jsonc
// every response carries a `maestro` block: this is the whole point
"maestro": {
  "route": [
    { "turn": 1, "model": "deepseek/deepseek-v4-pro",  "verdict": "REVISE" },
    { "turn": 2, "model": "anthropic/claude-opus-4.8", "verdict": "ACCEPT" }
  ],
  "classify": { "task": "code", "difficulty": 0.78, "confidence": 0.85 },
  "cost_usd": 0.0182,
  "cost_vs_frontier_only_usd": 0.0241,
  "savings_pct": 24
}
```

## 30-second quickstart

Maestro runs with **zero API keys** out of the box (built-in mock provider), so you can see it work instantly:

```bash
npx openmaestro serve            # or: docker run -p 8080:8080 ghcr.io/youruser/maestro
```

```bash
curl -s localhost:8080/v1/chat/completions \
  -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"Translate good morning to Spanish"}]}' \
  | jq .maestro
```

Add a real key and it routes to real models. Your existing OpenAI code doesn't change, just the base URL:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="unused")

client.chat.completions.create(
    model="maestro-auto",                       # or maestro-fugu, or a real model id (passthrough)
    messages=[{"role": "user", "content": "Design a multi-region rate limiter."}],
)
```

```bash
export OPENROUTER_API_KEY=sk-or-...     # open + closed models, BYOK
# or AI_GATEWAY_API_KEY=...             # Vercel AI Gateway (0% markup, also serves Fugu)
# or LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama / vLLM / llama.cpp, 100% offline
```

Want the full demo with **real prices but no spend**? `MAESTRO_FORCE_MOCK=true npx openmaestro serve` routes over the priced registry and executes on the mock provider.

## Use it in your coding agent

Maestro speaks **both** the OpenAI and the Anthropic wire formats, so it drops into the popular agents. Your harness keeps its own tool loop; Maestro just picks the model behind it.

**Claude Code** (Maestro exposes `/v1/messages`):
```bash
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_API_KEY=unused claude
```

**opencode / Cursor / Continue / any OpenAI-compatible tool**: point the base URL at Maestro and use model `maestro-auto`:
```jsonc
// opencode.json
{ "provider": { "maestro": { "npm": "@ai-sdk/openai-compatible",
  "options": { "baseURL": "http://localhost:8080/v1" },
  "models": { "maestro-auto": {} } } } }
```

**Tool calls pass straight through.** Maestro is a transparent proxy (like OpenRouter): it decides *which model* handles each call and returns the model's output (text or `tool_calls`) verbatim. Your app executes the tools and calls Maestro again with the result. Full setup for opencode / Claude Code / Cursor / Continue: **[docs/INTEGRATIONS.md](docs/INTEGRATIONS.md)**.

## Swap in your own models

The model pool is **100% yours**: it's a dated JSON registry of `ModelSpec`s. Copy [`maestro.config.example.json`](maestro.config.example.json), edit it, and run with `MAESTRO_REGISTRY=./maestro.config.json`. Slot labels are abstract and remappable (the key lesson from Fugu): the router reasons over slots/tiers, and you map them to any model id on any provider with no retraining and no code change. Mix open, closed, and your local Ollama models freely. The defaults track the current frontier and are dated, so `maestro registry check` warns when they go stale.

## Why Maestro

| | **Maestro** | Sakana Fugu | Raw gateway (OpenRouter/Vercel) |
|---|:--:|:--:|:--:|
| Open source / self-host | ✅ MIT | ❌ closed API | ⚠️ hosted |
| Model pool | **any** (open + closed, BYOK) | 3 closed models | any (but no routing) |
| Routing brain | ✅ cheap then verify then escalate | ✅ learned | ❌ you pick |
| Cost transparency | ✅ per-request, per-model | ❌ undisclosed | ⚠️ totals only |
| Refuses to overspend | ✅ confidence/verify gate | ⚠️ | ❌ |
| Run 100% locally | ✅ local-openai adapter | ❌ | ❌ |
| EU / region control | ✅ region filter | ❌ EU-blocked | ⚠️ |
| Drop-in OpenAI API | ✅ | ✅ | ✅ |

## How it works

```mermaid
flowchart LR
  A[OpenAI request<br/>model: maestro-auto] --> B[classify<br/>task / difficulty / caps]
  B --> C[route<br/>filter then tier then guardrail score<br/>then escalation ladder]
  C --> D[execute worker<br/>via gateway / local]
  D --> E{verify}
  E -- ACCEPT --> F[answer + maestro block]
  E -- REVISE and budget left --> G[escalate up] --> D
  C -. reads .-> R[(dated model registry<br/>slots to model ids)]
  D -. logs .-> T[(trace + cost)]
```

1. **Classify**: a fast, zero-cost heuristic tags the task (code/math/reasoning/...), estimates difficulty, and required capabilities.
2. **Route**: filter the registry by capability/policy/region/budget, pick a starting tier by difficulty, rank with a guardrail score, build an **escalation ladder** of `(model, effort)`.
3. **Execute**: call the chosen model through the provider adapter (streaming supported).
4. **Verify**: a verifier judges the answer. `ACCEPT` stops; `REVISE` escalates to the next rung (bounded by `maxTurns`). For code/tools, a real executable check is on the roadmap.
5. **Report**: the answer ships with the full route, per-model tokens, cost, and a "what frontier-only would have cost" comparison.

> We make no dishonest claims. Maestro does *not* promise it always beats a single frontier model. It promises to try cheaper first, verify, escalate when needed, and **show you the receipts**.

## Benchmarks

Reproduce locally with `npm run eval` (runs **offline**, deterministic, free; routes over the *priced* registry while executing on the mock provider, graded against ground-truth difficulty with oracle-route regret):

```
strategy             pass%      mean $      pass/$    regret $   fails
----------------------------------------------------------------------
maestro                92%     0.00053      1747.9     0.00035       2
best-single           100%     0.01507        66.3     0.01421       0
cheapest-single        56%     0.00016      3566.0     0.00000      11
random-route           88%     0.00689        127.7     0.00705       3

calibration (classifier confidence vs first-rung-correct):
  Brier = 0.181   ECE = 0.110   (lower is better)
```

**Read:** Maestro reaches **92%** of best-single quality at **97% lower mean cost** (~26x more successful answers per dollar, 1748 vs 66), and beats a **random** router on *both* quality (+4 pts) and cost (**13x cheaper**). `cheapest-single` is cheaper per call but fails 44% of tasks; Maestro is the best *balance*. The gap to best-single and the 2 failures are *real* (the heuristic classifier sometimes mis-estimates); we show them on purpose. This is a routing benchmark on a realistic-mix fixture set, not a leaderboard claim.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/chat/completions` | OpenAI-compatible (+ `stream`). `model` selects the mode. |
| `POST` | `/v1/messages` | Anthropic-compatible (+ stream), drop-in for Claude Code. |
| `GET` | `/v1/models` | `maestro-auto/fugu/ultra` + every registry model. |
| `POST` | `/v1/route` | **Dry-run** the router: see the decision, no model call. |
| `GET` | `/v1/traces/:id` | Inspect a past request's full trace. |
| `GET` | `/healthz` | Liveness + which providers are configured. |

**Modes** (the `model` field): `maestro-auto` (classify + route + verify), `maestro-fugu` (single worker + verify, low-latency), `maestro-ultra` (multi-step decompose, *roadmap*, falls back to fugu), or any real model id for **passthrough** (no routing, always a safe drop-in).

## Configuration

All via env (zero-config defaults shown):

| Variable | Default | Meaning |
|---|---|---|
| `MAESTRO_PORT` | `8080` | server port |
| `MAESTRO_DEFAULT_MODE` | `fugu` | mode when `model` is `maestro` |
| `MAESTRO_MAX_TURNS` | `3` | max escalation turns |
| `MAESTRO_DIFFICULTY_LOW` / `_HIGH` | `0.33` / `0.7` | tier thresholds |
| `MAESTRO_VERIFY` | `true` | enable the verify/escalate loop |
| `MAESTRO_REQUEST_TIMEOUT_MS` | `120000` | per-request upstream timeout (no hangs) |
| `MAESTRO_REGISTRY` | *(built-in)* | path to your own model registry JSON |
| `OPENROUTER_API_KEY` / `AI_GATEWAY_API_KEY` / `LOCAL_OPENAI_BASE_URL` | - | providers (BYOK) |
| `MAESTRO_FORCE_MOCK` | `false` | demo: route priced registry, execute mock |
| `MAESTRO_TRACE_FILE` | - | append every trace as JSONL |

Per-request overrides via the `maestro` field: `{ "budget": 0.02, "maxTurns": 2, "verify": true, "region": "eu", "pin": "anthropic/claude-opus-4.8" }`.

## CLI

```bash
maestro serve [--port N]      # start the OpenAI-compatible server
maestro route "<prompt>"      # dry-run the router (no model call)
maestro registry check        # report registry staleness
```

## Roadmap

This is v0.1, a ~5-hour build. It works and is tested, but there is a lot of headroom. Contributions welcome.

**Done (v0.1)**
- [x] OpenAI-compatible endpoint (`/v1/chat/completions`, `/v1/models`, `/v1/route`, `/v1/traces/:id`, `/healthz`)
- [x] Anthropic-compatible `/v1/messages` (works in Claude Code)
- [x] Two clean paths: concrete-model passthrough + `maestro-*` routed
- [x] Classify, route (tier + guardrail + escalation ladder), verify/escalate loop
- [x] Transparent tool-calling pass-through (agent loops keep their own loop)
- [x] Full request pass-through (`response_format`, `provider`, `seed`, `session_id`, `metadata`, `trace`, `reasoning`, `plugins`, ...) + upstream response preservation
- [x] OpenRouter / Vercel / local / mock adapters, BYO key, per-request timeouts
- [x] Cost-aware routing + transparency block + offline reproducible eval (43 tests)

**Next (v0.2, hardening)**
- [ ] Incremental indexed streaming tool-call deltas (currently best-effort)
- [ ] Full Anthropic `tool_use` block mapping for `/v1/messages`
- [ ] Retries + provider fallback on 429/5xx; circuit breaking
- [ ] Trace redaction (tool arguments / PII) + opt-in persistence (SQLite)
- [ ] Auth (API keys), rate limits, per-org budgets
- [ ] Latency: run the verifier async, or skip it when the classifier is highly confident

**Later (v1, smarter)**
- [ ] Executable verifier for code/tools (run the tests, do not ask an LLM)
- [ ] Semantic cache; prompt/version registry; a tiny trace-viewer UI
- [ ] Orchestration **profiles** (cheap / balanced / quality) + per-task best-model presets across ~40 models
- [ ] Large, long-running, tool-calling agent eval (success = task completion, not text match)

**Research (v2-v3)**
- [ ] Learned **TRINITY-style router** (frozen Qwen3-0.6B + tiny head, CPU-only, trained offline) as a drop-in for the heuristic classifier
- [ ] `maestro-ultra`: multi-step decomposition (Conductor-style) behind a quality gate
- [ ] Custom direct providers (Groq/Together/Fireworks/...) beyond the OpenRouter-first default

## FAQ

**Isn't this just a wrapper?** The wrapper part (gateways, OpenAI serving) is *deliberately* not reinvented; that is solved. The value is the **routing policy + verify/escalate loop + transparency + a reproducible eval**. The roadmap's learned router is where it stops being "just a wrapper".

**Does it really save money?** On the bundled realistic-mix eval: ~97% cheaper than always-frontier (~26x better cost-per-success), and it beats random routing on both quality and cost. Your mileage depends on your traffic mix and registry, which is why the eval is in the repo and the cost is on every response.

**Is it better than \<top single model\>?** Maestro isn't a model, it's the router. It *uses* the best models (including frontier ones) and reaches frontier-level results at a fraction of the cost by sending only the hard requests to the expensive models. The honest claim is "frontier-quality routing, open-source, far cheaper". Reproduce it yourself with `npm run eval` (offline) or `bash scripts/verify.sh` with your own keys.

**Can I run it fully offline?** Yes. Set `LOCAL_OPENAI_BASE_URL` to your Ollama/vLLM/llama.cpp server. Maestro never hosts a model itself.

## Acknowledgements

Maestro is an **open-source implementation of the ideas in [Sakana's Fugu](https://sakana.ai/fugu-release/)**, described in two papers included in [`papers/`](papers): **TRINITY** ([arXiv 2512.04695](papers/TRINITY-fugu-2512.04695v3.pdf), the route-then-verify loop) and **Conductor** ([arXiv 2512.04388](papers/Conductor-fugu-ultra-2512.04388v5.pdf), multi-step decomposition). The cheap-then-escalate pattern descends from [FrugalGPT](https://arxiv.org/abs/2305.05176); community reverse-engineering: [OpenFugu](https://github.com/trotsky1997/OpenFugu). Full design notes and our analysis (with corrections against Sakana's official report) are in [`docs/`](docs).

**Maestro is not affiliated with Sakana AI.** "Fugu" is Sakana AI's product/research; Maestro contains no Sakana code or weights. It is an independent open-source build from the public papers.

## License

[MIT](LICENSE).
