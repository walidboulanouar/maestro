# Maestro Roadmap

The plan for turning Maestro from a runnable v0.1 into the default open-source LLM routing layer. This is the canonical roadmap; the README carries a short summary that links here.

> Status today: **v0.1 (early, ~5-hour build).** The core works and is tested live on real models. Not production-hardened. Everything below is what gets it there.

> ### Shipped on the `roadmap` branch (built + tested)
> Retries + transient fallback, per-request timeouts, **auth + rate-limit + budget**, trace redaction, latency skip-verify, **multi-provider** (Groq/Together/Fireworks/DeepInfra/OpenAI/custom), **orchestration profiles** (cheap/balanced/quality), **dedupe cache**, **trace-viewer UI** (`/ui`), `maestro init`, **prompt/version registry**, **`maestro-ultra`** decomposition, **executable code verifier**, **full Anthropic `tool_use` mapping**, **`npm run eval --report`** + rule-only baseline, and the **npm/ghcr release workflow**. Still needing a paid run / 2FA: the long-running agentic eval, true incremental streaming tool deltas, and the npm publish.

## North star

> One self-hostable, OpenAI- and Anthropic-compatible endpoint that routes every request to the right model, cheap-first with verify and escalate, and shows you the exact route and cost. Drop-in for any agent harness. Open, honest, no GPU.

## Principles (what guides every decision)

1. **A model endpoint, not an agent runtime.** Your harness owns the tool loop. We route and preserve the protocol. We never execute tools.
2. **Transparent by default.** Every response shows route, models, tokens, and cost. No hidden behavior.
3. **Honesty is the moat.** No claim ships without a command that reproduces it. We show failures, not just wins.
4. **Wire-compatible.** Forward every request field; preserve every upstream response field. Do not break existing OpenAI/OpenRouter clients.
5. **No lock-in.** Bring your own key, swap models in a JSON file, self-host anywhere, MIT.
6. **Small surface.** Reuse what is solved (gateways, inference). Build only the routing brain.

## How to read this

Each milestone has a **goal**, **items** (checkboxes), and **done when** (acceptance criteria). Items are roughly in priority order. Help wanted on anything; see [Contributing](#contributing).

---

## v0.1 - Foundation (DONE)

Goal: a runnable, honest, wire-compatible router.

- [x] OpenAI-compatible API: `/v1/chat/completions` (+stream), `/v1/models`, `/v1/route`, `/v1/traces/:id`, `/healthz`
- [x] Anthropic-compatible `/v1/messages` (Claude Code)
- [x] Two paths: concrete-model passthrough + `maestro-*` routed
- [x] classify -> route (tier + guardrail + escalation ladder) -> verify/escalate loop
- [x] Transparent tool-calling pass-through (no verify on tool turns)
- [x] Full request pass-through (`extra`) + upstream response preservation
- [x] OpenRouter / Vercel / local / mock adapters, BYO key, per-request timeouts
- [x] Cost-aware routing + `maestro` transparency block + offline reproducible eval (43 tests)

---

## v0.2 - Hardening + ship (NEXT)

Goal: trustworthy under real traffic, and installable by anyone.

- [ ] **Ship it:** publish `openmaestro` to npm, push a `ghcr.io` Docker image, tag `v0.1.0` release
- [ ] **Retries + provider fallback** on 429 / 5xx / timeout, with backoff and a circuit breaker
- [ ] **Streaming tool calls:** real incremental indexed `delta.tool_calls` (currently best-effort)
- [ ] **Anthropic `tool_use`** full block mapping for `/v1/messages` (so Claude Code tool use is first-class)
- [ ] **Trace redaction:** scrub tool arguments / PII before storing; opt-in SQLite persistence
- [ ] **Auth + limits:** API keys, per-key rate limits, per-org/user budget caps
- [ ] **Latency:** run the verifier asynchronously, or skip it when the classifier is highly confident
- [ ] **Demo asset:** a 20s GIF (base_url swap -> routed answer + cost) in the README

Done when: a hostile load test passes (no hangs, graceful 429/5xx/timeout handling), `npx openmaestro` and `docker run` both work for a stranger, and tool-calling is verified against the OpenAI and Vercel AI SDKs.

## v0.3 - DX + adoption

Goal: pleasant to configure, easy to trust, easy to tune.

- [ ] **Orchestration profiles:** `cheap` / `balanced` / `quality` presets (one flag changes thresholds + pool)
- [ ] **Trace-viewer UI:** a tiny local page to inspect routes, costs, and verdicts over time
- [ ] **Semantic cache:** skip the model call on near-duplicate requests (configurable, off by default)
- [ ] **Prompt / version registry:** named, versioned system prompts and routing configs
- [ ] **Config UX:** `maestro init`, config validation, clearer registry-staleness warnings
- [ ] **More providers:** first-class Groq / Together / Fireworks / DeepInfra adapters (beyond OpenRouter-first)
- [ ] **Per-task best-model presets** across ~40 curated models (best model per task type, dated)

Done when: a new user goes from install to a tuned, cost-profiled deployment in minutes, and can see why each request routed the way it did.

## v1.0 - Smarter routing (proven)

Goal: routing that is measurably better than naive baselines on real, hard work.

- [ ] **Executable verifier** for code/tools: run the tests instead of asking an LLM (huge reliability win)
- [ ] **Embedding router:** semantic task classification beyond heuristics, with calibration
- [ ] **Calibrated confidence:** gate escalation on a measured confidence signal (track ECE/Brier)
- [ ] **Large, long-running, tool-calling agent eval:** success = task completion via tools, not text match (the real validation; budget a paid run)
- [ ] **Public benchmark report:** reproducible numbers across coding / math / tool / long-context, with ablations and the losses shown

Done when: on a public agentic benchmark, Maestro matches best-single quality within a confidence interval at materially lower cost and bounded p95 latency, reproducible by one command.

> Note: a learned (TRINITY-style, GPU-trained) router is intentionally **out of scope**. Maestro stays a no-GPU routing layer; the heuristic classifier plus the verify/escalate loop is the routing brain. The bar for "smarter routing" is raised with better heuristics and the eval, not a trained model.

## v3.0 - Ultra (multi-step)

Goal: Conductor-style decomposition for the hardest tasks, behind a quality gate.

- [ ] **`maestro-ultra`:** decompose a hard request into a small DAG of subtasks with shared memory (<= 5 steps)
- [ ] **Bounded recursion** + cost ceiling
- [ ] **Quality gate:** only engage Ultra when an eval shows it beats single-pass routing for that task class

Done when: on tasks where decomposition helps, Ultra improves quality-per-dollar over `maestro-auto`, proven by eval, and never silently costs more elsewhere.

---

## Cross-cutting tracks (continuous)

- **Evals:** every routing change is judged by the eval; grow the fixture set toward real traffic; never publish an unreproducible number.
- **Security:** secret hygiene, trace redaction, dependency pinning, SBOM, plugin/tool safety, region/data-policy controls.
- **Observability:** OpenTelemetry traces, Prometheus metrics, structured logs.
- **Docs + community:** keep README honest and current, integration guides per harness, examples, good-first-issues.
- **Integrations:** verified recipes for Claude Code, opencode, Cursor, Continue, OpenAI SDK, Vercel AI SDK, LangChain.

## Success metrics (how we know it is working)

- Cost-per-successful-task vs best-single (target: large, reproducible reduction)
- Quality within a confidence interval of best-single on public evals
- p95 latency overhead from routing (target: small; verifier async)
- Adapter/harness compatibility (no breakage across the supported clients)
- Adoption: stars, npm installs, external integrations, contributors

## Non-goals (explicitly out of scope)

- Running model inference / hosting GPUs
- Becoming an agent framework or owning the tool-execution loop
- A closed or hosted-only product (Maestro stays open and self-hostable)
- Unreproducible benchmark claims

## Contributing

Pick any unchecked item. Good first issues: a new provider adapter, eval fixtures (especially mixed-capability and tool-use traps), the trace-viewer UI, or the executable code verifier. See [CONTRIBUTING.md](CONTRIBUTING.md). Open an issue to claim an item so we do not duplicate work.

> This roadmap is a living document. Dates are intentionally omitted; priorities shift with what the eval and real usage reveal.
