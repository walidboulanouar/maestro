# OpenRouter Fusion & AI Gateways — backend decision for Maestro

> Live research, June 2026. Every claim cites URL + access date. Items flagged **[UNCERTAIN]** are not directly confirmed from a primary source. Vendor benchmark numbers are self-reported unless noted; treat them as marketing, not ground truth.
>
> **Bottom line up front:** Build Maestro's provider layer on **Vercel AI Gateway as the primary backend** (native to the Vercel AI SDK we already plan to use, zero inference markup, and it already exposes **`sakana/fugu-ultra`** as a callable model — so we get Fugu itself as a worker), with **OpenRouter as a secondary/fallback backend** (widest catalog, 400+ models, but a 5.5% credit fee). Both are OpenAI-compatible, so support both behind our `providers/` registry. **Do not** build Maestro *as* OpenRouter Fusion — Fusion is a fixed panel+judge ensemble with no tool execution and no cost gate; that is exactly the layer Maestro replaces.

---

## 1) What OpenRouter Fusion actually is

**Fusion is a panel + judge ensemble synthesizer, not a per-query router.** It runs your prompt across multiple models in parallel, has a judge model produce a structured cross-model analysis, then your outer model writes the final answer.

Mechanism (3 stages), verbatim from the docs:
1. **Panel stage** — "Up to 8 models process your prompt in parallel," each with access to `openrouter:web_search` and `openrouter:web_fetch`.
2. **Judge stage** — the judge "doesn't merge them. It returns structured analysis as JSON: what all or most models agreed on (treated as higher-confidence consensus), where they disagreed, what only some models covered, unique insights from individual models, and blind spots none of them addressed."
3. **Synthesis** — "Your outer model receives the judge's structured analysis and produces the final answer."
— https://openrouter.ai/docs/guides/routing/routers/fusion-router (accessed 2026-06-22)

- **Open or proprietary?** Proprietary. "The Fusion Router is proprietary to OpenRouter—no open-source claims appear in the documentation." Recursion is blocked (panel/judge models can't re-invoke fusion). — same doc, 2026-06-22
- **API shape:** OpenAI-compatible. Two ways to invoke: model alias `{"model":"openrouter/fusion"}`, or server-tool `{"tools":[{"type":"openrouter:fusion"}]}`. Config via `plugins`/`tools`: `analysis_models` (panel, 1–8, default a quality preset of Claude Opus / GPT-latest / Gemini Pro), judge = your outer `model`, `max_tool_calls` (default 8, range 1–16), plus `temperature`/`reasoning`. — same doc, 2026-06-22
- **Models covered:** any models in OpenRouter's catalog can be panel/judge members (defaults Claude Opus, GPT-latest, Gemini Pro). — same doc, 2026-06-22
- **Pricing:** "With the default 3-model panel, expect roughly 4–5× the cost of a single completion" (N panel calls + 1 judge call; scales linearly with panel size), and latency "often 2–3× longer than a standard call." — same doc + https://openrouter.ai/blog/announcements/fusion-beats-frontier/ (launched **2026-06-12**, FAQ updated 6/14; accessed 2026-06-22)
- **Benchmark claims (DRACO Deep Research, self-reported):** Fable 5 + GPT-5.5 fused = 69.0% (vs Fable 5 solo 65.3%); a budget panel (Gemini 3 Flash + Kimi K2.6 + DeepSeek V4 Pro) = 64.7%, beating GPT-5.5 (60.0%) and Opus 4.8 (58.8%); Opus-paired-with-itself = 65.5% (+6.7 over solo). — https://openrouter.ai/blog/announcements/fusion-beats-frontier/ (accessed 2026-06-22). **[Treat as vendor marketing.]**

### The tweet claim: "Sakana Fugu is reselling OpenRouter's Fusion idea" — assessment

Both are "many models behind one call, exposed as one answer," so the one-liner is *directionally* fair. But the mechanisms are different:

| | **OpenRouter Fusion** | **Sakana Fugu / Fugu Ultra** |
|---|---|---|
| Core mechanism | **Ensemble**: fan-out to a fixed panel (up to 8) **every time**, judge → synthesize | **Learned router/orchestrator**: routes to **1–3 agents depending on the problem**; can answer directly or delegate; recursion |
| "Brain" | A judge prompt producing structured JSON | A *trained* model (TRINITY evolved router + Conductor RL workflow author; ICLR 2026 papers) |
| Always multi-model? | Yes (panel always runs) | No — routes per request; may use one worker |
| Cost profile | ~4–5× a single call (panel + judge) by design | Variable; "Fugu" tier is latency-oriented/cheap |
| Who picks models | **You** (you configure the panel) | **The system learns** the policy |
| Open? | Proprietary | Proprietary |

Sources: Fusion docs (2026-06-22); https://sakana.ai/fugu/ and https://vercel.com/changelog/sakana-fugu-ultra-now-available-on-ai-gateway (2026-06-22); Fugu mechanism "routing work to 1-3 agents depending on the problem and combining their results" — Vercel changelog (2026-06-22).

**Genuinely different:** Fusion is a *static ensemble you assemble* (always pay N×); Fugu is a *learned policy that decides how many models to use*. Fusion is closer to Maestro's **ensemble/debate strategy (#8/#12 in MAESTRO-DESIGN §9)** than to Fugu's **learned router (#5)**. Neither has a confidence/cost gate that *refuses to orchestrate when one model wins* — which is Maestro's differentiator. So "reselling the idea" overstates it: same goal, different machinery (ensemble vs learned router).

---

## 2) OpenRouter as a backend

- **One OpenAI-compatible endpoint, 400+ models across 70+ providers**; "most SDKs work by just swapping the base URL." — https://openrouter.ai/docs/guides/overview/models, https://openrouter.ai/ (accessed 2026-06-22)
- **Auto Router (`openrouter/auto`)**: per-prompt model selection "powered by NotDiamond"; pins selected model+provider for cache consistency; restrictable via `plugins`; "You pay the standard rate for whichever model is selected, with no additional fee for using the Auto Router." — https://openrouter.ai/docs/guides/routing/routers/auto-router (accessed 2026-06-22)
- **Provider routing / fallbacks / preferences**: routes across multiple upstream providers per model with fallbacks and cost/latency sorting (provider-preferences). — https://openrouter.ai/blog/insights/model-routing/ (accessed 2026-06-22). *(Note: the canonical `/docs/features/provider-routing` URL 404'd on 2026-06-22; behavior confirmed via the routing blog + overview.)*
- **Pricing / markup:** **no markup on model inference** (pass-through provider rates), but a **5.5% credit-card platform fee** ($0.80 minimum, so small top-ups effectively pay 10–20%; crypto 5%). **BYOK:** first 1M BYOK requests/month free, then **5% of the equivalent OpenRouter cost** per request. Budget ~5–7% overhead. — https://openrouter.ai/pricing, https://openrouter.ai/docs/faq (accessed 2026-06-22)
- **Is Sakana Fugu on OpenRouter?** **[UNCERTAIN — likely no as of 2026-06-22].** Multiple searches of the OpenRouter catalog did not surface a `sakana/*` / `fugu` model; Fugu is sold via Sakana's own OpenAI-compatible API (`https://api.sakana.ai/v1`) and is confirmed on **Vercel AI Gateway**, not OpenRouter. Re-check `openrouter.ai/models` at build time. — searches + https://console.sakana.ai/models, https://sakana.ai/fugu/ (accessed 2026-06-22)

**As a Maestro backend:** widest model coverage and a built-in `auto` router and fallbacks; OpenAI-compatible so trivial to plug into `providers/`. Downsides: the 5.5% credit fee (real cost on top of inference), it's a proprietary SaaS dependency, and Fugu-as-a-worker isn't (yet) available there.

---

## 3) Vercel AI Gateway as a backend

- **Unified API to hundreds of models** (text/image/video) with one key + dashboard; "automatic fallbacks during provider outages," retries/failover, budgets per API key, custom reporting, Zero Data Retention. — https://vercel.com/ai-gateway, https://vercel.com/docs/ai-gateway (accessed 2026-06-22)
- **OpenAI-compatible:** supports "OpenAI Chat Completions" and "OpenAI Responses." — https://vercel.com/ai-gateway (accessed 2026-06-22)
- **Native to the Vercel AI SDK:** you literally pass the model id as a string — `streamText({ model: 'sakana/fugu-ultra', prompt: ... })`. No extra provider wiring. — https://vercel.com/changelog/sakana-fugu-ultra-now-available-on-ai-gateway (accessed 2026-06-22)
- **Pricing / markup:** **no markup, list price** — "pay exactly what providers charge with no platform fees"; **0% markup including on BYOK**; no platform fee on inference. Vercel monetizes routing/failover/observability/unified-billing + value-adds (Observability Plus; team-wide ZDR $0.10/1k requests). Note payment-processing fees still apply (a $100 credit top-up showed a ~3.2% Stripe fee). — https://vercel.com/ai-gateway, https://vercel.com/docs/ai-gateway/pricing, https://vercel.com/docs/ai-gateway/authentication-and-byok/byok (accessed 2026-06-22)
- **Sakana Fugu Ultra IS available** — model id **`sakana/fugu-ultra`**, published **2026-06-22**. "Fugu Ultra is built on a pool of publicly accessible frontier models… It coordinates several models, routing work to 1-3 agents depending on the problem and combining their results," with "capabilities similar to those of Claude Mythos Preview and Fable 5." A playground exists at `vercel.com/ai-gateway/models/fugu-ultra/playground`. — https://vercel.com/changelog/sakana-fugu-ultra-now-available-on-ai-gateway (accessed 2026-06-22). **[Plain `fugu` / `fugu-mini` on the Gateway: UNCERTAIN — only `fugu-ultra` confirmed.]**

**As a Maestro backend:** the cleanest fit. We already plan to use the **Vercel AI SDK** for provider calls/streaming (MAESTRO-DESIGN §11). Gateway = the SDK's native multi-provider plane with **zero inference markup** and **Fugu itself as one callable worker** — meaning Maestro can route to Fugu Ultra as just another model in the pool (e.g., a hard-tier worker, or a baseline to benchmark Maestro against). Downsides: fewer total models than OpenRouter's 400+ **[verify exact count at build time]**, and it ties part of the stack to Vercel's platform.

---

## 4) Comparison: Fusion vs Fugu vs Maestro

| Dimension | OpenRouter Fusion | Sakana Fugu (Ultra) | **Maestro** |
|---|---|---|---|
| What it is | Fixed panel + judge **ensemble** synthesizer | **Learned** router/orchestrator over a model pool | Open orchestrator: cascade + multi-agent + **tool execution** |
| Model selection | You assemble the panel; always fan out | Learned policy; **1–3 agents per problem** | Pluggable strategies, **confidence/cost gate** chooses depth |
| Always multi-model? | Yes (≈4–5× cost) | No | **No — single model when it wins** (anti-Fugu-loss design) |
| Tool execution | Panel gets web_search/web_fetch only | **Plans tools, cannot execute** (TRINITY limit) | **Executes tools** (sandboxed via coset/probe) |
| Cost/latency gate | None (panel always runs) | None exposed | **Budgets + cascade + confidence gate** |
| Verifier/critic | Judge = analysis JSON, no exec/verify | Verifier role (ACCEPT loop) | Planner/Executor/Verifier/Critic/Synthesizer |
| Transparency/evals | Benchmark blog (vendor) | Vendor benchmarks, closed | **First-class eval harness, baselines, protocol_hash** |
| Open / self-host | Proprietary SaaS | Proprietary SaaS, EU-blocked | **Open, self-hostable, EU-ok, BYO-keys** |
| API | OpenAI-compatible | OpenAI-compatible | OpenAI-compatible + control APIs |

Sources: as cited in §1–§3, all accessed 2026-06-22; Maestro column from `docs/MAESTRO-DESIGN.md`.

**Where they sit relative to Maestro:** Fusion ≈ Maestro's *ensemble/debate* strategy as a managed product. Fugu ≈ Maestro's *learned-router* strategy as a managed product. Maestro is the **superset that gates between them** (and single-model, and cascade) on confidence/cost, **executes tools**, and is **open + measured**. Crucially, both Fusion and Fugu are reachable *from inside* Maestro as backend models/workers — they are commoditized inputs, not competitors at the orchestration layer.

---

## 5) Recommendation for Maestro's provider layer

**Yes — Maestro's provider layer should BE these gateways, abstracted behind our own `providers/` registry. Do not reinvent the transport plane.**

**Primary: Vercel AI Gateway.**
- Native to the Vercel AI SDK we already chose (string model id, no wiring) — lowest integration cost.
- **0% inference markup, including BYOK** — best economics; users keep their own keys.
- **`sakana/fugu-ultra` is a callable model today** → we get Fugu both as (a) a worker Maestro can route to, and (b) the headline **baseline to benchmark Maestro against** in the eval harness. This directly serves MAESTRO-DESIGN §15 ("never lose to best-single," and beat Fugu on its weak flanks).
- Built-in failover/retries/budgets/observability reduce what we must build (but we keep our own budget controller + tracing as the source of truth).

**Secondary / fallback: OpenRouter.**
- 400+ models / 70+ providers = the widest catalog and a backstop when a model isn't on the Gateway.
- `openrouter/auto` and provider-routing are useful **baselines** to benchmark Maestro's router against (and to cite as prior art).
- Cost caveat: 5.5% credit fee (or 5% BYOK after 1M/mo). Prefer BYOK; treat as the fallback lane.

Both are OpenAI-compatible, so the `providers/` registry stays clean: model rows carry `{provider: gateway|openrouter|direct, model_id, caps, $/token, latency, region}` and the router is provider-agnostic. **BYO-keys + config-as-code** means the operator picks the lane.

**Is "build on a gateway, differentiate on orchestration + evals + tools" the right play?** Yes. The transport/catalog/failover layer is commoditizing fast (both gateways now offer it at ~0% markup) — building our own is undifferentiated toil. Maestro's defensible value is exactly what neither Fusion nor Fugu nor the gateways provide:
1. **Confidence/cost cascade** that refuses to orchestrate when a single model wins (fixes Fugu's documented losses and Fusion's always-4–5× cost).
2. **Real tool execution** (sandboxed) — Fugu only plans tools; Fusion only does web_search/fetch.
3. **Verifier/critic agents** + typed task protocol.
4. **Transparency + an eval harness with baselines** (best-single, cheapest-single, `openrouter/auto`, **and Fugu Ultra via the Gateway**) — no claim without a protocol_hash.
5. **Open, self-hostable, EU-ok, BYO-keys.**

**Net:** depend on Vercel AI Gateway (primary) + OpenRouter (fallback) for the model plane; spend our engineering on the orchestration + gate + tools + evals on top. Route to Fusion and Fugu Ultra *as backends* and *beat them on the metered cost/quality trade.*

**Build-time TODOs:** re-pull `openrouter.ai/models` to confirm Fugu is/ isn't there; confirm whether plain `fugu`/`fugu-mini` (not just `fugu-ultra`) reach the Gateway; verify exact Gateway model count; re-verify both pricing pages (fees change).
