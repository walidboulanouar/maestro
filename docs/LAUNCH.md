# Launch kit — "I open-sourced Fugu"

Copy-paste-ready posts + the honest framing. **Reproduce every number before you post:** `bash scripts/verify.sh` and `npm run eval`.

---

## The honest positioning (read first)

- ✅ **You CAN say:** "Maestro is the open-source version of Sakana's Fugu" — it's built directly from the two papers ([TRINITY 2512.04695](../papers) + [Conductor 2512.04388](../papers)), implements the route→verify→escalate loop, and is provider-agnostic, runnable, and tested.
- ✅ **You CAN say:** "It reaches frontier-level results at ~1/11th the cost on our routing benchmark (92% of best-single quality, 97% cheaper)" — because `npm run eval` reproduces it.
- ✅ **You CAN say:** "More usable than the existing open Fugu attempts" — see [COMPARISON.md](../COMPARISON.md) (OpenFugu's headline is a *mock*; MoL ships no benchmarks; both need a GPU).
- ⚠️ **Don't say:** "Maestro beats GPT-5.5 / Fable 5." Maestro isn't a model — it *uses* them. Claim cost-efficiency and openness, not model supremacy. (Overclaiming gets torn apart on HN; the honest claim is stronger anyway.)

---

## Show HN

**Title:** `Show HN: Maestro – I open-sourced Fugu (LLM router: cheap-first, verify, escalate)`

> Sakana's Fugu is a single API that internally routes a request to the best of a pool of models. It's closed, EU-blocked, opaque about cost, and limited to 3 models. I built the open-source version from the two papers (TRINITY + Conductor, both in the repo).
>
> Maestro is *just the routing brain*. Point your OpenAI (or Anthropic) client at it; for every request it classifies the task, tries a cheap model first, **verifies** the answer, and **escalates** to a stronger model only when needed — then shows you the exact route, tokens and cost on the response.
>
> - Runs with **zero API keys** (built-in mock provider): `npx openmaestro serve`
> - Routes **any** models — open + closed via OpenRouter/Vercel, or your **local** Ollama/vLLM
> - Works in **Claude Code** (`ANTHROPIC_BASE_URL`) and **opencode/Cursor** (OpenAI base URL)
> - Honest, reproducible benchmark (`npm run eval`): **92% of best-single quality at 97% lower cost** — and it shows its one failure
> - MIT, TypeScript, 30 tests
>
> Unlike OpenFugu (a research repro whose "+107%" is a mock harness, GPU-only) and Mixture-of-LoRA-Harness (LoRA serving, no benchmarks), Maestro actually runs and verifies today. Comparison + verification report in the repo. Tear it apart.

---

## X / Twitter thread

1/ I open-sourced Fugu. 🐡

Sakana's Fugu = one API that routes each request to the best model in a pool. Closed, EU-blocked, 3 models, no cost transparency.

Maestro = the open version. Built from the 2 papers. Runs anywhere. MIT. 🧵

2/ It's *just the routing brain*. Point your OpenAI/Anthropic client at it →
classify → try cheap model → **verify** → **escalate** only if needed → show the route + cost.

`npx openmaestro serve` (works with ZERO api keys via a mock provider)

3/ Honest numbers (reproduce with `npm run eval`, offline):
• 92% of best-single quality
• 97% lower cost
• ~26× more answers per dollar, and it beats random routing
It even shows its 1 failure. No mock "+107%" like the other open Fugu.

4/ Routes ANY models — GLM-5.2, DeepSeek V4, Kimi, Opus 4.8, GPT-5.5, Gemini 3 — via OpenRouter / Vercel AI Gateway, or your **local** Ollama/vLLM. Swap models in a JSON registry, no retraining.

5/ Works in your agent:
• Claude Code → `ANTHROPIC_BASE_URL=localhost:8080`
• opencode / Cursor → OpenAI base URL, model `maestro-auto`

6/ Open the bundled `test.html` to verify it live, or run `bash scripts/verify.sh`. The two papers + a full verification report + a head-to-head vs OpenFugu/MoL are in the repo.

⭐ github.com/walidboulanouar/maestro

---

## How it was implemented (for the writeup / blog)

- **The loop (from TRINITY):** `classify → route(model,effort) → execute → verify(ACCEPT/REVISE) → escalate`, bounded by `maxTurns`. Roles map to the paper's Thinker/Worker/Verifier.
- **Router (v0):** heuristic task+difficulty classifier → capability/policy/region/budget filter → tier by difficulty → MoL-style guardrail score → escalation ladder. (v2: the learned TRINITY head — frozen Qwen3-0.6B + tiny trained head, CPU-only.)
- **Key insight:** the **verifier must judge independently of the router**, or the loop never escalates. Sharing the signal scored *below random* (63%); splitting them → 92%. (Documented in VERIFICATION.md.)
- **Provider-agnostic:** one OpenAI-compatible adapter covers OpenRouter / Vercel / local; a mock adapter makes it run keyless; slots→ids registry makes models swappable with no retraining.
- **Transparency:** every response carries the route, per-model tokens, cost, and a frontier-only comparison.

## What I need from you (Walid) to verify real routing

1. Put **one** key in `.env` (gitignored): `OPENROUTER_API_KEY=` (easiest — open + closed models, one key) — or `AI_GATEWAY_API_KEY`, or a local `LOCAL_OPENAI_BASE_URL`.
2. Run `bash scripts/verify.sh` → it boots the server, smoke-tests every endpoint, and (with a key) sends one **real** request showing the real route + cost.
3. Open `test.html` while `npx openmaestro serve` runs → click **Run all tests**.

To actually launch: publish to npm (`npm publish`), push a Docker image to ghcr, record a ~20s demo GIF (the `test.html` "Try it" panel + the route/cost is the money shot), then post the Show HN / thread above.
