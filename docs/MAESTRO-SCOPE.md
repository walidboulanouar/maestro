# Maestro — locked scope (the only doc that matters for v0)

> Supersedes the broad `MAESTRO-DESIGN.md` for what we actually build first.
> `MAESTRO-DESIGN.md` / `MAESTRO-ANALYSIS.md` stay as research/background.

## One line
**Maestro is an open-source, self-hostable *orchestration/routing brain* — like [OpenFugu](https://github.com/trotsky1997/OpenFugu)/Sakana Fugu — that routes each request across a pool of models (consumed through an AI gateway) to reach ~Fable-5-level quality, behind one OpenAI-compatible endpoint.**

It is **just the routing/orchestration**. Nothing more, for now.

## What "local" means here (resolves a recurring confusion)
A second reviewer flagged that this scope looks "cloud-gateway-first" vs a "local-first" goal. Clarifying per Walid's explicit correction *("just the routing… models consumed using ai gateway… like opus or gemini… **but no gpu server**")*:
- **"Local" = you self-host the *router* (Maestro) on your own machine.** It's an open-source binary/process you run; nobody else hosts it. That is the local-first promise.
- **"Local" does NOT mean running the models on your own GPU.** Walid ruled out a GPU server, and the target pool includes closed models (Opus/GPT-5.5/Gemini) that *only* exist behind an API. So models come through a gateway. This is a deliberate decision, not scope drift.
- **But running models locally is supported for free** — a local OpenAI-compatible server (Ollama / vLLM / llama.cpp / LM Studio) is *just another provider adapter* in the registry (`provider: "local-openai"`, `baseUrl: http://localhost:11434/v1`). So anyone WITH a GPU can point Maestro at their own models and run 100% offline, without us building any inference. Best of both, at zero extra cost to the design.

## What is ALREADY SOLVED — reuse, do NOT build
We don't reinvent any of this. Maestro sits on top of it.
- **Model access (open + closed):** GLM-5.2, DeepSeek V4, Kimi K2.7, *and* Opus 4.8, GPT-5.5/Codex, Gemini — via an **AI gateway** (OpenRouter / Vercel AI Gateway, BYOK) **or a local OpenAI-compatible backend** (Ollama/vLLM/llama.cpp) as an optional adapter. We build no inference and host no GPU.
- **OpenAI-compatible serving:** a thin HTTP layer exposing `/v1/chat/completions` (+ stream, `/v1/models`). Trivial; we just expose it.
- **Observability/caching primitives:** borrow (Langfuse/OTel/semantic-cache lib) only if needed; not v0.

## What Maestro IS — the only novel part
**The orchestration/routing policy: given a request, decide *which model(s), at what effort, in how many turns*, then verify.** This is the entire product. This is what OpenFugu reverse-engineers from Fugu and what we make real, open, and honest.

### The pipeline (the "brain")
1. **Classify** — task type (code / math / reasoning / translation / factual / fresh-or-tool-needed), difficulty, required capabilities, constraints (budget, region, pinned model).
   - v0: a cheap/fast gateway model (or heuristics/embeddings) does this.
   - later: a tiny *learned* router (TRINITY-style Qwen3-0.6B + trained head) — runs on CPU; the real research lever, trained offline using OpenFugu's CMA-ES code.
2. **Route** — map (task, difficulty, capability, budget) → an ordered plan of `(model, effort)`. Easy → cheap open model; hard → frontier (Opus/GPT-5.5/Fable-5). Deterministic guardrails (MoL-style score: `priority + strong*100 + positive*10 − negative*250`) layered over the classifier's suggestion. **This is where Fable-5-level comes from: the hard stuff actually reaches a frontier model; the easy stuff stays cheap.**
3. **Execute** — call the chosen model via the gateway adapter (streaming supported).
4. **Verify loop** (TRINITY-style) — a verifier checks the answer (rubric; for code later: run tests). On reject + budget left → escalate to a stronger model / revise. Max K turns (3–5), terminate on ACCEPT. *This is the misroute safety net — and replaces the dishonest "we never lose to a single model" claim with a measured fallback.*
5. **Return** — answer **+ transparency metadata**: models used, turns, tokens, cost, route reason. (Fugu hides this; we don't.)

Two modes (Fugu parallel):
- **`fugu` mode (v0):** single worker + verify loop. Low latency.
- **`ultra` mode (later):** Conductor-style decompose into ≤5 sequential subtasks with shared memory. Only after v0 proves out.

## The non-negotiable bar (from the critique — it's right)
Because the product *is* routing, the routing has to be *proven*, or it's vibes:
- A small **offline eval** (script, not a platform) on a few public sets with executable/exact answers + a couple mixed-capability fixtures (e.g. "fix a React hydration bug AND explain GDPR impact").
- Metrics: **oracle-route regret**, cost-normalized quality, p50/p95 latency, fallback precision, **calibration (ECE/Brier)** of the difficulty/confidence signal.
- Baselines: best-single, cheapest-single, random-route, Maestro.
- **Success = matches best-single quality within CI at lower median cost, bounded p95** — *not* "beats Fugu/Fable-5."

## How this differs from OpenFugu (why build it)
OpenFugu is research-grade: its `+107%` is a **mock** eval, Ultra uses a **prompted stand-in** (no trained Conductor weights), no production endpoint discipline. Maestro = the *honest, usable* version: real gateway pool (open+closed), real verifier loop, **honest eval (no mocks)**, production OpenAI-compatible endpoint, full transparency — and *optionally* a properly trained router later. (Deep analysis: `archive/docs/01-openfugu-runtime.md`, `02-openfugu-training.md`.)

## Explicitly OUT (nothing more for now)
Local GPU inference / model hosting · multi-agent debate/critic/synthesizer/memory-manager · tools/sandbox execution · MoL/LoRA serving · learned-router *training* (deferred, not v0) · eval *platform/UI* · Postgres/Redis/BullMQ (a single self-hosted process + SQLite/files is enough) · "cost-arbitrage cloud gateway" framing.

## Implementation notes borrowed from OpenFugu's own docs (verified 2026-06-22)
From `OpenFugu/docs/{HOW_FUGU_IS_IMPLEMENTED,ARCHITECTURE,handoff}.md` — evidence-graded; these directly shape Maestro:
- **Slot labels are remappable metadata.** Fugu *trains* on a 7-slot pool (`es_log.json`: gpt-5, claude-sonnet-4, gemini-2.5-pro, DeepSeek-R1-Distill-Qwen-32B, gemma-3-27b-it, Qwen3-32B×2) but *deploys* against the frontier 3 (Gemini-3.1-Pro/Opus-4.8/GPT-5.5). "Slot labels are training metadata, remappable to any provider at deployment — no retraining." → **This validates Maestro's dated model registry + provider abstraction directly. Route over abstract slots; map slots → gateway model IDs in config.** (Also explains "gemini-2.5" — it's the training-log model.)
- **No weight merging — pure macro-level API composition.** Exactly Maestro's model: orchestrate calls, never touch weights → provider rotation needs no retraining.
- **The coordination loop is buildable WITHOUT the learned router.** `core.py:879-1000`: ≤5 turns, route→inject role prompt→call worker, terminate on Verifier `ACCEPT` (or max-turns → return latest, or Verifier-with-no-response → done/0-reward). A **Thinker can emit `<suggested_role>` to override the next selection.** v0 does this loop with *prompt-based* roles; the learned head is a later drop-in for the "route" step.
- **If/when we build the learned head — two gotchas that cost OpenFugu weeks:** (1) **router input must be the RAW transcript `"role: content\n"`, NOT a chat template** — chat-template scored 11% role / 5% joint vs 95%/100% raw; (2) read the **penultimate-token (pos −2)** hidden state with an **early-exit monkeypatch** (skip LM head + autoregressive decode → cheap, CPU-viable for Qwen3-0.6B). Params = 19,456 floats: `[0:9216]` SVF offsets, `[9216:19456]` → head reshape (10,1024) = 7 agent + 3 role logits (product Fugu may drop the 3 role logits).
- **Training is simpler than the "sep-CMA-ES" branding suggests.** handoff/probe: at 19,456-D the per-coordinate σ stays ~frozen (std 6e-4); only scalar σ moves (0.03→0.002) → "dynamics reduce to isotropic ES." So our later learned-router training can start with a plain isotropic ES, not a heavy CMA build. (Production also prepends a soft-KL SFT stage, τ undisclosed.)
- **Honesty check, reinforced:** OpenFugu's *training code is stripped* (~78% reconstructed) and **Conductor/Ultra has NO execution proof — code-read + report only.** → Maestro keeps `ultra`/decompose firmly post-v0; v0 = TRINITY-style single-worker + verify loop, which *is* fully reproduced and runnable.

## v0 deliverable
A single self-hostable service (`docker run` / `npx`): OpenAI-compatible `/v1/chat/completions` → **classify → route → execute(via gateway) → verify-loop → return + transparency**, with a config-driven **dated model registry** and the small offline eval proving the routing. Two gateway adapters (OpenRouter + Vercel AI Gateway), BYOK.
