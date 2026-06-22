# Maestro Verdict

## Verdict First

`docs/MAESTRO-SCOPE.md` is the source of truth for v0.

The previous "local inference first" framing is superseded. Maestro v0 should **not** be a local model-hosting project and should **not** center on vLLM/SGLang/llama.cpp as the first product surface.

The correct direction is:

> Maestro is an open-source, self-hostable routing/orchestration brain that exposes one OpenAI-compatible API and routes requests across a configurable pool of models through AI gateways, with benchmarkable routing and verification.

The v0 goal is to get as close as possible to Fugu/Fable-level practical quality by routing across strong models via gateways, while staying self-hosted, transparent, and benchmark-driven.

## Correct Scope

Maestro is not trying to host models.

Maestro is the brain that decides:

- what the task is
- how hard it is
- which model or effort tier should handle it
- whether the answer needs verification or escalation
- what trace and benchmark evidence should be recorded

The v0 request flow is:

```text
OpenAI-compatible client
  -> Maestro /v1/chat/completions
  -> classify request
  -> route to model/effort via gateway adapter
  -> execute
  -> verify/fallback if needed
  -> return answer + transparent routing metadata
```

## What v0 Should Use For Model Access

Gateway-first for v0:

- OpenRouter
- Vercel AI Gateway
- any OpenAI-compatible gateway/provider endpoint
- BYOK where possible
- optional `local-openai` adapter for users who already run Ollama, vLLM, SGLang, llama.cpp, LM Studio, or another OpenAI-compatible server

This is the fastest path to strong benchmark performance because it lets Maestro route across strong open and closed models without building GPU hosting, model loading, quantization, batching, or inference infrastructure.

## What Is Explicitly Out For v0

Do not build these now:

- local GPU inference
- local model hosting
- vLLM/SGLang/llama.cpp hosting as the primary product
- Ollama-centered architecture
- LoRA/MoL serving
- learned router training
- CMA-ES
- GRPO
- full Conductor/Ultra decomposition
- dashboard
- Postgres/Redis/BullMQ
- tool execution
- multi-agent debate swarm
- plugin marketplace

Local OpenAI-compatible adapters are allowed because they do not require Maestro to host models. They are provider adapters, not inference infrastructure.

## What Research Reveals

The important Fugu/OpenFugu insight is not "host local models."

The important insight is:

- one model-like endpoint can hide a pool of models
- model selection can be separated from model execution
- slot labels can be remapped to different provider/model IDs
- no worker weights need to be merged or modified
- a small router or classifier can make useful per-request decisions
- verification/fallback can reduce misroute damage

OpenFugu validates the architecture shape, but it remains research-heavy:

- its `+107%` result is mock-heavy
- the TRINITY inference path is the strongest reproduced part
- the trained Ultra/Conductor product behavior is not fully reproduced as a production system
- OpenFugu is not yet the clean productized routing brain Maestro should become

## What Makes Maestro Better Than OpenFugu

Maestro can be better than OpenFugu if it is:

- easier to run
- gateway-ready from day one
- OpenAI-compatible from day one
- config-driven
- transparent about routing decisions
- benchmarkable from day one
- honest about claims
- practical for developers to self-host
- not dependent on Sakana artifacts
- not locked into one model pool

The goal is not to claim "beats Fugu" in the README before evidence exists.

The goal is to build a benchmark harness where Maestro can prove it beats:

- always strongest model
- always cheapest model
- random route
- keyword-only route
- OpenFugu-style baseline if available

## v0 Build Target

Build a working prototype with:

- `GET /health`
- `GET /v1/models`
- `POST /v1/chat/completions`
- streaming if feasible
- config-driven dated model registry
- gateway adapter interface
- at least one real gateway adapter
- optional local OpenAI-compatible adapter
- mock adapter for offline tests
- router policies:
  - explicit model override
  - capability filter
  - keyword/metadata guardrails
  - cheap classifier/router model if configured
  - strongest fallback
- TRINITY-style verify/fallback loop with a hard turn cap
- response metadata:
  - selected model
  - strategy
  - route reason
  - models used
  - turns
  - latency
  - token usage/cost if available
- JSONL benchmark runner
- sample benchmark dataset
- README quickstart

## Benchmark Standard

Because the product is routing, routing must be measured.

Minimum metrics:

- route accuracy
- oracle route regret
- quality score where validators exist
- cost-normalized quality
- latency-normalized quality
- p50/p95 latency
- fallback rate
- verifier accept/reject rate
- router parse failure rate
- category-level win/loss

Minimum baselines:

- always strongest
- always cheapest
- random route
- keyword route
- Maestro hybrid route

No benchmark superiority claim should be accepted without these baselines.

## OpenFugu Details Worth Keeping

Keep these lessons from the OpenFugu docs:

- Academic TRINITY uses a small Qwen3-0.6B router plus a 19,456-float vector.
- Academic TRINITY emits 7 worker logits plus 3 role logits.
- Product Fugu appears to simplify toward model selection for latency.
- Router input formatting matters: raw `"role: content"` beats chat template heavily in the reproduced fixture.
- Slot labels are training metadata and can be remapped to deployment providers.
- The bounded verify loop is useful without learned routing.
- Conductor/Ultra should stay post-v0.

## Final Recommendation

Follow `docs/MAESTRO-SCOPE.md`.

Build Maestro v0 as:

> a self-hosted, gateway-first, OpenAI-compatible routing brain with transparent traces, a verifier/fallback loop, and a benchmark harness.

Do not build a local model-hosting platform.

Do not build a broad multi-agent platform.

Do not make unsupported "beats Fugu" claims.

Build the thing that can actually run benchmarks and earn the claim.
