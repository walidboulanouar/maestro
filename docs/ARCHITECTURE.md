# maestro — architecture

Four pieces. Each has a **mock (today)** and a **real (next)** implementation behind the same interface — like coset/probe's sandbox adapters.

```
                ┌─────────────────────────── one OpenAI-compatible endpoint ──────────────────────────┐
 request ──►    │  Conductor.answer(query, history)                                                    │
                │     1) role = Router.route(query, history)        which worker handles this?          │
                │     2) out  = WorkerPool.run(role, query)         the chosen worker answers           │
                │     3) (Ultra) verify / recurse / synthesize      revise own output = test-time scale │
                │     ◄── one answer                                                                    │
                └──────────────────────────────────────────────────────────────────────────────────────┘
```

## Interfaces (`maestro/`)
- **Router** (`router.py`) — `route(query, history) -> role`.
  - `HeuristicRouter` (mock): keyword/priority rules → role; `general` fallback (our "L0").
  - `LearnedRouter` (next): features/hidden-state → linear head → role; trained by CMA-ES (Trinity-style). Stub today.
  - Optional **metadata guardrail** (MoL-style): a strong library signal can override a noisy route.
- **WorkerPool** (`pool.py`) — `run(role, query) -> str`.
  - `MockPool` (today): canned per-role response — proves the loop offline.
  - `LiteLLMPool` (next): role → frontier model via litellm (needs `FUGU_API_KEY`/`FUGU_BASE_URL`).
  - `LoRAPool` (next): role → specialist LoRA adapter (MoL-Harness / SGLang).
- **Conductor** (`conductor.py`) — ties Router + Pool; `ultra=True` adds one verify/refine pass (recursion later).
- **serve** (`serve.py`) — stdlib HTTP, `POST /v1/chat/completions` → `Conductor.answer` → OpenAI-shaped JSON. Pool stays hidden.

## Roles (our "LoRA library" / worker families)
`general` (L0 entry + chat) · `code` · `math` · `translate` · `research` · `extract`. Extend freely; each maps to a model and/or a LoRA in real pools.

## Design choices (locked early)
- **Mock-first, real-ready** — runs with zero deps; real pools/router switch on via env, same interfaces.
- **Pool holds both** frontier models and LoRA adapters (take OpenFugu *and* MoL).
- **Router evolves**: heuristic → CMA-ES linear head → GRPO conductor. Never block on training to ship the plumbing.
- **One endpoint** — consumers never see the pool (Fugu's promise).
- **Honest eval gate** — a router only "wins" if it beats the best single worker on a held-out fixture (OpenFugu's eval bar).
