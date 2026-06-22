# maestro — roadmap

Mirrors OpenFugu's **read → run → train → serve**, adapted, with an honest "needs infra" column.

| Milestone | What | Status | Needs |
|---|---|---|---|
| **M1 — mock coordinator + serve** | Router/Pool/Conductor interfaces, HeuristicRouter, MockPool, OpenAI-compatible serve, self-test fixture | ✅ in this scaffold | nothing (stdlib) |
| **M2 — real worker pool** | `LiteLLMPool`: role → frontier model; live `route` + `serve` against a real pool | next | `litellm` (pinned) + `FUGU_API_KEY`/`FUGU_BASE_URL` |
| **M3 — learned router (Trinity)** | features/hidden-state → linear head; train by **sep-CMA-ES** on a routing fixture; **eval gate**: beat best single worker | next | a base model for features; a labeled fixture; `numpy` |
| **M4 — LoRA pool (MoL-style)** | `LoRAPool`: role → specialist LoRA via SGLang; hybrid router + metadata guardrail + L0 fallback; KV reuse | later | SGLang + LoRA adapters + GPU |
| **M5 — Ultra (Conductor)** | language workflow/DAG, **GRPO** training, **recursion** (revise own output) | later | tool dataset (ToolScale-like) + 8×A800-class |

## Build order (no blocking on GPUs)
1. Ship M1 (done) — the plumbing runs and is testable today.
2. M2: one pinned dep + keys → real answers. Highest value/effort.
3. M3: train the router on a fixture; prove it beats best-single-worker before trusting it.
4. M4/M5: only when GPUs/datasets are available; keep them behind the same interfaces.

## Eval bar (don't skip)
A learned router must beat the **best single worker** on a held-out fixture (per-query routing). Track: routing accuracy, win-rate vs best single, latency, $/query. (OpenFugu reports +107% query-level; aim to reproduce on our fixture.)

## Open questions
- Per-**query** routing (simpler, proven) vs per-**step** coordination (harder, the real Fugu Ultra promise) — start with per-query.
- Frontier-model pool vs LoRA pool vs both — scaffold supports both; pick first target by cost.
- Where features come from for the learned router (prompt embeddings vs base-model hidden states).
