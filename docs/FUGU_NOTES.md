# FUGU notes — what it is, how it works (sourced)

Reverse-understood from Sakana's release + OpenFugu + Mixture-of-LoRA-Harness. Evidence-graded so we don't build on guesses.

## The core mechanism (high confidence)
Fugu = a **coordinator policy over a pool of models**. Per request it either answers directly or selects/delegates to expert workers, then verifies + synthesizes — all internal, exposed as one model. *(Sakana release; OpenFugu README.)*

Two named research lines (ICLR 2026):
- **Trinity — "An Evolved LLM Coordinator."** A *tiny* coordinator that maps a query (or hidden state) to a worker choice. OpenFugu implements it as **hidden-state → linear head → worker**, trained with **sep-CMA-ES** (evolution strategy — no gradients, no Sakana weights). *(OpenFugu: `mini.py`, `train/train_trinity.py`; self-test ~95% agent / 100% role on a 37-case fixture with real Qwen3-0.6B weights.)*
- **Conductor — "Learning to Orchestrate Agents in Natural Language."** A larger coordinator that emits a **workflow/DAG** in language, trained with **GRPO** (RL) on a tool dataset (`nvidia/ToolScale`). Supports **recursion** (revise its own output = test-time scaling). *(OpenFugu: `ultra.py`, `train/train_conductor.py`, `train_recursion.py`.)*

Products: **Fugu** (latency) vs **Fugu Ultra** (quality, the recursive/multi-step one). *(Sakana.)*

## Routing styles we can mix
1. **Across frontier models** (OpenFugu): the pool is many LLMs via **litellm**; the router picks which model answers each query/step. Eval claim: query-level routing **+107%** over the best single worker (caveat: that's per-*query* routing, not per-*step* coordination).
2. **Across specialist LoRA adapters** (MoL-Harness): the pool is LoRA adapters on one base, served via SGLang behind one OpenAI endpoint. Router is **hybrid**: a model prompt-route + a **metadata guardrail** that can override noisy output + **L0 fallback** (L0 = entry router *and* general chat). Multi-turn policy: L0 routing sees only the current query; specialists see current query + same-task history; cross-task history masked; KV reuse optional.

→ **Our design takes both:** a pool that can hold frontier models *and* LoRA adapters; a router that starts heuristic, becomes learned (CMA-ES → GRPO); Ultra adds verify/recurse.

## What's hard / needs infra (be honest)
- Real Trinity needs a base model's **hidden states** + a fixture to train the linear head (CMA-ES). OpenFugu's mock trainer "chance→optimal in ~5 generations" runs anywhere; the real one needs the model.
- Conductor (GRPO) needs **8× A800-class GPUs** + a tool dataset.
- A real pool needs API keys (litellm) or served LoRAs (SGLang). All deferred behind env/flags; the scaffold runs mock without them.

## What we DON'T copy
No Sakana weights or product code. OpenFugu/MoL are references for the *mechanism*; we write our own. Any third-party artifact is fetched from its licensed source, never vendored.
