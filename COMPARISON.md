# Maestro vs other open-source Fugu attempts

An honest, verifiable head-to-head. We compare on the axes that matter for a **usable open-source router** — not vibes. Every Maestro claim is reproducible from this repo (`npm test`, `npm run eval`, `bash scripts/verify.sh`); claims about the others come from reading their own repos/docs (see [`archive/docs/`](archive/docs) for our code-traced analysis).

## The two prior projects

- **[OpenFugu](https://github.com/trotsky1997/OpenFugu)** — a faithful *research* reverse-engineering of Sakana Fugu (the TRINITY learned router + a Conductor stand-in). Its goal is to reproduce the *mechanism*.
- **[Mixture-of-LoRA-Harness](https://github.com/MindLab-Research/Mixture-of-LoRA-Harness)** — a router + SGLang overlay that serves multiple **LoRA adapters** on one base model. Different problem: it routes *adapters*, not *models/providers*.

## Feature matrix

| | **Maestro** | OpenFugu | Mixture-of-LoRA-Harness |
|---|:--:|:--:|:--:|
| Runs with **zero setup / no GPU** | ✅ `npx openmaestro serve` | ❌ needs Python + GPU + Qwen3-0.6B weights | ❌ needs GPU + SGLang + LoRA adapters |
| **Routes across providers** (open + closed) | ✅ any, BYOK | ⚠️ litellm pool (research) | ❌ LoRA adapters on one base only |
| **OpenAI-compatible** endpoint | ✅ | ✅ (last-message only) | ✅ |
| **Anthropic / Claude Code** support | ✅ `/v1/messages` | ❌ | ❌ |
| **Verify → escalate** loop, runnable | ✅ (tested) | ⚠️ present, but Conductor/Ultra has **no execution proof** (their handoff) | ❌ not an agent loop |
| **Honest, reproducible benchmark** | ✅ `npm run eval` (offline, oracle-regret, calibration) | ❌ headline **+107% is an explicit MOCK** harness | ❌ **no weights/logs/benchmarks** shipped |
| **Cost/route transparency** per request | ✅ `maestro` block + `x-maestro-*` | ❌ returns only `usage.fugu_turns` | ❌ |
| **Tests / CI** | ✅ 30 tests + CI | ⚠️ partial; training code **stripped (~78% reconstructed)** | ❌ none shipped |
| **Swap models without retraining** | ✅ dated registry, slots→ids | ⚠️ slots remappable but research harness | ⚠️ adapter library, version-coupled overlay |
| **Production endpoint discipline** | ✅ validation, errors, streaming, traces | ⚠️ research scaffold | ⚠️ serving pattern, low-to-medium readiness |
| **License** | MIT | check repo | check repo |

## Where Maestro is honestly *better*

1. **It actually runs, for anyone, right now** — no GPU, no weights, no Python. `npx openmaestro serve` + curl works in 30 seconds. OpenFugu and MoL both require a GPU box and model artifacts.
2. **Honest, reproducible numbers.** OpenFugu's famous "+107% over best single" is, by its own code, a **mock per-question routing harness** — not production proof. MoL ships **no benchmarks at all**. Maestro's `npm run eval` is offline, deterministic, graded against ground truth with oracle-route regret and calibration — and it *shows its one failure*.
3. **Real product surface.** OpenAI **and** Anthropic APIs (so Claude Code / opencode / Cursor work), request validation, streaming, per-request cost transparency, a dry-run `/v1/route`, trace inspection. The others are research scaffolds.
4. **Provider-agnostic by design.** Maestro routes across *any* models (open + closed + local), swappable via a JSON registry with no retraining. MoL routes LoRA adapters on a single base; OpenFugu targets a research pool.

## Where the others are ahead (so this stays honest)

- **OpenFugu reproduces the actual learned router** (frozen Qwen3-0.6B + trained head, the papers' real mechanism). Maestro v0 uses a *heuristic* classifier — the learned TRINITY router is our **roadmap v2**. If you want to study Fugu's ML internals, OpenFugu is the reference (and we credit it).
- **MoL solves a different, real problem** (efficient multi-LoRA serving with KV-prefix reuse). If you self-host one base model with many fine-tunes, that's its lane, not Maestro's. A LoRA-serving backend could even sit *behind* Maestro as a `local-openai` provider.

## The honest one-line claim you can make

> **Maestro is the open-source version of Fugu that you can actually run and verify today** — provider-agnostic, OpenAI- *and* Anthropic-compatible, with transparent cost routing and a reproducible benchmark — where OpenFugu is a research reproduction (mock benchmark, GPU-only) and Mixture-of-LoRA-Harness solves a different (LoRA-serving) problem.

Reproduce all of the above: `bash scripts/verify.sh` (and `npm run eval`). Don't take our word for it.
