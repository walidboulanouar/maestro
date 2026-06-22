# Maestro — verification report

> For a human or another AI agent to **independently verify** that this project does what it claims. Be skeptical. Every claim below has a command that reproduces it. If something doesn't check out, it's a bug — open an issue.

## TL;DR claims (and how to falsify each)

| Claim | Verify with | Expected |
|---|---|---|
| It builds & typechecks | `npm install && npm run typecheck && npm run build` | exit 0, `dist/cli.js` exists |
| It's tested | `npm test` | **65 tests pass** |
| The server boots without a GPU | `npm run serve` then `curl localhost:8080/healthz` | `{"status":"ok",...}` (add a provider key to route to real models) |
| It routes & escalates | `bash scripts/verify.sh` | all endpoint checks ✅ |
| It saves money (routing) | `npm run eval` | maestro ≈ 92% pass at ~97% lower cost than best-single (and beats random) |
| It works in Claude Code | `ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_API_KEY=unused claude` | Claude Code talks to Maestro |
| It routes REAL models | put a key in `.env`, `bash scripts/verify.sh` | "real routing works" + real cost shown |

**One command does all of the above:** `bash scripts/verify.sh`

## What was actually built (no hand-waving)

A self-hostable router, in TypeScript, that is *only* the routing brain (it hosts no models). Files that matter:

```
src/
  server.ts              Hono app: /v1/chat/completions, /v1/messages, /v1/models, /v1/route, /v1/traces/:id, /healthz
  cli.ts                 maestro serve | route | registry check | version
  core/
    classify.ts          heuristic task/difficulty classifier (+ independent judgeDifficulty for the verifier)
    route.ts             capability filter → tier by difficulty → guardrail score → escalation ladder
    orchestrator.ts      the loop: classify → route → execute → verify → escalate
    verify.ts            ACCEPT/REVISE (deterministic for mock; strict JSON rubric for real models)
    cost.ts, transcript.ts
  registry/{models,registry}.ts   dated model registry, slots → model ids
  providers/             openai-compatible (OpenRouter/Vercel/local) + mock + index
  api/{shape,anthropic}.ts        OpenAI + Anthropic response shaping
  transparency/trace.ts  in-memory + JSONL trace store
eval/{run,metrics}.ts + fixtures/tasks.jsonl   offline routing benchmark
test/*.test.ts          65 vitest tests
papers/                 the two source papers (TRINITY 2512.04695, Conductor 2512.04388)
```

## The eval methodology (read this before trusting the numbers)

`npm run eval` is **deliberately honest**:

- **Offline & deterministic.** It routes over the *priced* registry (real model prices + strengths) but **executes on the mock provider** — no network, no spend, reproducible. No `Math.random`.
- **Graded against ground truth, not the router's own opinion.** Each fixture has a hand-set `difficulty`. A route "passes" iff the **final model's strength ≥ the strength required by the fixture's true difficulty** (`required = 50 + 45·difficulty`). The classifier's estimate is *not* used for grading — so a mis-estimate shows up as a real failure.
- **The verifier is independent of the router.** The router uses `classify`; the verifier uses a separate, more conservative `judgeDifficulty`. This is why the escalation loop actually fires (an earlier version where they shared a signal scored *worse than random* — that bug is documented in the learnings below).
- **Baselines:** best-single (always strongest), cheapest-single, random-route (deterministic), plus an **oracle** (cheapest model that would pass) for regret.
- **Metrics:** pass rate, mean cost, passes-per-dollar, oracle-route **regret**, and **calibration** (Brier + ECE) of the classifier's confidence.

Current result (25 fixtures ≈ realistic traffic mix, 7 priced models, real OpenRouter ids/prices):

```
strategy             pass%      mean $      pass/$    regret $   fails
maestro                92%     0.00053      1747.9     0.00035       2
best-single           100%     0.01507        66.3     0.01421       0
cheapest-single        56%     0.00016      3566.0     0.00000      11
random-route           88%     0.00689        127.7     0.00705       3
Brier = 0.181   ECE = 0.110
```

**Honest reading:** Maestro hits 92% of best-single quality at ~1/28th the cost (~26× more answers per dollar), and beats a **random** router on both quality (+4 pts) and cost (13× cheaper). `cheapest-single` is cheaper per call but fails 44% of tasks — Maestro is the best *balance*. It is **not** 100% — the heuristic classifier mis-estimates two fixtures, and full escalation can occasionally cost ~the same as going straight to frontier. These are real and shown on purpose. This is a *routing* benchmark on a realistic-mix fixture set, **not** a model-quality leaderboard. (Note: an earlier version of this harness had a constant "random" baseline — a bug we found and fixed; see the eval source.)

## Assumptions made (so you can attack them)

1. **The default registry's model ids, prices, and strengths are indicative** and dated `2026-06-22`. They are not live-fetched. Swap in your own via `MAESTRO_REGISTRY`. (`maestro registry check` warns when stale.)
2. **The mock provider models answer quality is simulated** from `strength` vs `judgeDifficulty`, not from real generations — that's what makes the eval free and deterministic. Real quality requires real keys (`scripts/verify.sh` does a real call if a key is set).
3. **The heuristic classifier is v0.** It's regex/cue-based. A learned (GPU-trained) router is intentionally OUT OF SCOPE; Maestro stays no-GPU (heuristic classifier + verify/escalate).
4. **`maestro-ultra` falls back to `fugu`** until the Conductor decomposition lands (v3).
5. **Verifier false-accepts default to ACCEPT** on unparseable real-LLM output (to avoid runaway cost) — a conservative cost choice, noted in `verify.ts`.

## Known limitations

- No real executable code verifier yet (the verifier is an LLM rubric / mock heuristic) — roadmap v1.
- Streaming for orchestrated modes streams the *final* answer after the loop completes (true token-streaming only in passthrough). Correctness/transparency first.
- A learned (GPU) router is intentionally out of scope. v0 is heuristic + verify/escalate.

## Learnings / analysis (the non-obvious stuff)

- **The verifier MUST be independent of the router**, or the escalation loop is inert (it only ever confirms its own start-tier choice). First implementation shared the difficulty signal → maestro scored **63% (below random's 88%)**. Splitting router (`classify`) from judge (`judgeDifficulty`) took it to **92%**. This mirrors the papers: the Verifier is a distinct role, and in production it's a strong model inspecting the *answer*, not the router's guess.
- **Slot labels are remappable metadata** (from the Fugu reverse-engineering): decoupling the routing policy from concrete model ids means you swap providers/models with zero retraining. Validated by the registry design.
- **Tier thresholds (0.33/0.7) ≠ pass bands** — a calibration gap is unavoidable with a cheap classifier; the verify/escalate loop exists precisely to absorb it. That's the whole point of cheap-first-then-escalate.
- **Honesty is the moat.** A suspicious 100% would be less credible than a reproducible 92% with the failures shown. (We even document a bug we found in our own random baseline.)
- **A cost optimizer must price-rank.** The router ignored price at first, so it tied a (buggy-but-cheap) baseline. Adding a price term to the guardrail score (`src/core/route.ts`) made it genuinely cost-optimal.

## Verification checklist for another agent

- [ ] `npm install && npm run typecheck` → exit 0
- [ ] `npm test` → 65 passing
- [ ] `npm run build` → `dist/cli.js` exists and `node dist/cli.js version` prints `0.1.0`
- [ ] `npm run eval` → maestro pass% and savings match the README
- [ ] `npm run eval -- --verbose` → inspect per-fixture routing decisions
- [ ] `bash scripts/verify.sh` → all endpoint smoke checks ✅
- [ ] Open `test.html` in a browser while `npm run serve` runs → all cards PASS
- [ ] Put a real key in `.env`, re-run `scripts/verify.sh` → "real routing works" with real cost
- [ ] Confirm `papers/` contains both source PDFs
- [ ] Grep for overclaims: the repo should never assert "beats model X" without a reproducible number
