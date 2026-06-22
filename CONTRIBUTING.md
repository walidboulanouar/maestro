# Contributing to Maestro

Thanks for helping! Maestro is small on purpose — the value is a sharp routing brain, not a kitchen sink.

## Dev setup

```bash
npm install
npm run dev          # tsx watch, serves on :8080 (mock provider, no keys needed)
npm test             # vitest
npm run typecheck    # tsc --noEmit
npm run eval         # offline routing benchmark
```

## Ground rules

- **TypeScript, strict.** `npm run typecheck` must pass.
- **Tests for behavior.** New routing/verify logic needs a test in `test/`. `npm test` must pass.
- **Honesty over hype.** No benchmark claim that `npm run eval` can't reproduce. Show failures and regret, not just wins.
- **Pin dependencies exactly** (`save-exact=true` is set). Keep the dependency surface tiny — adding a dep needs a good reason.
- **Don't reinvent solved things** (gateways, inference, OpenAI serving). Maestro is the routing layer only.

## Good first issues

- New provider adapter quirks (header/format differences).
- Better heuristics in `src/core/classify.ts` (with eval evidence it helps).
- More eval fixtures in `eval/fixtures/` (especially mixed-capability traps).
- An executable code verifier (`src/core/verify.ts`) — run tests instead of asking an LLM.

## Architecture map

See [`docs/MAESTRO-PLAN.md`](docs/MAESTRO-PLAN.md) for the full design. Core flow lives in `src/core/` (`classify → route → orchestrator → verify`); the registry is in `src/registry/`; adapters in `src/providers/`.

By contributing you agree your work is MIT-licensed.
