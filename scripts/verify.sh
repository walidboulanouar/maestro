#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
#  Maestro — full local verification. Run from the repo root:
#      bash scripts/verify.sh
#  It installs deps, typechecks, tests, runs the offline eval, builds, then
#  boots the server and smoke-tests every endpoint. If you set a provider key
#  in .env it also sends ONE real request so you can confirm real routing.
# ──────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")/.."

# load .env if present (export every non-comment line)
if [ -f .env ]; then set -a; . ./.env; set +a; fi

PORT="${MAESTRO_PORT:-8080}"
PASS=0; FAIL=0
ok()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
bad()  { echo "  ❌ $1"; FAIL=$((FAIL+1)); }
step() { echo; echo "▶ $1"; }

step "1/6 install deps"
if [ ! -d node_modules ]; then npm install --no-audit --no-fund >/dev/null 2>&1; fi
[ -d node_modules ] && ok "node_modules present" || bad "npm install failed"

step "2/6 typecheck";  npm run typecheck >/dev/null 2>&1 && ok "tsc clean" || bad "typecheck failed"
step "3/6 unit tests"; npm test          >/dev/null 2>&1 && ok "vitest passed" || bad "tests failed"

step "4/6 offline routing eval (deterministic, free)"
EVAL_OUT="$(npm run eval 2>/dev/null)"
echo "$EVAL_OUT" | grep -E "maestro|best-single|summary" | sed 's/^/    /'
echo "$EVAL_OUT" | grep -q "summary: maestro" && ok "eval ran" || bad "eval failed"

step "5/6 build"; npm run build >/dev/null 2>&1 && [ -f dist/cli.js ] && ok "dist/cli.js built" || bad "build failed"

step "6/6 live server smoke (port $PORT)"
MAESTRO_PORT="$PORT" node dist/cli.js serve >/tmp/maestro_verify.log 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null' EXIT
# wait for readiness
for i in $(seq 1 20); do curl -fsS "localhost:$PORT/healthz" >/dev/null 2>&1 && break; sleep 0.5; done

curl -fsS "localhost:$PORT/healthz"   >/dev/null 2>&1 && ok "GET /healthz" || bad "GET /healthz"
curl -fsS "localhost:$PORT/v1/models" >/dev/null 2>&1 && ok "GET /v1/models" || bad "GET /v1/models"

CHAT="$(curl -fsS "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"hello"}]}' 2>/dev/null)"
echo "$CHAT" | grep -q '"maestro"' && ok "POST /v1/chat/completions (has maestro block)" || bad "chat completions"

MSG="$(curl -fsS "localhost:$PORT/v1/messages" -H 'content-type: application/json' \
  -d '{"model":"claude-opus-4.8","max_tokens":128,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null)"
echo "$MSG" | grep -q '"type":"message"' && ok "POST /v1/messages (Anthropic / Claude Code)" || bad "messages"

curl -fsS "localhost:$PORT/v1/route" -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"write a regex"}]}' 2>/dev/null \
  | grep -q '"ladder"' && ok "POST /v1/route (dry-run)" || bad "route"

# real provider check
if [ -n "${OPENROUTER_API_KEY:-}${AI_GATEWAY_API_KEY:-}${LOCAL_OPENAI_BASE_URL:-}" ]; then
  step "BONUS: real model request (a provider key is set)"
  REAL="$(curl -fsS "localhost:$PORT/v1/chat/completions" -H 'content-type: application/json' \
    -d '{"model":"maestro-auto","messages":[{"role":"user","content":"In one sentence, what is a thread?"}]}' 2>/dev/null)"
  echo "$REAL" | python3 -c "import sys,json;d=json.load(sys.stdin);m=d['maestro'];print('    routed to:',[t['model'] for t in m['route']]);print('    cost USD:',m['cost_usd'],' vs frontier-only:',m['cost_vs_frontier_only_usd'],' savings:',str(m['savings_pct'])+'%');print('    answer:',d['choices'][0]['message']['content'][:160])" 2>/dev/null \
    && ok "real routing works" || bad "real request failed (check key/credit) — see /tmp/maestro_verify.log"
else
  echo; echo "  ℹ  no provider key in .env → ran on the built-in MOCK provider."
  echo "     add OPENROUTER_API_KEY to .env to verify REAL routing."
fi

echo; echo "──────────────────────────────"
echo "  RESULT: $PASS passed, $FAIL failed"
echo "──────────────────────────────"
[ "$FAIL" -eq 0 ] && echo "  🎉 Maestro verified end-to-end." || echo "  ⚠ see failures above."
exit $FAIL
