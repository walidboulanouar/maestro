#!/usr/bin/env bash
# Verify a running Maestro server end-to-end. Usage:
#   npm run serve            # in another terminal
#   bash examples/setup-test.sh
# Override the URL with:  BASE=http://host:port bash examples/setup-test.sh
set -uo pipefail
BASE="${BASE:-http://localhost:8080}"
pass=0; fail=0
ok(){ echo "  ok   $1"; pass=$((pass+1)); }
no(){ echo "  FAIL $1"; fail=$((fail+1)); }

echo "Testing Maestro at $BASE"

curl -fsS "$BASE/healthz" >/dev/null 2>&1 && ok "GET /healthz" || { no "GET /healthz (is the server running? run: npm run serve)"; echo; echo "$pass passed, $fail failed"; exit 1; }

curl -fsS "$BASE/v1/models" 2>/dev/null | grep -q "maestro-auto" && ok "GET /v1/models lists maestro-auto" || no "GET /v1/models"

CHAT=$(curl -fsS "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"hello"}]}' 2>/dev/null)
echo "$CHAT" | grep -q '"maestro"' && ok "POST /v1/chat/completions (has maestro block)" || no "POST /v1/chat/completions"

curl -fsS "$BASE/v1/route" -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"write a regex"}]}' 2>/dev/null \
  | grep -q '"ladder"' && ok "POST /v1/route (dry-run)" || no "POST /v1/route"

curl -fsS "$BASE/v1/messages" -H 'content-type: application/json' \
  -d '{"model":"claude-opus-4.8","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}' 2>/dev/null \
  | grep -q '"type":"message"' && ok "POST /v1/messages (Claude Code)" || no "POST /v1/messages"

curl -fsS "$BASE/ui" 2>/dev/null | grep -q "Maestro" && ok "GET /ui (trace viewer)" || no "GET /ui"

echo
echo "$pass passed, $fail failed"
[ "$fail" -eq 0 ] && echo "Maestro is set up correctly." || echo "Some checks failed (see above)."
exit "$fail"
