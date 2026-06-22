#!/usr/bin/env bash
# Maestro examples. Start the server first:  npx maestro serve  (or MAESTRO_FORCE_MOCK=true for a priced demo)
set -euo pipefail
BASE=${BASE:-http://localhost:8080}

echo "# health"
curl -s "$BASE/healthz" | jq .

echo "# easy task -> stays cheap"
curl -s "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"Translate good morning to Spanish"}]}' \
  | jq '.maestro | {route, cost_usd, savings_pct}'

echo "# hard task -> escalates"
curl -s "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"Design a production-grade multi-region rate limiter, analyzing trade-offs and failure modes."}]}' \
  | jq '.maestro | {route, turns, cost_usd}'

echo "# dry-run the router (no model call)"
curl -s "$BASE/v1/route" -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"write a regex for emails"}]}' | jq .

echo "# pin a specific model (passthrough, no routing)"
curl -s "$BASE/v1/chat/completions" -H 'content-type: application/json' \
  -d '{"model":"maestro-auto","messages":[{"role":"user","content":"hi"}],"maestro":{"pin":"anthropic/claude-opus-4.8"}}' \
  | jq '.maestro.route'
