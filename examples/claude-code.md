# Use Maestro in Claude Code

Maestro exposes the Anthropic Messages API at `/v1/messages`, so Claude Code can talk to it directly. Every request is routed across your model pool (cheap-first, verify, escalate) and your tool loop is preserved.

## Setup

1. Start Maestro with a provider key:

```bash
export OPENROUTER_API_KEY=sk-or-...
npm run serve            # http://localhost:8080
```

2. Point Claude Code at it:

```bash
export ANTHROPIC_BASE_URL=http://localhost:8080
export ANTHROPIC_API_KEY=unused          # Maestro uses YOUR provider key server-side
claude
```

That is it. Claude Code now sends every request through Maestro, which picks the best model for each turn.

## What works

- Text + streaming via `/v1/messages`.
- **Tool calls**: Claude Code keeps its own tool loop. Maestro forwards your tool schema, returns the model's `tool_use` blocks unchanged (`stop_reason: "tool_use"`), and you execute the tool and call back with a `tool_result`. Verified end-to-end on real models.

## Verify the tool loop yourself (raw Anthropic API)

```bash
# step 1: ask something that needs a tool
curl -s localhost:8080/v1/messages -H 'content-type: application/json' -d '{
  "model":"claude-opus-4.8","max_tokens":300,
  "tools":[{"name":"get_weather","description":"weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}],
  "messages":[{"role":"user","content":"Weather in Tokyo right now? Use get_weather."}]
}' | jq '.stop_reason, .content'
# -> stop_reason "tool_use", a content block {type:"tool_use", name:"get_weather", input:{city:"Tokyo"}}
# then append the assistant tool_use + a {type:"tool_result"} and call again to get the final text.
```

## Notes

- The `model` you set in Claude Code does not matter much; Maestro routes regardless. Use `MAESTRO_PROFILE=cheap|balanced|quality` to bias the routing.
- Anthropic `tool_use` mapping is implemented; streaming tool-calls are best-effort (use non-streaming for heavy tool turns if your client needs incremental deltas).
