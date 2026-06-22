# Using Maestro in your tools

Maestro is a **transparent OpenAI-compatible proxy** — exactly like pointing at OpenRouter, except multiple models are orchestrated behind one stable endpoint. So anything that speaks the OpenAI (or Anthropic) API works. **Bring your own OpenRouter key** and you instantly get access to every model OpenRouter serves, routed automatically.

## 1. Start Maestro

```bash
# zero keys (mock provider, for trying it):
npx openmaestro serve

# real models — one OpenRouter key unlocks the whole catalog:
OPENROUTER_API_KEY=sk-or-... npx openmaestro serve
```

It listens on `http://localhost:8080`. Use model id **`maestro-auto`** (routed) or any real model id (pass-through).

## 2. opencode

opencode supports any OpenAI-compatible provider via `@ai-sdk/openai-compatible`. Add to `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "maestro": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Maestro",
      "options": { "baseURL": "http://localhost:8080/v1" },
      "models": {
        "maestro-auto": { "name": "Maestro (auto-route)" }
      }
    }
  }
}
```

Then run `/models` in opencode and pick **Maestro → maestro-auto**. Tool calls pass straight through to the routed model.

## 3. Claude Code

Maestro exposes the Anthropic Messages API at `/v1/messages`, so Claude Code can talk to it directly:

```bash
ANTHROPIC_BASE_URL=http://localhost:8080 ANTHROPIC_API_KEY=unused claude
```

> Note (early): text + streaming work via `/v1/messages`. Anthropic-style `tool_use` block mapping is partial — for heavy tool-calling agents, the OpenAI-format integrations (opencode/Cursor) are the most complete today.

## 4. Cursor / Continue / any OpenAI client

Set the base URL to `http://localhost:8080/v1`, the API key to anything (Maestro uses your provider key server-side), and the model to `maestro-auto`.

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8080/v1", api_key="maestro")
client.chat.completions.create(model="maestro-auto", messages=[{"role":"user","content":"hi"}])
```

## Bring your own key

Maestro never ships keys. Put **your** OpenRouter key in `.env` (gitignored) or the environment:

```bash
OPENROUTER_API_KEY=sk-or-...     # one key → the full OpenRouter catalog, orchestrated
# or point at anything OpenAI-compatible you already run:
LOCAL_OPENAI_BASE_URL=http://localhost:11434/v1   # Ollama / vLLM / llama.cpp
```

## Tool calling & agent loops

Maestro is a **smarter OpenAI-compatible model endpoint, not a tool runtime** — your app/agent keeps its own tool loop. The contract:

1. You send messages + `tools` + `tool_choice`.
2. Maestro routes to the best model and **forwards the tool schema unchanged**.
3. Maestro returns the model's `tool_calls` **unchanged** (`finish_reason: "tool_calls"`), then stops.
4. **You** execute the tool and call Maestro again with the `tool` result message.
5. Maestro routes that follow-up and returns the final answer.

Verified end-to-end on real models (request → `tool_calls` → tool result → final answer). Maestro accepts full OpenAI tool-loop messages: assistant turns with `content: null` + `tool_calls`, `role: "tool"` + `tool_call_id`, and multimodal content arrays.

Notes (early / honest):
- **Requests that include `tools` are single-turn routed calls** — Maestro does *not* verify/escalate them, because the outer client owns the agent loop.
- **Streaming tool calls are best-effort:** Maestro buffers the model's output and emits the `tool_calls` in the final stream delta (not incremental indexed deltas yet). Non-streaming tool calls are fully OpenAI-shaped. Tested against the OpenAI SDK; if your client needs incremental tool-call streaming, use non-streaming for tool turns.
- Traces may contain tool-call **arguments** the model generated; the JSONL trace file is opt-in (`MAESTRO_TRACE_FILE`).

## What you get that raw OpenRouter doesn't

Same drop-in experience, plus: automatic cheap-first routing with a verify/escalate loop, and a `maestro` block on every response showing exactly which model(s) answered, why, and the cost. Tools and messages are passed through untouched — Maestro only decides *which model* handles each call.
