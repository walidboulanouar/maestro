# Use Maestro in opencode

opencode supports any OpenAI-compatible provider via `@ai-sdk/openai-compatible`. Maestro is exactly that, so it drops in and routes every call across your pool.

## Setup

1. Start Maestro:

```bash
export OPENROUTER_API_KEY=sk-or-...
npm run serve            # http://localhost:8080
```

2. Add Maestro to your `opencode.json` (copy [`opencode.json`](opencode.json) into your project or `~/.config/opencode/`):

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "maestro": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Maestro",
      "options": { "baseURL": "http://localhost:8080/v1" },
      "models": { "maestro-auto": { "name": "Maestro (auto-route)" } }
    }
  }
}
```

3. In opencode, run `/models` and pick **Maestro -> maestro-auto**.

## What works

- Routing + verify/escalate behind one model id.
- Tool calls pass straight through; opencode keeps its own tool loop.
- Use `maestro-auto` (routed), `maestro-fugu` (fast), `maestro-ultra` (multi-step), or pin any real model id.

## Tip

Bias cost vs quality without touching opencode: start Maestro with `MAESTRO_PROFILE=cheap` (stay on cheaper tiers) or `MAESTRO_PROFILE=quality` (escalate sooner). Watch decisions live at `http://localhost:8080/ui`.
