# Examples & use cases

Working setups for using Maestro. Start the server first (from the repo root):

```bash
export OPENROUTER_API_KEY=sk-or-...   # or AI_GATEWAY_API_KEY / LOCAL_OPENAI_BASE_URL
npm run serve                         # http://localhost:8080
```

Then pick your tool:

| File | Use case |
|---|---|
| [`claude-code.md`](claude-code.md) | Use Maestro as the model behind **Claude Code** (Anthropic API). |
| [`opencode.md`](opencode.md) + [`opencode.json`](opencode.json) | Use Maestro in **opencode** (OpenAI-compatible provider). |
| [`openai-sdk.py`](openai-sdk.py) | Python OpenAI SDK pointed at Maestro. |
| [`curl.sh`](curl.sh) | Raw curl: routing, escalation, dry-run, pinning. |
| [`setup-test.sh`](setup-test.sh) | Verify your local setup end-to-end (healthz, models, a routed chat). |

Everything works the same whether Maestro routes to OpenRouter, the Vercel AI Gateway, or your local Ollama/vLLM. Use model id **`maestro-auto`** to route, `maestro-fugu` for low-latency single+verify, `maestro-ultra` for multi-step decomposition, or any real model id for pass-through.
