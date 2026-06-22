# Maestro system prompts (optional personas)

These are **optional** role/persona prompts. They are **not** auto-injected — Maestro is a transparent proxy and never adds anything to your messages unless *you* do. Use one only if you want Maestro to answer with a specific persona/flow when you call it as an assistant.

| File | What it is |
|---|---|
| [`maestro-system-prompt.md`](maestro-system-prompt.md) | A long, full-featured assistant persona for "Maestro": defines tone, refusal/safety flow, tool etiquette, formatting, etc. Adapted and rebranded from a publicly-circulated frontier-assistant system prompt; every brand mention was replaced with "Maestro". Use it to give Maestro a strong default role. |

## How to use

Prepend the file's contents as a `system` message in your request:

```bash
SYS=$(cat prompts/maestro-system-prompt.md)
curl -s localhost:8080/v1/chat/completions -H 'content-type: application/json' -d "$(jq -n --arg s "$SYS" '{
  model:"maestro-auto",
  messages:[{role:"system",content:$s},{role:"user",content:"Help me design an API."}]
}')"
```

Or download it directly (raw) and load it in your client:

```
https://raw.githubusercontent.com/walidboulanouar/maestro/main/prompts/maestro-system-prompt.md
```

## Notes / honesty

- **Opt-in only.** Maestro's value is routing + transparency, not a forced persona. Injecting a system prompt by default would break drop-in OpenAI/agent compatibility, so we don't.
- **Provenance.** This persona was adapted/rebranded from a publicly-circulated third-party assistant prompt. Review and adjust it for your own product before shipping it to users; do not treat it as legally vetted boilerplate.
- It's long (~3.8k lines). That's intentional: it's a complete persona. Trim it to taste.
