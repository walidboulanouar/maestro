# Executable code verifier via Vercel Sandbox

The executable verifier (run the model's code/tests instead of asking an LLM) was removed because `node:vm` is **not** a real sandbox - model code could escape and read your API key. The correct way to bring it back is to run untrusted code in a real isolated micro-VM. **Vercel Sandbox** is a great fit: ephemeral Firecracker micro-VMs, no access to your host or your keys, with a hard timeout.

This doc is the setup. If you do the 3 steps below and give me the credentials, I will wire it into Maestro behind `MAESTRO_CODE_VERIFY=vercel` and test it.

## What you do (instructions)

### 1. Have a Vercel project
Any Vercel account + one project. Sandbox runs are billed to it.

### 2. Get credentials (for self-hosting Maestro outside Vercel)
From the Vercel dashboard:
- **VERCEL_TOKEN** - Account Settings -> Tokens -> Create Token (scope it to the team).
- **VERCEL_TEAM_ID** - Team Settings -> General (the `team_...` id).
- **VERCEL_PROJECT_ID** - Project Settings -> General (the `prj_...` id).

(If Maestro is deployed *on* Vercel, you don't need these - it uses the automatic OIDC token. Locally/self-hosted, the 3 above are required.)

### 3. Put them in `.env` (gitignored) and enable the verifier
```bash
VERCEL_TOKEN=...
VERCEL_TEAM_ID=team_...
VERCEL_PROJECT_ID=prj_...
MAESTRO_CODE_VERIFY=vercel
```

That's it. Send me confirmation that those are set (don't paste the token here - it lives only in your local `.env`), and I'll implement + verify.

## What I will implement (so you know the shape)

A `src/core/sandbox-verify.ts` used only when `MAESTRO_CODE_VERIFY=vercel`:

```ts
import { Sandbox } from "@vercel/sandbox";   // pinned, added to deps

export async function verifyInSandbox(code: string, timeoutMs = 8000) {
  const sandbox = await Sandbox.create({ timeout: timeoutMs });   // ephemeral micro-VM
  try {
    await sandbox.writeFiles([{ path: "snippet.mjs", content: Buffer.from(code) }]);
    const run = await sandbox.runCommand({ cmd: "node", args: ["snippet.mjs"] });
    const ok = run.exitCode === 0;
    return { ok, error: ok ? undefined : await run.stderr() };
  } finally {
    await sandbox.stop();   // always tear down
  }
}
```

Then `verify()` for `task === "code"` extracts the fenced code, runs it in the sandbox, and returns ACCEPT (exit 0) / REVISE (non-zero, with the stderr as the reason). Isolation guarantees: the model's code runs in a fresh micro-VM with **no access to Maestro's process, env, keys, or filesystem**, and is killed at the timeout. This is the safe version of what `node:vm` could not do.

## Why this is safe (vs the removed node:vm version)
- `node:vm` shares the host process - `this.constructor.constructor('return process')()` escapes to the real `process` (env keys, fs, `process.exit`). Verified live during the security audit.
- Vercel Sandbox is a separate micro-VM - even a full breakout of the guest cannot touch the Maestro host or its secrets.

Status: **scaffolded in this doc, not yet wired** (needs your Vercel credentials). It stays off by default and a roadmap item until then.
