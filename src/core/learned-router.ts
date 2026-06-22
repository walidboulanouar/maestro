/**
 * Learned router seam (v2 roadmap).
 *
 * The TRINITY-style learned router (frozen Qwen3-0.6B + a tiny head reading the
 * penultimate-token hidden state) cannot run in Node, so it runs as a small
 * Python sidecar exposing one endpoint. Set MAESTRO_ROUTER_URL to enable it;
 * Maestro POSTs the raw transcript and the sidecar returns the model/slot to
 * start with. Anything goes wrong -> we return null and fall back to the
 * heuristic router, so this is always safe to leave unconfigured.
 *
 * Sidecar contract:
 *   POST {url}   { "transcript": "role: content\n..." }
 *   200          { "model": "anthropic/claude-opus-4.8" }  // or { "slot": "frontier-coder" }
 *
 * Train it with `training/train_router.py` (see docs/LEARNED-ROUTER.md).
 */
export async function learnedRoute(
  transcript: string,
  url: string,
  timeoutMs = 1500,
): Promise<string | null> {
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ transcript }),
      signal: AbortSignal.timeout(timeoutMs),
    });
    if (!res.ok) return null;
    const json = (await res.json()) as { model?: string; slot?: string };
    return json.model ?? json.slot ?? null;
  } catch {
    return null; // never let the learned router break a request
  }
}
