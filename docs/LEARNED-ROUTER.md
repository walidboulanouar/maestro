# Learned router (v2) - training recipe

Maestro v0/v1 use a heuristic classifier. v2 replaces it with the actual Fugu
mechanism: a tiny **learned** router. This is the recipe. It is scaffolded, not
shipped (it needs a GPU to train and a Python sidecar to serve), but the seam in
Maestro is real: set `MAESTRO_ROUTER_URL` and Maestro will ask the sidecar which
model to start with (falling back to the heuristic if anything fails).

## The mechanism (from TRINITY, arXiv 2512.04695)

- Backbone: **frozen Qwen3-0.6B**, forward pass exited early at a chosen layer.
- Read the **penultimate-token (-2) hidden state** `h in R^1024`.
- A tiny **linear head** `W in R^(K x 1024)` maps `h` to K logits = one per model
  slot (+ optional role logits). `argmax` picks the model.
- Trainable surface is tiny (~10-20k params: the head, optionally SVF offsets).

Two gotchas that cost the community weeks (do not skip):
1. Feed the **raw transcript** `"role: content\n"`, NOT a chat template
   (raw scored 95% vs 11% role accuracy in the reverse-engineering).
2. Read **position -2** and early-exit (skip the LM head + autoregressive decode);
   this is what makes it cheap (CPU-viable at inference).

## Data

Turn Maestro traces into `(transcript -> best model)` labels:
- For each logged request, the **oracle label** is the cheapest model whose answer
  passes (from an offline sweep, or from the verify outcomes in the trace).
- Redact before training (Maestro already redacts traces by default).
- Export JSONL: `{"transcript": "...", "label": "<slot>"}`.

## Train

`training/train_router.py` (stub) implements:
- load frozen Qwen3-0.6B, register a forward hook to capture `h[-2]`,
- a linear head over the K slots,
- optimize with **isotropic ES** (the paper's separable CMA-ES collapses to this
  at ~19k dims) on terminal task reward, or a simple cross-entropy on oracle
  labels to start. Optionally a soft-KL SFT warmup.
- save `router.npz` (head weights + slot list).

## Serve

`training/router_sidecar.py` (stub) loads `router.npz` + Qwen3-0.6B and exposes:

```
POST /   {"transcript": "role: content\n..."}  ->  {"slot": "frontier-coder"}
```

Run it, then point Maestro at it:

```bash
python training/router_sidecar.py            # serves on :8900
MAESTRO_ROUTER_URL=http://localhost:8900 maestro serve
```

## Done when

The learned router beats the heuristic on **oracle-route regret** on a held-out
set (run `npm run eval` style grading against ground truth), runs on CPU at the
sidecar, and is a single env var away. Until then, Maestro stays on the honest
heuristic and says so.
