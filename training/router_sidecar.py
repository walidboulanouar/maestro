#!/usr/bin/env python3
"""
Maestro learned-router sidecar (SCAFFOLD / v2 roadmap).

Serves the TRINITY-style learned router so Maestro can call it via
MAESTRO_ROUTER_URL. This is a documented stub: the model-loading + head are
sketched, not production code, and it needs `router.npz` from train_router.py.

Contract (what Maestro expects):
    POST /   {"transcript": "role: content\\n..."}  ->  {"slot": "<slot>"}

Run (after training produces router.npz):
    pip install fastapi uvicorn transformers torch numpy
    python training/router_sidecar.py
    # then: MAESTRO_ROUTER_URL=http://localhost:8900 maestro serve
"""
import json
import os

# --- the real implementation would load these (kept commented; needs a GPU/CPU + weights) ---
# import numpy as np, torch
# from transformers import AutoModelForCausalLM, AutoTokenizer
#
# WEIGHTS = np.load("router.npz")          # { "W": (K,1024), "slots": [...] }
# tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype="auto")
# model.eval()
#
# def route(transcript: str) -> str:
#     ids = tok(transcript, return_tensors="pt").input_ids      # RAW transcript, not chat template
#     with torch.no_grad():
#         out = model(ids, output_hidden_states=True)
#     h = out.hidden_states[-1][0, -2]                          # penultimate token
#     logits = WEIGHTS["W"] @ h.numpy()
#     return WEIGHTS["slots"][int(logits.argmax())]

SLOTS = ["cheap-generalist", "mid-reasoner", "frontier-coder"]  # placeholder pool


def route(transcript: str) -> str:
    # placeholder heuristic so the sidecar is runnable before weights exist
    t = transcript.lower()
    if any(k in t for k in ("prove", "design", "architecture", "rigorous", "end-to-end")):
        return "frontier-coder"
    if any(k in t for k in ("code", "function", "bug", "math", "solve")):
        return "mid-reasoner"
    return "cheap-generalist"


def main() -> None:
    try:
        from fastapi import FastAPI  # type: ignore
        import uvicorn  # type: ignore
    except Exception:
        print("install: pip install fastapi uvicorn (and transformers torch numpy for the real model)")
        return
    app = FastAPI()

    @app.post("/")
    async def _route(body: dict) -> dict:  # noqa: ANN001
        return {"slot": route(str(body.get("transcript", "")))}

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8900")))


if __name__ == "__main__":
    main()
