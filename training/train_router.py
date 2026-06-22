#!/usr/bin/env python3
"""
Train the Maestro learned router (SCAFFOLD / v2 roadmap). See docs/LEARNED-ROUTER.md.

Input:  data.jsonl  with lines {"transcript": "role: content\\n...", "label": "<slot>"}
Output: router.npz  with { "W": (K, 1024) head, "slots": [...] }

This is a documented sketch, not runnable as-is (needs a GPU/CPU + transformers/torch).
The real loop: freeze Qwen3-0.6B, capture the penultimate-token hidden state for
each RAW transcript, then fit the linear head (cross-entropy on oracle labels to
start; isotropic ES on terminal reward for the full TRINITY objective).
"""
import json
import sys

# import numpy as np, torch
# from transformers import AutoModelForCausalLM, AutoTokenizer


def load_data(path: str):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    slots = sorted({r["label"] for r in rows})
    return rows, slots


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "data.jsonl"
    rows, slots = load_data(path)
    print(f"loaded {len(rows)} examples, {len(slots)} slots: {slots}")
    print("TODO (needs GPU): capture penultimate-token hidden states from frozen")
    print("Qwen3-0.6B over RAW transcripts, fit linear head, save router.npz.")
    print("See docs/LEARNED-ROUTER.md for the full recipe.")


if __name__ == "__main__":
    main()
