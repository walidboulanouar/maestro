# archive/docs — how the two Fugu implementations work

Deep, code-traced architecture notes (with mermaid diagrams) for the two reference implementations vendored under `archive/`. Written from reading the actual source, not the READMEs. These are **study material** for building our own — see `../../docs/` for our design.

| Doc | Covers | Diagrams |
|---|---|---|
| [`01-openfugu-runtime.md`](01-openfugu-runtime.md) | **OpenFugu — inference/serve.** TRINITY routing (Qwen3-0.6B + SVF + 10-logit head, penultimate-token hidden state → 7 agent / 3 role logits), mini vs ultra (Conductor DAG + recursion), the OpenAI `/v1/chat/completions` flow, eval (+107% / caveats). | routing flowchart · serve sequence · mini-vs-ultra |
| [`02-openfugu-training.md`](02-openfugu-training.md) | **OpenFugu — training.** sep-CMA-ES on the head/SVF vector (mock→real), GRPO Conductor on ToolScale (1.21→1.64), recursion (+9% mock), adaptive k-of-n pool (+44%). Real run numbers from `results/`. | CMA-ES loop · GRPO loop · recursion/adaptive-pool |
| [`03-mol-routing.md`](03-mol-routing.md) | **Mixture-of-LoRA — routing.** LoRA-library markdown schema, hybrid router (model prompt-route + metadata guardrail + L0 fallback), multi-turn history masking, `route_decode_v2` + `lora_adapter_id` auto/L0–L4. | hybrid-router flowchart · multi-turn masking · combined flow |
| [`04-mol-serving.md`](04-mol-serving.md) | **Mixture-of-LoRA — serving.** PYTHONPATH SGLang overlay (no source edits), shadow-LoRA prep, the KV-reuse v2 patch (scheduler/mem_cache/lora-layer hunks), request path + adapter switch mid-decode. | request path · route-decode sequence · shadow-LoRA + overlay |

## The two systems in one line each
- **OpenFugu** = route across a pool of **frontier models** (litellm); a tiny *learned* coordinator (CMA-ES/GRPO) picks the worker per query/step, behind one endpoint. (Sakana Fugu reimplementation.)
- **Mixture-of-LoRA-Harness** = route across **specialist LoRA adapters** on one base (SGLang), hybrid router + L0 fallback, with KV reuse for the shared prefix.

> Two routing substrates — *which model* vs *which adapter* — that our own version (`maestro`) is designed to unify. See `../../docs/ARCHITECTURE.md`.

Source repos: [trotsky1997/OpenFugu](https://github.com/trotsky1997/OpenFugu) · [MindLab-Research/Mixture-of-LoRA-Harness](https://github.com/MindLab-Research/Mixture-of-LoRA-Harness) · [Sakana Fugu](https://sakana.ai/fugu-release/). See `../NOTICE`.
