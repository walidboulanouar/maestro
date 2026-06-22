# PDF Verification Notes

Sources verified locally:

- `/Users/boulanouarwalid/Downloads/2512.04695v3.pdf` — **TRINITY: An Evolved LLM Coordinator**
- `/Users/boulanouarwalid/Downloads/2512.04388v5.pdf` — **Learning to Orchestrate Agents in Natural Language with the Conductor**

## TRINITY Verification

The TRINITY paper confirms the core Fugu-style router idea:

- A small coordinator model uses hidden states as the routing representation.
- A lightweight head projects hidden state `h` into `L + 3` logits: `L` worker/model logits plus 3 role logits.
- The three roles are Thinker, Worker, and Verifier.
- The coordinator's generated text is discarded; only routing logits matter.
- SVF adapts a small set of backbone matrices by learning singular-value scales while keeping orthogonal matrices fixed.
- The learnable surface is below 20K parameters.
- Training uses sep-CMA-ES because the reward is sparse/noisy and imitation/RL are expensive.
- The serving loop halts when a Verifier accepts, or when the fixed turn budget is reached.
- The paper explicitly says tool integration is future work, so TRINITY should not be treated as a real tool-execution system.

Implementation nuance:

- The paper states the head can use hidden states from an earlier token to make rapid decisions.
- OpenFugu's reproduced checkpoint specifically verifies raw transcript formatting and the penultimate/position `-2` hidden state. That is an implementation finding from OpenFugu, not just a generic paper claim.

## Conductor Verification

The Conductor paper confirms the Ultra-style architecture:

- A 7B Conductor model writes full agentic workflows in natural language.
- Each workflow is parsed into three same-length Python lists:
  - worker/model IDs
  - natural-language subtasks
  - access lists describing which previous step outputs each worker sees
- Workflow steps execute sequentially.
- The final step output is the final Conductor response.
- Rewards first check whether the workflow format is parseable, then whether the executed workflow solves the task.
- Training uses GRPO.
- The main Conductor training setup starts from Qwen2.5, uses up to five workflow steps, and trains over a mixed reasoning dataset.
- Reported workers include closed models such as Gemini 2.5 Pro, Claude Sonnet 4, and GPT-5, plus open models such as DeepSeek-R1-Distill-Qwen-32B, Gemma3-27B, and Qwen3-32B.
- The paper reports training on 2 NVIDIA H100 80GB GPUs.
- Recursion is trained by allowing the Conductor to call itself as a worker, with an extra finetuning phase.

## Implications For Maestro

The papers support Maestro's gateway-first routing brain:

- Do not host or merge model weights.
- Route across model slots that can map to gateway model IDs.
- Keep v0 focused on classify, route, execute, verify/fallback, and trace.
- Treat learned routing and Conductor-style Ultra as post-v0.
- Build the benchmark harness before making superiority claims.
- Use the OpenFugu implementation notes carefully: some details are paper-level, while raw-transcript/position `-2` behavior is an OpenFugu reproduction detail.

## Bottom Line

The PDFs strengthen the Maestro scope in `docs/MAESTRO-SCOPE.md`: a self-hosted, gateway-first routing/orchestration brain is the practical product. The papers do not justify building local GPU inference, a broad multi-agent platform, or unsupported "beats Fugu" claims in v0.
