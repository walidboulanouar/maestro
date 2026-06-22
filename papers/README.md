# The papers Maestro is built on

Maestro is an **open-source implementation of the ideas in Sakana AI's Fugu**, which are described in these two papers (included here for reference / reproducibility):

| File | Paper | What Maestro takes from it |
|---|---|---|
| [`TRINITY-fugu-2512.04695v3.pdf`](TRINITY-fugu-2512.04695v3.pdf) | **TRINITY** (arXiv 2512.04695) — the low-latency Fugu line | The per-request **route → worker → verifier loop** (≤K turns, terminate on ACCEPT), roles (Thinker/Worker/Verifier), and the principle that a tiny coordinator picks which model acts next. Maestro v0 implements this with a heuristic router + verify/escalate loop; the **learned** TRINITY head (frozen Qwen3-0.6B + tiny head) is the v2 roadmap. |
| [`Conductor-fugu-ultra-2512.04388v5.pdf`](Conductor-fugu-ultra-2512.04388v5.pdf) | **Conductor** (arXiv 2512.04388) — the Fugu-Ultra line | Multi-step **decomposition into a DAG of subtasks** with shared memory (≤5 steps). This is Maestro's `maestro-ultra` mode (v3 roadmap). |

## How Maestro maps to Fugu

- **Fugu = a single API that internally orchestrates a pool of models.** Maestro = the same idea, open-source, over *any* gateway (open + closed models, or local).
- **Fugu (TRINITY):** single worker per turn + verifier, ≤K turns → Maestro's `maestro-fugu` / `maestro-auto`.
- **Fugu Ultra (Conductor):** decompose → DAG → execute → Maestro's `maestro-ultra` (roadmap).
- **Slot labels are remappable metadata** (a key detail from the reverse-engineering): Maestro's registry maps abstract slots → concrete model ids, no retraining to swap providers.

A deeper, code-traced analysis (with our corrections against Sakana's official technical report) lives in [`../docs/`](../docs) and [`../archive/docs/`](../archive/docs).

> **Not affiliated with Sakana AI.** "Fugu" is Sakana AI's product/research. These PDFs are the authors' arXiv preprints, included for study and reproducibility under arXiv's distribution terms. Maestro contains no Sakana code or weights.
