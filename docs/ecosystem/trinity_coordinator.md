# nshkrdotcom/trinity_coordinator — TRINITY in Elixir

> Ecosystem note for **Maestro**. Reviewed 2026-06-22 against a fresh clone of
> `nshkrdotcom/trinity_coordinator` @ `64144a2` (last push 2026-05-29).
> Repo: <https://github.com/nshkrdotcom/trinity_coordinator>

---

## 1. Overview & maturity

`trinity_coordinator` is an **Elixir/Nx** re-implementation of the *inference path*
of Sakana AI's **TRINITY: An Evolved LLM Coordinator** (arXiv:2512.04695, ICLR 2026).
A frozen small model (`Qwen/Qwen3-0.6B`, Sakana-SVF-adapted) reads a transcript, the
runtime extracts a penultimate-token hidden state, a tiny learned head maps it to
agent + role logits, and a configurable provider boundary dispatches the selected
work to a live LLM. (README L19–24.)

| Signal | Value |
|---|---|
| Language | **Elixir** (orchestration); **Python** only for the Sakana parity/export reference under `priv/sakana_trinity/scripts/` |
| Stars | 37 |
| License | MIT |
| Last push | 2026-05-29 |
| Tests | ~322 test cases across 50+ files; README reports the fast suite as "1 doctest, 302 tests, 0 failures (24 excluded)" on CUDA (README L82–84). 24 tests gated (live-provider, large-SVD). |
| Maturity | **Active development; a faithful re-implementation of the TRINITY *inference/runtime* path, not the training path.** Self-described "active development" (README L30). |

**What it is / isn't.** It is a genuine, well-engineered, heavily-tested re-implementation
of TRINITY's *runtime* — real Qwen3-0.6B forward pass through Bumblebee/EXLA, real
penultimate-hidden extraction, a real Axon routing head loaded from the actual Sakana
artifact, and the Thinker/Worker/Verifier loop with verifier-`ACCEPT` termination. It is
**not** a training reproduction: the **sep-CMA-ES training lane has been explicitly
removed** from the mainline (README L411–418, L436–439). The project *consumes* the
published Sakana-adapted artifact (a 624 MB safetensors bundle on HuggingFace) rather than
re-training. So: faithful runtime reimpl + serious artifact/parity tooling, with training
deliberately out of scope.

A notable strength is the **parity rigor**: the Elixir SVD/SVF reconstruction is checked
byte-for-byte against the Python reference (`priv/sakana_trinity/reference/sakana_decompose_model.original.py`,
a 33-line `torch.svd` decomposer; full export in `scripts/export_sakana_trinity_safetensors.py`).
The Apple-Silicon lane was independently validated by Nx-core maintainers (README L400–404).

---

## 2. Architecture

### 2.1 Pipeline (faithful to the paper)

```
transcript -> Extractor.format -> Bumblebee Qwen3-0.6B (EXLA/EMLX/Emily)
            -> hidden state @ position -2 (penultimate token)
            -> CoordinationHead (imported Sakana {10,1024} linear head)
            -> {agent_logits :: {7}, role_logits :: {3}}
            -> RoleInjector (role-specific system prompt)
            -> AgentPool.dispatch -> :inference provider boundary
            -> Verifier (ACCEPT/REVISE) -> Trace (JSONL)
```
(README L455–463.)

The mapping to the paper is tight and the code cites it directly:

- **SLM + penultimate-token routing.** `Extractor.extract_penultimate_hidden_state/1`
  slices `seq_len - 2` from the final-layer hidden states, with a `seq_len <= 1` fallback
  to index 0 (`lib/trinity_coordinator/extractor.ex` L15–27, L278–280). This is exactly the
  paper's `<Head Input>` choice (TRINITY-deep §3.1) — and the paper's ablation shows using
  the *last* token instead collapses accuracy, so this detail matters.
- **Lightweight linear head.** `CoordinationHead.build_model/4` default is a single biasless
  `Axon.dense` of size `num_agents + num_roles` (7+3=10) over a 1024-dim input — i.e. the
  paper's winning linear head `z = Wh`, `W ∈ ℝ^{10×1024}` = 10,240 params
  (`coordination_head.ex` L46–48). It also implements the paper's `:block_diagonal` and
  `:sparse` ablation heads (L51–55, L309–342), matching Appendix A.4.
- **Tri-role loop.** Roles are `0 => Worker, 1 => Thinker, 2 => Verifier`
  (`orchestrator.ex` L29) — note the imported Sakana checkpoint emits role logits in the
  Python order `solver / thinker / verifier`, and "solver" is the paper's "Worker"
  (`role_injector.ex` L1–11; README L443–446). Default `max_turns = 5` (`orchestrator.ex` L30),
  matching the paper's K=5.
- **Verifier termination.** `Verifier.parse/2` extracts `ACCEPT`/`REVISE` (+ optional
  diagnosis); the loop terminates the first turn where role=Verifier and status=accepted
  (`verifier.ex` L42–79; `orchestrator.ex` L298–371). Unknown verifier text is treated as
  `:revised` for safety (`verifier.ex` L104–106). This is the paper's
  `τ = min{k : R_k=V and u_k=ACCEPT}` rule.
- **SVF.** The Sakana router vector splits into 9,216 SVF singular-value offsets + a
  {10,1024} router head (README L342–354); SVD/SVF reconstruction lives in
  `lib/trinity_coordinator/sakana/` (`svd.ex`, `head.ex`, `safetensors_slice.ex`).

### 2.2 One faithful *extension* beyond the paper

The orchestrator honors **Thinker steering**: a Thinker turn can emit
`<suggestion>...</suggestion>` + `<suggested_role>solver|verifier</suggested_role>`, and the
next turn's role is overridden accordingly (`thinker.ex` L27–46; `orchestrator.ex` L467–504).
The paper mentions the Thinker "may specify the role of the next agent" (TRINITY-deep §3.2),
so this is a reasonable reading of an under-specified part of the paper, taken from the
"supplemental Python submission" the authors treat as the executable spec.

### 2.3 Why Elixir, and the OTP angle

The orchestration loop is plain recursive functions, not a process-per-agent design. The
only OTP primitive used for the loop is `StateManager`, a **`use Agent`** holding the
conversation transcript (`state_manager.ex` L1–6). Budgets (wall-time, provider-calls,
verifier-revisions, cost) are tracked with Erlang `:counters` atomics
(`orchestrator.ex` L111–117, L758–1001). So the concurrency story is modest:

- **Elixir is used because Nx/EXLA/Axon/Bumblebee give a credible native ML stack on the
  BEAM** (real Qwen forward pass + SVD on GPU/Apple/CPU), plus first-class supervised tooling
  (`mix`, ExUnit, Credo, Dialyzer). That, not heavy concurrency, is the real reason.
- The TRINITY *inference* loop is **inherently sequential** (each turn depends on the prior
  turn's transcript), so OTP's actor concurrency buys little for a single run. Where BEAM
  *would* shine is **many concurrent coordination runs** behind one service — that's a
  service-layer concern this repo flags as future work, not something it leans on today.

### 2.4 Provider / model pool integration

- A configurable **`ProviderPool`** maps agent ids to provider specs via app env
  (`provider_pool.ex` L1–19). The built-in `:default` pool maps **all seven** agent ids to
  OpenAI `gpt-4o-mini` — the Sakana checkpoint's slot labels (`gpt-5`, `gemini-2.5-pro`, …)
  are treated as **training metadata, not provider bindings** (README L575–580; see
  `docs/agent_slot_provider_mapping.md`). A `gemini_cli_asm` pool routes through the
  Gemini CLI SDK.
- All providers go through one **`:inference` boundary** (`AgentPool` → `Inference` adapter;
  `agent_pool.ex` L145–198), with a `Mock` adapter for deterministic, key-free runs.
- Live calls are **gated** behind `--allow-live` or a "governed authority" packet that
  carries opaque credential refs and redacts secrets in traces
  (`governed_authority.ex`; README L600–624).

---

## 3. How to run it

```bash
git clone https://github.com/nshkrdotcom/trinity_coordinator.git && cd trinity_coordinator
mix deps.get
XLA_TARGET=cuda12 mix trinity.env.check        # preflight (rejects cuda13)
XLA_TARGET=cuda12 mix test                      # fast suite (live + big-SVD gated out)
mix trinity.artifact.fetch                      # ~624 MB SHA-pinned Qwen3 bundle from HF
# End-to-end, no provider budget spent:
XLA_TARGET=cuda12 mix trinity.route.demo --mock-provider --trace-out tmp/demo.jsonl
# Router-correctness eval (37 fixed cases):
XLA_TARGET=cuda12 mix run examples/qwen_router_prompt_eval.exs
```
Backends: `:cuda_exla` (default), `:host_exla`, `:binary` (pure-CPU), `:emlx`/`:emily`
(Apple Silicon), `:mock_tiny` (README L487–500). Apple/CPU need no `XLA_TARGET`.
Real GPU is needed for non-trivial routing; the mock lane runs anywhere.

---

## 4. Faithfulness vs the paper and vs OpenFugu's TRINITY

| Dimension | Paper (TRINITY-deep.md) | nshkrdotcom/trinity_coordinator | OpenFugu TRINITY (`archive/OpenFugu`) |
|---|---|---|---|
| Backbone | Qwen3-0.6B, frozen + SVF | **Same** (real Qwen3-0.6B, Sakana-SVF artifact) | Hidden-state featurizer + linear head (Python) |
| Head input | Penultimate-token hidden | **Same** (pos -2) | Hidden-state features |
| Head | Linear `{10,1024}` (winner) + 3 ablation variants | **Same** + block-diagonal/sparse variants | Linear head |
| Roles | Thinker / Worker / Verifier, K=5 | **Same** (+ Thinker role-steering) | Thinker/Worker pattern |
| Termination | first Verifier ACCEPT, else K | **Same** | similar |
| **Training** | **sep-CMA-ES** (the paper's core contribution) | **Removed from mainline** — consumes published artifact | **Implements sep-CMA-ES** (`train/train_trinity*.py`) + GRPO Conductor |
| Pool | 7 real models (3 closed + 4 open) | Configurable; default = 7× gpt-4o-mini (slots = metadata) | litellm pool |
| Serving | n/a | gated provider boundary + JSONL traces | OpenAI-compatible `/v1` serve |

**Verdict on faithfulness.** For the **runtime/inference path**, `trinity_coordinator` is
*more faithful and more rigorous* than OpenFugu — it loads the actual Sakana-adapted Qwen
weights and validates SVD/SVF parity against the Python reference, where OpenFugu builds a
fresh head over generic hidden features. For the **training path**, OpenFugu is more faithful
(it actually runs sep-CMA-ES and GRPO), whereas `trinity_coordinator` deliberately drops
training. They are complementary: OpenFugu = read→**train**→serve in Python; nshkr = a
production-shaped, parity-checked **inference runtime** in Elixir with no training.

---

## 5. Reusable for Maestro

> **Heads-up on the premise:** the task framed Maestro as "TS". As of this review the
> Maestro repo (`maestro/README.md`) is a **Python** scaffold (`python3 -m maestro.cli …`),
> while `docs/ecosystem/routing-tools-survey.md` describes the *target* as TS/Next.js. The
> notes below assume the TS/Next.js target but flag where language matters.

What's worth borrowing (ideas, not code — it's Elixir/Nx-specific and GPL-free MIT but
deeply tied to BEAM ML libs):

1. **The penultimate-token detail is load-bearing — copy it exactly.** Both the paper and
   this repo confirm last-token collapses. If Maestro ever does hidden-state routing, slice
   `seq_len-2` of the final layer (`extractor.ex` L15–27).
2. **Verifier-as-terminator contract.** The `ACCEPT`/`REVISE` + diagnosis parser with
   "unknown ⇒ treat as revise" is a clean, directly portable control contract for Maestro's
   Ultra verify/recurse stage (`verifier.ex`). Trivial to reimplement in TS.
3. **Enforceable budgets on the loop.** Wall-time / provider-calls / verifier-revisions /
   latency / estimated-cost budgets, each emitting a structured `:run_failed` trace
   (`orchestrator.ex` L758–1001). This is exactly the "confidence/cost gating" Maestro's
   survey calls the driving requirement — a good spec to mirror.
4. **Structured JSONL trace events** (`run_started`, `slm_extracted`, `route_selected`,
   `provider_called`, `turn_completed`, `run_completed`/`run_failed`) with hashed transcripts
   and redacted secrets — a strong template for Maestro's eval/audit logging.
5. **"Slot labels are training metadata, not provider bindings"** + a pluggable provider pool
   with a mock adapter (`provider_pool.ex`, `agent_pool/mock.ex`). Maestro should likewise
   decouple the *learned* agent index from the *deployed* model behind it.
6. **Artifact pinning discipline** — SHA-256 per-file manifest + verified fetch
   (`artifact_fetch/pin.ex`). Good model-asset hygiene if Maestro ever ships adapter/router
   weights.

### Is Elixir/OTP a better fit than Node workers for the orchestration loop? (honest take)

**For a single coordination run: no meaningful advantage.** The TRINITY loop is sequential
(turn N+1 needs turn N's output), so there's no intra-run parallelism to exploit — this repo
itself uses a plain recursive function + one `Agent`, not a process tree. Node/TS handles a
sequential async loop of awaited LLM calls perfectly well.

**For a multi-tenant *service* hosting many concurrent runs: BEAM has a real edge** —
cheap isolated processes, per-run supervision/crash isolation, preemptive scheduling so one
slow provider call can't starve others, and back-pressure primitives. But Maestro's actual
bottleneck is network-bound LLM latency, where Node's event loop is already adequate, and the
decisive reason to choose a runtime here is the **ML stack** (Qwen forward pass + SVD), not the
concurrency model. If Maestro stays a thin router that calls *hosted* models over HTTP and
does its routing with a small classifier or a remote scorer (per the survey's RouteLLM/FrugalGPT
pattern), **TS/Next.js is the right call** and OTP would be over-engineering. Elixir only wins
if Maestro decides to run the SLM + SVF locally on the BEAM the way this repo does — which is a
much bigger commitment than Maestro's current "route → hosted worker → verify" scope.

**Verdict:** Mine `trinity_coordinator` for *contracts and discipline* — penultimate-token
routing, the verifier-terminator, loop budgets, JSONL tracing, provider-pool/metadata
decoupling, and artifact pinning. Do **not** adopt Elixir/OTP for Maestro on concurrency
grounds alone; it's justified only if Maestro chooses to host the coordinator SLM locally.
For training, look to OpenFugu (sep-CMA-ES), since this repo intentionally omits it.
