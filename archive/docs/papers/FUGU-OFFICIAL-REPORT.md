# Sakana Fugu — OFFICIAL Technical Report (ground truth)

> **Source of record:** `Fugu_technical_report.pdf`, "Sakana Fugu Technical Report," Fugu Team, Sakana AI, dated **2026-6-22**, **31 pages** (19 pages main body + refs pp.20–25 + Appendices A–C pp.26–31). 6.3 MB.
> Cloned from `github.com/SakanaAI/fugu` (HEAD pushed 2026-06-22T07:46Z). Repo has **no license / no description** set (`gh api repos/SakanaAI/fugu`). Read in full via the Read tool (pages 1–31), every page incl. tables, figures, references, appendices.
> Every claim below cites the page / section / table / figure it came from. Where a prior in-house doc conflicts with the report, the **report wins** and the conflict is logged in §9.
>
> **Repo contents (is there code?):** No model code, no training code, no inference code. The repo is the **report + a Codex installer**. Concretely: `Fugu_technical_report.pdf`, `README.md`, four figure PNGs in `assets/figures/`, and an **install harness** — `scripts/install.sh` (~1150 lines), `scripts/codex-fugu`, `configs/files/fugu.json` (model card for the `fugu` / `fugu-ultra` slugs), `configs/injects/model_providers.sakana.toml` (points Codex at `https://api.sakana.ai/v1`, `wire_api = "responses"`), and TOML format/inject files. So Fugu ships as a **hosted API consumed through OpenAI Codex**, not as open weights or open orchestration code. (Supply-chain note: ran `pin-guard scan` on the clone — only benign npmrc WARN/INFO, **no `ioc-*` findings**. The README's `curl … | bash` installer and `install.sh` are a legit Codex-installer wrapper; not executed.)

---

## 1. One-paragraph overview

Sakana Fugu is a **family of learned orchestrator models exposed as a single LLM API** ("a multi-agent system delivered as one model"). Given a query, a Fugu model builds an agentic scaffold over a pool of frontier worker LLMs — deciding which workers to involve, what instructions/roles to assign, how outputs are combined/verified, and when to synthesize (p.4 §3). Two public variants: **Fugu** (latency-aware, selects a **single worker per input**, decision-only routing head, builds on **Trinity**) and **Fugu-Ultra** (quality-first, composes **multi-agent workflows of up to 5 steps per input** with shared memory, builds on the **Conductor**, trained with GRPO). The report positions "learned orchestration" as a **new scaling axis** complementary to bigger models (p.2–3 §1). It reports SOTA-or-near-SOTA on a suite of coding/reasoning/agentic benchmarks (Table 1, p.11) plus qualitative AutoResearch, classical-Japanese reading-order, CAD, Rubik's-cube-solver, blindfold-chess, and online-trading case studies (§4.3, Appendix B).

---

## 2. Architecture — Fugu vs Fugu-Ultra (exactly as the report states it)

### 2.1 Fugu (p.5–7, §3.1) — "Balancing Performance and Latency"
- **Builds on Trinity** (cites "Xu et al., 2025"), "scales and adapts the learned-orchestration idea to a production setting" with low-latency, multi-turn end-to-end constraints (p.5 §3.1).
- **Parametrization (p.5–6 §3.1.1, Figure 2):** a **lightweight selection head operating in parallel to the base model's LM head**. It takes a hidden state `h ∈ ℝ^d` from the orchestrator backbone and outputs **L logits, one per worker model** in the pool. Figure 2 caption: head families are **"Linear | Low-rank | Sparse | Block-diagonal."**
- **Key difference vs Trinity (Figure 2 caption + p.6):** *"Unlike the Trinity coordinator, Fugu does **not** assign roles. The selected model is always invoked as a worker"* — this narrows the coordination space to **model selection alone**, lowering latency.
- Also **fine-tunes the singular-value scales** of selected parameter matrices in the backbone's LM layers (Figure 2 caption; p.6 — "singular-value fine-tuning," cites Sun et al. 2025 = Transformer²).
- **Decision-only, logits-not-text (p.6):** "Fugu uses the orchestrator's **logits rather than its generated text** … compute a hidden state at an early token position, apply the selection head, and immediately dispatch the query to the selected worker, without the expensive autoregressive decoding process." This is what makes it cheap/low-latency and makes evolutionary optimization practical.
- **Training (two stages):**
  1. **SFT on single-step tasks (p.6 §3.1.2):** run every worker `M_j` (`j=1…K`) on each question `q_i` for `n` repetitions, get a reward vector `s_i`, convert to a **soft target distribution over workers via softmax with temperature τ** (Eq. 1), train the head + SVF scales to **minimize KL** to that distribution (Eq. 2, `L_SFT = (1/|D|) Σ D_KL(p_i(·) ‖ π_θ(·|q_i))`).
  2. **Evolutionary strategies on end-to-end tasks (p.6–7 §3.1.3):** **sep-CMA-ES** (same approach as Trinity). Maximize expected terminal reward `J(θ)=E_{τ∼π_θ}[R(τ)]` (Eq. 3), `R(τ)∈{0,1}` = task completed. Eqs. 4–5 give the sep-CMA-ES sample/recombine updates (parent `θ_t`, step `σ_t`, diagonal covariance `D_t`, population `λ`, top-`μ` fitness-weighted recombination). End-to-end tasks are **real multi-turn coding-assistant trajectories** from Claude Code, Codex, OpenCode (p.7).

### 2.2 Fugu-Ultra (p.7–9, §3.2) — "Prioritizing Performance"
- **Builds on the Conductor** (cites "Nielsen et al., 2025"), "adding novel extensions to accommodate long-horizon function-calling and multi-agent workflows through adaptive agent memory" (p.7 §3.2).
- **Conductor task (p.8 §3.2.1):** solves tasks **indirectly** by designing **agentic workflows in natural language**. Each workflow is a sequence of **workflow steps**; each step = a string **subtask** + an integer **assigned worker agent id** + an **access list** indexing which prior steps' subtask-solutions enter that worker's context. Final step's output = the answer `o_i`. Topologies range from best-of-N / sequential chains to **arbitrary parallelizable tree-structured** approaches.
- **Reward & RL (p.8 §3.2.1):** two progressive conditions — (1) **format**: `r_i=0` if subtask/agent/access lists can't be parsed; (2) **correctness**: `r_i=1` if final output matches solution `s_i`, **`r_i=0.5` otherwise**. Trained with **GRPO** (Shao et al. 2024), Eq. 6 = clipped surrogate with KL term, Eq. 7 = group-normalized advantage `A_i=(r_i−mean)/std`. The Conductor framework also **allows specifying the orchestrator itself as a worker agent** (p.8) — this is the report's only nod to recursion (see §9).
- **Function-calling extensions (p.8–9 §3.2.2):** Ultra must track **workflow state** (selected models, communication topology, assigned subtasks) so each agent's function-call loop is routed correctly across a multi-turn conversation.
  - **Intra-workflow agent isolation (p.9):** each agent observes others **only through the access list**, otherwise seeing a transcript of only its own prior actions — prevents "**orchestration collapse**" (first agent's trajectory dragging all others into redundant work).
  - **Persistent shared memory (p.9):** inter-workflow shared memory across agents lets them observe tool calls from *previous* workflows in a multi-turn conversation (avoid redundant re-discovery), while staying isolated within the current workflow.
- **Training setup (p.9 §3.2.3):** Ultra designs agentic workflows of **up to 5 steps** over a pool that **includes Gemini-3.1-Pro (Google DeepMind 2026), Claude-Opus-4.8 (Anthropic 2026c), and GPT-5.5 (OpenAI 2026a)**. Trained with the Conductor reward + GRPO and **without any KL divergence penalty**. Dataset = mix of public data + expert-designed end-to-end agent-user environments (the same end-to-end tasks from §3.1.3).

### 2.3 Mapping to Trinity / Conductor — confirmed
| Public model | Built on | Mechanism (per report) |
|---|---|---|
| **Fugu** | **Trinity** (Xu et al. 2025) | Lightweight **single-worker selection head** (logits, no role assignment) + SVF; SFT-to-soft-distribution then **sep-CMA-ES** on end-to-end coding trajectories. |
| **Fugu-Ultra** | **Conductor** (Nielsen et al. 2025) | RL-trained NL **workflow generator** (subtasks + agent ids + access lists), ≤5 steps; **GRPO, no KL**; + function-calling workflow-state tracking, intra-workflow isolation, persistent shared memory. |

This matches our prior TRINITY-deep / CONDUCTOR-deep mappings (§9).

---

## 3. The agent pool (§3.2.3, p.9 + §4.1.1 p.10)

**Exactly three frontier workers are named, in two places:**
- p.9 §3.2.3: pool "**includes Gemini-3.1-Pro** (Google DeepMind, 2026), **Claude-Opus-4.8** (Anthropic, 2026c), and **GPT-5.5** (OpenAI, 2026a)."
- p.10 §4.1.1: "a large pool of diverse models that includes state-of-the-art (SOTA) frontier LLMs such as **Gemini-3.1-Pro, Claude-Opus-4.8, and GPT-5.5**." Fugu is evaluated against "these same models, with the same maximum reasoning effort."

**How each is invoked:**
- In **Fugu**: a single worker is selected per input via the head's argmax over `L` logits, then the query is dispatched to that one worker (p.5–6).
- In **Fugu-Ultra**: workers are named by **integer id** inside the Conductor's `model_id` list, each given a NL subtask + an access list, executed sequentially, up to 5 steps (p.8). Qualitative §4.4 examples show Gemini/Opus/GPT placed as planners, builders, debuggers, verifiers, and **aggregators** (e.g. "Gemini-3.1-Pro at the head of the tree," "GPT-as-aggregator," p.17).

**Caveat — "recursive Fugu" is NOT in the report.** The report never states Fugu calls *instances of itself* in the pool. The closest statement is generic: the Conductor framework "**allows specifying the orchestrator itself as a worker agent**" (p.8) and Trinity's recursion lineage. The "+ recursive Fugu" pool membership in our FUGU-claims.md came from the **release/marketing page, not the report** — see §9 (newly-corrected).

**Worker pool is configurable (p.3 §1):** "agent pools can be configured to favor particular providers, exclude specific models, or respect data, privacy, and compliance constraints, without retraining."

---

## 4. FULL benchmark table — Table 1 (Model Card), p.11

> Table 1 caption (p.11): "Fugu models, through intelligent orchestration of leading frontier models, harness and amplify their differing skillsets to achieve new SOTA capabilities. **Best scores are in bold and second-best are underlined. Baseline scores are provider-reported wherever available.**"
> **Critical:** Table 1 has exactly **5 columns: Fugu-Ultra, Fugu, Claude Opus 4.8, Gemini 3.1, GPT-5.5.** There is **NO Fable-5 column and NO Mythos column in Table 1.** Fable-5 / Mythos Preview appear **only in Figure 4** (p.12) and on **3 benchmarks** (GPQA-D, CharXiv, Terminal Bench). (bold) = best, (underline) = 2nd-best, as printed.

| Benchmark | Fugu-Ultra | Fugu | Claude Opus 4.8 | Gemini 3.1 | GPT-5.5 |
|---|---|---|---|---|---|
| SWE Bench Pro | **73.7** | 59.0 | _69.2_ | 54.2 | 58.6 |
| Terminal Bench 2.1 | **82.1** | _80.2_ | 74.6 | 70.3 | 78.2 |
| LiveCodeBench | **93.2** | _92.9_ | 87.8 | 88.5 | 85.3 |
| LiveCodeBench Pro | **90.8** | 87.8 | 84.8 | 82.9 | _88.4_ |
| Humanity's Last Exam | **50.0** | 47.2 | _49.8_ | 44.4 | 41.4 |
| CharXiv Reasoning | **86.6** | _85.1_ | 84.2 | 83.3 | 84.1 |
| GPQA Diamond | **95.5** | **95.5** | 92.0 | _94.3_ | 93.6 |
| SciCode | 58.7 | **60.1** | 53.5 | _58.9_ | 56.1 |
| τ³ Banking | _20.6_ | **21.7** | _20.6_ | 8.4 | _20.6_ |
| Long Context Reasoning | 73.3 | **74.7** | 67.7 | 72.7 | _74.3_ |
| MRCRv2 | _93.6_ | 86.6 | 87.9 | 84.9 | **94.8** |

(GPQA-D: both Fugu and Fugu-Ultra print **95.5** in bold — joint best. τ³ Banking: Opus, Gemini-baseline and GPT all tie at 20.6 underlined; Fugu best at 21.7. Reading per printed bold/underline.)

**Where Fugu does NOT win in Table 1:**
- **MRCRv2** → **GPT-5.5 wins (94.8)**; Fugu-Ultra 2nd (93.6).
- **SciCode** → **plain Fugu wins (60.1)**, Fugu-Ultra only 58.7 (orchestration *hurts* here — Ultra < Fugu < none-of-the-frontier-but Gemini 58.9 is 2nd).
- **τ³ Banking / Long Context Reasoning** → **plain Fugu** beats Fugu-Ultra (latency variant wins the quality variant).

### 4.1 Figure 4 (p.12) — the ONLY place Fable-5 / Mythos appear in scored bars
Figure 4 caption: "Fugu models exceed the capabilities of the **Mythos Preview and Fable 5** model class purely through intelligent orchestration." Bars (read from chart), 6 models per panel = Fugu-Ultra, Fugu, **Mythos Preview, Fable 5**, Gemini 3.1 Pro, GPT-5.5, Opus 4.8:
- **GPQA-Diamond:** Fugu-Ultra **95.5**, Fugu **95.5**, Mythos ≈94.6, Fable5 ≈92.6, Gemini ≈94.3, GPT ≈93.6, Opus ≈92.0.
- **CharXiv Reasoning:** Fugu-Ultra **86.6**, Fugu **85.1**, Mythos ≈86.1, Fable5 ≈83.3-area, GPT ≈84.1, Opus ≈84.2.
- **Terminal Bench 2.1:** Fugu-Ultra **82.1**, Fugu **80.2**, Mythos ≈80.4, Gemini ≈70.3, GPT ≈78.2, Opus ≈74.6.

### 4.2 Figure 1 (p.1) — the headline 8-panel grid
Figure 1 plots 8 benchmarks: **Terminal Bench 2.1, CharXiv Reasoning, GPQA-D, LiveCodeBench, SciCode, SWEBench Pro, Humanity's Last Exam (text), CTI-REALM.** Numbers match Table 1 for the seven that overlap. (**Note:** Figure 1's last panel is labeled **"CTI-REALM"** and HLE is labeled **"Humanity's Last Exam (text)"** — but **neither CTI-REALM nor an HLE-text split appears in the Table-1 model card**; Table 1 instead lists LiveCodeBench Pro, τ³ Banking, Long Context Reasoning, MRCRv2. CTI-REALM is mentioned **nowhere in the report's prose or Appendix A**. Treat Table 1 as authoritative; CTI-REALM in Figure 1 is unexplained — flagged in §9.)

### 4.3 Methodology & caveats (Figure 1 caption p.1; §4.1.1 p.10; Appendix A pp.26–27)
- **All baseline scores are provider-reported wherever available** (Fig 1 caption: "All scores other than Fugu's are reported by the model providers"; Table 1 caption repeats this).
- **"max-of-Fable5/Mythos" rule (Fig 1 caption):** "For Fable 5 and Mythos Preview, we **report the max of the two** if both scores are available on the same benchmark."
- **"not in pool" caveat (Fig 1 caption):** "**Neither of them [Fable 5, Mythos Preview] is in Fugu's agent pool as they are not publicly accessible.**" (Confirmed again p.11: Mythos/Fable5 "not publicly available.")
- **Per-benchmark eval config (Appendix A, p.26–27):** SWE Bench Pro = max 1000 turns, Mini-SWE-Agent, EvalScope v1.8.1; Terminal Bench 2.1 = EvalScope v1.8.1, 500 max turns, Terminus 2 harness or tbench.ai leaderboard; LiveCodeBench v6 = May 2023–Apr 2025, 1055 questions, scores from vals.ai; LiveCodeBench Pro = text-only, no tools, 2025 Q2 split, 5 retries; HLE = full 2500 samples incl. multimodal, **no tools**; CharXiv = **gpt-4o as LLM judge**, Opus-4.8 self-computed; GPQA-D = EvalScope default; SciCode = Artificial Analysis impl., Inspect AI, 288 subproblems, had to version-bump numpy/scipy/sympy; τ³ Banking = pass@4, **GPT-5.2 (low) as user simulator**; Long Context Reasoning = Artificial Analysis setup, Qwen3-235B equality checker; MRCRv2 = 8-needle up to 128k context. Mythos/Fable5 numbers pulled from "official Anthropic-reported material … and from official leaderboards and third-party services (tbench.ai, Artificial Analysis, Benchmarklist)."

### 4.4 Qualitative / expert-task results (§4.3, Appendix B — exact numbers)
- **AutoResearch (Table 2, p.14):** 123 autonomous experiments, 3 seeds, single H100. Final validation **BPB (lower better)**: **Fugu-Ultra 0.9774 ± 0.0019** (best single-seed **0.9748**); Model C 0.9781 (0.9766); Model B 0.9793 (0.9758); Model A 0.9822 (0.9799). Gains modest (0.0010–0.0051 BPB) but consistent. (Baselines anonymized Model A/B/C = Gemini 3.1 Pro high / Opus 4.8 max / GPT-5.5 xhigh, mapping intentionally varied.)
- **Classical Japanese reading order (Table 3, p.16):** mean NED (higher better) over 25 expert-annotated kana letters. **Fugu-Ultra 0.776**, Model A 0.642, **Fugu 0.473**, Model B 0.449, Model C "No completed run," seed heuristic baseline 0.116.
- **Rubik's cube solver (Table 4, p.27):** 300 scrambles. Fugu-Ultra 300/300 solved, **mean 19.72 HTM** (optimal 20), 72.6 s; Model A 300/300, 19.76 HTM, 67.2 s; **Fugu** 300/300, 21.15 HTM, **1.9 s** (~35× faster than deep solvers' ~70 s); Model B & Model C **0/300** (their solvers crashed).
- **Blindfold chess (Figure 9, Appendix B.2/C, p.28–31):** 4 illustrative games, **Fugu wins all 4** (incl. vs ~2100-Elo Stockfish 18); Fugu ACPL ≈18–30 vs opponents ≈46–72. Full move records given p.30–31. Explicitly "selected, illustrative games, not an aggregate or win rate."
- **Online sequential stock trading (Figure 10, Appendix B.3, p.29):** 50-week run on one anonymized equity ("STOCK_X"), $10k start. **Fugu-Ultra $11,943.22 ± $633.86 (+19.43% mean over 5 runs)**; other frontier models < +15%. Footnote: single equity / single 50-week window, "not generalizable trading performance."

---

## 5. TRANSPARENCY — what's disclosed vs hidden (the exact thing elie & the community asked for)

The community asked Sakana to disclose, **per benchmark: (a) % usage of each model, (b) cost per task, (c) number of output tokens.** Verdict after reading all 31 pages incl. appendices:

| Disclosure requested | Disclosed in report? | Where / what exactly |
|---|---|---|
| **Per-benchmark % usage of each model (routing distribution)** | **PARTIAL / QUALITATIVE ONLY.** No numeric table. | **Figure 5 (p.13)** "Per-task model distribution" — a bar chart giving **per-model distribution for only 3 tasks** (HLE, Terminal Bench, GPQA-Diamond) for **both** Fugu-Ultra and Fugu. Readable bar values (approx): **HLE** — Gemini .34/.10, Opus .31/.38, GPT .35/.52 (Ultra/Fugu); **Terminal Bench** — Gemini .02/.00, Opus .34/.14, GPT .64/.86; **GPQA-D** — Gemini .56/.46, Opus .25/.06, GPT .19/.48. Plus prose (p.12 §4.2): TB peaks on GPT-5.5; GPQA-D focuses on Gemini; HLE balanced; HLE math→GPT, chem/bio→Gemini. **No full per-benchmark usage table for the other 8+ benchmarks. No exact percentages printed (chart only).** |
| **Cost per task ($)** | **NOT DISCLOSED anywhere in the report.** | The report gives **no $ cost-per-task, no per-benchmark cost, no cost-adjusted-performance table.** (Contrast: the underlying Conductor *paper* has cost tables — Tables 5/6 — but the **Fugu report does not**.) Latency is described only qualitatively ("Fugu latency comparable to a direct frontier call," "Ultra trades latency for quality"); **no latency numbers, no $/token, no $/task.** Pricing ($/1M tokens, subscription tiers) lives on the **console/marketing pages, not the report.** |
| **Number of output tokens per benchmark** | **NOT DISCLOSED.** | The report reports **no output-token counts per benchmark / per task.** The only token-ish numbers are eval-config caps in Appendix A (e.g. SWE Bench Pro "max turns 1000," Terminal Bench "500 max turns," EvalScope 32GB/4CPU resource limits) and the config file's `truncation_policy limit: 10000` / `context_window: 1000000` — **none of which is a measured per-benchmark token spend.** |
| Number of agent calls / workflow steps per benchmark | **NOT DISCLOSED numerically.** | Only "up to 5 steps" cap (p.9) and qualitative trajectory descriptions (§4.4). No per-benchmark mean-#-calls table (the Conductor paper had ~3 avg calls; the Fugu report does not repeat it). |
| Recursion depth | **NOT DISCLOSED for Fugu.** | "up to 5 steps" workflow depth (Ultra). No recursion-depth knob/number stated; recursion only implied via "orchestrator-as-worker" (p.8). |
| Verification/synthesis mechanism | **DISCLOSED (qualitative).** | Ultra's access-list + final-step-as-answer (p.8); intra-workflow isolation + persistent shared memory (p.9); §4.4 "debate and aggregation," GPT/Gemini/Opus-as-aggregator, build-and-debug alternation, "bring in a specialist." No formal verifier role (unlike Trinity's Verifier). |

**Bottom line for elie's question:** **No.** The report does **not** disclose per-benchmark cost-per-task or output-token counts at all, and discloses model-usage % only as a **3-task bar chart (Figure 5)** with no exact numbers and no coverage of the full benchmark suite. The transparency the community asked for is **largely not provided** in the official report.

---

## 6. Verification / synthesis, recursion depth, latency, pricing — what the report actually says

- **Verification/synthesis:** Ultra synthesizes by making the **final workflow step's output the answer**; aggregation/verification happen as *emergent* workflow steps (a worker tasked to verify/aggregate), not a hard-coded verifier. §4.4 documents debate/tree topologies, dynamic choice of aggregator (Gemini for trivia, GPT for math), build-then-debug (GPT builds, Opus debugs), and bringing in a specialist (p.16–18). Fugu (non-Ultra) does **no synthesis** — it routes to one worker.
- **Recursion depth:** not quantified for Fugu; only "orchestrator-as-worker is allowed" (p.8) + "up to 5 steps" (p.9).
- **Latency:** **no numeric latency claims.** Qualitative only: Fugu latency "comparable to a direct call to a frontier model" (p.4, p.5); Ultra "trades additional latency for higher quality" (p.4, p.7). Decision-only head avoids autoregressive decoding to keep latency low (p.6).
- **Pricing:** **not in the report.** (Pricing/subscriptions are on console.sakana.ai per FUGU-claims.md, not the PDF.)

---

## 7. Authors & provenance (p.19)

**Team & Project Lead:** Yujin Tang. **Core (model):** Edoardo Cetin, Jinglue Xu, Qi Sun, Stefan Nielsen, Vincent Richard. **Core (infra):** Haruto Goda, Iaroslav Tymchenko, Nhan Nguyen. **Contributors:** Hyunin Lee, Mari Ashiga, Shashank Kotyan, So Kuroki, Tarin Clanuwat. Cite as "Sakana AI (2026)." (Overlaps the Trinity/Conductor author sets — Xu, Sun, Nielsen, Cetin, Tang all recur, confirming lineage.)

---

## 8. Config-file facts (from the repo, not the PDF — useful corroboration)

`configs/files/fugu.json`: both `fugu` and `fugu-ultra` have **context_window 1,000,000**, single reasoning level **"high"**, input modalities **text + image**, `supports_parallel_tool_calls: true`, `truncation_policy {mode: tokens, limit: 10000}`. Difference: **`fugu-ultra` has `supports_reasoning_summaries: true`**, plain `fugu` does **not** (priority 0 = fugu, 1 = fugu-ultra). `configs/injects/model_providers.sakana.toml`: base_url `https://api.sakana.ai/v1`, `wire_api = "responses"`, `env_key = SAKANA_API_KEY`, `stream_idle_timeout_ms = 7200000` (2h). Both model `base_instructions` are **safety guardrails about not killing your own runtime/PIDs** — i.e. Codex-agent safety text, consistent with Fugu being shipped as a Codex backend.

---

## 9. DOUBLE-VERIFY: prior in-house docs vs the OFFICIAL report

### 9.1 ✅ VERIFIED (prior docs agree with the official report)
1. **Product framing** — "multi-agent system delivered as one model," learned orchestration as a new scaling axis, hides selection/delegation/verification/synthesis. ✅ (FUGU-claims §1; report p.2–4.)
2. **Fugu = single-worker router, no role assignment; Ultra = ≤5-step multi-agent NL workflows w/ shared memory.** ✅ Exact match (FUGU-claims §2; report p.5–6, p.8–9). Quote confirmed verbatim: "Unlike the Trinity coordinator, Fugu does not assign roles."
3. **Lineage:** Fugu←Trinity (sep-CMA-ES, SVF, lightweight head, logits-not-text); Ultra←Conductor (GRPO, 0/0.5/1 reward, no KL, subtasks+access-list workflows). ✅ Match across FUGU-claims §3, TRINITY-deep, CONDUCTOR-deep, and report p.5–9 (Eqs. 1–7 all consistent: Conductor reward tiers 0/0.5/1, GRPO advantage, sep-CMA-ES updates).
4. **Named pool members = Gemini-3.1-Pro + Claude-Opus-4.8 + GPT-5.5.** ✅ (FUGU-claims §4; report p.9 §3.2.3 & p.10 §4.1.1.)
5. **Fable-5 / Mythos NOT in pool ("not publicly accessible"); "max-of-the-two" rule.** ✅ (FUGU-claims §4; report Fig 1 caption + p.11.)
6. **Pool is configurable / opt-out providers without retraining.** ✅ (FUGU-claims §4; report p.3 §1.)
7. **Numbers that DO match Table 1:** Terminal Bench 2.1 (FU 82.1 / F 80.2), GPQA-D (95.5/95.5), LiveCodeBench (93.2/92.9), CharXiv (86.6/85.1), SWEBench Pro (73.7/59.0), SciCode (58.7/60.1), LiveCodeBench Pro (90.8/87.8), τ³ Banking (20.6/21.7), Long Context Reasoning (73.3/74.7), MRCRv2 (93.6/86.6). ✅ All match.
8. **Weakness signals are real:** SciCode — Fugu-Ultra (58.7) < plain Fugu (60.1); SWEBench Pro plain Fugu only 59.0 (< Opus 69.2). ✅ Confirmed in Table 1. **MRCRv2 — GPT-5.5 (94.8) beats Fugu-Ultra.** ✅ Confirmed.

### 9.2 ❌ CONTRADICTED / NOT SUPPORTED BY THE REPORT (prior docs were wrong or marketing-sourced)
1. **❌ "CTI-REALM" as a scored benchmark with Opus 4.8 winning (69.6 > FU 69.4).** The official **Table 1 does NOT contain CTI-REALM at all.** CTI-REALM appears only as a **panel label in Figure 1 (p.1)** and is **mentioned nowhere in prose or Appendix A.** No Opus-69.6/FU-69.4 numbers appear anywhere in the report. → **The whole "CTI-REALM: Opus beats Ultra" claim in FUGU-claims §5/§8 is unsupported by the report.** (Flag: Figure 1 itself listing CTI-REALM but Table 1 omitting it is an internal report inconsistency — the report never gives CTI-REALM numbers.)
2. **❌ "Fugu LOSES to Fable 5 on SWEBench Pro (80.0), HLE (53.3), SciCode (60.2)."** The report's **Table 1 has no Fable-5 column**, and Fable-5/Mythos appear **only in Figure 4 on GPQA-D, CharXiv, Terminal Bench** — **none of which are SWEBench Pro / HLE / SciCode.** There is **no Fable5 80.0 / 53.3 / 60.2 anywhere in the report.** → The "Fugu loses to Fable5" narrative (FUGU-claims §5 table, §7, §8 targets) is **not substantiated by the official report.** (Those Fable5 numbers came from the earlier chart-image/marketing extraction, not the PDF.)
3. **❌ HLE numbers.** FUGU-claims §5 had Fugu **48.5** (text) and a separate product-page "47.2." **Report Table 1: HLE = Fugu-Ultra 50.0, Fugu 47.2, Opus 49.8.** So **Fugu = 47.2 (not 48.5)**, and **HLE has no Fable5 53.3**; Fugu-Ultra (50.0) is **best** on HLE in the report, Opus 2nd. → The "Fugu Ultra loses HLE to Fable5" claim is **contradicted**.
4. **❌ "+ recursive Fugu in the agent pool."** The report **never** states Fugu calls instances of itself in the worker pool. It only says the Conductor framework *allows* the orchestrator to be specified as a worker (p.8). The explicit "Fugu is itself an LLM that calls instances of itself recursively" is a **release-page/marketing quote**, **not in the technical report.** → Downgrade from "confirmed (report §3.2.3)" to "marketing claim; report only permits orchestrator-as-worker generically."
5. **❌ "tech report §3.2.3 = Gemini+Opus+GPT+recursive Fugu."** §3.2.3 (p.9) names only the three frontier models and says "**includes**" (non-exhaustive). It does **not** enumerate recursive Fugu. The §3.2.3 citation in FUGU-claims §4 is right about the 3 models, wrong about recursive Fugu.
6. **⚠️ Benchmark suite mismatch.** FUGU-claims §5 main table = {Terminal Bench, CharXiv, GPQA-D, LiveCodeBench, SciCode, SWEBench Pro, HLE, CTI-REALM} with Fable5/Mythos columns. **Report Table 1 = {SWEBench Pro, Terminal Bench, LiveCodeBench, LiveCodeBench Pro, HLE, CharXiv, GPQA-D, SciCode, τ³ Banking, Long Context Reasoning, MRCRv2}, 5 model columns, no Fable5/Mythos.** The prior doc conflated the marketing-grid (with Fable5/Mythos + CTI-REALM) and the report's model card. → Use the §4 table above as canonical.

### 9.3 🆕 NEWLY-LEARNED (in the official report, absent/under-stated in prior docs)
1. **🆕 Fugu's SFT stage uses a soft KL-to-worker-performance distribution (Eqs. 1–2, p.6)** — train head+SVF to match a temperature-softmaxed reward distribution over workers, *then* sep-CMA-ES. (Trinity-deep had sep-CMA-ES; the explicit **two-stage SFT-then-ES for Fugu** with the KL objective is new detail.)
2. **🆕 End-to-end training data = real multi-turn trajectories from Claude Code, Codex, OpenCode** (p.7) — Fugu is tuned on actual coding-assistant harness transcripts.
3. **🆕 "Orchestration collapse" + intra-workflow isolation + persistent shared memory** (p.9) — Ultra's specific memory architecture for multi-agent function calling. Richer than CONDUCTOR-deep's access-list description.
4. **🆕 Five qualitative expert benchmarks with hard numbers:** AutoResearch BPB (Table 2), classical-Japanese kana reading-order NED (Table 3), Rubik's-cube solver (Table 4), blindfold chess (Fig 9 + full PGNs), online stock trading +19.43% (Fig 10). None were in prior docs.
5. **🆕 Table 1 contains MRCRv2, Long Context Reasoning, τ³ Banking, LiveCodeBench Pro as first-class rows** with full 5-model comparisons — and **GPT-5.5 beats Fugu on MRCRv2 (94.8 vs 93.6).** New head-to-head loss.
6. **🆕 Figure 5 per-task model-distribution bars** (HLE/TB/GPQA only) — the closest the report gets to usage-% transparency.
7. **🆕 Eval harness specifics (Appendix A):** EvalScope v1.8.1, Mini-SWE-Agent, Terminus 2, gpt-4o as CharXiv judge, GPT-5.2-low as τ³ user simulator, vals.ai for LiveCodeBench, version-bumped numpy/scipy/sympy for SciCode. Strengthens reproducibility claims but also reveals **baseline scores are provider-reported, not re-run by Sakana** (a comparability caveat).
8. **🆕 Repo ships as a Codex backend** (install.sh + model_providers.sakana.toml → api.sakana.ai), 1M context, image input, Ultra-only reasoning summaries. No open weights/code.

---

## 10. Net takeaways for Maestro positioning (corrected against ground truth)

- Fugu's real, report-backed weak spots are: **SciCode (Ultra 58.7 < Fugu 60.1 — orchestration hurts)**, **plain Fugu on SWEBench Pro (59.0, far below its own Ultra 73.7 and below Opus 69.2)**, **MRCRv2 (GPT-5.5 94.8 beats Fugu-Ultra 93.6)**, and **τ³ Banking / Long Context where the cheap Fugu beats Ultra** (Ultra's extra cost buys nothing there).
- **Do NOT** repeat the "Fugu loses to Fable5 on SWEBench Pro / HLE / SciCode" or "Opus beats Ultra on CTI-REALM" lines — **the official report does not support them** (Fable5/Mythos aren't in those benchmarks; CTI-REALM has no numbers; HLE-text 53.3 doesn't exist in the report). The biggest *honest* "ceiling" critique is the **transparency gap**: Sakana publishes **no cost-per-task, no output-token counts, and only a 3-task usage chart** — exactly what the community asked for and didn't get (§5).
- Fugu's capability is **bounded by {Gemini-3.1-Pro, Opus-4.8, GPT-5.5}** plus orchestration uplift; gains over the best single member are **real but modest** (e.g. AutoResearch 0.001–0.005 BPB; benchmark deltas often 1–6 pts) and **sometimes negative** (SciCode, MRCRv2).

---

*End. All architecture/pool/benchmark/transparency facts above are transcribed from the official `Fugu_technical_report.pdf` (Sakana AI, 2026, 31 pp.) with page/section/table/figure citations; the §9 comparison is against archive/docs/papers/{FUGU-claims, TRINITY-deep, CONDUCTOR-deep}.md.*
