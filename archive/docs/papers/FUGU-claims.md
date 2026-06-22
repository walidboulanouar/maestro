# Sakana Fugu — Extracted Claims & Benchmark Audit

Sources (fetched 2026-06-22, quoted not from memory):
- Release blog: https://sakana.ai/fugu-release/
- Product page: https://sakana.ai/fugu/
- Console: https://console.sakana.ai
- Technical report PDF: https://github.com/SakanaAI/fugu/blob/main/Fugu_technical_report.pdf
  (raw: https://raw.githubusercontent.com/SakanaAI/fugu/main/Fugu_technical_report.pdf — dated 2026-6-22)
- Papers: TRINITY https://arxiv.org/abs/2512.04695 · Conductor https://arxiv.org/abs/2512.04388

> Note on benchmark sourcing: the per-benchmark numbers live in **chart images** (Figure 1, "benchmark-fugu-grid.png" / "benchmark-table.png") on the release page, NOT as an HTML table. The numbers below were recovered by extracting text from the technical-report PDF chart (`pdftotext`), which renders the same Figure 1. They cross-check against the chart-image numbers supplied in the task. Cells confirmed from the PDF are marked; cells not present in the chart are marked "not scored / blank."

---

## 1. Product framing

Fugu is pitched as **a single-model API that is secretly a multi-agent system** — the orchestration (model selection, delegation, verification, synthesis) is hidden from the caller.

- "Sakana Fugu is a multi-agent system that behaves like a single model." (release page)
- "Sakana Fugu is, a Multi-Agent System, Delivered as One Model" / tagline "One Model to Command Them All" (product page)
- "Fugu is itself a language model trained to call various LLMs in an agent pool, including instances of itself **recursively**." (release page) — recursion confirmed.
- "Fugu is itself a language model specialized to understand when to delegate, how agents should communicate, and how to combine their work into a single, reliable answer." (release page)
- "It manages **model selection, delegation, verification, and synthesis internally**, so the complexity of a multi-agent system never reaches your code." (release page)
- Tech report §3: "Given a user query, a Fugu model constructs an agentic scaffold over a pool of frontier LLM workers, deciding which workers to involve, what instructions or roles to assign, how intermediate outputs should be combined or verified, and when to synthesize the final answer. The user interacts with Fugu as if calling a single model, while internally the system can route, delegate, and coordinate across multiple specialized agents."
- "learned orchestration" is the explicit framing — tech report: "we believe **learned orchestration** carries broader implications… progress in AI need not depend solely on access to the largest training runs." Sakana positions orchestration as "a new complementary scaling axis beyond ever larger and expensive language models."

## 2. Fugu vs Fugu Ultra (the exact stated difference)

The stated axis is **latency vs maximum quality / multi-step depth**, and crucially **how many workers run per query**:

- **Fugu** = latency-aware default. "Fugu balances strong performance with low latency, making it the ideal default for everyday work." Tech report: Fugu "**selecting a single worker per input** so that its latency is comparable to a direct call to a frontier model, while still routing each query to the most capable agent."
- **Fugu Ultra** = max quality. "Fugu Ultra coordinates a **deeper pool of expert agents** to maximize answer quality on hard, high-stakes problems." Tech report: Ultra "**composing workflows of multiple agents per input**… trades additional latency for higher quality… for the most complex tasks." Ultra designs "agentic workflows of **up to 5 steps**."
- Engineering distinction (from report): **Fugu = single-worker routing only (no role assignment)**; **Ultra = multi-step, multi-agent workflows with shared memory**. They are two different architectures, not just a knob.

## 3. The two underlying papers

- **TRINITY** — *"TRINITY: An Evolved LLM Coordinator"*, Xu, Sun, Schwendeman, Nielsen, Cetin, Tang, ICLR 2026 (arxiv 2512.04695). Described: "Trinity uses a lightweight **evolved** coordinator to orchestrate multiple LLMs over several turns," assigning **Thinker / Worker / Verifier** roles. **Fugu (non-Ultra) builds on Trinity** but simplifies it — "Unlike the Trinity coordinator, Fugu does **not** assign roles. The selected model is always invoked as a worker," to cut coordination space and latency. Trained with sep-CMA-ES (evolutionary).
- **Conductor** — *"Learning to Orchestrate Agents in Natural Language with the Conductor"*, Nielsen, Cetin, Schwendeman, Sun, Xu, Tang, ICLR 2026 (arxiv 2512.04388). Described: "The Conductor is trained with **reinforcement learning** to discover natural-language coordination strategies." It "outputs full agentic workflows as natural language that divide an input task, allocate arbitrary subtasks, and define targeted communication." **Fugu Ultra builds on Conductor**, adding long-horizon function-calling, multi-agent workflows, and adaptive/shared agent memory. Trained with GRPO, no KL penalty.
- **Technical report**: "Sakana Fugu Technical Report," Fugu Team, Sakana AI, 2026 — confirmed downloadable (6.3 MB PDF, dated 2026-6-22).

## 4. The agent pool + caveats

- **Roster actually named in the tech report (§3.2.3):** the worker pool is "a diverse pool of frontier LLMs that includes **Gemini-3.1-Pro** (Google DeepMind, 2026), **Claude-Opus-4.8** (Anthropic, 2026c), and **GPT-5.5** (OpenAI, 2026a)." Plus Fugu can call **instances of itself recursively**. (The marketing pages call the pool "swappable / dynamically orchestrated" but do NOT name members; only the report names them.)
- **NOT in the pool:** "**Fable 5 and Mythos Preview** … Neither of them is in Fugu's agent pool as they are **not publicly accessible**." (tech report Figure 1 caption + release page). For these two, Sakana reports "the max of the two [provider-reported scores] if both scores are available on the same benchmark."
- **Export-controls framing:** Fugu delivers "frontier capability **without the risk of export controls**" (release page + tech report). Plus resilience claim: "If a single provider restricts access, Fugu dynamically routes around the disruption."
- **Opt-out / compliance:** "Control which agents can participate in Fugu's model pool. Opt out of specific providers or models to meet data, privacy, compliance, or organizational requirements." (Fugu only.) Report: "agent pools can be configured to favor particular providers, exclude specific models, or respect data, privacy, and compliance constraints, without retraining."

## 5. Full benchmark table

Models: **FU = Fugu Ultra**, **F = Fugu**, **Fable5**, **Mythos** (Mythos Preview), **Gem = Gemini 3.1 Pro**, **GPT = GPT 5.5**, **Opus = Opus 4.8**.
Note: Fable 5 and Mythos Preview each appear on only a SUBSET of benchmarks in Figure 1 (each benchmark's chart shows EITHER Fable5 OR Mythos as the closed-model comparator, never both). Confirmed cells = read from the report PDF chart; matches task-supplied chart values.

| Benchmark | Fugu Ultra | Fugu | Fable 5 | Mythos Prev | Gemini 3.1 Pro | GPT 5.5 | Opus 4.8 | Fugu Ultra wins? |
|---|---|---|---|---|---|---|---|---|
| Terminal Bench 2.1 | **82.1** | 80.2 | 80.4 | — (not in chart) | 70.3 | 78.2 | 74.6 | Wins overall (beats Fable5 80.4) |
| CharXiv Reasoning | **86.6** | 85.1 | — | 86.1 | 83.3 | 84.1 | 84.2 | Wins (beats Mythos 86.1) |
| GPQA-D | **95.5** | 95.5 | — | 94.6 | 94.3 | 93.6 | 92.0 | Wins / ties Fugu; beats Mythos |
| LiveCodeBench | **93.2** | 92.9 | 89.8 | — | 88.5 | 85.3 | 87.8 | Wins |
| SciCode | 58.7 | **60.1** | 60.2 | — | 58.9 | 56.1 | 53.5 | **LOSES** — Fable5 60.2 > Fugu 60.1 > FU 58.7 |
| SWEBench Pro | 73.7 | 59.0 | **80.0** | — | 54.2 | 58.6 | 69.2 | **LOSES big** — Fable5 80.0 ≫ FU 73.7; plain Fugu 59.0 is weak |
| Humanity's Last Exam (text) | 50.0 | 48.5 | **53.3** | — | 44.7 | 44.3 | 45.7 | **LOSES** — Fable5 53.3 > FU 50.0 |
| CTI-REALM | 69.4 | 67.5 | — | 68.5 | 56.0 | 67.3 | **69.6** | **LOSES** — Opus 4.8 69.6 > FU 69.4 (statistical tie) |

Extra benchmarks listed on the **product page** (HTML table, no Fable5/Mythos columns there):
LiveCodeBench Pro: FU 90.8 / F 87.8 / Opus 84.8 / Gem 82.9 / GPT 88.4.
τ³ Banking: FU 20.6 / F 21.7 / Opus 20.6 / Gem 8.4 / GPT 20.6.
Long Context Reasoning: FU 73.3 / F 74.7 / Opus 67.7 / Gem 72.7 / GPT 74.3.
MRCRv2: FU 93.6 / F 86.6 / Opus 87.9 / Gem 84.9 / GPT 94.8 (GPT 5.5 wins).
(The product-page table reports HLE as 47.2/50.0 for F/FU vs Opus 49.8 — slightly different framing than the Figure-1 "text" subset above; the release-page Figure 1 is the authoritative "(text)" split.)

## 6. API / availability / pricing

- "Sakana Fugu is generally available today." Both models via one **OpenAI-compatible API** (`fugu` and `fugu-ultra`, e.g. version `fugu-ultra-20260615`).
- **Not available in EU/EEA** ("while we work toward compliance with GDPR and EU-specific regulations").
- Subscriptions: **Standard $20/mo**, **Pro $100/mo** (10× Standard), **Max $200/mo** (20× Standard). Promo: subscribe before end of July 2026 → free second month.
- Pay-as-you-go (Fugu Ultra, per 1M tokens): **$5 input / $30 output / $0.50 cached input**; above 272K-token context: **$10 / $45 / $1.00**.
- Entry: https://console.sakana.ai/login · support fugu-support@sakana.ai

---

## 7. Marketing vs engineering fact

**Marketing (claims to discount / treat skeptically):**
- "One Model to Command Them All" / "single model" — it is NOT a model in the trained-weights sense; it's a thin routing/orchestration head over **other companies' APIs** (Gemini, Opus, GPT). The "single model" is a UX wrapper.
- "State-of-the-art" / "shoulder-to-shoulder with Fable 5 and Mythos Preview" — true only on a curated subset, and **the SOTA comparison deliberately excludes the strongest closed models from being beaten head-to-head** by declaring them "not publicly accessible," then comparing to provider-reported numbers. Fugu **loses to Fable 5 on SWEBench Pro, HLE, and SciCode.**
- "Without the risk of export controls" — a positioning slogan; Fugu's whole capability still **depends on the very frontier APIs** (US labs) it routes to. If those get export-controlled, Fugu's pool degrades.
- "Frontier capability" — Fugu's ceiling is **the max of its pool members**, not beyond it (except where orchestration/verification recovers a few points). On any benchmark, Fugu cannot exceed what a perfect router over {Gemini, Opus, GPT} could achieve — and indeed several Fugu numbers sit ≈ the best pool member.

**Engineering fact (credible, load-bearing):**
- The pool is concretely **Gemini-3.1-Pro + Claude-Opus-4.8 + GPT-5.5 + recursive Fugu** (report §3.2.3). Everything rests on these three vendor APIs.
- **Fugu = single-worker router** (decision-only head, logits not text → cheap, low latency); **Ultra = RL-trained Conductor producing ≤5-step multi-agent workflows with shared memory.** Two genuinely different mechanisms.
- The gains over the best single pool member are **real but modest** on most benchmarks (often 1–3 pts), and orchestration sometimes **underperforms** the best individual model (CTI-REALM: Opus alone 69.6 > Ultra 69.4; SWEBench Pro: Opus alone 69.2 > plain Fugu 59.0).
- Methods are real published research (TRINITY evolutionary coordinator; Conductor RL orchestrator), both ICLR 2026.

## 8. What Maestro must beat / where Fugu is beatable

**Where Fugu is weak / beatable:**
1. **SWEBench Pro** — plain Fugu only 59.0 (worse than Opus 4.8 alone at 69.2; far behind Fable5 80.0). Fugu Ultra 73.7 still loses to Fable5. **Agentic software-engineering is Fugu's softest flank** — beat it here.
2. **Humanity's Last Exam (text)** — Fugu Ultra 50.0 loses to Fable5 53.3. Hard-reasoning ceiling is not actually frontier.
3. **SciCode** — Fugu *Ultra* (58.7) is **worse than plain Fugu** (60.1) and loses to Fable5 (60.2): a sign orchestration adds noise on some scientific-coding tasks. Multi-agent overhead can hurt.
4. **CTI-REALM** — Opus alone (69.6) ≥ Fugu Ultra (69.4): on cyber-threat-intel, a single specialist model is as good as the whole orchestra → orchestration premium ≈ 0.
5. **Pool ceiling** — Fugu can only be as good as {Gemini, Opus, GPT}. A system that (a) includes a stronger member (e.g. Fable 5 / Mythos, the very models Fugu admits it can't access) or (b) does genuine cross-model synthesis that exceeds the best member, would structurally beat it.
6. **EU/EEA gap** — Fugu is unavailable in EU/EEA. A GDPR-compliant orchestrator wins that market by default.
7. **Latency cost of Ultra** — Ultra's quality comes from up-to-5-step multi-agent workflows = high latency + high token cost ($30/1M output, $45 above 272K). A cheaper/faster orchestrator at similar quality is a wedge.

**What Maestro should aim to beat (targets, Fugu Ultra unless noted):**
- Terminal Bench 2.1 > 82.1 · CharXiv Reasoning > 86.6 · GPQA-D > 95.5 · LiveCodeBench > 93.2 · SciCode > 60.2 (the Fable5 high) · SWEBench Pro > 80.0 (the Fable5 high — the real bar, not Fugu's 73.7) · HLE text > 53.3 (Fable5) · CTI-REALM > 69.6 (Opus).
- Strategic: include or match the closed frontier models Fugu excludes; prove synthesis that **exceeds the best pool member** (Fugu mostly doesn't); ship **EU-compliant**; and undercut Ultra's latency/$ per quality point.
