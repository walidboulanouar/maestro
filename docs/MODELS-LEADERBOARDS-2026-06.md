# Maestro Model Pool — Live Leaderboard Cross-Check (June 2026)

> **Research date: 2026-06-22.** Every claim is from a live web fetch on 2026-06-21/22,
> cited inline with a URL + the date the leaderboard itself reports. Nothing here is from
> model memory (training cutoff is stale). **Vendor-reported vs independent scores are
> flagged.** Anything I could not verify is marked ⚠️.
>
> Companion doc: [`OSS-MODELS-2026-06.md`](./OSS-MODELS-2026-06.md) (deeper open-weight detail).

---

## 0. TL;DR — which board to trust for what

Different boards measure different things and **disagree by design**:

| Board | What it actually measures | June-2026 #1 |
|---|---|---|
| **Arena.ai Agent Arena** | Tool orchestration: reliability, task completion, steerability (the EXECUTOR signal) | Claude Fable 5 (High) |
| **Arena.ai Text Arena** (ex-LMArena/Chatbot Arena) | Human blind-vote chat preference | Claude Fable 5 |
| **Arena.ai WebDev/Code Arena** | Front-end + multi-file build preference | Claude Fable 5 |
| **Artificial Analysis Intelligence Index** | Composite of 10 evals (reasoning, coding, agentic, science, long-ctx) | Claude Fable 5 (max, w/ Opus 4.8 fallback) |
| **llm-stats.com** | Aggregated public benchmarks | Claude Mythos Preview |
| **SWE-bench Pro (Scale SEAL, standardized harness)** | Real repo bug-fix under identical scaffolding | GPT-5.4 (xHigh) |

**Key takeaway:** Anthropic's Fable 5 / Opus 4.x tops *preference + agentic* boards; OpenAI's
GPT-5.4/5.5 tops the *standardized SWE-bench Pro* coding board. Open weights (GLM-5.2) is the
clear open leader, cracking the closed top-10 on agentic and beating GPT-5.5 on WebDev preference.

⚠️ **Export-control caveat (recurring):** Anthropic's top closed models **Claude Fable 5** and
**Claude Mythos 5** were suspended for foreign nationals/non-US customers under a US export-control
directive dated 2026-06-12/13. They appear on leaderboards but **may be unavailable in EU** — a
direct argument for Maestro's open fallback tier. ([AA article](https://artificialanalysis.ai/articles/claude-fable-5-mythos-intelligence-index), 2026-06; [morphllm SWE-bench Pro](https://www.morphllm.com/swe-bench-pro), 2026-06-18)

---

## 1. Per-leaderboard top-10s

### 1.1 Arena.ai **Agent Arena** — agentic tool orchestration *(the user-provided board — VERIFIED)*
- **URL:** https://arena.ai/leaderboard/agent
- **Date:** 2026-06-18 · **909,592 sessions · 28 models**
- **Measures:** "how well models orchestrate tools for real-world agentic tasks, based on signals like tool reliability, task completion, and steerability."
- **Source identified:** This is **Arena.ai's Agent Arena** (the rebranded LMArena / LMSYS Chatbot Arena org). The user's 26-row list **matches exactly**, top to bottom.

| # | Model (variant) | Org | # | Model (variant) | Org |
|---|---|---|---|---|---|
| 1 | Claude Fable 5 (High) | Anthropic | 14 | Gemini 3.5 Flash | Google |
| 2 | Claude Opus 4.8 (Thinking) | Anthropic | 15 | Gemini 3.1 Pro Preview | Google |
| 3 | GPT-5.5 (xHigh) | OpenAI | 16 | DeepSeek V4 Pro | DeepSeek |
| 4 | Claude Opus 4.7 (Thinking) | Anthropic | 17 | Kimi K2.6 | Moonshot |
| 5 | GPT-5.5 (High) | OpenAI | 18 | Kimi K2.7 Code | Moonshot |
| 6 | Claude Opus 4.7 | Anthropic | 19 | DeepSeek V4 Flash | DeepSeek |
| 7 | Claude Opus 4.6 | Anthropic | 20 | MiniMax M3 | MiniMax |
| 8 | GPT-5.5 | OpenAI | 21 | Qwen 3.6 Plus | Alibaba |
| 9 | GPT-5.4 (High) | OpenAI | 22 | Grok Build 0.1 | xAI |
| 10 | GLM-5.2 (Max) [MIT] | Z.ai | 23 | Grok 4.3 (High) | xAI |
| 11 | Claude Opus 4.8 | Anthropic | 24 | Nemotron 3 Ultra | Nvidia |
| 12 | Claude Sonnet 4.6 | Anthropic | 25 | MiniMax M2.7 | MiniMax |
| 13 | GLM 5.1 | Z.ai | 26 | Gemini 3 Flash | Google |

Source: [arena.ai/leaderboard/agent](https://arena.ai/leaderboard/agent) (fetched 2026-06-22, board dated 2026-06-18).
**Cross-check status: CONFIRMED 1:1.** Note effort variants outrank their base model (Opus 4.8 Thinking #2 vs base Opus 4.8 #11; GPT-5.5 xHigh #3 vs base #8) — variants are distinct routable targets.

### 1.2 Arena.ai **Text Arena** (ex-Chatbot Arena) — human chat preference
- **URL:** https://arena.ai/leaderboard/text · **Date:** 2026-06-16 · **6,917,183 votes · 367 models**

| # | Model | Org | Elo | Open? |
|---|---|---|---|---|
| 1 | claude-fable-5 | Anthropic | 1508 | Closed |
| 2 | claude-opus-4-6-thinking | Anthropic | 1504 | Closed |
| 3 | claude-opus-4-7-thinking | Anthropic | 1502 | Closed |
| 4 | claude-opus-4-6 | Anthropic | 1499 | Closed |
| 5 | claude-opus-4-7 | Anthropic | 1493 | Closed |
| 6 | muse-spark | Meta | 1487 | Closed |
| 7 | gemini-3.1-pro-preview | Google | 1486 | Closed |
| 8 | gemini-3-pro | Google | 1486 | Closed |
| 9 | claude-opus-4-8-thinking | Anthropic | 1483 | Closed |
| 10 | gpt-5.5-high | OpenAI | 1481 | Closed |

(11 gpt-5.4-high 1478 · 12 claude-opus-4-8 1478 · 13 gemini-3.5-flash 1476 · 14 gpt-5.2-chat 1475 · 15 **glm-5.1 1475 — top open, MIT**)
Source: [arena.ai/leaderboard/text](https://arena.ai/leaderboard/text) (board dated 2026-06-16).
**Note divergence:** On *preference* Opus 4.6/4.7 sit ABOVE Opus 4.8 — newer ≠ higher Elo. Top tier clustered in ~33 Elo.

### 1.3 Arena.ai **WebDev / Code Arena** — build-task preference
- **URL:** https://arena.ai/leaderboard/code · **Date:** 2026-06-19 · **391,241 votes** (muse-spark deprecated 06-18)

| # | Model | Org | Score | Open? |
|---|---|---|---|---|
| 1 | claude-fable-5 | Anthropic | 1654 | Closed |
| 2 | **glm-5.2 (max)** | Z.ai | 1593 | **Open (MIT)** |
| 3 | claude-opus-4-8-thinking | Anthropic | 1565 | Closed |
| 4 | claude-opus-4-7-thinking | Anthropic | 1563 | Closed |
| 5 | claude-opus-4-7 | Anthropic | 1557 | Closed |
| 6 | claude-opus-4-8 | Anthropic | 1542 | Closed |
| 7 | claude-opus-4-6-thinking | Anthropic | 1542 | Closed |
| 8 | seed-2.1-pro-preview | Bytedance | 1539 | Closed |
| 9 | claude-opus-4-6 | Anthropic | 1538 | Closed |
| 10 | qwen3.7-max-20260517 | Alibaba | 1530 | Closed |

(11 glm-5.1 1529 open · 12 claude-sonnet-4-6 1521)
Source: [arena.ai/leaderboard/code](https://arena.ai/leaderboard/code) (board dated 2026-06-19).
**Standout:** GLM-5.2 is #2 — an open MIT model beating every GPT and Gemini on front-end build preference.

### 1.4 **Artificial Analysis Intelligence Index** — composite (10 evals)
- **URL:** https://artificialanalysis.ai/leaderboards/models · **Date:** live (rolling 72h), fetched 2026-06-22

| # | Model (variant) | Org | Index | Open? |
|---|---|---|---|---|
| 1 | Claude Fable 5 (max, Opus 4.8 fallback) | Anthropic | 60 | Closed |
| 2 | Claude Opus 4.8 (max) | Anthropic | 56 | Closed |
| 3 | GPT-5.5 (xhigh) | OpenAI | 55 | Closed |
| 4 | Claude Opus 4.7 (max) | Anthropic | 54 | Closed |
| 5 | GPT-5.5 (high) | OpenAI | 53 | Closed |
| 6 | **GLM-5.2 (max)** | Z.ai | 51 | **Open (MIT)** |
| 7 | GPT-5.5 (medium) | OpenAI | 50 | Closed |
| 8 | Gemini 3.5 Flash | Google | 50 | Closed |
| 9 | Claude Sonnet 4.6 (max) | Anthropic | 47 | Closed |
| 10 | Gemini 3.1 Pro Preview | Google | 46 | Closed |

(11 Qwen3.7 Max 46 · 13 MiniMax-M3 44 open · 14 DeepSeek V4 Pro 44)
Source: [artificialanalysis.ai/leaderboards/models](https://artificialanalysis.ai/leaderboards/models) (fetched 2026-06-22).
⚠️ Note: a separate AA *article* quotes Opus 4.8 at **61.4** ("first above 60") under index v4.1 — versioning/scale differs from the live board's compressed scale; treat absolute numbers as version-dependent. ([AA article](https://artificialanalysis.ai/articles/claude-fable-5-mythos-intelligence-index)).
**Same effort-variant pattern:** GPT-5.5 appears 3× (xhigh 55 / high 53 / medium 50) — one model, three routable price-quality points.

### 1.5 **llm-stats.com** — aggregated public benchmarks
- **URL:** https://llm-stats.com · **Date:** continuous, fetched 2026-06-22

| # | Model | Org | Score | Ctx | $/M (blended) | Open? |
|---|---|---|---|---|---|---|
| 1 | Claude Mythos Preview | Anthropic | 65.2 | — | — | Closed |
| 2 | GPT-5.2 Pro | OpenAI | 63.3 | — | — | Closed |
| 3 | Claude Opus 4.8 | Anthropic | 61.3 | 1.0M | $7.22 | Closed |
| 4 | **GLM-5.2** | Zhipu/Z.ai | 59.0 | 1.0M | $1.73 | **Open (MIT)** |
| 5 | Claude Fable 5 | Anthropic | 58.6 | — | — | Closed |
| 6 | GPT-5.4 | OpenAI | 57.3 | 1.0M | $3.89 | Closed |
| 7 | GPT-5.1 | OpenAI | 56.4 | 400K | $2.22 | Closed |
| 8 | GPT-5.5 | OpenAI | 56.2 | 1.1M | $7.78 | Closed |
| 9 | Seed 2.0 Pro | ByteDance | 55.5 | — | — | Closed |
| 10 | GPT-5.1 Instant | OpenAI | 55.1 | 400K | $2.22 | Closed |

Source: [llm-stats.com](https://llm-stats.com) (fetched 2026-06-22). ⚠️ Mythos Preview / Fable 5 are export-restricted.
**Divergence:** llm-stats puts *Mythos Preview* and *GPT-5.2 Pro* above Fable 5/Opus 4.8 — different eval mix than AA or the arenas.

### 1.6 **SWE-bench Pro** — standardized vs vendor-reported coding
- **URL (standardized):** https://www.morphllm.com/swe-bench-pro · also [Scale SEAL](https://labs.scale.com/leaderboard/swe_bench_pro_public) · **Date:** 2026-06-18

| # | Model (variant) | Standardized (Scale SEAL, identical scaffolding) | Open? |
|---|---|---|---|
| 1 | GPT-5.4 (xHigh) | **59.1%** | No |
| 2 | Muse Spark | 55.0% | No |
| 3 | Claude Opus 4.6 (thinking) | 51.9% | No |
| 4 | Gemini 3.1 Pro (thinking) | 46.1% | No |
| 5 | Claude Opus 4.5 | 45.9% | No |
| 6 | Claude Sonnet 4.5 | 43.6% | No |
| 7 | Gemini 3 Pro | 43.3% | No |
| 9 | GPT-5 (High) | 41.8% | No |
| 12 | **Qwen3 Coder 480B** | 38.7% | **Yes** |

**Vendor-reported (each lab's own tuned harness):** Claude Opus 4.8 = **69.2%** vs its standardized best of 51.9%. Gaps of **10–30 points** between vendor and standardized. ⚠️ **Always treat vendor SWE-bench as non-comparable.**
Source: [morphllm.com/swe-bench-pro](https://www.morphllm.com/swe-bench-pro) (2026-06-18). Open leaders elsewhere: GLM-5.2 ~62.1 (vendor), MiniMax M3 59.0, Kimi K2.6 58.6 — all ⚠️ vendor-reported (see OSS doc).

### 1.7 **Aider Polyglot** — code-edit pass rate ⚠️ STALE
- **URL:** https://llm-stats.com/benchmarks/aider-polyglot · top rows still show GPT-5 (88.0%), Gemini 2.5 Pro (82.2%), o3 (81.3%), DeepSeek-V3.2 (74.5%) — i.e. **previous-gen models**.
- **Verdict:** This mirror has not been refreshed for the 5.x/Opus-4.x/GLM-5 generation. **Do not use as a current signal.** Prefer Arena Code + SWE-bench Pro for coding. (Aider's own board: [aider.chat/docs/leaderboards](https://aider.chat/docs/leaderboards/).)

---

## 2. Combined capability table (~15 models, closed + open)

Prices are per **1M tokens (input / output)**, USD, standard tier. "AA" = Artificial Analysis Intelligence Index (live board, §1.4). Effort variant is a **distinct routable target**.

| Model + variant | Org | Open/Closed | License | Context | Best board placements | Access | ~Price in/out |
|---|---|---|---|---|---|---|---|
| **Claude Fable 5 (High)** | Anthropic | Closed | proprietary | 1M | Agent #1, Text #1, WebDev #1, AA #1(60) | Anthropic API ⚠️ US-only (export) | ⚠️ premium, ~Opus-tier |
| **Claude Opus 4.8 (Thinking)** | Anthropic | Closed | proprietary | 1M | Agent #2; base AA #2(56) | Anthropic/Bedrock/Vertex | $5 / $25 (Fast mode $10/$50) |
| **Claude Opus 4.7 (Thinking)** | Anthropic | Closed | proprietary | 1M | Agent #4, Text #3 | Anthropic/Bedrock/Vertex | ~$5 / $25 |
| **Claude Opus 4.6 (Thinking)** | Anthropic | Closed | proprietary | 1M | Text #2, SWE-Pro #3 (51.9% std) | Anthropic/Bedrock/Vertex | ~$5 / $25 |
| **Claude Sonnet 4.6** | Anthropic | Closed | proprietary | 1M | Agent #12, AA #9(47) | Anthropic/Bedrock/Vertex | ~$3 / $15 |
| **GPT-5.5 (xHigh)** | OpenAI | Closed | proprietary | ~1.1M | Agent #3, AA #3(55) | OpenAI API / Azure | $5 / $30 (cache $0.50) |
| **GPT-5.5 (High)** | OpenAI | Closed | proprietary | ~1.1M | Agent #5, Text #10, AA #5(53) | OpenAI / Azure | $5 / $30 |
| **GPT-5.4 (High/xHigh)** | OpenAI | Closed | proprietary | 1M | Agent #9, **SWE-Pro #1 (59.1% std)** | OpenAI / Azure | $3.89 blended (llm-stats) |
| **Gemini 3.1 Pro Preview** | Google | Closed | proprietary | ~1M+ | Text #7, AA #10(46), Agent #15 | Vertex / AI Studio | $2 / $12 |
| **Gemini 3.5 Flash** | Google | Closed | proprietary | ~1M | Agent #14, AA #8(50) | Vertex / AI Studio | $2.33 blended (llm-stats) |
| **Grok 4.3 (High)** | xAI | Closed | proprietary | (large) | Agent #23 | xAI API / OpenRouter | $1.25 / $2.50 (cache $0.20) |
| **GLM-5.2 (Max)** | Z.ai / Zhipu | **Open** | **MIT** | 1M in / 131K out | **Agent #10, WebDev #2, AA #6(51)** — top open everywhere | HF `zai-org/GLM-5.2`; Z.ai/OpenRouter | $1.40 / $4.40 (cache $0.26) |
| **DeepSeek V4 Pro** | DeepSeek | **Open** | **MIT** | 1M in / 384K out | Agent #16, AA #14(44), SWE-Verified ~80.6% (vendor) | HF; DeepSeek API | $0.435 / $0.87 (promo) |
| **DeepSeek V4 Flash** | DeepSeek | **Open** | **MIT** | 1M | Agent #19 (cheapest strong router) | HF; DeepSeek API | $0.14 / $0.28 |
| **Kimi K2.7 Code** | Moonshot | **Open** | Modified MIT | 256K | Agent #18 (coding-specialized) | HF; Moonshot API | ~$0.95 / $4.00 ⚠️ |
| **Kimi K2.6** | Moonshot | **Open** | Modified MIT | 256K | Agent #17, AA ~53.9 (one roundup) | HF; Moonshot API | ~$0.60 blended |
| **MiniMax M3** | MiniMax | **Open**⚠️ | ⚠️ license unconfirmed | 1M, native multimodal | Agent #20, AA #13(44), SWE-Pro 59.0% (vendor) | HF `MiniMaxAI/MiniMax-M3`; OpenRouter | $0.30 / $1.20 (launch 50% off) |
| **Qwen 3.6 Plus** | Alibaba | **Open** | Apache-2.0 | 1M | Agent #21 | HF; Qwen/OpenRouter | low (undercuts frontier) |
| **Nemotron 3 Ultra** | Nvidia | **Open** | Nvidia open | (large) | Agent #24 | HF; NIM | self-host |

Closed pricing/context: [devtk.ai pricing](https://devtk.ai/en/blog/ai-api-pricing-comparison-2026/), [finout Opus 4.8](https://www.finout.io/blog/claude-opus-4.8-pricing-2026-everything-you-need-to-know), [Claude pricing docs](https://platform.claude.com/docs/en/about-claude/pricing) — all fetched 2026-06-22.
Open pricing/context: [OpenRouter GLM-5.2](https://openrouter.ai/z-ai/glm-5.2), [Grok 4.3 OpenRouter](https://openrouter.ai/x-ai/grok-4.3), [llm-stats](https://llm-stats.com), plus [`OSS-MODELS-2026-06.md`](./OSS-MODELS-2026-06.md).
⚠️ Flags: Fable 5 export-restricted; MiniMax M3 license unconfirmed at launch; Kimi K2.7 Code pricing & all open SWE-bench Pro numbers are vendor-reported.

---

## 3. Recommended Maestro HYBRID pool — by ROLE and by EFFORT TIER

Design principle: **mix closed frontier (max quality + best agentic tool use) with open weights
(cost, self-host, EU-safe).** Each role lists a closed primary and an open fallback so Maestro
keeps running if a closed model is rate-limited, too pricey, or export-blocked in EU.

### By ROLE

| Role | Closed primary (quality) | Open fallback (cost / self-host / EU) | Why |
|---|---|---|---|
| **Generalist / orchestrator** | Claude Opus 4.8 (Thinking) | GLM-5.2 (Max) | Opus 4.8 Thinking = Agent #2; GLM-5.2 only open model in the closed agentic top-10 |
| **Agentic-coder / EXECUTOR (tool steps)** | Claude Fable 5 (High) → Opus 4.8 (Thinking) | GLM-5.2 (Max) | Use the **Agent Arena** order: Fable #1, Opus 4.8 Th #2, GPT-5.5 xHigh #3; GLM-5.2 #10 open |
| **Standardized bug-fix coding** | GPT-5.4 (xHigh) | Qwen3 Coder 480B / Kimi K2.7 Code | GPT-5.4 = **SWE-bench Pro #1 standardized**; pick by independent (not vendor) scores |
| **Cheap-fast router / classifier** | Gemini 3.5 Flash | DeepSeek V4 Flash ($0.14/$0.28) | High AA (50) at low cost; V4 Flash is the cheapest strong open router |
| **Long-context (≥1M)** | Gemini 3.1 Pro / Opus 4.8 (1M) | DeepSeek V4 Pro (1M) / MiniMax M3 (1M) | All 1M-class; open options for big-context at fraction of cost |
| **Math / reasoning** | GPT-5.5 (xHigh) / Opus 4.8 (Thinking) | GLM-5.2 (Max) / DeepSeek V4 Pro | Top AA composite incl. reasoning; GLM-5.2 strong on AIME/GPQA (⚠️ vendor) |
| **Vision / multimodal** | Gemini 3.1 Pro / Gemini 3.5 Flash | MiniMax M3 (native multimodal, open) | Gemini = strongest vision; MiniMax M3 the only open native-multimodal frontier |
| **WebDev / front-end build** | Claude Fable 5 | **GLM-5.2 (Max)** (WebDev #2, beats all GPT/Gemini) | Open model genuinely competitive here |

### By EFFORT TIER — when to spend

Treat **effort as a dial on the same model**, not a different model class. The arenas prove the
high-effort variant materially out-ranks its base (Opus 4.8 Thinking #2 vs base #11; GPT-5.5 xHigh
#3 vs base #8). Spend up only when the step justifies it.

| Tier | Use when | Targets |
|---|---|---|
| **xHigh / Thinking (max spend)** | Final-answer agentic steps, hard multi-tool plans, ambiguous reqs, code that must pass | Claude Fable 5 (High), Opus 4.8 (Thinking), GPT-5.5 (xHigh), GPT-5.4 (xHigh) |
| **High / Medium (balanced)** | Normal sub-tasks, drafting, mid-difficulty tool calls | GPT-5.5 (High/Medium), Opus 4.7, GLM-5.2 (Max), Gemini 3.1 Pro |
| **Cheap pass (first-attempt / fan-out / routing)** | Classification, routing, retrieval, large parallel fan-out, draft-then-verify | Gemini 3.5 Flash, DeepSeek V4 Flash, Qwen 3.6 Plus, Kimi K2.6 |

**Routing heuristic:** start cheap → escalate effort/model only on low confidence, verification
failure, or high task stakes. Reserve xHigh/Thinking for the executor's decisive tool steps.

---

## 4. Routing note — "model + effort" is the unit; Agent Arena = the executor signal

Maestro must route to a **(model, effort)** pair, never a bare model name. The live boards make
this unavoidable: GPT-5.5 occupies three AA rows (xhigh 55 / high 53 / medium 50) and Opus 4.8
Thinking sits 9 places above base Opus 4.8 on Agent Arena — same weights, different cost, latency,
and quality. So the catalog key is e.g. `anthropic/opus-4.8#thinking`, `openai/gpt-5.5#xhigh`,
`zai/glm-5.2#max`, each with its own price/latency/quality profile and independent rank.

For the **EXECUTOR / tool-step worker specifically**, use the **Arena.ai Agent Arena** order
(§1.1) as the primary signal — it directly measures tool reliability + task completion +
steerability over 909,592 real sessions, which is exactly what a tool-calling worker does. Do
**not** rank executors by Text Arena (chat preference) or by vendor SWE-bench (non-comparable
harnesses). For the *planner/reasoner* role, weight AA Intelligence Index + Text Arena; for the
*coder* role, weight standardized SWE-bench Pro. Keep an EU/export-safe open lane (GLM-5.2,
DeepSeek V4, MiniMax M3) wired in parallel so a Fable-5 export block never stalls a pipeline.

---

## One-paragraph summary

As of 2026-06-22, six live leaderboards agree the frontier is **Anthropic Claude Fable 5 /
Opus 4.8** and **OpenAI GPT-5.5/5.4**, but they rank by different things: **Arena.ai Agent Arena**
(the user-supplied board — VERIFIED 1:1, 909,592 sessions, 28 models, 2026-06-18) puts Fable 5
(High) #1 for tool orchestration; **Text** and **WebDev arenas** and the **Artificial Analysis
Intelligence Index** also crown Fable 5 / Opus 4.8; **llm-stats** instead tops out at Claude Mythos
Preview / GPT-5.2 Pro; and the **standardized SWE-bench Pro** (Scale SEAL) is led by **GPT-5.4
(xHigh) at 59.1%**, with vendor-reported numbers (Opus 4.8 69.2%) inflated 10–30 pts and not
comparable. The clear open-weight leader is **GLM-5.2 (MIT)** — Agent #10, WebDev #2 (beating every
GPT/Gemini), AA #6 — backed by **DeepSeek V4 Pro/Flash, MiniMax M3 (native multimodal), Kimi K2.x,
Qwen 3.6**. Two caveats matter for Maestro: Anthropic's Fable 5/Mythos 5 are **export-restricted
for non-US users** (strong reason for an open EU lane), and effort variants (Thinking/xHigh) are
genuinely distinct routable targets — so Maestro should key its catalog on **(model + effort)** and
use **Agent Arena** as the ranking signal for the executor/tool-step worker specifically. Open
SWE-bench Pro and Kimi-pricing claims remain **vendor-reported/unverified** and are flagged.
