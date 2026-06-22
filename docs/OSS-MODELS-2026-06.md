# Top Open-Weight LLMs — Builder's Shortlist for Maestro (June 2026)

> Research date: **2026-06-22**. Every claim below is from a live web source fetched on
> 2026-06-21/22, cited inline with a URL. The author's training cutoff is stale, so nothing
> here is from memory. **Unverified / vendor-only claims are flagged explicitly.**
>
> Context: Maestro is our open answer to **Sakana Fugu** — an "LLM that orchestrates other
> LLMs". Fugu's pool is closed US APIs (Gemini 3.1 Pro, Claude Opus 4.8, GPT-5.5). Maestro's
> differentiator is a pool of **open-weight, self-hostable models** with no export-control or
> regional lockout. This doc picks that pool.

---

## 1. Ranked table — top open-weight models (June 2026)

Ranked primarily by **Artificial Analysis Intelligence Index** standing where available, then
by agentic/coding strength and practical usability (license + cost). "Open?" = weights public
+ self-hostable + commercially usable.

| # | Model (version) | Org | Released | Params (total / active) | Context | License | Open? | Headline benchmarks | Access & approx price | Sources |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | **GLM-5.2 (max)** | Z.ai / Zhipu | weights 2026-06-16 (announced 06-13) | 753B MoE / ~40B active | 1M in / 131K out | **MIT** | ✅ full | **#1 open weight on AA Intelligence Index = 51**. Vendor: SWE-bench Pro 62.1, Terminal-Bench 2.1 81.0, AIME 2026 99.2, GPQA Diamond 91.2, HLE+tools 54.7 ⚠️ vendor self-reported | HF `zai-org/GLM-5.2`; OpenRouter ~**$1.40 in / $4.40 out** per M | [AA leaderboard](https://artificialanalysis.ai/leaderboards/models), [TechTimes 06-17](https://www.techtimes.com/articles/318543/20260617/glm-52-open-weights-live-top-coding-benchmark-api-use-carries-china-data-risk.htm), [aimadetools guide](https://www.aimadetools.com/blog/glm-5-2-complete-guide), [OpenRouter GLM-5.2](https://openrouter.ai/z-ai/glm-5.2), [aiforanything review](https://aiforanything.io/blog/glm-5-2-review-2026) |
| 2 | **Kimi K2.7-Code** | Moonshot AI | 2026-06-12 | 1T MoE / 32B active (384 experts) | 256K | **Modified MIT** | ✅ (modified) | Coding-specialized, forced thinking, -30% reasoning tokens vs K2.6; +21.8% on Kimi Code Bench v2 ⚠️ vendor-only, no 3rd-party SWE-bench yet | HF weights; Kimi API ~**$0.95 in / $4.00 out** per M | [MarkTechPost 06-12](https://www.marktechpost.com/2026/06/12/moonshot-ai-releases-kimi-k2-7-code-a-coding-model-reporting-21-8-on-kimi-code-bench-v2-over-k2-6/), [explainx](https://www.explainx.ai/blog/kimi-k2-7-code-open-source-coding-model-2026), [codersera K2.7](https://codersera.com/blog/kimi-k2-7-complete-guide-2026/) |
| 3 | **Kimi K2.6** | Moonshot AI | 2026-04 | 1T MoE / 32B active | 256K | Modified MIT | ✅ (modified) | AA Intelligence Index **53.9** (top open per one June roundup); SWE-bench Verified 80.2%, **SWE-bench Pro 58.6%** — first open model to beat GPT-5.4 (xhigh) on SWE-bench Pro | HF weights; Kimi API / OpenRouter | [deeplearning.ai The Batch](https://www.deeplearning.ai/the-batch/kimi-k2-6-matches-open-qwen3-6-max-anddeepseek-v4-falls-just-behind-top-closed-models), [buildfastwithai June ranking](https://www.buildfastwithai.com/blogs/best-ai-models-june-2026), [atlascloud comparison](https://www.atlascloud.ai/blog/kimi-k2-6-vs-glm-5-1-vs-qwen-3-6-plus-vs-minimax-m2-7-coding-2026) |
| 4 | **DeepSeek V4-Pro (Max)** | DeepSeek | 2026-04-24 | 1.6T MoE / 49B active | 1M in / 384K out | **MIT** | ✅ full | AA Intelligence Index ~**44–51.5** (largest open weight model); **SWE-bench Verified 80.6%** (highest open, tied Gemini 3.1 Pro); matches GPT-5.5/Opus at ~10–13x lower cost | HF weights; DeepSeek API **$0.435 in / $0.87 out** per M | [winbuzzer 04-27](https://winbuzzer.com/2026/04/27/deepseek-v4-open-weights-launch-xcxwbn/), [morphllm](https://www.morphllm.com/deepseek-v4), [aimadetools V4 Pro](https://www.aimadetools.com/blog/deepseek-v4-pro-complete-guide/), [AA leaderboard](https://artificialanalysis.ai/leaderboards/models) |
| 5 | **DeepSeek V4-Flash** | DeepSeek | 2026-04-24 | 284B MoE / 13B active | 1M | MIT | ✅ full | Cheapest strong open router model; AA index ~40 (one variant 29); top-3 OpenRouter token volume | HF weights; API **$0.14 in / $0.28 out** per M | [morphllm](https://www.morphllm.com/deepseek-v4), [costgoat / OpenRouter rankings](https://officechai.com/miscellaneous/these-are-the-most-popular-ai-models-on-openrouter-june-2026/) |
| 6 | **MiniMax M3** | MiniMax | 2026-06-01 | ~428B MoE / ~23B active | 1M | ⚠️ **license unconfirmed at launch** (M2.7 restricted commercial use w/o written OK) | ⚠️ partial | First open-weight to combine frontier coding + 1M ctx + **native multimodal**; **SWE-bench Pro 59.0%** (tops open weight); AA index ~44 | HF `MiniMaxAI/MiniMax-M3`; OpenRouter **$0.30 in / $1.20 out** (launch 50% off; reg $0.60/$2.40) | [the-decoder](https://the-decoder.com/minimax-m3-open-weight-model-with-a-million-token-context-challenges-proprietary-leaders/), [felloai M3](https://felloai.com/minimax-m3/), [HF MiniMax-M3](https://huggingface.co/MiniMaxAI/MiniMax-M3) |
| 7 | **Qwen3.6-Max-Preview** | Alibaba Qwen | 2026-04-20 | MoE (Max tier; weights status varies) | up to 1M | Apache-2.0 (open tiers) | ✅ (open tiers) | Claims **top on 6 programming benchmarks** incl. SWE-bench Pro & Terminal-Bench 2.0 ⚠️ vendor | OpenRouter / Qwen API | [qwen.ai blog](https://qwen.ai/blog?id=qwen3.6-max-preview), [mindstudio Qwen3.6](https://www.mindstudio.ai/blog/kimmy-k2-6-qwen-3-6-open-source-frontier-models) |
| 8 | **Qwen3.6-Plus** | Alibaba Qwen | 2026-03-30 | MoE / 1M ctx tier | 1M | Apache-2.0 | ✅ full | 1M ctx, always-on CoT, undercuts most frontier on price | OpenRouter `qwen/qwen3.6-plus-preview` | [OpenRouter Qwen3.6-Plus](https://openrouter.ai/qwen/qwen3.6-plus-preview), [aimadetools Qwen3.6](https://www.aimadetools.com/blog/qwen-3-6-complete-guide/) |
| 9 | **Qwen3.6-35B-A3B** (open weight) | Alibaba Qwen | 2026-04 | 35B MoE / **3B active** | 262K native → ~1M | **Apache-2.0** | ✅ full | Best small/efficient self-host pick; runs on modest GPUs | HF `Qwen/Qwen3.6-35B-A3B` | [HF Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B), [unsloth docs](https://unsloth.ai/docs/models/qwen3.6) |
| 10 | **Mistral Large 3** | Mistral AI | 2025-12 | 675B MoE / 41B active | 256K | **Apache-2.0** | ✅ full | Western/EU-friendly flagship; no MAU restriction (unlike Llama) | HF weights; Mistral API | [mistral.ai/news/mistral-3](https://mistral.ai/news/mistral-3/), [intuitionlabs](https://intuitionlabs.ai/articles/mistral-large-3-moe-llm-explained) |
| 11 | **gpt-oss-120b** | OpenAI | 2025-08 | ~120B MoE (MXFP4) | — | **Apache-2.0** | ✅ full | ≈ o4-mini on reasoning; runs on single 80GB GPU; widely served (Azure, AWS, Together, Vercel, Cloudflare, Ollama) | HF `openai/gpt-oss-120b`; many providers | [OpenAI gpt-oss](https://openai.com/index/introducing-gpt-oss/), [HF gpt-oss-120b](https://huggingface.co/openai/gpt-oss-120b) |
| 12 | **NVIDIA Nemotron 3 Ultra** | NVIDIA | 2026-06-04 | 550B | — | "fully permissive" | ✅ full | Permissive-license large model; strong reasoning per June roundup ⚠️ specifics light | HF weights | [buildfastwithai June ranking](https://www.buildfastwithai.com/blogs/best-ai-models-june-2026) |

Notes:
- **Llama 4.x** (Meta) is *not* recommended for the pool: it carries the **Meta community license**
  with a >700M-MAU usage restriction — it is not unconditionally open. Llama 4 Scout's headline is its
  10M-token context. ([techsy](https://techsy.io/en/blog/best-open-source-llms-2026), [mistral.ai/news/mistral-3](https://mistral.ai/news/mistral-3/))
- **gpt-oss-20b** (Apache-2.0, runs in 16GB) is the edge/tiny option if you want an OpenAI-lineage
  cheap router. ([OpenAI gpt-oss](https://openai.com/index/introducing-gpt-oss/))

---

## 2. Current open-source leaderboard (June 2026) — summary

- **Artificial Analysis Intelligence Index** (live, fetched 2026-06-21): the highest-ranked
  **open-weight** model is **GLM-5.2 (max) = 51**, ahead of **MiniMax-M3 = 44** and
  **DeepSeek V4-Pro (Max) = 44**. 26 of the top 50 models overall are open-weight variants.
  ([artificialanalysis.ai](https://artificialanalysis.ai/leaderboards/models))
- Independent June roundups put **Kimi K2.6 at the very top of open** (AA index **53.9**) with
  **DeepSeek V4-Pro 51.5** and **GLM-5.1 51.4** close behind — i.e. the exact ordering shifts
  by source and snapshot date, but the **same ~4 names (GLM-5.2/5.1, Kimi K2.6/K2.7, DeepSeek
  V4, MiniMax M3) own the top.** ([buildfastwithai June 2026](https://www.buildfastwithai.com/blogs/best-ai-models-june-2026))
- **Coding / agentic leaders:** GLM-5.2 leads coding-oriented open metrics; **MiniMax M3 tops
  open-weight SWE-bench Pro at 59.0%**, just above **Kimi K2.6 (58.6%)**; **DeepSeek V4-Pro
  leads SWE-bench Verified among open at 80.6%**. ([benchlm](https://benchlm.ai/best/open-source),
  [the-decoder](https://the-decoder.com/minimax-m3-open-weight-model-with-a-million-token-context-challenges-proprietary-leaders/),
  [morphllm](https://www.morphllm.com/deepseek-v4))
- **OpenRouter usage:** DeepSeek holds 3 of the top 10 slots (ranks 1/5/8, ~1.08T tokens/day);
  **Chinese models crossed ~60% of all OpenRouter token consumption** in 2026.
  ([officechai June 2026](https://officechai.com/miscellaneous/these-are-the-most-popular-ai-models-on-openrouter-june-2026/),
  [chatforest](https://chatforest.com/reviews/chinese-ai-models-openrouter-dominance-deepseek-kimi-minimax-glm-2026/))
- **Big picture:** the open frontier in June 2026 is **almost entirely Chinese labs** (Z.ai,
  DeepSeek, Moonshot, MiniMax, Alibaba), with Western open options being **Mistral Large 3
  (Apache, EU-friendly)** and **OpenAI gpt-oss / NVIDIA Nemotron 3 (Apache/permissive)**.
  ([buildfastwithai](https://www.buildfastwithai.com/blogs/best-ai-models-june-2026))

> ⚠️ **Verification caveat:** Almost all headline benchmark numbers for the newest models
> (GLM-5.2, Kimi K2.7, MiniMax M3, Qwen3.6-Max) are **vendor self-reported** and not yet
> confirmed by a neutral harness. AA index scores are the most independent signal available.

---

## 3. Recommended Maestro open pool (5 complementary models)

A balanced pool covering: top generalist, top coder, cheap/fast router, long-context, math/reasoning.
All are weights-public and self-hostable; licenses noted.

| Role | Pick | Why | License | Self-host reality |
|---|---|---|---|---|
| **Strong generalist (default Thinker)** | **GLM-5.2 (max)** | #1 open-weight on AA Index (51); strongest all-round agentic/coding open model; clean **MIT** | MIT ✅ | 753B MoE — needs a multi-GPU node, but MIT means no strings ([TechTimes](https://www.techtimes.com/articles/318543/20260617/glm-52-open-weights-live-top-coding-benchmark-api-use-carries-china-data-risk.htm)) |
| **Top coder (Worker for code/agentic)** | **DeepSeek V4-Pro** *(or Kimi K2.7-Code for pure coding)* | SWE-bench Verified 80.6% (highest open, ties Gemini 3.1 Pro); MIT; cheap. K2.7-Code is the coding specialist alternative | MIT ✅ / Modified-MIT | V4-Pro 1.6T is heavy to host; use the API at $0.435/$0.87 per M and keep the option to self-host ([morphllm](https://www.morphllm.com/deepseek-v4)) |
| **Cheap/fast router (triage + Verifier)** | **DeepSeek V4-Flash** *(or Qwen3.6-35B-A3B)* | $0.14/$0.28 per M, 1M ctx — ideal for the orchestrator's cheap first-pass routing & verification. Qwen3.6-35B-A3B (3B active, Apache) if you want a tiny self-hostable router | MIT ✅ / Apache ✅ | Qwen3.6-35B-A3B runs on a single modest GPU ([HF](https://huggingface.co/Qwen/Qwen3.6-35B-A3B)) |
| **Long-context** | **Qwen3.6-Plus (1M)** *(or MiniMax M3 for multimodal + 1M)* | True 1M-token tier, Apache-2.0, undercuts frontier on price; MiniMax M3 adds native multimodality if Maestro needs vision | Apache ✅ / ⚠️ M3 license TBD | Qwen3.6 open tiers self-hostable; MiniMax M3 license must be confirmed before commercial use ([OpenRouter](https://openrouter.ai/qwen/qwen3.6-plus-preview)) |
| **Math / reasoning** | **Kimi K2.6** *(or DeepSeek V4-Pro reasoning mode)* | Top open AA index (53.9 per roundup); forced/strong reasoning; first open model to beat GPT-5.4 on SWE-bench Pro — strong reasoning Verifier | Modified MIT ✅ | 1T MoE; API-first is practical, weights available ([deeplearning.ai](https://www.deeplearning.ai/the-batch/kimi-k2-6-matches-open-qwen3-6-max-anddeepseek-v4-falls-just-behind-top-closed-models)) |

**Western/EU-clean variant** (if Maestro needs to avoid Chinese-lab data/jurisdiction concerns,
which the GLM-5.2 coverage explicitly flags for *API* use — self-hosting the weights removes this):
swap to **Mistral Large 3 (Apache, EU)** as generalist, **gpt-oss-120b (Apache)** as coder/router,
**NVIDIA Nemotron 3 Ultra** as the large reasoner. All Apache/permissive, all self-hostable.
([TechTimes data-risk note](https://www.techtimes.com/articles/318543/20260617/glm-52-open-weights-live-top-coding-benchmark-api-use-carries-china-data-risk.htm),
[mistral.ai](https://mistral.ai/news/mistral-3/), [OpenAI gpt-oss](https://openai.com/index/introducing-gpt-oss/))

---

## 4. All-open Maestro pool vs Fugu's closed pool

**Fugu (Sakana AI, GA 2026-06-22):** a ~7B-class orchestrator LLM trained to call an agent pool
of **closed US APIs — GPT-5.5, Claude Opus 4.8, Gemini 3.1 Pro** (+ others), built on the ICLR
2026 TRINITY + Conductor papers, claiming **73.7 on SWE-bench Pro** (beating Opus 4.8). Sold as
Fugu / Fugu Ultra at $20/$100/$200 tiers. ([oflight deep dive](https://www.oflight.co.jp/en/columns/sakana-fugu-orchestration-model-2026-06),
[venturebeat](https://venturebeat.com/orchestration/how-sakana-trained-a-7b-model-to-orchestrate-gpt-5-claude-sonnet-4-and-gemini-2-5-pro),
[buildfastwithai Fugu review](https://www.buildfastwithai.com/blogs/sakana-ai-fugu-review-the-orchestration-model-that-routes-around-export-controls))

| Dimension | Fugu (closed pool) | Maestro (open pool) |
|---|---|---|
| **Pool models** | GPT-5.5, Opus 4.8, Gemini 3.1 Pro — proprietary, API-only | GLM-5.2, DeepSeek V4, Kimi K2.x, Qwen3.6, MiniMax M3 / Mistral, gpt-oss |
| **Weights access** | None — black box | **Public on Hugging Face**, inspectable & fine-tunable |
| **Self-hostable** | No (must hit vendor APIs) | **Yes** — run the whole pool on your own GPUs / VPC |
| **Export controls / regional lockout** | Fugu itself is **blocked in EU/EEA** (GDPR pending); Sakana also notes export-controlled models (Fable 5, Mythos) can't be in the pool | **No real export-control exposure for self-hosted open weights** — you hold the artifacts; runs anywhere you have hardware |
| **Data residency** | Prompts leave to 3 US vendors | Fully in your jurisdiction when self-hosted (avoids the China-API data-risk *and* US-API exposure) |
| **License to build a product on** | Vendor ToS, no redistribution | MIT / Apache / Modified-MIT — commercial use + redistribution allowed |
| **Cost** | $20–$200 subscription + underlying frontier API costs | Open APIs ~$0.14–$4.40 per M (10–13x cheaper than closed), or fixed self-host compute |
| **Frontier ceiling** | Higher raw single-model ceiling (Opus 4.8 / GPT-5.5) | Slightly lower per-model, but **top open is now within striking distance** (DeepSeek V4 ties Gemini 3.1 Pro on SWE-bench Verified; Kimi beat GPT-5.4 on SWE-bench Pro) |

**The honest "truly open" angle:** Maestro can claim what Fugu cannot —
(1) **every model in the pool is open-weight and self-hostable**, so the orchestrator and its
workers can run entirely inside a customer's VPC with **no data leaving and no API dependency**;
(2) **no export-control or regional lockout** for self-hosted weights (Fugu is literally
unavailable in the EU today and has acknowledged export-controlled models it can't touch);
(3) **inspectable + fine-tunable** pool members vs three black boxes; (4) **~10x cheaper**.
The trade-off to state plainly: the *closed* pool still has a somewhat higher top-end single-model
ceiling, so Maestro's pitch is **"95% of the capability, 100% open, self-hostable, and a fraction
of the cost"** — not "beats Opus 4.8 head-to-head."

> ⚠️ **Caveats to verify before shipping claims:** (a) Fugu's 73.7 SWE-bench Pro and the open
> models' headline numbers are mostly **vendor/lab self-reported**; (b) **MiniMax M3's license
> was unconfirmed at launch** (M2.7 restricted commercial use) — confirm before adding to a
> commercial pool; (c) "Modified MIT" (Kimi) and Z.ai/DeepSeek MIT terms should get a legal
> read for redistribution-in-a-product; (d) AA Index scores shift snapshot-to-snapshot — re-pull
> [artificialanalysis.ai/leaderboards/models](https://artificialanalysis.ai/leaderboards/models)
> at build time.

---

### One-paragraph summary

As of June 22, 2026, the open-weight frontier is led by Chinese labs and is genuinely close to the
closed frontier: **GLM-5.2 (Z.ai, 753B MoE, MIT, 1M ctx, weights live 06-16)** is the
top-ranked open model on the live Artificial Analysis Intelligence Index (51), with
**DeepSeek V4-Pro (1.6T MoE, MIT, SWE-bench Verified 80.6% — ties Gemini 3.1 Pro)**,
**Kimi K2.6/K2.7-Code (1T MoE, Modified-MIT, SWE-bench Pro ~58.6%, first open to beat GPT-5.4)**,
**MiniMax M3 (multimodal, 1M ctx, SWE-bench Pro 59.0% — license TBD)**, and the Apache-licensed
**Qwen3.6**, **Mistral Large 3**, and **gpt-oss** rounding out a strong, commercially-usable set.
For Maestro I recommend a 5-model pool — **GLM-5.2 (generalist)**, **DeepSeek V4-Pro / Kimi K2.7-Code
(coder)**, **DeepSeek V4-Flash / Qwen3.6-35B-A3B (cheap router)**, **Qwen3.6-Plus / MiniMax M3
(long-context, optional multimodal)**, **Kimi K2.6 (reasoner)** — with an all-Apache Western variant
(Mistral Large 3 + gpt-oss + Nemotron 3) for EU/data-clean deployments. Against Fugu's closed pool
(GPT-5.5 / Opus 4.8 / Gemini 3.1 Pro, EU-blocked today), an all-open pool wins decisively on being
**self-hostable, inspectable, export-control-free, and ~10x cheaper**, at the cost of a modestly
lower single-model ceiling. Note that most headline benchmarks are vendor-reported and MiniMax M3's
license needs confirmation before commercial use.
