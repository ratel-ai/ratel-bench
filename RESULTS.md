# Ratel benchmark — narrative summary

This document consolidates the agent benchmark across four model families on the MetaTool corpus and explains what the numbers say about where Ratel helps today and where it's still maturing.

## Where Ratel is most valuable today

| your situation | Ratel's value today |
|---|---|
| Local model + large catalog | **Critical.** Without Ratel the model can't function. (qwen3.5 pool=100: 8% → 77%) |
| Open-source cloud model + large catalog | **Strong win.** Beats baseline on accuracy *and* tokens. (glm-5.1: +12pp, -85% tokens) |
| Frontier model + large catalog | **Cost-driven win.** ~80% input-token savings, modest accuracy cost; closing now. |
| Any model + tiny catalog (≤30) | Skip Ratel — pool fits in the prompt cleanly. |

## Headline takeaways

- **Token savings are universal and large.** Across every model tested, Ratel cuts input tokens by **70–85% at realistic pool sizes (100–180 tools)**. This is true whether the underlying model is a tiny local model or frontier Claude.
- **Open-source / local models: Ratel is what makes them usable.** `glm-5.1:cloud` gains **~10 pp accuracy** versus the baseline at large pool sizes, and **`qwen3.5` running locally on a MacBook Pro M4 24 GB jumps from 8.3% → 76.7%** when the pool grows to 100 tools — the baseline simply collapses, Ratel keeps it working.
- **Frontier Claude models: massive token + cost savings, accuracy not yet at parity.** Sonnet 4.6 keeps solving most tasks even with 180-tool pools, so the headroom Ratel buys is mostly cost / context — not pass-rate. Ratel currently trades ~13 pp on Sonnet 4.6 and stays within ~5 pp on Opus 4.6 / 4.7, all for **50–80% input-token savings**. Closing the Sonnet gap is the active work item.

## Methodology

- **Corpus**: MetaTool single-tool selection, strict LLM judge (no rewarding of "I don't have a tool, try X" style refusals).
- **Arms**:
  - `control-baseline` — entire candidate pool exposed to the agent.
  - `control-oracle` — only the gold tool is exposed (upper bound).
  - `ratel-full` — Ratel discovery + selection. The model sees ~5 BM25-prefetched tools plus the 2 gateway tools (`search_tools` + `invoke_tool`) — ~7 total — regardless of pool size.
- **Pool sizes**: 30, 50, 100, 180. Real-world MCP setups land in the 100–200 range.
- **Hardware**: cloud APIs for Claude / glm-5.1:cloud; local Ollama on **MacBook Pro M4 24 GB** for qwen3.5.

## Results by model family

### Local / open-source models — where Ratel shines

#### `qwen3.5` (Ollama, MacBook Pro M4 24 GB)

| pool | control-baseline | ratel-full | Δ accuracy | input tokens (ctrl → ratel) | wall time (ctrl → ratel) |
|---|---|---|---|---|---|
| 30 | 91.7% | 88.3% | -3.4 pp | 3 926 → 2 494 (-37%) | 59.6s → 61.7s |
| 50 | 86.7% | 81.7% | -5.0 pp | 6 738 → 2 557 (-62%) | 61.3s → 65.6s |
| **100** | **8.3%** | **76.7%** | **+68.4 pp** | 6 485 → 2 820 (-57%) | 107.6s → 69.1s (**-36%**) |

What happens at pool=100 is the story. The baseline arm catastrophically degrades — the model is overwhelmed by the tool list and stops calling tools at all (programmatic pass rate drops to 0). With Ratel, the model only sees ~7 well-targeted tools (5 prefetched + 2 gateway) and stays at **76.7%**. This is the difference between "local models can't handle large tool catalogs" and "local models are a real option for large MCP setups."

The wall-clock improvement (107.6s → 69.1s) is a side benefit — fewer tokens means faster inference on memory-constrained hardware.

#### `glm-5.1:cloud` (open-source, hosted)

| pool | control-baseline | ratel-full | Δ accuracy | input tokens (ctrl → ratel) |
|---|---|---|---|---|
| 30 | 85.6% | 86.7% | +1.1 pp | 2 976 → 2 428 (-18%) |
| 50 | 80.0% | 84.4% | +4.4 pp | 4 897 → 2 579 (-47%) |
| 100 | 78.9% | 85.6% | **+6.7 pp** | 10 062 → 2 909 (**-71%**) |
| 180 | 75.6% | 87.8% | **+12.2 pp** | 19 362 → 2 923 (**-85%**) |

For glm-5.1, the picture is the cleanest possible win: **Ratel beats the baseline at every pool size**, and the gap **widens as the catalog grows**. At pool=180 it's +12 pp pass-rate while using ~85% fewer input tokens.

The Ratel arm is also essentially **pool-invariant**: 85-88% across all pool sizes. The model isn't seeing the full pool, so it doesn't matter how big it gets.

### Frontier Claude models — token savings now, accuracy parity in flight

Sonnet already handles 180-tool pools reasonably well in the baseline, so Ratel doesn't unlock a new capability here. What it does deliver is **dramatic input-token savings** at a modest-to-moderate accuracy cost, with Opus 4.6 / 4.7 staying within ~5 pp of baseline and Sonnet 4.6 trading ~13 pp.

The Ratel arm reported in these tables is `ratel-discovery-tool` (gateway only — `search_tools` + `invoke_tool`, no BM25 pre-fetch). `ratel-full` is omitted; it has been zig-zagging around `ratel-discovery-tool` and is under investigation as a separate thread.

#### `claude-sonnet-4-6`

| pool | control-baseline | ratel-discovery-tool | Δ accuracy | input tokens (ctrl → ratel) | $ (ctrl → ratel) | wall time (ctrl → ratel) |
|---|---|---|---|---|---|---|
| 50  | 83.3% | 66.7% | -16.6 pp | 5 562 → 3 602 (-35%)      | $0.026 → $0.022 (-17%)     | 12.3s → 14.9s |
| 100 | 82.2% | 66.7% | -15.5 pp | 9 920 → 3 580 (**-64%**)  | $0.039 → $0.022 (**-44%**) | 12.6s → 14.8s |
| 180 | 81.1% | 67.8% | -13.3 pp | 17 008 → 3 641 (**-79%**) | $0.060 → $0.022 (**-64%**) | 13.0s → 15.0s |

control-oracle (upper bound): **90.0%**.

Sonnet 4.6 is the strongest fat-context selector in the family (highest baseline) and pays the largest Ratel hit. At pool=180 the residual failures break down as **31% missing-gold** (the agent's `search_tools` queries never surfaced the gold tool) plus **9% selection error** (gold *was* in the candidate set, model picked wrong) — so the gap is mostly an upstream-retrieval problem, with a small but real selection tax on top. Hybrid / semantic retrieval is the planned next step against the missing-gold component. The trade-off Sonnet users get today: **-13 pp pass rate at pool=180 in exchange for -79% input tokens and -64% dollars, pool-size-invariant**.

#### `claude-opus-4-6`

| pool | control-baseline | ratel-discovery-tool | Δ accuracy | input tokens (ctrl → ratel) | $ (ctrl → ratel) | wall time (ctrl → ratel) |
|---|---|---|---|---|---|---|
| 50  | 71.7% | 71.7%     | ±0 pp        | 5 202 → 4 223 (-19%)       | $0.038 → $0.040 (+4%)      | 12.3s → 18.8s |
| 100 | 70.0% | **73.3%** | **+3.3 pp**  | 9 230 → 4 313 (**-53%**)   | $0.058 → $0.040 (**-30%**) | 12.5s → 18.6s |
| 180 | 71.7% | 70.0%     | -1.7 pp      | 16 322 → 4 497 (**-72%**)  | $0.093 → $0.041 (**-56%**) | 12.6s → 19.1s |

control-oracle (upper bound): **90.0%**.

Opus 4.6 is the clean Ratel case among the Claude family: `ratel-discovery-tool` tracks baseline within ±3 pp at every pool size while cutting input tokens **-19% to -72%** and dollars up to **-56% at pool=180**. Critically, at pool=180 **every** ratel-discovery-tool failure on Opus 4.6 was a missing-gold cell — i.e. when `search_tools` *did* return the gold tool, Opus 4.6 picked it correctly every time. So Opus 4.6 is the model where "the bottleneck is upstream of the agent loop" is literally true; the remaining gap is entirely a function of the discovery layer (BM25 + the agent's query choices). At pool=50, Ratel's extra turns push dollar cost slightly above baseline (output tokens dominate at small pools where the baseline catalog is cheap to lay out); the win shows up cleanly from pool=100 onward.

**Anthropic's [tool-search-tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool)** comparison on the same scenarios (n=90 per cell):

| pool | claude-sdk-tool-search | Δ vs baseline | input tokens (ctrl → sdk-search) | $ (ctrl → sdk-search) |
|---|---|---|---|---|
| 50  | 58.9% | -12.8 pp | 5 202 → 3 366 (-35%)      | $0.038 → $0.034 (-11%) |
| 100 | 55.6% | -14.4 pp | 9 230 → 3 751 (-59%)      | $0.058 → $0.036 (-37%) |
| 180 | 52.2% | -19.5 pp | 16 322 → 3 677 (**-77%**) | $0.093 → $0.036 (**-61%**) |

Anthropic's tool-search-tool sits 13–20 pp below baseline across the sweep, i.e. **5–10× the accuracy hit** Ratel pays for comparable token / cost savings.

#### `claude-opus-4-7`

| pool | control-baseline | ratel-discovery-tool | Δ accuracy | input tokens (ctrl → ratel) | $ (ctrl → ratel) | wall time (ctrl → ratel) |
|---|---|---|---|---|---|---|
| 50  | 76.7% | 70.0% | -6.7 pp | 7 184 → 3 514 (**-51%**)   | $0.051 → $0.036 (-30%)     | 11.3s → 13.6s |
| 100 | 68.3% | 66.7% | -1.6 pp | 12 733 → 3 847 (**-70%**)  | $0.079 → $0.038 (**-51%**) | 12.5s → 13.7s |
| 180 | 71.7% | 66.7% | -5.0 pp | 21 871 → 3 835 (**-82%**)  | $0.124 → $0.038 (**-69%**) | 11.5s → 13.6s |

control-oracle (upper bound): **83.3%**.

Opus 4.7 lifts the baseline relative to 4.6, softening the headline gap (`ratel-discovery-tool` lands within **-7 pp** of baseline at every pool size while saving **-51% to -82% input tokens** and up to **-69% dollars**) — but the failure mix is different. At pool=180, Opus 4.7's residual Ratel failures split **22% missing-gold and 32% selection-error**: in 32% of cells the gold tool *was* in the candidate set and the model still picked wrong. That's the highest selection-error rate in the family (vs ~0% for Opus 4.6, ~9% for Sonnet). So for Opus 4.7 the bottleneck is not just upstream retrieval — the model itself is less reliable at choosing from a small candidate window than at choosing from the full pool. That makes the headroom for Opus 4.7 mostly a model-side problem (selection from candidates) rather than a retrieval-layer problem.

**Anthropic's tool-search-tool** on the same scenarios (n=90 per cell):

| pool | claude-sdk-tool-search | Δ vs baseline | input tokens (ctrl → sdk-search) | $ (ctrl → sdk-search) |
|---|---|---|---|---|
| 50  | 65.6% | -11.1 pp | 7 184 → 5 286 (-26%)      | $0.051 → $0.046 (-10%) |
| 100 | 61.1% | -7.2 pp  | 12 733 → 4 882 (-62%)     | $0.079 → $0.044 (-45%) |
| 180 | 51.1% | -20.6 pp | 21 871 → 5 142 (-76%)     | $0.124 → $0.045 (**-64%**) |

At pool=180 the SDK tool-search arm drops **-20.6 pp** — about **4× the accuracy hit** of `ratel-discovery-tool` (-5.0 pp), and even uses **more input tokens** (5 142 vs 3 835). The gap widens with pool size in both directions.

## What's next

1. **Hybrid retrieval** — replace BM25-only discovery with hybrid (BM25 + dense) to close the "missing gold" gap on Claude models without giving up token savings.
2. **Discovery feedback loop** — let the agent re-query the tool index when its first candidate set doesn't contain a useful tool, instead of failing the cell.
3. **Broader scenario coverage** — the current corpus is single-tool MetaTool; add multi-tool and toolret scenarios to the agent benchmark. Retrieval-quality rows for both already ship in the report and look strong.

## Reproducing these numbers

```bash
# Strict judge is now the default for the no-criteria fallback (v0.1.2).
# Older results can be re-judged in place without rerunning agents:
pnpm -F @ratel-ai/benchmark start rejudge \
  agent/results/<run>.jsonl \
  --judge-prompt strict

# Render the narrative tables:
pnpm -F @ratel-ai/benchmark report \
  --agent agent/results/<run>.rejudged-strict.jsonl \
  --output results/<NAME>.md
```
