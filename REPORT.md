# ratel-bench — MetaTool report

Re-baseline of the refactored harness on the **MetaTool** corpus. Two parts: a
retrieval (tool-selection) sweep over the full corpus, and an agent campaign on a
30-scenario subset comparing the three arms.

> **Not a before/after vs. the old harness.** Scoring, arms, and gold semantics
> all changed in the refactor (old pass = programmatic-OR-LLM-judge union over a
> `gold_tools[]` set; new pass = exact `expectedTool ∈ effectiveToolIds`, no
> judge). The old per-cell traces are gone, so these are fresh numbers for the
> new harness, not a delta against `RESULTS.md`.

## TL;DR

- **Retrieval (BM25, full corpus, N=20,614):** MRR@5 = **0.663 → 0.569 → 0.517**
  at pool 30 / 100 / 180.
- **Agents (N=30, claude-sonnet-4-6):** selection pass-rate
  **oracle 73.3% > baseline 60.0% > ratel-full 53.3%**.
- **ratel-full now mirrors real discovery.** The pre-discovered top-K is injected
  as a synthetic `search_tools` **tool-result message** (not as extra tool
  definitions), so the model can't tell pre-discovery from self-discovery and
  must `invoke_tool` a hit to use it.
- **The Ratel trade:** ratel-full holds **~flat input across pool sizes** (5.0k at
  pool 180 vs baseline's 16.6k — **70% fewer input tokens**, **74% lower
  cache-priced cost**), at a cost of **~7pp selection accuracy**, driven by a
  higher *refusal* rate (the gateway indirection makes the model select-but-not-
  invoke more often).
- **Pre-discovery K 5→15 lifted large-pool recall +9pp** (the gold tool reaches
  the model more often), which tracks the pool-180 pass-rate gain (40%→50% vs the
  earlier K=5 arm).
- **Ceiling effect:** even oracle (only the gold tool exposed) tops out at ~73% —
  ~27% of MetaTool queries the model answers in prose without invoking. Pure-
  selection scoring counts that as a miss.

---

## What changed since the first cut

The agent campaign was re-run after two changes to `ratel-full` plus a metering
fix; retrieval (Part A) is unchanged.

1. **Pre-discovery is injected as a tool-result message** (`src/agents/ratel-full.ts`).
   The arm now exposes only the `search_tools` / `invoke_tool` gateway, runs the
   real gateway search for the turn's query, and injects its exact `{groups:[…]}`
   payload as a synthetic *assistant tool-call + tool-result* pair after the user
   turn. Byte-for-byte what a model-issued `search_tools` call returns — so the
   model must `invoke_tool` a hit (no first-class shortcut), exactly like real
   discovery. The tool block is now constant (2 gateway tools) → cacheable.
2. **All-arms Anthropic prompt caching** (`src/agents/baseline.ts`). A single
   ephemeral `cacheControl` breakpoint on the last tool of every arm, plus one on
   ratel-full's injected discovery. Uniform policy; no behavioral change (the
   marker is stripped before the model sees content).
3. **Metering fix.** AI SDK v6 reports cache splits under
   `usage.inputTokenDetails.{cacheReadTokens,cacheWriteTokens}` — there is no
   `cacheCreationInputTokens`. The old `summarize` read that nonexistent field, so
   cache *writes* were silently zero; now fresh / cache-read / cache-write are all
   captured.
4. **Pre-discovery K 5 → 15** to raise the chance the gold tool is in the injected
   set (see *Pre-discovery recall*).

---

## Setup

| | |
|---|---|
| Harness | `ratel-bench` (`src/`), AI SDK `ToolLoopAgent`, `@ratel-ai/sdk` 0.1.5 catalog |
| Dataset | `datasets/metatool.json` — 199 tools, 20,614 single-turn scenarios |
| Model | `claude-sonnet-4-6` |
| Pools | 30, 100, 180 (universe = 199 plugins) |
| Agent subset | 30 scenarios, deterministic seed 0 (`--sample 30`) |
| Arms | oracle (gold only), baseline (full pool), ratel-full (BM25 pre-discovery via injected `search_tools` result + gateway) |
| Pre-discovery K | 15 injected candidates |
| Caching | ephemeral breakpoint on the tool block (all arms) + injected discovery (ratel-full) |
| Concurrency | 8 — 210 cells, 0 errors, **~450s** wall (≈8× over ~60 min sequential-equivalent) |

**Scoring**

- *Retrieval (MRR@5):* `1/position` if the gold tool is in BM25 top-5, else 0.
  For MetaTool the curated `expectedQuery` equals the user message, so the
  `input-only` and `expected-query` modes are identical — one number reported.
- *Agent (pass):* `expectedTool ∈ effectiveToolIds` for the turn
  (`search_tools` dropped, `invoke_tool` unwrapped to its inner tool). Pure
  selection — no LLM judge.

**Pooling.** Each scenario gets its **own** pool: its gold tool first, then
`poolSize − 1` deterministic distractors (`buildScenarioPool`, seed folds in the
scenario id). Without this, MetaTool's global gold union (≈ all 199 tools) would
fill every pool and flatten the sweep.

## Dataset: MetaTool

Source: [HowieHwong/MetaTool](https://github.com/HowieHwong/MetaTool) (MIT),
ingested Lane-B (data-only) by `src/ingest/metatool.ts` →
`pnpm ingest:metatool`. The single-tool slice maps cleanly: each `Query,Tool`
row → one scenario with `turns.length === 1`, `expectedTool` = the gold plugin.
199 plugins (name + description, **no parameter schemas**); 20,615 query rows →
20,614 scenarios (1 malformed row skipped). Corpus is gitignored — regenerate
deterministically from pinned upstream URLs.

---

## Part A — Retrieval (tool-selection)

BM25 over plugin descriptions, full corpus (N = 20,614):

| pool | MRR@5 | run-to-run \|Δ\| |
|---:|---:|---:|
| 30 | **0.6633** | 0.0001 |
| 100 | **0.5691** | 0.0001 |
| 180 | **0.5167** | 0.0001 |

MRR decays as the pool grows (more distractors), as expected. Unchanged by the
agent rework. (On the 30-scenario agent subset the same metric reads
0.567 / 0.511 / 0.473 — small-sample, shown only for context.)

### ⚠ Non-determinism in `@ratel-ai/sdk` search

The native `ToolRegistry` search **randomizes tied-score results per call** — five
identical in-process searches return five different orderings of tied tools. This
perturbs MRR (and the agents' gateway search) whenever the gold tool ties with
distractors. **Impact is bounded:** ≤0.0001 at full N (it averages out); visible
on small samples. **Recommendation (Ratel owns the SDK):** add a seedable /
deterministic ranking mode for benchmarking and reproducible traces.

---

## Part B — Agent campaign (N = 30, claude-sonnet-4-6, K=15)

### Pass rate, tokens, latency per arm × pool

Token columns split the input into **fresh** / **cache-read** / **cache-write**
(`totIn` = their sum). `cost-eq` prices cache traffic the way Anthropic bills it
(`fresh + 0.1·read + 1.25·write`) — the honest like-for-like number.

| arm | pool | pass % | fresh | cache-read | cache-write | totIn | cost-eq | out | wall s | calls |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| oracle | — | **73.3** | 1,113 | 0 | 0 | 1,113 | 1,113 | 723 | 15.3 | 1.27 |
| baseline | 30 | 66.7 | 456 | 1,273 | 1,958 | 3,687 | 3,030 | 748 | 15.7 | 1.37 |
| baseline | 100 | 60.0 | 449 | 3,519 | 5,310 | 9,277 | 7,438 | 645 | 14.0 | 1.33 |
| baseline | 180 | 53.3 | 469 | 6,855 | 9,251 | 16,575 | 12,718 | 711 | 15.6 | 1.67 |
| ratel-full | 30 | 56.7 | 308 | 1,927 | 1,668 | 3,902 | 2,585 | 893 | 18.9 | 1.53 |
| ratel-full | 100 | 53.3 | 351 | 2,607 | 2,167 | 5,125 | 3,321 | 949 | 19.8 | 1.40 |
| ratel-full | 180 | 50.0 | 304 | 2,479 | 2,242 | 5,025 | 3,354 | 1,006 | 20.4 | 1.67 |

**Rollup (all pools):** oracle 73.3% · baseline 60.0% · ratel-full 53.3%.

**ratel-full vs the earlier direct-tool / K=5 arm** (same harness, prior run):
57/53/50% vs 53/50/**40**% — the gain concentrates at pool 180 (+10pp), matching
the K=15 recall lift there.

### Input: flat (ratel) vs. scaling (baseline)

| pool | baseline totIn | ratel-full totIn | fewer | baseline cost-eq | ratel-full cost-eq | cheaper |
|---:|---:|---:|---:|---:|---:|---:|
| 30 | 3,687 | 3,902 | −6% | 3,030 | 2,585 | 15% |
| 100 | 9,277 | 5,125 | 45% | 7,438 | 3,321 | 55% |
| 180 | 16,575 | 5,025 | **70%** | 12,718 | 3,354 | **74%** |

This is the core Ratel result: **context cost is decoupled from catalog size.**
Baseline's input grows linearly with the pool; ratel-full stays ~flat because the
gateway feeds a fixed, small surface. The savings *grow* with the catalog. (K=15
costs ratel-full ~1.5k more input than K=5 did — the price of the +9pp large-pool
recall — so the headline is now "70% fewer" rather than the old "78%.")

### Accuracy: ratel-full trails baseline by ~7pp

| pool | baseline | ratel-full | Δ |
|---:|---:|---:|---:|
| 30 | 66.7% | 56.7% | −10.0 |
| 100 | 60.0% | 53.3% | −6.7 |
| 180 | 53.3% | 50.0% | −3.3 |

The gap narrows at pool 180 because baseline degrades fastest there (wrong-tool
picks spike). The ratel-full cost is mostly *refusals*, not wrong picks (below).

### Pre-discovery recall (why K=15)

The model can only `invoke_tool` the gold tool if it's in the injected set (or it
re-searches). Recall of the injected set — fraction of scenarios with the gold
tool in the BM25 top-K, measured directly over n=1,200 (no API):

| pool | recall@5 | recall@15 | lift |
|---:|---:|---:|---:|
| 30 | 78.8% | 80.5% | +1.8pp |
| 100 | 69.8% | 78.8% | **+9.0pp** |
| 180 | 65.7% | 74.7% | **+9.0pp** |

K=15 raises the achievable ceiling **+9pp at pools 100/180** (where ratel-full was
weakest) and barely moves pool 30 (already near the BM25 lexical ceiling, yet
injecting 15 of 30 tools). It only counters the *retrieval-miss* failure — the
remaining ~20–25% the lexical search can't surface at any K rely on the model
re-searching.

### Prompt caching

The metering fix makes cache traffic visible. Within a single run the loop's
later steps re-read the prefix, so **both** arms cache (`cache-read > 0`) — but the
cost lands very differently:

- **ratel-full** caches a *tiny constant* gateway surface → small cache traffic
  (~2.5k read + ~2.2k write at pool 180).
- **baseline** caches a *large, per-scenario-unique* tool block → it pays the 1.25×
  cache-write premium on 9.3k tokens every call, only partly offset by the within-
  run read. Net it's still cheaper than uncached, but far above ratel-full.
- **oracle** shows **0 cache** — its 1-tool block is below Anthropic's ~1024-token
  minimum cacheable prefix. Same reason the *cross-call* caching of ratel-full's
  gateway block doesn't fire on this corpus: realized caching here is within-run,
  not cross-call. The cross-catalog "constant prefix cached across calls" win needs
  a larger system/tools surface or multi-turn scenarios.

### Failure breakdown

`refusal` = no effective tool selected (prose answer, or `search_tools` without a
follow-up `invoke_tool`); `wrong` = invoked the wrong tool.

| arm | pool | fails | refusal | wrong | error |
|---|---:|---:|---:|---:|---:|
| oracle | — | 8 | 8 | 0 | 0 |
| baseline | 30 | 10 | 7 | 3 | 0 |
| baseline | 100 | 12 | 9 | 3 | 0 |
| baseline | 180 | 14 | 7 | 7 | 0 |
| ratel-full | 30 | 13 | 12 | 1 | 0 |
| ratel-full | 100 | 14 | 10 | 4 | 0 |
| ratel-full | 180 | 15 | 9 | 6 | 0 |

Two failure modes, and the new arm shifts the mix:

1. **Refusals dominate ratel-full** (≈10/30). The injected discovery + gateway
   indirection makes the model more likely to *select but not invoke* — e.g. it
   reads the discovery, names the right tool ("I'll use **Glowing**…"), then asks
   the user for a missing argument instead of calling `invoke_tool`. Pure-selection
   scoring can't see that selection, so it counts as a miss. This is the bulk of
   the faithfulness cost and the same artifact that caps oracle at 73%.
2. **Wrong-tool picks grow with the pool** for both arms (baseline 3→7,
   ratel-full 1→6) — more candidates, more ways to pick wrong. K=15 injecting more
   candidates contributes to ratel-full's rise here.

### On speed

ratel-full is **slower per cell** (18.9–20.4s vs baseline's 14–16s): the gateway
adds round-trips (the model issues a real `invoke_tool`, sometimes a second
`search_tools`). Its win is *tokens/cost, not latency*.

---

## Key takeaways

1. **Ratel decouples context cost from catalog size** — ~flat input vs baseline's
   linear growth; 70% fewer input tokens / 74% lower cache-priced cost at pool 180,
   widening with the catalog.
2. **Faithful discovery costs ~7pp of selection accuracy here**, mostly via a
   higher *refusal* rate from the gateway indirection (select-but-don't-invoke),
   not wrong picks. The token/accuracy trade is the headline.
3. **Pre-discovery K matters at scale** — K 5→15 buys +9pp injected-set recall at
   pools 100/180 and tracks a 40%→50% pool-180 pass-rate gain; negligible at
   pool 30.
4. **Caching is within-run on this corpus** — both arms cache across loop steps;
   oracle/cross-call don't (below the ~1024-token minimum). The structural win is
   that ratel-full's tiny surface generates far less cache traffic.
5. **Oracle ceiling ≈ 73%** — pure-selection scoring penalizes the ~27% of queries
   the model answers directly. An LLM-judge or "no-tool-needed" gold label would
   lift the ceiling and change the comparison.

## Caveats

- **N = 30** subset — pass rates carry ±~9pp sampling/model noise; the recall
  numbers (n=1,200) are the robust evidence for the K effect. Directional, not
  definitive.
- The **before/after for ratel-full** conflates two changes (injection mechanism +
  K). The recall diagnostic isolates K cleanly; the pass-rate delta does not.
- Single model (`claude-sonnet-4-6`), single seed, one run per cell.
- MetaTool is **single-turn** with **no parameter schemas** — it exercises tool
  *selection*, not argument construction or multi-turn context (which is exactly
  where cross-call caching and the "ask for a missing arg" refusals would resolve
  differently).
- Pure-selection scoring, no LLM judge (by design).

## Reproduce

```bash
pnpm ingest:metatool                       # → datasets/metatool.json (199 tools, 20,614 scenarios)

# Retrieval, full corpus (no API key needed):
pnpm start --dataset datasets/metatool.json --pools 30,100,180 \
  --model retrieval-only --out results/mt-full

# Agent campaign, 30-scenario subset (needs ANTHROPIC_API_KEY):
pnpm start --dataset datasets/metatool.json --pools 30,100,180 \
  --sample 30 --model claude-sonnet-4-6 --agents ratel-full \
  --concurrency 8 --out results/mt-agents
```

Outputs (gitignored): `results/<run>/tool-selection.jsonl`,
`results/<run>/agents.jsonl`, `results/<run>/summary.json`.
