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
  at pool 30 / 100 / 180. Reproducible to ±0.0001 at full N.
- **Agents (N=30, claude-sonnet-4-6):** selection pass-rate
  **oracle 80.0% > baseline 58.9% > ratel-full 47.8%**.
- **The Ratel trade:** ratel-full holds a **flat ~3.3k-token context at every
  pool size** while baseline grows to 15.6k — **78% fewer input tokens at pool
  180** — but costs **~11pp of selection accuracy** on this subset, driven by
  more wrong-tool picks (25 vs 11).
- **Ceiling effect:** even oracle (only the gold tool exposed) tops out at 80% —
  ~20% of MetaTool queries the model just answers in prose without calling any
  tool. Pure-selection scoring counts that as a miss.

---

## Setup

| | |
|---|---|
| Harness | `ratel-bench` (`src/`), AI SDK `ToolLoopAgent`, `@ratel-ai/sdk` 0.1.5 catalog |
| Dataset | `datasets/metatool.json` — 199 tools, 20,614 single-turn scenarios |
| Model | `claude-sonnet-4-6` |
| Pools | 30, 100, 180 (universe = 199 plugins) |
| Agent subset | 30 scenarios, deterministic seed 0 (`--sample 30`) |
| Arms | oracle (gold only), baseline (full pool), ratel-full (BM25 pre-discovery + gateway) |
| Concurrency | 8 (`runConcurrent`) — 210 cells, 0 errors, **438s** wall |

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

- 199 plugins (name + description, **no parameter schemas**).
- 20,615 query rows → **20,614 scenarios** (1 malformed upstream row skipped as
  unknown-gold).
- The multi-tool slice is **not** ingested (a `Turn` carries a single
  `expectedTool`).
- Corpus is gitignored — regenerate deterministically from pinned upstream URLs.

---

## Part A — Retrieval (tool-selection)

BM25 over plugin descriptions, full corpus (N = 20,614):

| pool | MRR@5 | run-to-run \|Δ\| |
|---:|---:|---:|
| 30 | **0.6633** | 0.0001 |
| 100 | **0.5691** | 0.0001 |
| 180 | **0.5167** | 0.0001 |

MRR decays as the pool grows (more distractors), as expected. Lexical BM25 puts
the gold plugin around rank ~1.5 within top-5 at pool 30.

### ⚠ Non-determinism in `@ratel-ai/sdk` search

The native `ToolRegistry` search **randomizes tied-score results per call** —
five identical in-process searches return five different orderings:

```
run0: ["tool_2","tool_9","tool_0","tool_1","tool_4"]
run1: ["tool_6","tool_5","tool_8","tool_2","tool_1"]   # same query, same catalog
...
```

This perturbs MRR whenever the gold tool ties with distractors, and it makes the
agents' gateway search non-reproducible too. **Impact is bounded:** at full N the
run-to-run delta is ≤0.0001 (it averages out); on small samples and per-scenario
it is visible. **Recommendation (Ratel owns the SDK):** add a seedable /
deterministic ranking mode for benchmarking and reproducible production traces.

---

## Part B — Agent campaign (N = 30, claude-sonnet-4-6)

### Pass rate, tokens, latency per arm × pool

| arm | pool | pass % | input tok | output tok | wall ms | tool calls |
|---|---:|---:|---:|---:|---:|---:|
| oracle | — | **80.0** | 1,166 | 727 | 15,183 | 1.33 |
| baseline | 30 | 66.7 | 3,687 | 732 | 15,287 | 1.37 |
| baseline | 100 | 63.3 | 9,461 | 674 | 14,197 | 1.30 |
| baseline | 180 | 46.7 | 15,639 | 674 | 14,317 | 1.47 |
| ratel-full | 30 | 53.3 | 3,163 | 847 | 18,296 | 1.73 |
| ratel-full | 100 | 50.0 | 3,376 | 780 | 16,690 | 1.77 |
| ratel-full | 180 | 40.0 | 3,368 | 809 | 17,016 | 1.73 |

**Rollup (all pools):** oracle 80.0% · baseline 58.9% · ratel-full 47.8%.

### Input tokens: flat (ratel) vs. scaling (baseline)

| pool | baseline | ratel-full | savings |
|---:|---:|---:|---:|
| 30 | 3,687 | 3,163 | 14% |
| 100 | 9,461 | 3,376 | 64% |
| 180 | 15,639 | 3,368 | **78%** |

This is the core Ratel result: **context cost is decoupled from catalog size.**
Baseline pays linearly for a bigger pool; ratel-full stays flat because
pre-discovery + the gateway feed a fixed, small surface. The savings *grow* with
the catalog.

### Accuracy: ratel-full trails baseline by ~11pp

| pool | baseline | ratel-full | Δ |
|---:|---:|---:|---:|
| 30 | 66.7% | 53.3% | −13.4 |
| 100 | 63.3% | 50.0% | −13.3 |
| 180 | 46.7% | 40.0% | −6.7 |

The gap narrows at pool 180 because baseline degrades fastest there. Consistent
with the old `RESULTS.md` Claude behavior (ratel arms traded a few pp of accuracy
for large token savings).

### Failure breakdown

| arm | fails | no tool call (refusal) | wrong tool | error |
|---|---:|---:|---:|---:|
| oracle | 6 | 6 | 0 | 0 |
| baseline | 37 | 26 | 11 | 0 |
| ratel-full | 47 | 22 | **25** | 0 |

Two distinct failure modes:

1. **No-tool-call refusals** — the model answers in prose instead of selecting a
   tool. Dominant for oracle/baseline and the reason oracle caps at 80%. A
   corpus/metric artifact: with pure-selection scoring (no LLM judge), a correct
   prose answer still counts as a miss.
2. **Wrong-tool picks** — ratel-full's main extra cost (25 vs baseline's 11): the
   discovery indirection sometimes surfaces or selects the wrong tool.

### On speed (a correction)

ratel-full is **not** faster per cell — it's slightly slower (17.3s vs 14.6s)
because of the extra gateway round-trips (≈1.75 steps vs ≈1.38). Its win is
*tokens, not latency*. Separately, within **every** arm, FAIL cells run much
faster than PASS cells (oracle 3.6s vs 18.1s) because many failures are short
zero-tool-call refusals — so fast cells skew toward failures, across all arms.

---

## Key takeaways

1. **Ratel decouples context cost from catalog size** — flat ~3.3k input tokens
   vs baseline's 15.6k at pool 180 (78% fewer), and the gap widens as the
   catalog grows.
2. **It costs ~11pp of selection accuracy here**, mostly via wrong-tool picks
   from the discovery layer. The token/accuracy trade is the headline.
3. **Oracle ceiling = 80%** — pure-selection scoring penalizes the ~20% of
   queries the model answers directly. An LLM-judge or "no-tool-needed" gold
   label would lift the ceiling and change the comparison.
4. **Reproducibility gap in the SDK search** (per-call tie randomization) —
   negligible at full N, real at small N; worth a deterministic mode.
5. **Parallel runner**: 210 cells in 438s vs 55.5 min sequential-equivalent
   (7.6× at concurrency 8).

## Caveats

- **N = 30** subset — pass rates carry ±~9pp sampling noise; directional, not
  definitive. Re-run with a larger `--sample` for tighter numbers.
- Single model (`claude-sonnet-4-6`), single seed, one run per cell (no variance
  bands).
- MetaTool is **single-turn** with **no parameter schemas** — it exercises tool
  *selection*, not argument construction or multi-turn context.
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
