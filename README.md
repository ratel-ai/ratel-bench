# ratel-bench

Two-layer benchmark harness for [**Ratel**](https://github.com/ratel-ai/ratel) — measures retrieval quality and agent-loop token savings. Backs every "Ratel does X better" claim. New product features that touch retrieval or context shape are expected to add a corresponding scenario before being declared done.

This repo is intentionally decoupled from the Ratel monorepo: the agent campaign pins `@ratel-ai/sdk@0.2.0` from npm and the retrieval crate pins `ratel-ai-core@0.1.5` from crates.io, so the harness measures the same artifacts users install — not whatever's on the working tree.

**Latest results: [`RESULTS.md`](RESULTS.md)** — narrative breakdown across Claude (Sonnet, Opus), `glm-5.1:cloud`, and local `qwen3.5`.

## Where Ratel is most valuable today

| your situation | Ratel's value today |
|---|---|
| Local model + large catalog | **Critical.** qwen3.5 at pool=100 goes from 8% → 77% (-57% input tokens, -36% wall time). |
| Open-source cloud + large catalog | **Strong win.** glm-5.1 at pool=180: **+1.7 pp** accuracy and **-85%** input tokens; pool-invariant accuracy as the catalog grows. |
| Frontier (Sonnet) + large catalog | **Cost-driven win.** Sonnet 4.6 at pool=180: **-82%** input tokens, **-68%** $; -8 pp accuracy. |
| Frontier (Opus) + large catalog | **Competitive win.** Opus 4.6 pool=180: **+8 pp** accuracy and **-72%** tokens (discovery-tool arm). Opus 4.7 pool=180: ≈parity (-1.7 pp) with **-81%** tokens — Anthropic's own tool-search-tool loses **-8 pp** on the same setup. |
| Any model + tiny catalog (≤30) | Skip Ratel — pool fits in the prompt cleanly. |

Full per-pool breakdown and methodology in [`RESULTS.md`](RESULTS.md).

Locked decisions live in:

- [`docs/adr/0005-benchmark-design.md`](docs/adr/0005-benchmark-design.md) — overall harness (arms, models, variance, results storage)
- [`docs/adr/0006-benchmark-corpus-and-eval-modes.md`](docs/adr/0006-benchmark-corpus-and-eval-modes.md) — corpus pivot + the three eval modes
- [`docs/adr/0007-benchmark-corpus-not-snapshotted.md`](docs/adr/0007-benchmark-corpus-not-snapshotted.md) — corpus is ingested locally; no committed snapshot, no MetaTool sampling
- [`docs/adr/0008-skill-retrieval-eval-mode.md`](docs/adr/0008-skill-retrieval-eval-mode.md) — skill retrieval evaluated separately on an authored skill corpus (SR-Agents) via `SkillRegistry`; per-dataset + aggregate summary

## Layout

```
retrieval/    Rust crate (ratel-benchmark-retrieval) — ingest + BM25 metrics
agent/        TypeScript pnpm package (@ratel-ai/benchmark) — MetaTool agent campaign + report
bfcl-loaded/  TypeScript + Python pnpm package (@ratel-ai/bfcl-loaded) — BFCL v4 multi-turn agent campaign
fixtures/     raw upstream downloads (gitignored)
test-data/    normalized JSONL produced by `ingest` (gitignored — ADR-0007)
results/      retrieval / agent JSONL outputs + REPORT.md (gitignored)
```

Both `retrieval/` and `agent/` have their own README with the details of how to run their layer.

## Eval modes

Per [ADR-0006](docs/adr/0006-benchmark-corpus-and-eval-modes.md), three eval modes split across two suites.

**Retrieval-only** — fast, deterministic, $0, no API keys. Backs claims about ranking quality. Lives in [`retrieval/`](retrieval/).

- **(a) MetaTool — pre-fetch retrieval (replace path).** Measures whether BM25 surfaces the right tool given a real user-task query, before the agent's turn. 199 OpenAI plugin descriptions + ~21k user queries (MIT). Single-tool and multi-tool queries are both scored as **tool retrieval** via the real `ToolRegistry`; the summary splits into `single-tool · tool` and `multi-tool · tool` (single-tool recall is binary, multi-tool recall is fractional, so they get separate panels).
- **(b) ToolRet — IR / autonomous-discovery retrieval (gateway path).** Measures whether the index ranks correctly when the agent emits an IR-shaped query mid-loop (e.g. `searchTools("a tool that converts currency")`). 7,961 retrieval tasks across 35 sub-corpora over a 44,453-tool catalog (Apache-2.0).
- **(d) SR-Agents — skill retrieval.** Measures whether BM25 surfaces the right authored **skill** document (not a tool) for a task. ~26k skills (`name` + `description` indexed; markdown `body` carried but not indexed) as the catalog/distractor universe, with ~5.4k instances across six datasets (`bigcodebench`, `champ`, `logicbench`, `medcalcbench`, `theoremqa`, `toolqa`). Scored via the real `SkillRegistry`. Fully separate from tool retrieval — its own corpus, ingester, run path, and output. Multi-mapping datasets (e.g. CHAMP) have several gold skills per instance, so recall@K is fractional and `complete@K` is the all-or-nothing bar. See [ADR-0008](docs/adr/0008-skill-retrieval-eval-mode.md).

**Agentic** — end-to-end agent runs with token cost + correctness signals. Requires API keys. Lives in [`agent/`](agent/).

- **(c) MetaTool tasks + LLM-as-judge.** Runs control (baseline + oracle) arms alongside three Ratel arms (full / pre-discovery only / discovery-tool only) on MetaTool user-task queries with stubbed tool responses. The full Ratel arm pre-discovers BM25 top-K from the prompt *and* exposes the gateway (`search_tools` / `invoke_tool`) so the model can recover when pre-discovery missed; the two ablations isolate which layer is doing the work. An optional local-only `claude-sdk-tool-search` arm can be wired up alongside as a competitive baseline against Anthropic's native tool-search-tool. Programmatic judge does selection-only intersection (`effective_tool_ids ∩ gold_tools ≠ ∅`); the LLM judge scores final-text coherence against the user prompt as a fallback / tiebreaker. Reports input/output tokens, cache hit rate, $-cost, and wall-clock time at realistic catalog sizes (default pool size 180), averaged per-scenario across runs and then across scenarios.

## Setup

### Prerequisites

- **Node.js 20+** and **pnpm 10+** (the repo pins `pnpm@10.28.2` via `packageManager`). Drives the agent campaign and the report renderer.
- **Rust stable** (1.85+ — the retrieval crate uses edition 2024). Drives ingest and retrieval-only modes.
- **API keys** (mode c only — retrieval-only is $0 and key-free):
  - `OPENAI_API_KEY` — required to score `gpt-5.4-mini`.
  - `ANTHROPIC_API_KEY` — required to score `claude-sonnet-4-6` / `claude-opus-4-7` **and** to power the default LLM judge.

  Set at least one. The harness skips models with no key rather than failing. Place them in `.env` at the repo root — `dotenv` is loaded automatically.

### Install

```bash
pnpm install                  # JS deps for agent/ + bfcl-loaded/ + mcpverse/
cargo build -p ratel-benchmark-retrieval --release   # optional — first `cargo run --release` compiles otherwise
```

`pnpm install` is the only required step for the agent layer. The retrieval crate's deps are pulled lazily on first `cargo run`, so the explicit `cargo build` above is just a warm-up.

## Ingest datasets

Per [ADR-0007](docs/adr/0007-benchmark-corpus-not-snapshotted.md), the corpora are **not** committed — neither the raw upstream downloads (`fixtures/`) nor the normalized JSONL (`test-data/`). You have to ingest once before any benchmark can run.

```bash
# MetaTool — feeds retrieval mode (a) and the agent campaign mode (c).
cargo run -p ratel-benchmark-retrieval --release -- ingest metatool --download

# ToolRet — feeds retrieval mode (b). Skip if you only care about the agent campaign.
cargo run -p ratel-benchmark-retrieval --release -- ingest toolret --download

# SR-Agents — feeds skill retrieval mode (d). Writes a skill catalog + instances.
cargo run -p ratel-benchmark-retrieval --release -- ingest sragents --download
```

`--download` pulls upstream sources (MetaTool: MIT, ToolRet: Apache-2.0, SR-Agents) into `fixtures/` via `curl` (and unzips the SR-Agents corpus), then writes normalized JSONL to `test-data/metatool.jsonl`, `test-data/toolret.jsonl`, and `test-data/sragents-skills.jsonl` + `test-data/sragents.jsonl`. Re-running without `--download` against the cached fixtures produces a byte-identical JSONL. Full ingest tunables in [`retrieval/README.md`](retrieval/README.md).

Or let `run-all` handle ingest for you — it no-ops when the snapshots already exist.

## Run the whole benchmark

```bash
pnpm -F @ratel-ai/benchmark run-all
```

This single command:

1. Ingests MetaTool, ToolRet, and SR-Agents (downloads upstream sources via `curl`) if their normalized JSONL isn't already present under `test-data/`.
2. Runs BM25 tool retrieval over each tool corpus (modes a + b) and skill retrieval over SR-Agents (mode d), at corpus-appropriate pool sizes.
3. Runs the mode-(c) MetaTool agent campaign with conservative defaults if `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set; otherwise prints a notice and skips. Defaults: 50 sampled scenarios × 1 run × every committed arm × available models, $5 global cap.
4. Renders `results/REPORT.md` from the retrieval and (if present) agent JSONL outputs.

Re-running `run-all` skips ingest when the snapshot already exists; pass `--force` to re-ingest, `--skip-ingest` to fail loudly if missing, `--skip-agent` to opt out of mode (c) even when keys are set, or `--only metatool|toolret|sragents` to restrict to one corpus. For the headline N=5 variance run, invoke `pnpm -F @ratel-ai/benchmark start` directly — see [`agent/README.md`](agent/README.md).

## Run only the agent benchmark (mode c)

Use this when you're iterating on agent behavior and don't want to repay retrieval each loop. Prerequisites: `pnpm install` done, MetaTool ingested (`test-data/metatool.jsonl` exists), and at least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` exported.

Fast local smoke (~$0.20–$1, ~50 scenarios × 1 run × 3 arms × 1 model):

```bash
pnpm -F @ratel-ai/benchmark start \
  --scenarios 50 --runs 1 \
  --arms control-baseline,control-oracle,ratel-full \
  --models claude-sonnet-4-6 \
  --pool-sizes 180 \
  --dollar-global 5 \
  --concurrency 10
```

Output: one JSONL row per `(scenario, arm, model, pool_size, run)` cell in `agent/results/agent.jsonl` (resumable — re-runs skip already-recorded cells unless `--force`). Render a report with `pnpm -F @ratel-ai/benchmark report`.

Full flag reference, the N=5 variance recipe, the local-Ollama path, and the cached-control-runs / `--ephemeral` workflow live in [`agent/README.md`](agent/README.md).

## Run only retrieval (modes a + b + d)

Fast, deterministic, $0, no API keys. Once the corpora are ingested:

```bash
# Tool retrieval (modes a / b) — one corpus JSONL.
cargo run -p ratel-benchmark-retrieval --release -- retrieval \
  --corpus test-data/metatool.jsonl \
  --output results/metatool-retrieval.jsonl \
  --top-k 1,3,5,10 --pool-sizes 30,100,180

# Skill retrieval (mode d) — separate subcommand, catalog + instances.
cargo run -p ratel-benchmark-retrieval --release -- skill-retrieval \
  --instances test-data/sragents.jsonl \
  --skills-catalog test-data/sragents-skills.jsonl \
  --output results/sragents-skill-retrieval.jsonl \
  --top-k 1,3,5,10 --pool-sizes 100,1000,26262
```

ToolRet uses the same `retrieval` runner with a different corpus and pool-size sweep. Skill retrieval uses the dedicated `skill-retrieval` subcommand (the skill catalog is the BM25 index; instances carry gold skill ids). Full reference in [`retrieval/README.md`](retrieval/README.md).

## BFCL (Berkeley Function-Calling Leaderboard)

[BFCL](https://gorilla.cs.berkeley.edu/leaderboard.html) (v3) measures whether a model calls the **right function with the right arguments** for a user request. ratel-bench runs it through the same two layers as the other suites:

- **Retrieval** (Rust, deterministic, $0, no keys) — can BM25 surface the gold function from a catalog of distractors, before the agent's turn?
- **Task completion** (agent campaign, needs an API key) — does the LLM actually emit the correct call? Scored against BFCL's `possible_answer` ground truth by the AST judge (right function **and** arguments). Tools are **stubbed**, so this verifies the *call*, not real execution/outcome.

Two categories are ingested: `bfcl-simple` (one gold call) and `bfcl-multiple` (pick the right function among several).

### Data type

Each line is a `Scenario` — the same schema every suite uses ([`retrieval/src/corpus.rs`](retrieval/src/corpus.rs)). BFCL rows additionally carry `gold_calls`: per-argument **lists of acceptable values** (BFCL's `possible_answer`). An empty string `""` in a list marks the argument optional.

```jsonc
{
  "id": "bfcl-simple-simple_0",
  "prompt": "Find the area of a triangle with a base of 10 units and height of 5 units.",
  "candidate_pool": [
    { "id": "calculate_triangle_area", "name": "calculate_triangle_area",
      "description": "Calculate the area of a triangle ...",
      "input_schema": { "type": "object", "properties": { /* base, height, unit */ },
                        "required": ["base", "height"] }, "output_schema": {} }
  ],
  "gold_tools": ["calculate_triangle_area"],   // drives the RETRIEVAL gold
  "category": "bfcl-simple",
  "gold_calls": [                              // drives the TASK-COMPLETION (AST) judge
    { "tool": "calculate_triangle_area",
      "args": { "base": [10], "height": [5], "unit": ["units", ""] } }   // "" ⇒ optional arg
  ]
}
```

Ingest the two corpora (writes `test-data/bfcl-simple.jsonl` + `test-data/bfcl-multiple.jsonl`):

```bash
cargo run -p ratel-benchmark-retrieval --release -- ingest bfcl --download
```

`--corpus` takes a single path. To run **simple + multiple together**, concatenate them — ids are prefixed (`bfcl-simple-*` / `bfcl-multiple-*`) so they don't collide:

```bash
cat test-data/bfcl-simple.jsonl test-data/bfcl-multiple.jsonl > test-data/bfcl-all.jsonl
```

### 1. Retrieval test

```bash
cargo run -p ratel-benchmark-retrieval --release -- retrieval \
  --corpus test-data/bfcl-all.jsonl \
  --output results/bfcl-retrieval.jsonl \
  --top-k 1,3,5,10 --pool-sizes 30,100,180
```

Writes two files: a per-`(scenario, pool_size, k)` **detail JSONL** at `--output` (overwritten each run) and an aggregate **summary JSONL** at `…-summary.jsonl` (appended — one line per run, a history). The summary auto-splits into `single-tool` and `multi-tool` buckets.

### 2. Task-completion task

Needs an API key (`OPENAI_API_KEY` for `gpt-*`, `ANTHROPIC_API_KEY` for `claude-*`). `--models` accepts a comma-separated list:

```bash
pnpm -F @ratel-ai/benchmark start \
  --corpus test-data/bfcl-all.jsonl \
  --models gpt-5.4-mini,claude-sonnet-4-6 \
  --ephemeral
```

Writes one row per `(scenario, arm, model, pool_size, run)` cell. `--ephemeral` lands in a fresh `agent/results/ephemeral/agent-<timestamp>.jsonl` (drop it + add `--output` to persist to the canonical `agent/results/agent.jsonl`, which is appended/resumable).

### 3. Consolidated report (optional)

Merge both layers into one file (`results/BFCL.json`, overwritten each run):

```bash
pnpm -F @ratel-ai/benchmark exec tsx src/bfcl-export.ts
```

### Export structures

Both run types carry **`run_type`**, **`run_id`**, and **`generated_at`**, and share `scenario_id` — so retrieval and task-completion rows are joinable.

**Retrieval detail row** (one per scenario × pool_size × k):

```jsonc
{
  "run_type": "retrieval",
  "run_id": "ret-1782146740544786",
  "generated_at": "2026-06-22T16:45:40.544786+00:00",
  "scenario_id": "bfcl-simple-simple_0",
  "query": "Find the area of a triangle with a base of 10 units and height of 5 units.",
  "golden_answer": ["calculate_triangle_area"],     // gold tool id(s)
  "category": "bfcl-simple",
  "target_pool_size": 30, "actual_pool_size": 30,
  "ratel_ai_core_version": "0.2.0",
  "k": 3, "pool_size": 30, "gold_count": 1,
  "recall_at_k": 1.0, "precision_at_k": 0.33, "reciprocal_rank": 1.0,
  "hit_at_k": true, "complete_at_k": true, "ndcg_at_k": 1.0,
  "gold_score": 4.06,
  "retrieved": [                                     // what BM25 returned, best-first
    { "id": "calculate_triangle_area", "score": 4.06 },
    { "id": "calculate_area", "score": 3.74 }
  ]
}
```

**Retrieval summary line** (one per run, appended):

```jsonc
{
  "run_id": "ret-1782146740544786",
  "generated_at": "2026-06-22T16:45:40.544786+00:00",
  "ratel_ai_core_version": "0.2.0",
  "corpus": "test-data/bfcl-all.jsonl", "output": "results/bfcl-retrieval.jsonl",
  "scenarios": 400, "rows_written": 4800,
  "top_k": [1, 3, 5, 10], "pool_sizes": [30, 100, 180], "seed": 42,
  "by_bucket": [
    { "subset": "single-tool", "mode": "tool", "scenarios": 200,
      "overall": { "bm25_gold_score": { "mean": 4.15, "coverage": 1.0, "...": "..." },
                   "by_k": [ { "k": 1, "mean_recall": 1.0, "hit_rate": 1.0, "mean_ndcg": 1.0, "mean_mrr": 1.0, "...": "..." } ] },
      "by_pool_size": [ { "pool_size": 30, "by_k": [ /* same shape, per pool */ ] } ] },
    { "subset": "multi-tool", "mode": "tool", "...": "..." }
  ]
}
```

**Task-completion row** (`agent.jsonl`; abbreviated — full schema in [`agent/src/types.ts`](agent/src/types.ts)):

```jsonc
{
  "run_type": "task_completion",
  "run_id": "5f3c…-uuid",
  "generated_at": "2026-06-22T16:45:40.000Z",
  "scenario_id": "bfcl-simple-simple_0",
  "category": "bfcl-simple",
  "arm": "ratel-full", "model": "gpt-5.4-mini", "run_index": 0,
  "ratel_version": "0.2.0",            // @ratel-ai/sdk version (NOT ratel-ai-core)
  "catalog_size": 5, "pool_size": 180, "seed": 42,
  "input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290,
  "tool_calls_total": 1, "turns": 1,
  "effective_tool_ids": ["calculate_triangle_area"],
  "programmatic_verdict": "pass",      // right function (name only)
  "ast_verdict": "pass",               // right function AND arguments (BFCL AST)
  "judge_verdict": "n/a",
  "wall_ms": 980, "dollar_cost": 0.0006,
  "tool_calls": [ { "toolId": "calculate_triangle_area", "args": { "base": 10, "height": 5 } } ]
}
```

**Consolidated `results/BFCL.json`** (pretty-printed; pools single+multi for task completion):

```jsonc
{
  "benchmark": "BFCL",
  "generated_at": "2026-06-22T16:45:40.000Z",
  "ratel_ai_core_version": "0.2.0",    // from retrieval rows
  "ratel_sdk_version": "0.2.0",        // from agent cells
  "counts": { "agent_cells": 1200, "retrieval_rows": 4800 },
  "retrieval_evaluation": {
    "bfcl-simple": [ { "k": 5, "pool_size": 180, "n": 200,
      "accuracy_at_k": 0.98, "complete_at_k": 0.98,
      "mean_recall": 0.98, "mean_ndcg": 0.96, "mean_mrr": 0.95, "...": "..." } ],
    "bfcl-multiple": [ /* same shape */ ]
  },
  "task_completion_evaluation": {
    "note": "selection_accuracy = right function called; task_completion_accuracy = right function AND arguments (BFCL AST). single+multi pooled.",
    "by_arm": [ { "arm": "ratel-full", "model": "gpt-5.4-mini", "category": "bfcl", "pool_size": 180,
      "scenarios": 400, "runs": 1, "selection_accuracy": 0.91, "task_completion_accuracy": 0.84,
      "mean_input_tokens": 1234, "mean_dollar_cost": 0.0006, "mean_wall_ms": 980, "...": "..." } ],
    "savings_ratel_vs_control": [ { "model": "gpt-5.4-mini", "pool_size": 180,
      "input_tokens": { "control": 9000, "ratel": 1234, "savings_pct": 86.3 }, "...": "..." } ]
  }
}
```

## Corpus format

All suites consume the same JSONL — one `Scenario` per line:

```jsonc
{
  "id": "fs-001",
  "prompt": "Show me /etc/hosts.",
  "candidate_pool": [ /* tools available in this scenario */ ],
  "gold_tools": ["fs.read_file"],
  "judge_criteria": "mentions localhost",
  "category": "filesystem"
}
```

The Rust definition in [`retrieval/src/corpus.rs`](retrieval/src/corpus.rs) is canonical. The TS mirror in [`agent/src/types.ts`](agent/src/types.ts) tracks it.
