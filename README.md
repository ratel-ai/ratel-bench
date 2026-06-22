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

**Three-stage contract.** Each eval produces a **per-row metrics** JSONL (overwritten each run) and appends to an **experiment-summary** JSONL (an append-only history). A separate **report producer** rebuilds one report from the summaries, keyed by **ratel-ai-core version**. We always run on **`bfcl-all`** (the two subsets concatenated); everything splits back into `simple` / `multiple` by the scenario-id prefix.

| File (`results/raw/bfcl/`) | Producer | Mode |
|---|---|---|
| `retrieval-rows.jsonl` | Rust `retrieval` | overwrite |
| `retrieval-summary.jsonl` | `bfcl-summarize` | **append** |
| `agent.jsonl` (intermediate cells) | agent `start` | append/resumable |
| `task-completion-rows.jsonl` | `bfcl-summarize` | overwrite |
| `task-completion-summary.jsonl` | `bfcl-summarize` | **append** |
| `results/reports/bfcl/report.json` | `bfcl-report` | overwrite (rebuilt) |

Both summary files share a flat contract `{ timestamp, ratel_ai_core_version, source, type, …metrics }` (`source` is `retriever_evaluation`/`task_completion`, `type` is `simple`/`multiple`). Task-completion rows add `model` (the LLM name) and `arm` (`control-baseline`/`control-oracle`/`ratel-full`); retrieval has a single retriever (ratel-ai-core BM25), so it carries neither. `ratel_ai_core_version` on both sides is resolved from `Cargo.lock` (Rust via `build.rs`; the agent via `agent/src/versions.ts`).

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

### 1. Retrieval eval → per-row

```bash
cargo run -p ratel-benchmark-retrieval --release -- retrieval \
  --corpus test-data/bfcl-all.jsonl \
  --output results/raw/bfcl/retrieval-rows.jsonl \
  --top-k 1,3,5,10 --pool-sizes 30,100,600
```

Writes `retrieval-rows.jsonl` (overwrite) — one row per `(scenario, pool_size, k)` with `ratel_ai_core_version`, `generated_at`, `category`, `query`, `golden_answer` (true answers = gold tool ids), `retrieved` (id+BM25-score), all metrics, and `gold_score` (per-row gold BM25 similarity).

### 2. Task-completion eval → cells

Needs an API key (`OPENAI_API_KEY` for `gpt-*`, `ANTHROPIC_API_KEY` for `claude-*`); `ollama:<tag>` needs none.

```bash
pnpm -F @ratel-ai/benchmark start \
  --corpus test-data/bfcl-all.jsonl \
  --models claude-haiku-4-5 \
  --arms ratel-full \
  --output results/raw/bfcl/agent.jsonl
```

Writes agent cells (intermediate); each carries `ratel_ai_core_version` (the `Cargo.lock` value) and `generated_at`.

### 3. Summarize → per-row + summaries

Pure transform (no recompute, no API): writes `task-completion-rows.jsonl` (overwrite) and **appends** `retrieval-summary.jsonl` + `task-completion-summary.jsonl`.

```bash
pnpm -F @ratel-ai/benchmark bfcl-summarize
```

Reads `retrieval-rows.jsonl` + `agent.jsonl` + `test-data/bfcl-all.jsonl` (override via `--retrieval-rows` / `--agent` / `--corpus`). **All arms** present in `agent.jsonl` are summarized per arm by default; pass `--arm ratel-full` to restrict to one.

### 4. Build the report

Rebuilds `results/reports/bfcl/report.json` from the append-only summaries — one entry per ratel-ai-core version, latest timestamp per (source, model, type):

```bash
pnpm -F @ratel-ai/benchmark bfcl-report
```

### End-to-end (local Ollama, $0)

`pnpm -F @ratel-ai/benchmark run-all --bfcl` runs it all: ingest → concat `bfcl-all` → retrieval → qwen3.5 campaign → `bfcl-summarize` → `bfcl-report` → cleanup (raw results + report kept). `--skip-agent` does retrieval + summarize + report only; `--keep-bfcl` keeps the ingested corpora.

### Export structures

**Retrieval per-row** (`retrieval-rows.jsonl`, overwrite):

```jsonc
{
  "generated_at": "2026-06-22T16:45:40+00:00",
  "ratel_ai_core_version": "0.2.0",
  "scenario_id": "bfcl-simple-simple_0",
  "category": "bfcl-simple",
  "query": "Find the area of a triangle ...",
  "golden_answer": ["calculate_triangle_area"],     // true answer = gold tool id(s)
  "k": 3, "pool_size": 30, "gold_count": 1,
  "recall_at_k": 1.0, "precision_at_k": 0.33, "reciprocal_rank": 1.0,
  "hit_at_k": true, "complete_at_k": true, "ndcg_at_k": 1.0,
  "gold_score": 4.06,                                // per-row gold BM25 similarity
  "retrieved": [ { "id": "calculate_triangle_area", "score": 4.06 }, { "id": "calculate_area", "score": 3.74 } ]
}
```

**Retrieval summary** (`retrieval-summary.jsonl`, append — one row per type × pool_size × k):

```jsonc
{
  "timestamp": "2026-06-22T16:45:40+00:00", "ratel_ai_core_version": "0.2.0",
  "source": "retriever_evaluation", "type": "simple",
  "pool_size": 30, "k": 1, "n": 399,
  "mean_precision": 0.9, "median_precision": 1.0, "mean_recall": 0.9, "median_recall": 1.0,
  "mean_mrr": 0.92, "median_mrr": 1.0, "mean_ndcg": 0.91, "median_ndcg": 1.0,
  "accuracy": 0.9, "complete_rate": 0.9,
  "gold_similarity": { "mean": 19.9, "median": 19.2, "stddev": 10.7, "coverage": 1.0 }
}
```

**Task-completion per-row** (`task-completion-rows.jsonl`, overwrite — one per scenario × arm):

```jsonc
{
  "ratel_ai_core_version": "0.2.0", "generated_at": "2026-06-22T10:00:00.000Z",
  "type": "simple", "model": "claude-haiku-4-5", "arm": "ratel-full",
  "scenario_id": "bfcl-simple-simple_0",
  "query": "Find the area of a triangle ...",
  "true_answers": { "gold_tools": ["calculate_triangle_area"],
                    "gold_calls": [ { "tool": "calculate_triangle_area", "args": { "base": [10], "height": [5] } } ] },
  "llm_answer": [ { "toolId": "calculate_triangle_area", "args": { "base": 10, "height": 5 } } ],
  "selection_pass": true, "task_completion_pass": true,
  "input_tokens": 1234, "output_tokens": 56, "total_tokens": 1290,
  "dollar_cost": 0.0006, "wall_ms": 980, "turns": 1
}
```

**Task-completion summary** (`task-completion-summary.jsonl`, append — one row per type × LLM × arm):

```jsonc
{
  "timestamp": "2026-06-22T10:00:00.000Z", "ratel_ai_core_version": "0.2.0",
  "source": "task_completion", "model": "claude-haiku-4-5", "arm": "ratel-full", "type": "simple",
  "scenarios": 399, "selection_accuracy": 0.91, "task_completion_accuracy": 0.84,
  "mean_input_tokens": 1234, "mean_total_tokens": 1290, "mean_dollar_cost": 0.0006,
  "mean_wall_ms": 980, "mean_turns": 1
}
```

**Report** (`results/reports/bfcl/report.json`, rebuilt each run — latest timestamp per group):

```jsonc
{
  "generated_at": "2026-06-22T16:45:40.000Z",
  "ratel_versions": {
    "0.2.0": {
      "retriever_evaluation": {
        "simple":   { "timestamp": "...", "metrics": [ { "pool_size": 30, "k": 1, "accuracy": 0.9, "gold_similarity": { "...": "..." } } ] },
        "multiple": { "timestamp": "...", "metrics": [ /* per pool_size × k */ ] }
      },
      "task_completion": {
        "claude-haiku-4-5": {
          "ratel-full":       { "simple": { "timestamp": "...", "metrics": { "scenarios": 399, "selection_accuracy": 0.91, "task_completion_accuracy": 0.84, "mean_input_tokens": 3471, "mean_dollar_cost": 0.0049, "...": "..." } },
                                "multiple": { "timestamp": "...", "metrics": { "...": "..." } } },
          "control-baseline": { "simple": { "metrics": { "task_completion_accuracy": 0.82, "mean_input_tokens": 30160, "...": "..." } }, "multiple": { "...": "..." } },
          "control-oracle":   { "simple": { "metrics": { "...": "..." } }, "multiple": { "...": "..." } }
        }
      }
    }
  }
}
```

Structure: `retriever_evaluation` is keyed by `type` directly (single retriever); `task_completion` is keyed by **LLM → arm → type**, so each arm (`control-baseline` / `control-oracle` / `ratel-full`) shows its own accuracy/tokens/cost — compare them directly per LLM and subset.

**Add/update + reproducibility.** Re-running an eval appends a new summary line (with the eval's `generated_at` timestamp); `bfcl-report` rebuilds deterministically, taking the **latest timestamp per group** (retrieval: version × type; task: version × LLM × arm × type) — so a new LLM/arm/version is **added** and an existing entry is **updated** to its latest run, with others untouched. Because the report is rebuilt from the append-only summaries (not edited in place), it's fully reproducible; the per-row files keep the latest run's detail for audit.

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
