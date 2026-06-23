# 9. BFCL function-calling evaluation + reproducible per-row / summary / report export

Date: 2026-06-23

## Status

Accepted. Adds a BFCL (Berkeley Function-Calling Leaderboard) evaluation
alongside the MetaTool / ToolRet / SR-Agents modes of ADR-0006/0007/0008, and
defines a reproducible export contract (per-row → append-only summaries →
rebuildable report) that the benchmark website consumes. Supersedes the earlier
single-`results/BFCL.json` snapshot.

## Context

MetaTool (mode a/c) and ToolRet (mode b) measure **retrieval**; SR-Agents
(mode d) measures **skill retrieval**. None measures whether a model calls a
function **correctly** — the right function *and* the right arguments. BFCL is
the standard for that, so it fills a real gap in the suite.

Two further problems prompted the export redesign:

1. **Reproducibility.** The first BFCL agent campaign was run under a `$`-cap;
   a capped run stops mid-dataset, so its aggregate numbers depend on *where it
   stopped* — not reproducible. A benchmark result must be rebuildable.
2. **Multi-version / multi-model accumulation.** We need to compare several
   `ratel-ai-core` versions and several LLMs over time without re-running prior
   work, and publish a single artifact to a separate website repo.

## Decision

### Data — BFCL v3 `simple` + `multiple`

Ingest BFCL v3 `simple` (one gold call) and `multiple` (pick the right function
among several) via `crate::ingest::bfcl` → `test-data/bfcl-simple.jsonl` and
`test-data/bfcl-multiple.jsonl`. Each scenario carries `gold_calls` (BFCL's
`possible_answer`: per-argument **lists of acceptable values**; `""` in a list
marks the arg optional). Runs use **`bfcl-all`** — the two subsets concatenated;
everything splits back to `simple`/`multiple` by the `bfcl-simple-*` /
`bfcl-multiple-*` scenario-id prefix.

### Two eval layers (reuse the harness)

- **Retrieval** (Rust, deterministic, $0): does BM25 surface the gold function
  from a distractor pool? Single retriever (ratel-ai-core BM25).
- **Task completion** (agent campaign): does the LLM emit the correct call?
  Scored by the **AST judge** against `gold_calls`. Tools are **stubbed**, so
  this verifies the *call*, not real execution/outcome (no executable / multi-turn
  BFCL categories are ingested).

### Export contract — three stages

1. **Per-row metrics** (OVERWRITE, latest run's detail):
   `results/raw/bfcl/retrieval-rows.jsonl` (Rust) and
   `results/raw/bfcl/task-completion-rows.jsonl` (built by `bfcl-summarize` from
   agent cells joined to the corpus — query, gold answers, the model's calls, and
   per-cell verdicts/tokens/cost/latency).
2. **Experiment summaries** (APPEND-only history):
   `retrieval-summary.jsonl` + `task-completion-summary.jsonl`. Flat contract
   `{ timestamp, ratel_ai_core_version, source, type, …metrics }`. Task rows add
   `model` (LLM) and `arm`; retrieval rows carry neither (single retriever).
3. **Report** (`bfcl-report` → `results/reports/bfcl/report.json`, OVERWRITE,
   rebuilt): keyed `ratel_versions[v].retriever_evaluation[type]` and
   `ratel_versions[v].task_completion[llm][arm][type]`, taking the **latest
   timestamp per group**. Rebuilding from the append-only summaries each run is
   deterministic — a new version/LLM/arm is *added*, an existing one *updated* to
   its latest run, others untouched. This is the single artifact the website reads.

`run-all --bfcl` orchestrates: ingest → concat `bfcl-all` → retrieval →
agent campaign → `bfcl-summarize` → `bfcl-report` → cleanup (raw + report kept).

### Task-completion metrics (the five reported)

Per `(ratel-ai-core version × LLM × arm × subset)`:

1. **`task_completion_accuracy`** — right function **and** arguments (AST).
2. **`selection_accuracy`** — right function (name only).
3. **`recall`** — mean **argument recall**: fraction of the gold call's required
   args supplied with an acceptable value (partial-credit complement to the
   all-or-nothing AST verdict; `astArgRecall` in `judges/ast.ts`). Tool-level
   recall was rejected — for single-gold BFCL it is identical to selection.
4. **token cost** — `mean_total_tokens`.
5. **`latency_p50_ms`** — p50 wall-clock.

Both retrieval and task completion are **split by subset** (`simple`/`multiple`).
Task completion is broken down **per arm** (`control-baseline` / `control-oracle`
/ `ratel-full`) so the Ratel-vs-control comparison is read directly off the report
rather than computed as a separate savings field.

### Version alignment

`ratel_ai_core_version` is resolved from the workspace **`Cargo.lock`** on both
layers — Rust via `build.rs` (`RATEL_AI_CORE_VERSION`), the agent via
`agent/src/versions.ts` (the agent reaches retrieval through `@ratel-ai/sdk`, so
its own `ratel_version` is recorded separately and informationally). The report
groups by this version; there is no cross-layer hard-fail.

### Tool-name sanitization collisions

Provider tool-name rules force id sanitization (`.`/`/` → `_`). BFCL contains
distinct ids that collapse to the same name (e.g. `solve.quadratic_equation` and
`solve_quadratic_equation`), and `bfcl-all` surfaces cross-subset clashes that
the per-subset ingest de-collision doesn't catch. `registerDirect`
(`agents/_shared.ts`) **disambiguates** (suffix `_2`, `_3`, …) instead of
throwing, keeping `nameToId` correct so judging is unaffected.

## Caveats

- **AST is a port** of BFCL's `possible_answer` matcher, not byte-identical —
  deeply nested generics and some exotic Python coercions are out of scope, so a
  few failures may be matcher strictness rather than true model errors.
- **No execution/outcome verification.** Tools are stubbed; correctness is at the
  call level. Executable / multi-turn BFCL categories are not ingested.
- **Task completion is largely arm-insensitive on BFCL.** Every arm exposes the
  gold tool (baseline = full pool incl. gold, oracle = gold only, ratel = discovers
  it), so accuracy barely moves across arms — the differentiator is cost/tokens,
  not completion. At small `--scenarios` samples, the per-arm/model accuracies are
  noisy and can coincide.
- **`control-baseline` is expensive** (the full pool, ~30k input tokens/cell);
  it dominates campaign cost and is the reason the `$`-cap was introduced.

## Consequences

- BFCL function-calling quality is reported per `ratel-ai-core` version, per LLM,
  per arm, per subset, with five leaderboard metrics — on a single, rebuildable
  `report.json` the website repo consumes.
- Results accumulate across versions/models without re-running prior work (append-
  only summaries + resumable agent cells keyed by
  `ratel_version::scenario::arm::model::run::pool`).
- New code: `agent/src/bfcl-types.ts`, `bfcl-summarize.ts`, `bfcl-report.ts`,
  `versions.ts`, `astArgRecall` in `judges/ast.ts`; retrieval gains per-row
  `query`/`golden_answer`/`retrieved`/gold similarity; `registerDirect` collision
  disambiguation. The earlier `bfcl-export.ts` → `results/BFCL.json` (single
  version-aligned snapshot) is removed.
