# `retrieval/`

Rust half of the benchmark harness. Crate: `ratel-benchmark-retrieval`. Houses the corpus loader, BM25 retrieval-only runner, ingestion adapters for MetaTool and ToolRet, and the `ratel-benchmark-retrieval` CLI. Depends on `ratel-ai-core` from crates.io (pinned, see top-level [`README.md`](../README.md)).

Pairs with the TS agent layer at [`agent/`](../agent). For the modes overview and the unified `run-all` entrypoint, see [`../README.md`](../README.md). Locked decisions in [`docs/adr/0005-benchmark-design.md`](../docs/adr/0005-benchmark-design.md), [`0006-benchmark-corpus-and-eval-modes.md`](../docs/adr/0006-benchmark-corpus-and-eval-modes.md), [`0007-benchmark-corpus-not-snapshotted.md`](../docs/adr/0007-benchmark-corpus-not-snapshotted.md).

## Layout

```
src/
  corpus.rs         scenario JSONL parser + ToolSpec
  retrieval.rs      BM25 ranking + recall/precision/MRR/nDCG
  runner.rs         scenario × pool_size × k cell driver + summary aggregation
  stats.rs          mean/median/population-stddev helpers
  ingest/
    metatool.rs     MetaTool → normalized JSONL adapter
    toolret.rs      ToolRet → normalized JSONL adapter
  main.rs           CLI: `retrieval` + `ingest <source>`
tests/
  metatool_ingest.rs end-to-end ingest → retrieval round-trip
```

## Quickstart: MetaTool retrieval-only (mode a)

Two commands. Step 1 ingests MetaTool into the harness's normalized JSONL; step 2 runs BM25 retrieval over it. Both raw downloads (`fixtures/metatool/`) and normalized JSONL (`test-data/metatool.jsonl`) are gitignored per [ADR-0007](../docs/adr/0007-benchmark-corpus-not-snapshotted.md).

### 1. Ingest MetaTool

```bash
cargo run -p ratel-benchmark-retrieval --release -- ingest metatool --download
```

`--download` pulls the three upstream MetaTool sources (MIT) via `curl` into `fixtures/metatool/`, then emits the **full** upstream query set (≈20,630 single-tool + ≈497 multi-tool, modulo unknown-gold filtering) to `test-data/metatool.jsonl`. Re-running without `--download` against the cached fixtures produces a byte-identical JSONL.

Tunables (`... ingest metatool --help` for the full list):

- `--download` — pull upstream sources via `curl` before ingesting. Drop the flag to re-ingest pre-existing files.
- `--fixtures-dir PATH` (default `fixtures/metatool`) — where downloaded files live (and are read from when `--download` is omitted).
- `--plugins / --single-tool / --multi-tool` — override individual source paths if your layout doesn't match upstream's. Pass `--multi-tool ""` to skip the multi-tool slice.

### 2. Run retrieval

```bash
cargo run -p ratel-benchmark-retrieval --release -- retrieval \
  --corpus test-data/metatool.jsonl \
  --output results/metatool-retrieval.jsonl \
  --top-k 1,3,5,10 --pool-sizes 30,100,180
```

Emits one JSONL row per `(scenario, pool_size, k)` cell with recall@K, precision@K, MRR@K, hit@K, and nDCG@K (binary relevance). One BM25 ranking per query is sliced at every K cutoff, so adding more K values is essentially free.

Alongside the detail JSONL, an aggregate **summary** is appended automatically (default
`results/metatool-retrieval-summary.jsonl`, derived from `--output`). Each run adds one
compact JSON line — existing lines are kept, so re-running this command (e.g. after a BM25
tuning change, or on a later date) builds up a comparable history of experiments rather than
clobbering the previous run; `generated_at` on each line is what distinguishes one run from
another. Each line has an `overall` block (mean/median precision/recall/nDCG/MRR + hit-rate per
K, aggregated across every pool size) and a `by_pool_size` breakdown with the same shape per
pool size. Both blocks also report `bm25_gold_score` — the mean/median/population-stddev of
the raw BM25 score assigned to the gold tool when it's found in the ranking, plus `coverage`
(the fraction of scenarios where it was found at all). This is the standalone "average
retriever performance" artifact; no TS report run required. (Note: unlike the summary, the
detail JSONL at `--output` is still overwritten on each run.)

Tunables:

- `--corpus PATH` — JSONL corpus to evaluate.
- `--output PATH` (default `results/retrieval.jsonl`) — where to write the metrics JSONL (overwritten each run).
- `--summary-output PATH` (default: `--output` with its extension replaced by `-summary.jsonl`) — where to append the aggregate overall-performance summary, one JSON line per run.
- `--top-k A,B,C` (default `1,3,5,10`) — comma-separated K cutoffs.
- `--pool-sizes A,B,C` (default `30,150,600`) — catalog scales to evaluate at. MetaTool's gold-tool universe is ≈199 plugins, so `30,100,180` keeps every cell meaningful (the default `30,150,600` would silently clamp).
- `--scenarios N` — limit to first N rows for a smoke run.
- `--seed N` (default `42`) — seed for distractor shuffling.

The merged report splits MetaTool into separate `single-tool` and `multi-tool` panels because their `recall@K` semantics differ — single-tool is binary (0 or 1), multi-tool is fractional.

## Quickstart: ToolRet retrieval-only (mode b)

Same shape as mode (a): one ingest, one retrieval pass.

### 1. Ingest ToolRet

```bash
cargo run -p ratel-benchmark-retrieval --release -- ingest toolret --download
```

`--download` pulls 38 Parquet files (3 tool subsets + 35 query sub-corpora) from the upstream HuggingFace datasets (Apache-2.0) via `curl` into `fixtures/toolret/`, then writes the full normalized corpus to `test-data/toolret.jsonl`. **No sampling** — every upstream query is kept (rows whose gold tools aren't in the published catalog are dropped, ~5 of 7,961). Re-running without `--download` against the cached fixtures produces a byte-identical JSONL.

The scenario `prompt` is ToolRet's `instruction` field with the `Given a … task, retrieve tools that …` wrapper stripped — the unwrapped instruction is the IR-shaped retrieval query an agent would emit at the gateway, which is the path mode (b) is meant to measure.

Tunables (`... ingest toolret --help` for the full list):

- `--download` — pull upstream parquet via `curl` before ingesting.
- `--fixtures-dir PATH` (default `fixtures/toolret`) — where downloaded parquet lives, laid out as `<dir>/tools/<subset>.parquet` and `<dir>/queries/<subset>.parquet`.
- `--output PATH` (default `test-data/toolret.jsonl`) — where to write the normalized JSONL.

### 2. Run retrieval

```bash
cargo run -p ratel-benchmark-retrieval --release -- retrieval \
  --corpus test-data/toolret.jsonl \
  --output results/toolret-retrieval.jsonl \
  --top-k 1,3,5,10 --pool-sizes 100,1000,7000
```

Same runner as mode (a); only the corpus and pool sizes change. ToolRet's effective universe under gold-only pooling (see below) is ~7,651 unique tools, so `100,1000,7000` spans a small / mid / full-haystack curve.

**Leaderboard caveat.** Per ADR-0006 the harness mirrors MetaTool's gold-only pooling: each scenario's `candidate_pool` carries only its gold tool(s); the runner adds distractors at retrieval time from the union of every other scenario's gold tools. That caps the universe at ~7,651 — well below ToolRet's published 44k pool. **Absolute nDCG numbers from this harness are NOT directly comparable to ToolRet's leaderboard**; relative deltas between arms / index variants are valid. Side-loading the full 44k catalog as a runner-time distractor universe is a tracked follow-up.

## Generating the merged report

After the suites have written their JSONL outputs:

```bash
pnpm -F @ratel-ai/benchmark report
```

By default this auto-discovers every `*retrieval.jsonl` under `results/` (so a MetaTool pass and a ToolRet pass appear side by side, one panel per corpus inferred from scenario-id prefix), reads `agent/results/agent.jsonl` if present, and writes `results/REPORT.md`. To pin the inputs explicitly:

```bash
pnpm -F @ratel-ai/benchmark report \
  --agent agent/results/agent.jsonl \
  --retrieval results/metatool-retrieval.jsonl \
  --retrieval results/toolret-retrieval.jsonl \
  --output results/REPORT.md
```

Pass `--retrieval` once per file. The retrieval section reports both mean and median per `(corpus, subset, k, pool_size)` cell — useful for MetaTool, where most queries hit gold at rank 1 but a long tail of misses pulls the mean below the median.

## Tests

```bash
cargo test -p ratel-benchmark-retrieval
```

Unit tests cover the corpus loader, BM25 metric primitives, runner cell iteration, and both ingest adapters (parsing + scenario assembly). The integration test under `tests/metatool_ingest.rs` drives an inline-fixture ingest end-to-end through the retrieval runner with finite metrics.
