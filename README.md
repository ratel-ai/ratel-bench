# ratel-bench

A minimal TypeScript harness for benchmarking [**Ratel**](https://github.com/ratel-ai/ratel)
context-engineering against a baseline — measuring **tool-selection retrieval
quality** and **agent-loop token cost** at varying catalog (pool) sizes.

The harness pins `@ratel-ai/sdk` from npm, so it measures the artifact users
install. Datasets are ingested locally and gitignored — nothing is snapshotted in
the repo.

**Latest results: [`REPORT.md`](REPORT.md)** — full MetaTool campaign (retrieval
sweep + 3-arm agent campaign with token/cost split, recall@K, and the
faithfulness/caching analysis).

## What it does

Two things, from one CLI (`src/cli.ts`):

1. **Tool-selection (retrieval).** BM25 over the catalog at each pool size; scores
   `MRR@5` (`1/position` if the gold tool is in the top-5). Deterministic, $0, no
   API key. (`src/tool-selection.ts`)
2. **Agent campaign.** Runs three arms over the scenarios × pool sizes, scoring
   `pass = expectedTool ∈ effectiveToolIds` (pure selection, no LLM judge).
   (`src/runner.ts`, `src/agents/`)

### Arms

All three share one AI-SDK `ToolLoopAgent` loop (`src/agents/baseline.ts`); they
differ only in the tool surface fed in.

- **oracle** — only the gold tool(s) exposed. Upper bound, pool-agnostic.
- **baseline** — the full pool registered as direct tools. The fat-context floor.
- **ratel-full** — gateway-only (`search_tools` / `invoke_tool`) with BM25
  pre-discovery **injected as a synthetic `search_tools` tool-result message**, so
  the model can't tell pre-discovery from self-discovery and must `invoke_tool` a
  hit. (`src/agents/ratel-full.ts`)

A uniform Anthropic ephemeral cache breakpoint is set on every arm's tool block
(plus ratel-full's injected discovery); token metering splits input into
fresh / cache-read / cache-write.

## Layout

```
src/
  cli.ts             entrypoint: load → tool-selection + agent campaign → JSONL + summary
  types.ts           Dataset / Scenario / Turn / ToolSpec / Agent / AgentRunResult
  dataset.ts         loadDataset()
  pool.ts            buildScenarioPool() — per-scenario gold-first + deterministic distractors
  catalog.ts         thin wrapper over @ratel-ai/sdk ToolCatalog
  tool-selection.ts  retrieval (MRR@5) sweep
  runner.ts          campaign task list + bounded-concurrency runner + summary
  agents/            baseline (+ base loop & metering), oracle, ratel-full
  ingest/metatool.ts MetaTool Lane-B ingest → datasets/metatool.json
datasets/
  example-dataset.json   committed toy dataset (30 tools, 2 scenarios)
  metatool.json          generated, gitignored (199 tools, 20,614 scenarios)
results/                 JSONL + summary.json outputs (gitignored)
REPORT.md                latest narrative results
```

## Setup

- **Node 20+**, **pnpm 10+** (pinned `pnpm@10.28.2`).
- **API keys** (agent campaign only — retrieval is $0/key-free): `ANTHROPIC_API_KEY`
  for `claude-*`, `OPENAI_API_KEY` for `gpt-*`. Put them in `.env` (loaded via
  `dotenv`). Any other model id (e.g. `retrieval-only`) skips the agent campaign.

```bash
pnpm install
```

## Ingest

The corpus is not committed — regenerate it deterministically from pinned upstream
URLs (cached under `datasets/.cache/`):

```bash
pnpm ingest:metatool      # → datasets/metatool.json
```

## Run

```bash
# Retrieval only, full corpus (no API key):
pnpm start --dataset datasets/metatool.json --pools 30,100,180 \
  --model retrieval-only --out results/mt-retrieval

# Agent campaign, 30-scenario subset (needs ANTHROPIC_API_KEY):
pnpm start --dataset datasets/metatool.json --pools 30,100,180 \
  --sample 30 --model claude-sonnet-4-6 --agents ratel-full \
  --concurrency 8 --out results/mt-agents
```

**Flags** (env fallbacks in parens): `--dataset` (`DATASET`), `--pools` (`POOLS`,
default `10,30`), `--model` (`MODEL`, default `claude-sonnet-4-6`), `--agents`
(`AGENTS`, default `ratel-full`; comma-list of non-control arms, baseline+oracle
always run), `--sample N` (`SAMPLE`, deterministic subset; 0 = all), `--seed`,
`--top-k` (retrieval cutoff), `--concurrency` (`CONCURRENCY`, default 8), `--out`.

**Outputs** (gitignored): `results/<run>/tool-selection.jsonl`, `agents.jsonl`,
`summary.json`. The summary JSON is the machine-readable report; `REPORT.md` is the
hand-written narrative.

```bash
pnpm typecheck && pnpm lint
```

## Corpus format

One `Dataset` JSON (`src/types.ts` is canonical):

```jsonc
{
  "tools": [{ "id": "weather.get", "name": "weather.get", "description": "…", "inputSchema": {} }],
  "scenarios": [{
    "id": "metatool-st-42",
    "turns": [{
      "input": { "messages": [{ "role": "user", "content": "what's the weather in Paris?" }] },
      "expectedTool": "weather.get",      // gold tool id for this turn
      "expectedQuery": "weather in Paris" // authored IR query (retrieval scoring)
    }]
  }]
}
```

MetaTool ingests to this shape (single-turn, no parameter schemas).

## Open questions / next steps

- **Cleaner K attribution.** ratel-full's pass-rate before/after conflates the
  injection-mechanism change with K 5→15. An A/B varying *only* K on the current
  injection would isolate it. (The recall@K diagnostic in `REPORT.md` already
  isolates the retrieval effect — n=1,200, no API.)
- **Selection-aware scoring.** Pure selection counts "names the right tool but
  doesn't `invoke_tool`" as a miss — the bulk of the refusals and the ~73% oracle
  ceiling. An LLM-judge / commit-extraction pass would credit selection across all
  arms.
- **SDK determinism.** `@ratel-ai/sdk` search randomizes tied results per call — a
  seedable mode upstream would make runs bit-reproducible (see `REPORT.md`).
- **Tighter numbers.** N=30 carries ±~9pp; a larger `--sample` and/or a second
  model would firm up the agent pass rates.
- **Reproduce recall@K.** The recall diagnostic was a throwaway script; add a small
  committed mode if it's worth re-running.
