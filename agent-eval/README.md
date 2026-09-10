# `agent-eval/`

The MCP-Atlas benchmark: measures what the ratel-local gateway costs and saves compared to
wiring MCP servers directly into the agent. Both arms run the same MCP-Atlas coding tasks against
the same sandbox (11 servers, 127 tools):

- **native** — upstream servers wired straight into the agent; every tool schema sits in context.
- **ratel** — the same servers behind ratel-local; the agent only sees `search_tools` /
  `invoke_tool`.

For the repo-wide corpora/arms overview see [`../README.md`](../README.md). This package is
separate from the fixed BFCL/SR-Agents retrieval+LLM-eval design described in
[`../EXPERIMENTS.md`](../EXPERIMENTS.md) — it benchmarks a different question (gateway vs native
MCP wiring, on real coding tasks) and has its own pipeline below.

## Pipeline

Five stages, run in order:

```bash
pnpm -F @ratel-ai/agent-eval mcpatlas-doctor      # preflight — asserts the sandbox's served
                                                   # tool set and catalog integrity
pnpm -F @ratel-ai/agent-eval mcpatlas-ingest      # pins the MCP-Atlas dataset revision,
                                                   # regenerates the catalog manifests
pnpm -F @ratel-ai/agent-eval mcpatlas-run         # the campaign driver — see flags below
pnpm -F @ratel-ai/agent-eval mcpatlas-summarize   # per-cell exports (tool-calls / search-events
                                                   # / retrieval-rows / agent rows)
pnpm -F @ratel-ai/agent-eval mcpatlas-report      # renders the markdown comparison report
```

`mcpatlas-shim` (`atlas-mcp-shim.ts`) isn't a pipeline stage you invoke directly — `mcpatlas-run`
spawns it to proxy the sandbox servers to the agent and log every tool call and gateway search.

### `mcpatlas-run` flags

| flag | default | what it does |
|---|---|---|
| `--scope` | `coding` | which MCP-Atlas task scope to run |
| `--arms` | `native,ratel` | comma-separated arms to run |
| `--tasks` | all | limit to N tasks |
| `--task-offset` | `0` | skip the first N tasks (isolate a single task without re-running earlier ones) |
| `--runs` | `1` | k>1 sampling — repeat each task k times |
| `--concurrency` | `1` | parallel cells |
| `--dollar-global` | `50` | spend cap across the run |
| `--catalog-tools` | full catalog | shrink the sandbox catalog to N tools (catalog size is an experimental variable) |
| `--retriever-method` | `bm25` | retriever method wired into the ratel arm |
| `--harness` | `claude-code` | agent harness — `claude-code` or `codex` (codex requires an explicit `--model`) |
| `--model` | `claude-haiku-4-5` | model id |
| `--judge-model` | none | LLM judge model (requires `ANTHROPIC_API_KEY`) |
| `--ratel-local` | `0.8.1` | ratel-local version pin |
| `--sandbox-url` | `http://localhost:1984` | sandbox endpoint |
| `--output` | `results/raw/mcpatlas/agent.jsonl` | output path |
| `--cache-source` | none | reuse cells from a prior run |
| `--max-turns` | `256` | per-cell turn cap |
| `--per-cell-timeout-ms` | `1800000` (30 min) | per-cell wall-clock timeout |
| `--force` | off | ignore the cache, re-run every cell |
| `--refresh-native` | off | force-refresh the native arm's tool listing |
| `--skip-doctor` | off | skip the preflight doctor check |
| `--keep-stack` / `--keep-artifacts` | off | leave the sandbox stack / per-cell artifacts running after the campaign for debugging |

## Layout

```
src/
  mcpatlas-build.ts       assembleCell() — assembles one McpAtlasCell (task + arm + config)
  mcpatlas-run.ts         mcpatlas-run — the campaign driver (process spawning, kill-signal
                           capture, process-group reaping, per-cell timeouts)
  mcpatlas-agent.ts       claude-code invocation, transcript/turn/token-usage parsing
  mcpatlas-codex.ts       Codex CLI harness (approval handling, config.toml building)

  atlas-mcp-shim.ts       proxies the sandbox MCP servers to the agent, logs tool calls +
                           gateway searches with turn indexes, gold markers, sizes
  mcpatlas-gateway.ts     ratel-local search_tools/invoke_tool gateway wiring
  mcpatlas-servers.ts     sandbox server definitions/config
  mcpatlas-ingest.ts      pins the MCP-Atlas dataset revision, regenerates catalog manifests
  mcpatlas-doctor.ts      preflight — asserts the served tool set and catalog integrity

  mcpatlas-claim-match.ts deterministic claim pre-screen
  mcpatlas-judge.ts       LLM judge scoring the pre-screen's residual (sees only task + claims +
                           final answer — never the trajectory, gold tools, or arm label)
  mcpatlas-prompt.ts      agent/judge prompts
  mcpatlas-stats.ts       confidence intervals and significance
  mcpatlas-summarize.ts   per-cell exports; task/retrieval/failure/cost summaries
  mcpatlas-report.ts      markdown comparison report
  mcpatlas-types.ts       shared types
```

## Sandbox

The MCP-Atlas sandbox (server set, image, patches) is documented in
[`sandbox/README.md`](sandbox/README.md).

## Fixtures

`fixtures/mcpatlas/` (task list + catalog manifests) is version-controlled — like the
`bfcl`/`sragents` reports, it's allowlisted in the repo's `.gitignore` so the experiment
definition stays reproducible even though raw dataset rows aren't snapshotted.

## Tests

```bash
pnpm -F @ratel-ai/agent-eval test
```

Unit tests cover the pipeline end-to-end: cell assembly, transcript/turn parsing for both
harnesses, the shim's call/search logging, the gateway wiring, claim matching, judge prompt
construction, statistics, and the summarize/report aggregations. Real LLM calls and sandbox I/O
are not exercised in unit tests.
