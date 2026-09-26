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
| `--force` | off | run even when the existing `--output` was built from a different task list (skips the task-list-hash guard); does not disable cache reuse |
| `--refresh-native` | off | ignore the native cache (`--cache-source`): every native cell runs live |
| `--skip-doctor` | off | skip the preflight doctor check |
| `--keep-stack` / `--keep-artifacts` | off | leave the sandbox stack / per-cell artifacts running after the campaign for debugging |

### Errored cells

`mcpatlas-cell-errors.ts` classifies a cell's `error` as `transient | access | request | timeout |
outcome` — a local copy of `agent/src/cell-errors.ts` (agent-eval never imports `agent/`), extended
with the shapes a cell carries. First match wins:

1. `finish_reason` `error_max_turns` → `outcome`, whatever the error text.
2. `error_timeout`, or runCell's envelope error with `(timedOut=true` → `timeout`.
3. runCell's `catalog integrity: …`, `ratel cell produced empty telemetry …` (gateway never
   started; affects `excluded_cells` only, ratel cells are never cached), and an envelope error
   with `(timedOut=false, exitCode=null, signal=SIGKILL)` (OOM / external kill) → `transient`.
   Any other runCell throw (`finish_reason` `error`: no scorable result came back) that no
   later rule recognises — spawn failures (`spawn ENOMEM`, `(timedOut=false, exitCode=null,
   signal=null): spawn … EAGAIN|ENOENT`), `ENOSPC`, a CLI exit without an envelope — is
   `transient` too (see 8).
4. The shared message rules: context-too-long / content-filter → `outcome`; model-access wording
   and codex's `Quota exceeded …` / `hit your usage limit` → `access`.
5. By HTTP status, read from Claude Code's `API Error: <status> …` and
   `API Error: Request rejected (<status>) · …` (the 429 shape of the pinned 2.1.246) and codex's
   `unexpected status <status> …` / `exceeded retry limit, last status: <status> …`:
   408/409/429/5xx → `transient`, 401/403/404 → `access`, other 4xx → `request`.
6. Connection/overload wording → `transient`: Claude Code's `API Error: Unable to connect to API`,
   `Connection dropped (…)`, `Connection refused …`, `Can't reach the API server …`,
   `No response from API`, `Connection error`, `Request timed out` (also bare), and the like;
   codex's `stream disconnected before completion`, high demand, `model is at capacity`,
   `Connection failed:`, and the retryable errors it surfaces once its retries run out:
   `rate limit exceeded: …`, `Error while reading the server response: …`, `request timed out`,
   `timeout waiting for child process to exit`, `internal error; agent loop died unexpectedly`,
   and network Io errnos (`Connection reset by peer (os error 104)`, `Broken pipe (os error 32)`
   …; a local one such as `No such file or directory (os error 2)` is not); `Overloaded`,
   `fetch failed`, `ECONNRESET`.
7. codex's raw `"type": "invalid_request_error"` body → `request`.
8. Anything else → `transient` on a runCell throw (`finish_reason` `error`), `outcome` on a
   harness envelope row (the model's own failure).

Claude Code's error subtypes (`error_during_execution`, `error_max_turns`, …) carry no `result`:
their `errors[]` lines (one per line) are the cell's `error`.

A codex turn that fails after an agent message records the failure (not the prose) as `error`;
`final_text` keeps the prose. A codex stream retry (`Reconnecting... n/m`) the turn recovers from
is not an error.

- **Native cache (`--cache-source`).** A prior native cell is never reused when its run errored
  `transient | access | request`, or when its `judge_error` starts `judge failed:`; that key runs
  live, and a later good cell on the same key wins. `timeout`, max-turns and `judge omitted …`
  cells are final measurements and stay reusable. `nativeCacheKey` is unchanged.
  `--refresh-native` disables reuse entirely.
- **Summaries — documented divergence.** BFCL/SR-Agents summaries drop infra-errored
  (`transient | access`) rows from every metric. `mcpatlas-summarize` does NOT: its denominators
  are unchanged, and those cells still score as fails. It only counts them as `excluded_cells` on
  each task-completion row (alongside `errored`, which counts every errored cell; `request`,
  `timeout` and `outcome` errors are in `errored` but not in `excluded_cells`). A non-zero
  `excluded_cells` means that group's rates are depressed by infra — re-run before quoting it.

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

  mcpatlas-cell-errors.ts error classes for a cell (local copy of agent/src/cell-errors.ts):
                           native-cache reuse and summary excluded_cells

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
