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
| `--max-output-tokens` | unset | agent output cap per response, sent as `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (claude-code only; an error with `--harness codex`) — see [Output caps](#output-caps) |
| `--judge-max-output-tokens` | unset | judge output cap per call (requires `--judge-model`) — see [Output caps](#output-caps) |
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

Environment: `RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS` (default `10`, `0` = off) — see
[Error circuit](#error-circuit).

### Output caps

Both caps are opt-in and have **no default**. Claude Code already sends its own `max_tokens` and
derives its auto-compaction threshold from it, so an implicit cap would silently change the agent
under test. Unset means no cap is sent and no cap field is recorded: `config_hash` and native cache
keys stay byte-identical to an uncapped run. Truncation *detection* (`truncated_turns`,
`truncated_cells`, judge truncation) applies to every run, capped or not.

- Values must be positive integers; anything else (including a flag with no value) is an error.
  `--judge-max-output-tokens` without `--judge-model` is an error too: a screen-only run makes no
  judge call, so the cap would never apply yet would still change `config_hash` and cache keys.
- Only `--max-output-tokens` caps Claude Code. An ambient `CLAUDE_CODE_MAX_OUTPUT_TOKENS` (shell,
  `agent-eval/.env`, a Claude Code settings `env`) is stripped from the claude spawn's environment,
  with a warning, so it can never cap a run recorded as uncapped. Codex never reads the variable,
  so a codex run neither strips it nor warns.
- A set cap is recorded in the frozen config (`max_output_tokens` / `judge_max_output_tokens`)
  and on every cell, and is appended to the native cache key. `--cache-source` keys each prior
  cell on its **own** recorded caps, so a capped native never serves an uncapped run, and the
  reverse never happens either.
- Summary task and cost rows carry `max_output_tokens` / `judge_max_output_tokens` when any cell
  in the group was capped: the shared value, or `mixed`. They are metrics, not group keys, so a
  capped run still replaces the uncapped rows of the same group in `report.json` — the field
  says so. Uncapped summaries carry neither key.
- Agent truncation is a scored outcome, never re-run. `truncated_turns` on every claude-code cell
  built from a result envelope (including max-turns and other `is_error` envelopes) counts
  assistant responses that stopped on `max_tokens` — the set cap, or Claude Code's own default
  when unset — deduped by message id because Claude Code writes one transcript line per content
  block. `mcpatlas-summarize` counts cells with `truncated_turns > 0` as `truncated_cells`.
  Denominators are unchanged. Codex cells, legacy cells and `runCell`'s `finish_reason: "error"`
  rows (no-envelope timeouts or kills, spawn failures, catalog-integrity or empty-telemetry
  refusals) carry no `truncated_turns`, so they never count in `truncated_cells`.
- A judge whose output is cut off (finish `length`) scores the task n/a (`coverage` null) with
  `judge_error: "judge truncated at N output tokens"`, or `"judge truncated at the provider's
  default output limit"` when no judge cap is set. If every claim was still answered, the scores
  stand and there is no `judge_error`. Like `judge omitted …`, both stay reusable; only
  `judge failed: …` re-runs. (Before this, an uncapped truncation read `judge failed: No object
  generated…` and re-ran.)

### Judge retries

The judge call passes `maxRetries: 6` (`JUDGE_MAX_RETRIES` in `mcpatlas-judge.ts`; the SDK
default is 2), so throttling and 5xx get the SDK's backoff (2s doubling, ~2 min in all; a
`Retry-After` under 60s replaces a step) before the task is scored `judge failed: …` (still
re-run, as above). A truncation after a retried attempt still reads `judge truncated …` (a
no-output truncation arrives inside the SDK's `RetryError`, whose `lastError` is classified; a
partial-JSON one arrives bare) and meters that attempt's tokens. No retry stats are recorded (the
judge call is not behind agent/'s `withRetry`). The count is a fixed constant, not a flag, and is
**not** recorded in the frozen config: it changes whether a judge call eventually succeeds, never
how it scores, so `config_hash` and native cache keys stay unchanged.

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
   (incl. Bedrock's daily cap `Too many tokens per day`) and codex's `Quota exceeded …` /
   `hit your usage limit` → `access`.
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

Not copied: rules for agent/'s `withRetry` wrapper's `RetriesExhaustedError` /
`FatalProviderError`. The agent under test is Claude Code or codex (their own retries), never an
`ai` SDK model, so neither can reach a cell; the judge only writes `judge_error` (see Judge
retries).

Claude Code's error subtypes (`error_during_execution`, `error_max_turns`, …) carry no `result`:
their `errors[]` lines (one per line) are the cell's `error`.

A codex turn that fails after an agent message records the failure (not the prose) as `error`;
`final_text` keeps the prose. A codex stream retry (`Reconnecting... n/m`) the turn recovers from
is not an error.

- **Native cache (`--cache-source`).** A prior native cell is never reused when its run errored
  `transient | access | request`, or when its `judge_error` starts `judge failed:`; that key runs
  live, and a later good cell on the same key wins. `timeout`, max-turns, `judge omitted …` and
  `judge truncated …` cells are final measurements and stay reusable. This rule leaves
  `nativeCacheKey` unchanged (only a set output cap extends it; see [Output caps](#output-caps)).
  `--refresh-native` disables reuse entirely.
- **Summaries — documented divergence.** BFCL/SR-Agents summaries drop infra-errored
  (`transient | access`) rows from every metric. `mcpatlas-summarize` does NOT: its denominators
  are unchanged, and those cells still score as fails. It only counts them as `excluded_cells` on
  each task-completion row (alongside `errored`, which counts every errored cell; `request`,
  `timeout` and `outcome` errors are in `errored` but not in `excluded_cells`). A non-zero
  `excluded_cells` means that group's rates are depressed by infra — re-run before quoting it.

### Error circuit

`mcpatlas-run` stops launching cells after `RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS` consecutive
infra-errored cells (`transient | access`, as classified above) — a gated, daily-capped or
unreachable model, where every further cell only writes another errored row. Same knob name and
default as the `agent/` breaker.

- Default `10`; `0` turns it off. Unset = default; any other value, empty included, must be a
  non-negative integer (surrounding spaces allowed, as in `agent/`), else it errors before anything
  is spent.
- An error-free cell resets the count. `timeout`, max-turns, `request` and `outcome` errors neither
  count nor reset: none is an infra failure (the provider answered, or the cell hit its deadline),
  but none is a clean run either. `request` is still re-run (see [Errored cells](#errored-cells)).
- Counted in completion order across both passes: a trip in the native pass means the ratel pass
  launches nothing. Cells already in flight finish and are written.
- The run ends with `aborted: <model> — N consecutive infra-errored cells (last: <class>: <error>)`,
  then the usual `done: … stopped=error_circuit` line (not-launched cells count as `skipped`), and
  exits **2**. A `--dollar-global` stop (`stopped=global_cap`) still exits 0.
- Not recorded in the frozen config: it decides when a run stops, never how a cell scores, so
  `config_hash` and native cache keys are unchanged.

## Layout

```
src/
  mcpatlas-build.ts       assembleCell() — assembles one McpAtlasCell (task + arm + config)
  mcpatlas-run.ts         mcpatlas-run — the campaign driver (process spawning, kill-signal
                           capture, process-group reaping, per-cell timeouts, error circuit)
  mcpatlas-agent.ts       claude-code invocation, transcript/turn/token-usage parsing
  mcpatlas-codex.ts       Codex CLI harness (approval handling, config.toml building)

  atlas-mcp-shim.ts       proxies the sandbox MCP servers to the agent, logs tool calls +
                           gateway searches with turn indexes, gold markers, sizes
  mcpatlas-gateway.ts     ratel-local search_tools/invoke_tool gateway wiring
  mcpatlas-servers.ts     sandbox server definitions/config
  mcpatlas-ingest.ts      pins the MCP-Atlas dataset revision, regenerates catalog manifests
  mcpatlas-doctor.ts      preflight — asserts the served tool set and catalog integrity

  mcpatlas-cell-errors.ts error classes for a cell (local copy of agent/src/cell-errors.ts):
                           native-cache reuse, summary excluded_cells and the error circuit

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
