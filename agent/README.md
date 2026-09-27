# `agent/`

End-to-end agent layer of the benchmark plus the unified suite orchestrator. Drives the Vercel AI SDK [`ToolLoopAgent`](https://ai-sdk.dev/docs/reference/ai-sdk-core/tool-loop-agent) across two control arms and several non-control arms (see below), meters token usage, judges correctness, and emits one JSONL row per `(scenario, arm, model, run)` cell. Mode (c) per ADR-0006 is the agent campaign this layer powers.

Headline metrics in `REPORT.md` average per scenario across runs first, then across scenarios — so a scenario passing 4/5 runs contributes a 0.8 success rate, and high-run-count scenarios can't drown out the rest.

Pairs with the Rust retrieval-only layer at [`retrieval/`](../retrieval). For modes overview see [`../README.md`](../README.md). Locked decisions in [`docs/adr/0005-benchmark-design.md`](../docs/adr/0005-benchmark-design.md), [`0006`](../docs/adr/0006-benchmark-corpus-and-eval-modes.md), [`0007`](../docs/adr/0007-benchmark-corpus-not-snapshotted.md).

## Arms

Each arm is an `AgentDescriptor` (`{ id, label, run(input) }`) defined in its own file under [`src/agents/`](src/agents/). The runner builds a registry at startup and dispatches each cell to the arm's `run` function. Reading any one file shows the full integration end-to-end (tool construction, optional Ratel wiring, agent loop) — no implicit framework magic.

| id | label | path | what it does |
|---|---|---|---|
| `control-baseline` | control (baseline) | `agents/control-baseline.ts` | Every tool in the expanded pool, registered directly. Fat-context floor. |
| `control-oracle`   | control (oracle)   | `agents/control-oracle.ts`   | Only the gold tools. Upper bound on what the model can do given perfect selection. **Pool-size-agnostic**: emits one cell per (scenario, model, run) regardless of `--pool-sizes`; the row's `pool_size` is `null` and the report shows it as `—` with the real catalog count (~1–2) in the `catalog` column. |
| `ratel-full`       | ratel (full)       | `agents/non-control/ratel-full.ts` | BM25 top-K of the prompt pre-fetched as direct tools, **plus** the `search_tools` / `invoke_tool` gateway. The canonical Ratel surface. |
| `ratel-pre-discovery`  | ratel (pre-discovery only) | `agents/non-control/ratel-pre-discovery.ts` | BM25 top-K only — no gateway. Ablation: did pre-fetch alone suffice? |
| `ratel-discovery-tool` | ratel (discovery-tool only) | `agents/non-control/ratel-discovery-tool.ts` | Gateway only — no pre-fetch. Ablation: can the agent self-discover with a strong index? |
| `claude-sdk-tool-search` | claude-sdk (tool-search tool) | `agents/non-control/ignore.claude-sdk-tool-search.ts` | **Local-only, gitignored.** Anthropic's native [tool-search-tool](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-search-tool) as a competitive baseline. Claude-only via `skipForModel`. |

The default `--arms` list excludes `claude-sdk-tool-search` (it lives behind a local-only gitignored file); opt in via `--arms ...,claude-sdk-tool-search` on a host that has wired it up.

`control-oracle` is committed in the default arm list, deviating from ADR-0006 ("oracle drops from the default arm list... stays available behind a flag"). Even under stubbed execution it's the cleanest selection-noise floor — the model's output coherence given exactly the gold tools and no distractors — and we want it in every report rather than gated behind a flag.

### Adding a local-only arm

Drop a new file in `src/agents/non-control/` whose name starts with `ignore.` (matches the local `.gitignore` rule). Export a `descriptor: AgentDescriptor` with a unique `id`. The runner's auto-discovery picks it up next time. Use this for prototypes, closed-SDK baselines, or any arm you don't want to commit yet.

## Layout

```
src/
  agents/
    _shared.ts                shared sanitization / AI SDK adapters / metered loop
    control-baseline.ts       control arm — all tools direct
    control-oracle.ts         control arm — gold tools only
    non-control/              auto-discovered; `ignore.*` is gitignored
      ratel-full.ts           ratel: BM25 pre-fetch + gateway
      ratel-pre-discovery.ts  ratel: BM25 pre-fetch only
      ratel-discovery-tool.ts ratel: gateway only
      ignore.claude-sdk-tool-search.ts  (local-only) Anthropic tool-search-tool
  cli.ts              entry — pnpm start (mode c agent campaign)
  corpus.ts           reads the shared JSONL scenario format
  judges/
    programmatic.ts   selection-intersection (per ADR-0006)
    llm.ts            Sonnet-as-judge primary for mode (c) (prompt-only fallback when no criteria)
  llm-retry.ts        LLM retry wrapper + pausable, abortable cell deadline
  metering.ts         tokens, calls, turns, cost wrapped around agent.generate
  pool.ts             builds the per-scenario tool pool (gold + seeded distractors)
  report.ts           aggregator (medians, savings, retrieval, taxonomy)
  report-cli.ts       entry — pnpm report
  results-audit.ts    entry — pnpm results-audit (read-only error / stale-cache audit of raw JSONL)
  run-all.ts          entry — pnpm run-all (whole benchmark: ingest + a + b + c + report)
  runner.ts           registry-based dispatch, resumable, dollar-capped
  types.ts            AgentDescriptor / AgentRunInput / CellResult / Scenario shapes
```

## Run the whole benchmark

```bash
pnpm -F @ratel-ai/benchmark run-all
```

Ingests both corpora (if missing), runs retrieval modes (a) + (b), runs the mode-(c) agent campaign with conservative defaults if a provider key is set (skipped with a notice otherwise — keeping `run-all` $0 by default), and renders REPORT.md. See [`../README.md`](../README.md) for the full description.

Flags: `--force` (re-ingest), `--skip-ingest`, `--skip-agent` (skip mode (c) even with keys), `--only metatool|toolret`.

The auto-invoked mode (c) defaults to: 50 sampled scenarios × 1 run × every committed arm (the two control arms plus the three ratel ablations), available models only (`claude-sonnet-4-6` and/or `gpt-5.4-mini` depending on which key is set), pool size 180, $5 global cap. The local-only `claude-sdk-tool-search` arm is excluded by default. For the headline variance run see the next section.

## Run the headline agent campaign (mode c)

```bash
# Required env (one or both):
#   OPENAI_API_KEY     — for gpt-5.4-mini
#   ANTHROPIC_API_KEY  — for claude-sonnet-4-6 (also powers the LLM judge)
#
# The default --corpus path expects the ingested MetaTool snapshot at
# test-data/metatool.jsonl. Run `pnpm -F @ratel-ai/benchmark run-all`
# (or `cargo run -p ratel-benchmark-retrieval --release -- ingest metatool --download`)
# first.

pnpm -F @ratel-ai/benchmark start \
  --output agent/results/agent.jsonl \
  --scenarios 200 \
  --arms control-baseline,control-oracle,ratel-full,ratel-pre-discovery,ratel-discovery-tool \
  --models gpt-5.4-mini,claude-sonnet-4-6 \
  --runs 5 \
  --top-k 5 \
  --pool-sizes 30,100,180 \
  --max-steps 12 \
  --dollar-global 25 \
  --concurrency 10
```

Resumable — re-runs skip cells already in `agent.jsonl` unless `--force`. Pass `--ephemeral` instead to write each smoke into a fresh `agent/results/ephemeral/agent-<timestamp>.jsonl` file so the canonical `agent.jsonl` stays untouched. `--scenarios N` samples a deterministic seeded subset of the full ~21k MetaTool query set; the same `--seed` reproduces the same subset across runs.

`--concurrency N` (default 10) controls how many cells run in parallel. The benchmark is wall-clock-bound on provider latency, so 10 typically yields ~10× speedup against cloud APIs. Dial down to `1` for Ollama (single-process server) or tight provider tiers. Dollar caps are best-effort under concurrency: in-flight cells finish, no new ones start, so overshoot is bounded by `concurrency × per-cell-cost` (~$0.30 at the defaults).

`--timeout-ms N` (default 60000) sets the per-cell **active-time** deadline: retry backoff doesn't count (see "LLM retries and deadlines"), and on expiry the request is aborted. Cloud models rarely need more, but local Ollama models (especially CPU-bound or large 70B+) can comfortably exceed a minute on a 12-step trace — bump to `300000` (5 min) or higher when you see `run timed out after 60000ms` errors in the trace.

`--retriever bm25|semantic|hybrid` (default `bm25`) picks the retrieval method the Ratel arms use — this is the **0.4.0** knob (sparse / dense / hybrid). Pre-0.4.0 versions don't accept it; their runs are unchanged. Since there is no Rust `ratel-ai-core` 0.4.0, generation runs SDK-side: pin the 0.4.0 SDK, tag each method as its own report layer with `RATEL_VERSION_LABEL=0.4.0-sparse|dense|hybrid`, and set `--retriever`. `control-baseline`/`control-oracle` are retriever-independent and reused from the canonical 0.2.0 cache (see "Cached control runs") — run the *first* method with `--force` to purge any stale/pre-fix cells so a gold-incomplete 0.2.0 pool can't skew the numbers. See [`EXPERIMENTS.md`](../EXPERIMENTS.md) for the full per-method recipe. `bfcl-candidates` and `sragents-candidates` take the same flag; `sragents-select` does not (it reads pre-ranked candidates).

`--pool-sizes` controls the per-scenario tool catalog (gold + distractors pulled from other scenarios). Accepts a comma-separated list (e.g. `--pool-sizes 30,100,180`) — each scenario is evaluated at every requested size, and the report breaks the headline / savings / failure tables down per pool. Pass a single value to skip the sweep. The legacy singular form `--pool-size 180` still works as an alias for one value but rejects commas. The default (180) sits at the MetaTool plugin universe ceiling; smaller values stress retrieval less, larger values are clamped at the universe size. Pool-size-agnostic arms (currently just `control-oracle`) ignore this flag and emit one cell per (scenario, model, run) regardless of how many sizes are listed.

## Output caps (`maxOutputTokens`)

Every agent LLM call requests the model's output cap, taken from one place: `models.json` `run[].maxOutputTokens` (4096 for claude-haiku-4-5 / claude-sonnet-4-6 / claude-opus-4-5; 16384 for claude-sonnet-5 / claude-opus-4-8 / claude-fable-5 / gpt-5.4-mini / gpt-5.6-luna). A test (`models-catalog.test.ts`) requires it on every entry with a `bedrockProfile` or a `claude-`/`gpt-` id. There is no silent default: a model without a catalog cap (self-hosted `qwen3-4b` / `mistral-7b-instruct`, `ollama:*`, ad-hoc `<url>#name` ids — matched to an entry by its `endpoint`) sends no cap, and the startup line says so: `caps: claude-haiku-4-5=4096, ollama:qwen3.5=none (no models.json cap)`. Judge calls send no cap unless `--judge-max-output-tokens` is set.

- `--max-output-tokens N|none` (`start`, `sragents-select`) overrides the catalog for every agent model; `none` sends no cap. Agent calls only.
- `--judge-max-output-tokens N` (`start`, `rejudge`) caps the LLM judge. Unset (the default) sends no cap.
- N must be a positive integer (`--max-output-tokens` also accepts `none`); `sragents-select` rejects `--judge-max-output-tokens` (it has no judge).

The cap is a per-call setting (`ToolLoopAgent` / `generateObject`), not `wrapLanguageModel` middleware, attached after model resolution so no resolver branch can skip it. Rows record the **requested** cap as `max_output_tokens` (`null` = none sent; legacy rows lack it); `@ai-sdk/anthropic` may clamp it further for ids it knows. Arms get it on `AgentRunInput.model.maxOutputTokens` and must honour and stamp it (`runMeteredLoop` does both).

- **Truncation is visible and final.** A BFCL step that stops on the cap counts in `truncated_steps` (`max_step_output_tokens` is the headroom signal); the cell is scored on its verdict and never re-run. An SR selection cut off at the cap is a `NoObjectGeneratedError` with finish `length` → `error_class: "outcome"`, a final scored miss. A truncated judge answer is `n/a` with `judge truncated at N output tokens`.
- **Resume refuses conflicting recorded caps.** A run throws when its output holds rows at the current version for one of its models with a *different recorded* `max_output_tokens` (`null` included): live rows, and reused rows that recorded a cap (served exact under an earlier invocation's cap). Live rows without the field only warn. Reused ones without it (legacy-tier serves) are skipped only when the run has no explicit `--max-output-tokens`: an explicit cap (N or `none`) serves exact-tier rows only, so it also refuses to resume over legacy controls an earlier invocation served into the output (under `none`, a same-provider pre-cap row is exact and stays). Infra-error (`transient|access`) rows, which carry no measured output, are skipped. Resume with the original cap, a fresh `--output`, or `--force`.
- **Summaries** carry `max_output_tokens` per group: the shared value, `null` when no kept row recorded one, or `"mixed"` (e.g. capped live rows beside legacy uncapped controls). A group served only legacy controls reads `null`, same as an uncapped run. Legacy-served rows are `cache_source: "reused"` rows with no `max_output_tokens` key (`provider` may be present; an explicit `null` means uncapped): `jq -c 'select(.cache_source=="reused" and (has("max_output_tokens")|not))'`. `results-audit` counts them per label · arm · model (`legacy-served controls (reused, no recorded cap)`), mixed group or not, and keys caps `none` (uncapped) apart from `unset` (not recorded).

## LLM retries and deadlines

Agent calls (every `start` arm via `runMeteredLoop`, and `sragents-select`) run with the AI SDK's own retries off (`maxRetries: 0`; ai@6 would retry only 2× at 2s/4s). Instead `llm-retry.ts` wraps each cell's model (`withRetry`, a Proxy over `doGenerate` that keeps `specificationVersion`/`provider`/`modelId`, so V2 anthropic/bedrock and V3 openai models alike):

- **Only `transient` errors are retried** (`classifyError`: 429, 5xx, overload, network drops). Backoff is equal jitter over a capped exponential: `d/2 + rand·d/2` with `d = min(max-delay, base·2^(n−1))`. A `Retry-After`/`retry-after-ms` header raises the wait to `max(jittered, header)`; a header beyond the max delay or the cell's remaining wait budget fails fast.
- **Gated or missing models fail at once**: a fatal provider marker or an `access` error (401/403/404, "not available for this account", Bedrock's daily cap "Too many tokens per day", OpenAI's exhausted credit: a 429 with code `insufficient_quota`) becomes a `FatalProviderError` (`error_class: "access"`). Running out of attempts or wait budget is a `RetriesExhaustedError` (`Failed after N attempts[ (reason)]. Last error: …`), classified by its cause (`transient`). Every other error passes through unchanged.
- **The deadline is active time and aborts.** `--timeout-ms` (BFCL, default 60000; `sragents-select`, default 300000) runs a `PausableDeadline` that pauses during retry sleeps and aborts the request (`abortSignal`) on expiry with `run timed out after Nms`. A layer that ignores the abort is cut off by a hard backstop at deadline + grace. A timeout after a retry is `transient` (excluded from metrics), even when that retry cost no active time (retry sleeps are off the clock, so an instant 429 early in a cell that later hangs still flips it); otherwise it's `timeout`, a scored fail. Worst case per cell ≈ timeout + max wait + grace.
- **Each retry logs one line** tagged with its cell (`[scenario · arm · model · #run] retry: provider/model attempt n/max failed (status); waiting Nms …`), off under `--quiet`. Exhaustion and fatal errors show as the cell's error.

| Env var (echoed at startup, with the active deadline: `retry: …`) | Default | Meaning |
|---|---|---|
| `RATEL_LLM_RETRY_MAX_ATTEMPTS` | 8 | calls per LLM request, first try included |
| `RATEL_LLM_RETRY_BASE_MS` | 2000 | backoff base |
| `RATEL_LLM_RETRY_MAX_DELAY_MS` | 60000 | cap on one backoff (and on an honoured `Retry-After`) |
| `RATEL_LLM_RETRY_MAX_WAIT_MS` | 180000 | total backoff budget per cell |
| `RATEL_CELL_TIMEOUT_GRACE_MS` | 30000 | backstop grace after the deadline |

Each must be a positive integer, and the two that become a timer delay (`RATEL_LLM_RETRY_MAX_DELAY_MS`, `RATEL_CELL_TIMEOUT_GRACE_MS`) at most 2147483647 ms, Node's timer limit, as must `--timeout-ms` (anything else exits with an error naming the variable). Rows (BFCL and SR) record `retries`, `throttled_retries` (after a 429/503/529), `retry_wait_ms` (backoff inside `wall_ms`) and `retry_policy`, e.g. `a8/b2000/c60000/w180000;timeout=active:180000+g30000`; legacy rows lack them. Summaries add `retries` and `throttled_retries` (summed over every superseding row that recorded them, excluded ones included; `null` for an all-legacy group, a lower bound beside legacy rows), `retry_policy` (the kept rows' shared value, `"mixed"`, or `null` for legacy) and `latency_p50_net_ms` (median `wall_ms − retry_wait_ms`); REPORT.md's Headline adds `mean wall (net)` (`mean_wall_net_ms`). **Retries change a cell's completion probability, not its answer**; `latency_p50_ms`, `mean_wall_ms` and `wall_savings_pct` stay wall-clock (waits included). SR's `--timeout-ms` used to be undici's ~300s *per attempt* with no abort; it now bounds the whole call, so SR can record a `timeout` row. The BFCL LLM judge is not wrapped: it passes `maxRetries: 6` (`JUDGE_MAX_RETRIES`) to the SDK and never feeds these counters.

## Pinned `@ratel-ai/sdk` version

The benchmark consumes `@ratel-ai/sdk` from the npm registry at a version pinned in `agent/package.json` (currently `0.1.5`). The resolved version is stamped on every JSONL row as `ratel_version` and rendered in the report header. To benchmark a new ratel release: bump the pinned version, `pnpm install`, re-run. The previous-version JSONL keeps its rows — they're keyed by version, so they neither collide with nor satisfy the new run.

Edits to the upstream SDK in [`ratel-ai/ratel`](https://github.com/ratel-ai/ratel) therefore **don't** flow into the benchmark unless and until they're published. This is deliberate: we want the campaign to measure the same artifact users install, not whatever's on the working tree.

## Cached control runs

`control-baseline` and `control-oracle` cells are cached across invocations — they don't depend on the ratel code path being iterated on, so re-running them per campaign is pure waste. The cache is version-agnostic — keyed by `(scenario_id, arm, model, pool_size, run_index)` — and backed by the canonical `agent.jsonl`; a reused row is re-stamped to the current `ratel_version` with `cache_source: "reused"` (live rows carry `"live"`).

- **Non-ephemeral runs** (default `--output`): control rows already in the output at the current version are skipped via the resume path.
- **Ephemeral runs** (`--ephemeral`): the canonical `agent.jsonl` is opened read-only at start. For each scheduled cell, if its key is in the cache, the cached row is re-stamped into the ephemeral output and the live agent loop is skipped. Ratel arms (`ratel-full` / `ratel-pre-discovery` / `ratel-discovery-tool`) still run live every time — that's the point of an ephemeral iteration.
- **`--force`** disables the cache (and truncates the output file in non-ephemeral mode), so the campaign always re-pays.
- **Errored rows are never served.** Rerunnable errors (`transient|access|request`, classified by `cell-errors.ts`) are skipped when indexing the cache, so a key whose only cached rows are errors runs live. `timeout`/`outcome` rows are final, scored results and stay reusable. `--cache-source a,b,…` indexes several files (best harness tier, then earliest, across them; see below).
- **Harness tiers.** A cached row must match the model's current harness, `provider|max_output_tokens`: exact-tier rows (same SDK provider and same requested cap) are served first, then legacy rows (no recorded cap, i.e. pre-cap builds; a recorded provider must still match), never a different recorded provider or cap. A same-provider row with no recorded cap is exact for an uncapped run (pre-cap builds sent no cap either). Within a tier the earliest row wins. So a REBASELINE (`--force`) run on a cap-recording build replaces older legacy rows in what later runs are served; a pre-cap re-baseline stays legacy-tier, where earlier rows win.
- **An explicit `--max-output-tokens` (N or `none`) serves exact-tier rows only.** Legacy rows ran under an unknown (SDK-default) cap, so they can't stand in for a run that names its cap; nor does resume keep ones an earlier invocation served into the output. Paired cap comparisons still need `--force` on every side: the catalog-cap side takes legacy controls, and a repeat run is served the first run's exact rows.

A run-start stderr line summarizes the hit count, calling out legacy-tier hits: `cache: 47 control cells reused (31 legacy-tier: no recorded cap) (re-stamped to 0.4.0), 153 will run`.

The `--output` cell summary at the end now reports `<run> cells run, <cached> cached, <skipped> skipped`.

For a fast local smoke (~$0.20–$1):

```bash
pnpm -F @ratel-ai/benchmark start \
  --scenarios 50 --runs 1 \
  --arms control-baseline,control-oracle,ratel-full \
  --models claude-sonnet-4-6 \
  --pool-sizes 180 \
  --dollar-global 5 \
  --concurrency 10
```

## Local models (Ollama)

The `ollama:` model prefix routes through a local [Ollama](https://ollama.com) server's OpenAI-compatible endpoint — no API keys, $0 cost. Tool calling depends on the model's native function-calling support: Qwen / Llama families work well, Gemma is hit-or-miss.

```bash
# Make sure Ollama is running and the model is pulled (`ollama pull qwen3.5`).

pnpm -F @ratel-ai/benchmark start \
  --scenarios 50 --runs 1 \
  --arms control-baseline,ratel-full \
  --models ollama:qwen3.5,ollama:gemma4 \
  --pool-sizes 30,180 \
  --judge-model ollama:qwen3.5 \  # cost-free local judge
  --concurrency 1 \               # local Ollama is single-process
  --timeout-ms 300000             # 5 min — local models often need more than 60s
```

Flags:
- `--ollama-base-url URL` — override the default `http://localhost:11434/v1` (or set `OLLAMA_BASE_URL` in the env). Useful for remote Ollama instances.
- `--judge-model MODEL` — pick any model id (cloud or `ollama:*`) for the LLM judge. Defaults to `claude-sonnet-4-6` when `ANTHROPIC_API_KEY` is set, otherwise the LLM judge is disabled and only the programmatic verdict is recorded.

`dollar_cost` is recorded as `0` for `ollama:*` cells — `--dollar-global` therefore never trips on local-only runs. If you mix cloud + local models in one run, the cap still bounds the cloud spend. The model id keeps its `ollama:` prefix in the JSONL row and the report so local vs cloud cells stay distinguishable.

## Remote / user-hosted endpoints

To evaluate a model you host yourself (vLLM, TGI, LM Studio, or an AWS API-Gateway-fronted model) that exposes an **OpenAI-compatible** `/v1` endpoint, embed the URL in the model id as `<baseURL>#<model-name>`. For setting up the AWS API Gateway endpoint (the `qwen3-4b` example below), see [ratel-inference-gateway](https://github.com/ratel-ai/ratel-inference-gateway).

```bash
pnpm -F @ratel-ai/benchmark start \
  --scenarios 50 --runs 1 \
  --arms control-baseline,ratel-full \
  --models 'https://<your-gateway>.execute-api.<region>.amazonaws.com/prod/v1#qwen3-4b' \
  --pool-sizes 30 --no-judge \
  --concurrency 2 \
  --timeout-ms 120000     # remote/cold models often need more than 60s
```

- **Auth:** `AWS_BEDROCK_BEARER` is simply the env var the benchmark reads for the endpoint's bearer token — put it in `agent/.env` (gitignored), or override with `--model-api-key <token>`. Unauthenticated endpoints work with no token.
- **Auto-warm:** before the run, each `<url>#<model>` endpoint is warmed via `POST <baseURL>/warm` and polled until ready, so scale-to-zero gateways don't fail every early cell on the cold start. Endpoints without a `/warm` route are skipped gracefully.
- **Cost / id:** like `ollama:*`, remote cells record `dollar_cost = 0` (you pay your own infra) and keep the full `<url>#<model>` string as the row/report id.
- **Same syntax for SR-Agents** — `pnpm -F @ratel-ai/benchmark sragents-select --models '<url>#<model>' …` (see the repo-root README, Scenario 3).

Tool calling and structured output still depend on the model's native capabilities — a small remote model may log zero tool calls (BFCL) or fail strict schema generation (SR-Agents) just as a small local model would.

## Generate the report only

```bash
pnpm -F @ratel-ai/benchmark report \
  --agent agent/results/agent.jsonl \
  --retrieval results/retrieval.jsonl \
  --output results/REPORT.md
```

Auto-discovers every `*retrieval.jsonl` under `results/` if `--retrieval` is omitted.

Cells are superseded per cell first (label + cell key; the last final row wins), then final infra errors (`transient|access`) are dropped; the header line reports both counts. Other errors stay scored fails, but token/turn/$/wall means skip them (`—` when a group has no clean cell). `bfcl-summarize` / `sragents-summarize` apply the same rules and add `excluded_cells`, `errored_cells`, `truncated_cells`, `max_output_tokens` (value, `"mixed"` or `null`; see "Output caps"), and `latency_p50_net_ms`, `retries`, `throttled_retries` and `retry_policy` (see "LLM retries and deadlines") to every summary row (metrics are `null` for a group with no kept rows); `--label L` limits them to one `ratel_ai_core_version` label. A re-summary replaces a published group only when its timestamp (the group's newest raw row) is >= the published one.

## Audit raw results

```bash
pnpm -F @ratel-ai/benchmark results-audit --bench bfcl \
  --agent results/raw/bfcl/agent.jsonl \
  --cache "$(ls results/raw/bfcl/agent*.jsonl | paste -sd, -)"
```

Prints errored/truncated row counts per file × label × arm × model × error class × `ratel_version`, the control keys the old earliest-wins cache would have re-served as errors from the `--cache` file set (default: the agent file), label × arm × model groups that mix providers or output caps (`none` = uncapped, `unset` = no recorded cap), and per label × arm × model the legacy-served controls (reused rows with no recorded cap), whether or not the group is mixed. For `--bench bfcl` it also lists label cells whose rows span more than one `ratel_version` (the cell key includes it, so each version counts as its own cell; `bfcl-summarize` and `report` warn about the same) and, per label, the `ratel_version` of its infra-errored rows: the `--ratel-version` a control re-drain must pass to supersede them. It never writes unless `--drop-infra-errors --out <path>` is given, which copies the agent file to `<path>` minus the `transient|access` rows a later row of the same cell supersedes (every other non-blank line byte-for-byte). Final infra rows stay, so the cleaned file summarizes exactly like the original (same `excluded_cells`).

## Tests

```bash
pnpm -F @ratel-ai/benchmark test
```

Unit tests cover the corpus reader, shared agent helpers (sanitization, schema normalization, tool-bundle assembly), per-arm bundle-builders (one test file per agent), agent registry auto-discovery, metering math (incl. `ratel_version` stamping), both judges (programmatic intersection + LLM prompt-only fallback), pool universe + distractor expansion, runner orchestration (resume / dollar caps / cell iteration / seeded sampling / pool-size-agnostic arms / control-row caching across ephemeral runs), and report aggregations (incl. null-`pool_size` handling and the `Catalog` column). Real LLM calls are not exercised in unit tests.
