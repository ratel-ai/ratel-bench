# Ratel Benchmark — Experiment Design & Commands

**The publishable suite design is fixed.** The pools, k-values, arms, scenarios,
seed, and LLM-eval setup do not change between campaigns. A campaign freezes one
latest stable compatible SDK/core release pair at submission and runs BM25
(`sparse`), dense (`semantic`), and hybrid against that pair. Historical labels
and commands below remain available for comparison; they are not additional
versions in a full campaign.

The v1 suite contract lives in `agent/src/suite-contract.ts`, with shared JSON
fixtures in `fixtures/suite/`. `models.json` `run[]` is the ordered 16-model
Bedrock default. The request's `runOnlyModels` replaces it; `excludeModels`
then removes canonical IDs. The default model concurrency is one and the
aggregate model/API budget is $1,000. Notification recipients are private
request data and never enter the public manifest or result.

The fixed work plan has 599 BFCL and a seed-42, six-dataset stratified 600
SR-Agents scenarios (100 per dataset), one repetition, LLM pool 100, top-k 5,
and baseline/oracle/Ratel arms. Controls run once per model/scenario and are
reused across retrievers; Ratel cells run once per retriever. The 16-model
default has 95,920 distinct LLM cells and 21,582 model-free retrieval cells
(pools BFCL 30/100, SR-Agents 50/100; k 1/3/5). The manifest freezes semantic
work-unit keys over the version pair, benchmark, scenario, model, arm, method,
pool and k. A smoke scenario cap makes the manifest non-publishable.

## ⚠️ RULES — always apply (these cost real money / comparability if forgotten)
1. **`control-baseline` and `control-oracle` are reused from cache automatically — never re-run.**
   Both `start` and `sragents-select` **default** to reusing version-independent baseline/oracle
   from the canonical `agent.jsonl` in the OUTPUT file's directory — **no flag needed**. A model
   with no cached 0.2.0 controls runs them fresh. `--cache-source <path>` overrides; `--force`
   disables reuse and truncates the `--output` file (both commands) before re-running everything. So writing to `results/raw/bfcl/agent-0.4.0-sparse.jsonl` auto-reuses from
   `results/raw/bfcl/agent.jsonl` → only `ratel-full` runs live (control-baseline is the
   expensive arm ≈ 8k input tokens/cell, so this is the big saving).
   **SR-Agents caveat:** control-baseline is built from the candidates' `pool_ids` (the full
   gold-complete pool), so it is genuinely retriever-independent and shared across versions. The
   reuse key ignores pool CONTENTS, so a stale pre-`pool_ids` cache would be silently re-adopted —
   when first regenerating on the corrected pool, run baseline once with `--force` to purge it,
   then let every other run reuse the fresh cells.
   **Errored rows are never reused, never double-counted** (taxonomy: `agent/src/cell-errors.ts`).
   - *Cache:* rerunnable errors (`transient|access|request`) are skipped, so a key whose only
     cached rows are errors runs live; `timeout|outcome` rows are final, scored results and stay
     reusable (except under `--retry-errors all`, which re-queues them, so the cache stops serving
     them too). `--cache-source a,b,…` reads several files (best harness tier, then earliest, across
     them — see *Harness*; e.g. canonical + a controls backfill); `--ratel-version V` stamps re-drained control rows
     with a label's original SDK version (control arms only). V must equal the `ratel_version` of
     the rows being replaced, as `results-audit` prints it per label (e.g. `0.1.5` for label
     `0.2.0`, `0.3.0-rc.1` for label `0.3.0-rc.1`). A different or omitted V (the installed SDK
     version) doesn't supersede them: each version counts as its own cell, so the label is
     double-counted (`bfcl-summarize` / `report` warn; `results-audit` lists the split cells).
   - *Resume:* a cell whose current-version rows are all rerunnable (`--retry-errors infra`,
     the default: `transient|access|request`; `all` adds `timeout|outcome`; `none` keeps them) is
     **re-queued**, not skipped, until it has spent `--max-attempts` (3; 0 = unlimited) live
     attempts (`cache_source: "reused"` rows don't count; rows record `attempt`). This covers every
     arm, `ratel-full` included, in both `start` and `sragents-select`. It is what makes a
     re-drain work: re-running a label's controls (`--ratel-version V --cache-source
     <canonical>,<backfill>`) re-queues its errored control rows, the cache serves the good rows,
     and the appended rows supersede the errors in summaries. Re-queued/exhausted counts print as
     `resume: K re-queued (transient a, access b, request c); M exhausted` (non-zero classes
     only). The resume cap guard skips rows whose cell the run re-queues, so re-running at a fixed
     cap replaces the `request` rows a rejected cap left.
   - *Harness:* controls are reused only under the same harness (`provider|max_output_tokens`):
     exact match first, then legacy rows (no recorded cap; a recorded provider must still match),
     earliest within a tier, never a different recorded backend or cap. A same-provider row with no
     recorded cap is exact for an uncapped run (pre-cap builds sent no cap either). Any explicit
     `--max-output-tokens` (N or `none`) serves exact-tier rows only, and refuses to resume over
     legacy controls an earlier invocation served into the same output. A REBASELINE (`--force`)
     changes what is served only on a build that records caps: its exact-tier rows then beat
     older legacy rows (a pre-cap re-baseline stays legacy-tier, where earlier rows win). A
     re-drain from a controls backfill is served its rows only under the backfill's backend and
     cap (same provider, no different `--max-output-tokens`); otherwise those keys run live. The
     `cache:` line counts legacy-tier hits. Summary `max_output_tokens` can't tell legacy rows from
     uncapped ones: a group served only legacy controls reads `null`, and `"mixed"` means
     legacy/uncapped rows sit beside capped ones. Legacy-served controls are `cache_source:
     "reused"` rows with no `max_output_tokens` key (`provider` may be present: U1/U2-era rows
     record it; an explicit `null` means uncapped). Under an uncapped run a same-provider such
     row is exact-tier, not legacy; either way its effective cap is none / the SDK default.
     `results-audit` prints them per label · arm · model (`legacy-served controls`), and its
     `caps` keep `none` (uncapped) apart from `unset` (no recorded cap).
   - *Summaries / reports:* rows are superseded per cell (label + cell key; the last final row
     wins, so a re-run replaces a transient error and a duplicated row counts once). Final
     `transient|access` rows are left out of every metric and counted in `excluded_cells`;
     `request|timeout|outcome` stay scored fails (`errored_cells`); output-limit cut-offs are kept
     and scored on their verdict (`truncated_cells`; they can pass). Token/cost/latency means use
     non-errored rows only (summaries: null when a group has none; REPORT.md: `—`). A re-summary
     replaces a published group only when its timestamp is >= the published one (on a tie the last
     appended row wins; retrieval: once per pool × k). A group's timestamp is its newest raw row,
     so a group whose raw rows were rewritten after it was summarized stays stale until new
     (re-stamped) rows are appended, e.g. by a re-drain: check that the report's timestamps for
     the label match the new summary rows. `--label L` summarizes one label only.
   - *Audit first:* `results-audit --bench bfcl|sragents --agent <file> --cache <a,b,…>` counts
     errors per file/label/arm/model/class/ratel_version, lists the control keys the old cache
     re-served as errors, flags mixed providers/caps, counts legacy-served controls and (BFCL)
     lists label cells under >1 `ratel_version` plus each label's infra-errored rows'
     `ratel_version`. Read-only unless `--drop-infra-errors --out <path>`: drops only `transient|access` rows a later row of the same
     cell supersedes, so no summary changes; final infra rows stay (still `excluded_cells`).
2. **SR-Agents LLM eval is ALWAYS 600 scenarios = 100/dataset × 6** — a seeded subset of the
   full 5,400. ALWAYS pin it: `sragents-candidates --scenarios-from results/raw/sragents/candidates.jsonl`.
   Without it, all 5,400 run (9× cost + incomparable set).

## Fixed variables (constant across ALL versions)

### Retrieval evaluation — model-free retriever quality (recall / precision / MRR / nDCG / hit@k)
| Benchmark | Pool sizes | top-k |
|---|---|---|
| **BFCL** (tools) | **30, 100** | **1, 3, 5** |
| **SR-Agents** (skills) | **50, 100** | **1, 3, 5** |

### LLM evaluation — task completion (BFCL) / skill selection (SR-Agents)
| Benchmark | Pool size | top-k | Arms |
|---|---|---|---|
| **BFCL** | **100** | **5** | control-baseline, control-oracle, ratel-full |
| **SR-Agents** | **100** | **5** | control-baseline, control-oracle, ratel-full |

- **Scenarios:** BFCL = 599 (all of `bfcl-all.jsonl`). SR-Agents = **600 = 100/dataset × 6**, a
  seeded subset of the full **5,400** in `sragents.jsonl` — always pin it with
  `sragents-candidates --scenarios-from results/raw/sragents/candidates.jsonl` (else all 5,400 run).
- **Pool rule:** pool = `gold + deterministic distractors`, truncated to pool size; gold
  always present (so recall@pool = 1.0). Mirrors `expandPool` (`agent/src/pool.ts`).
- **Output caps:** every agent call requests the model's models.json `maxOutputTokens` (4096:
  haiku-4-5 / sonnet-4-6 / opus-4-5; 16384: sonnet-5 / opus-4-8 / fable-5 / gpt-5.4-mini /
  gpt-5.6-luna; self-hosted: none sent), recorded per row as `max_output_tokens`. Resume won't
  mix recorded caps: it throws when the output holds current-version rows of a run model (live,
  or reused with a recorded cap) whose `max_output_tokens` differs (`null` included); live rows
  with no recorded cap only warn, and infra-error (`transient|access`) rows are ignored. Uncapped
  legacy controls can still be served into a capped label through the legacy tier (see
  *Harness*); re-baseline with `--force` for a single-cap label. `--max-output-tokens N|none`
  (`start`, `sragents-select`) overrides the catalog for a whole run and serves exact-tier
  controls only; `--judge-max-output-tokens N` (`start`, `rejudge`) caps the BFCL judge
  (default: none). Cut-offs stay scored and final (`truncated_cells`); a truncated judge answer
  is `n/a` (`judge truncated at N`). Check the startup `caps:` line. Paired cap checks (capped vs
  `none`, plus the noise repeat) must run every side with `--force` (or a fresh output and
  `--cache-source <nonexistent path>`) so controls run live: otherwise the catalog-cap side takes
  legacy controls and the repeat is served the first run's exact rows (noise 0 by construction).
- **Retries & deadlines:** agent calls (`start` arms, `sragents-select`) run with the SDK's
  retries off (`maxRetries: 0`) under one policy (`agent/src/llm-retry.ts`): only `transient`
  errors (429/5xx/overload/network) are retried, with equal-jitter capped exponential backoff
  (`Retry-After` honoured up to the cap; beyond the cap or the remaining wait budget the call fails
  fast as `transient`); gated/missing models, Bedrock's daily token cap ("Too many tokens per
  day") and OpenAI's exhausted credit (429 `insufficient_quota`) fail at once (`access`). Each retry logs one line tagged with its cell (off under
  `--quiet`). Env knobs, echoed at startup with the active deadline (`retry: …` line):
  `RATEL_LLM_RETRY_MAX_ATTEMPTS` (8), `RATEL_LLM_RETRY_BASE_MS` (2000),
  `RATEL_LLM_RETRY_MAX_DELAY_MS` (60000), `RATEL_LLM_RETRY_MAX_WAIT_MS` (180000, per cell),
  `RATEL_CELL_TIMEOUT_GRACE_MS` (30000). `--timeout-ms` is **active time** (retry sleeps don't
  count) and actually aborts the request; a layer that ignores the abort is cut off at deadline +
  grace. A timeout after any retry is `transient` (excluded) — even when that retry cost no active
  time (an instant 429 early in a cell that later hangs) — otherwise `timeout` (a scored fail).
  **SR:** `sragents-select --timeout-ms` (default 300000) now bounds the whole call, active time,
  and aborts it — previously undici allowed ~300s *per attempt* with no abort — so SR gains a
  `timeout` outcome. Rows record `retries`, `throttled_retries` (429/503/529), `retry_wait_ms` and
  `retry_policy` (e.g. `a8/b2000/c60000/w180000;timeout=active:180000+g30000`); summaries add
  `retries`, `throttled_retries` (`null` for all-legacy groups, a lower bound beside legacy
  rows), `retry_policy` (value, `"mixed"` beside legacy rows, or `null`)
  and `latency_p50_net_ms` (median of `wall_ms − retry_wait_ms`); REPORT.md's Headline adds
  `mean wall (net)` (`mean_wall_net_ms`). Retries change a cell's completion probability, **not
  its answer**, so accuracy stays comparable across policies; `latency_p50_ms`, `mean_wall_ms` and
  `wall_savings_pct` are unchanged (wall-clock, waits included). The BFCL LLM judge is not
  wrapped: it passes `maxRetries: 6` to the SDK and never feeds the agent's counters.
- **Breaker & re-runs** (`start`, `sragents-select`): a fatal provider error (gated model, daily
  cap) writes **no row** and aborts that model (its spend still counts); so do
  `RATEL_ABORT_AFTER_CONSECUTIVE_ERRORS` (10; 0 = off) consecutive `transient|access` rows of one
  model (`error_circuit`; those rows are written, a clean row resets the streak). After the main
  pass, `--retry-rounds` (1) in-process rounds, `--retry-delay-s` (60) apart, re-run this run's
  rerunnable cells with attempts left, under the same dollar cap and output (never re-truncated),
  skipping aborted models. `retry_policy` gains the rerun policy (`;rerun=infra/3`). The run ends
  with one line, `done: N cells run, C cached, S skipped, $X spent, stopped=R, R retries (T
  throttled), E errors, Q re-queued, X exhausted` (SR prints it after its own summary), plus one
  `aborted: <model> — <fatal|error_circuit>: <detail>` line per aborted model; **any abort exits
  with code 2**. `stopped` ranks `fatal` > `error_circuit` > `global_cap` > `completed`; an abort
  that outranks a hit cap reads `stopped=fatal+global_cap` (or `error_circuit+global_cap`). `N`
  and `E` count live **rows**, retry-round attempts included: a cell a round recovered adds 2 to
  `N` and 1 to `E`; final failures are `X exhausted` plus the non-rerunnable errors. The AWS
  runner's `bench-flags.mjs` passes each flag only when the file that parses it (`cli-args.ts`,
  `sragents-select.ts`) spells it as a quoted literal (`"--retry-errors"`); a test pins this.
- **Models:** whatever LLM(s) under test — cloud (`claude-sonnet-4-6`, `gpt-5.4-mini`) or
  user-hosted via URL (`https://<your-gateway>.execute-api.<region>.amazonaws.com/prod/v1#qwen3-4b`).

## Versions & labels
| Label | Retriever |
|---|---|
| `0.2.0` | BM25 (lexical) |
| `0.3.0-rc.1` | BM25 |
| `0.4.0-sparse` | 0.4.0 BM25 (lexical) |
| `0.4.0-dense` | 0.4.0 semantic (embeddings) |
| `0.4.0-hybrid` | 0.4.0 hybrid |

The report groups layers by `ratel_ai_core_version`. For 0.4.0 the three methods are run as
three separate labels (same SDK, different `--retriever`).

## What changed in 0.4.0 — and ONLY this
1. **Three retrieval methods** (sparse / dense / hybrid) selectable via `--retriever`, chosen
   **benchmark-side** (no Ratel change).
2. **Embeddings computed at registration** for semantic/hybrid (bm25 unchanged — no embeddings).
3. **No Rust core 0.4.0** → generation is **SDK-based** (`ToolCatalog` / `SkillCatalog`) instead
   of the Rust retriever. Everything downstream (summarize/report) is unchanged.

## Setup (once)
```bash
cd /Users/bercaakbayir/Desktop/ratel-ai/ratel-bench
# user-hosted model + token (token lives in agent/.env as AWS_BEDROCK_BEARER, gitignored)
# gateway setup: https://github.com/ratel-ai/ratel-inference-gateway
M='https://<your-gateway>.execute-api.<region>.amazonaws.com/prod/v1#qwen3-4b'
# 0.4.0 SDK:
sed -i '' 's#npm:@ratel-ai/sdk@[^"]*#npm:@ratel-ai/sdk@0.4.0#' agent/package.json && pnpm install
```
`RATEL_VERSION_LABEL=<label>` stamps `ratel_ai_core_version` on every row (that's how the three
0.4.0 method-layers are labeled).

## Commands — run per method `<m>` ∈ {sparse=bm25, dense=semantic, hybrid=hybrid}

### BFCL
```bash
# --- LLM eval (retrieves live via SDK; pool 100, top 5) ---
# --cache-source reuses baseline/oracle from canonical → only ratel-full runs live (RULE 1).
RATEL_VERSION_LABEL=0.4.0-<m> pnpm -F @ratel-ai/benchmark start \
  --corpus test-data/bfcl-all.jsonl --output results/raw/bfcl/agent-0.4.0-<m>.jsonl \
  --cache-source results/raw/bfcl/agent.jsonl \
  --arms control-baseline,control-oracle,ratel-full --models "$M" \
  --retriever <method> --pool-sizes 100 --top-k 5 --runs 1 --no-judge --concurrency 4 --timeout-ms 120000

# --- Retrieval eval (pools 30,100 × k 1,3,5) ---
RATEL_VERSION_LABEL=0.4.0-<m> pnpm -F @ratel-ai/benchmark bfcl-candidates \
  --retriever <method> --pool-sizes 30,100 --output results/raw/bfcl/retrieval-0.4.0-<m>.jsonl

# --- summarize + report ---
pnpm -F @ratel-ai/benchmark bfcl-summarize --label 0.4.0-<m> \
  --retrieval-rows results/raw/bfcl/retrieval-0.4.0-<m>.jsonl --agent results/raw/bfcl/agent-0.4.0-<m>.jsonl
pnpm -F @ratel-ai/benchmark bfcl-report
```

### SR-Agents
```bash
# --- Reference prep (ONCE, before any method): the --pool-from reference must carry
#     pool_ids for BOTH pool sizes (50, 100). The SDK looks pool_ids up by `target_pool_size`
#     and SILENTLY SKIPS any size the reference lacks — a `--pool-sizes 100`-only reference is
#     why pool-50 retrieval eval was missing for 0.4.0. Same seed/scenarios ⇒ deterministic:
#     regenerating with 50,100 reproduces the identical 100-pool and just adds the 50-pool.
cargo run -p ratel-benchmark-retrieval --release -- skill-retrieval \
  --instances test-data/sragents.jsonl --skills-catalog test-data/sragents-skills.jsonl \
  --output results/raw/sragents/candidates.jsonl \
  --scenarios 600 --top-k 5 --pool-sizes 50,100 --seed 42

# --- Candidate gen (SDK): re-rank the FIXED 0.2.0 pool with the 0.4.0 retriever ---
# --pool-from reads the canonical pool from the reference's `pool_ids` field (the FULL
# gold-complete membership — NOT the lossy `retrieved` ranking). Same 50/100-pool for every
# version, so control-baseline/oracle stay identical & reusable; only ratel-full's ranking
# changes. Pins the canonical 600. REQUIRES a pool_ids-aware reference carrying BOTH 50 & 100
# (see "Reference prep" above; else pool 50 is silently dropped / generation throws "no pool_ids").
# Retrieval eval = pools 50,100 (fixed design); LLM eval below stays pool 100 only.
RATEL_VERSION_LABEL=0.4.0-<m> pnpm -F @ratel-ai/benchmark sragents-candidates \
  --retriever <method> --pool-sizes 50,100 --top-k 5 \
  --pool-from results/raw/sragents/candidates.jsonl \
  --output results/raw/sragents/candidates-0.4.0-<m>.jsonl

# --- LLM eval (pool 100, top 5) ---
# control-baseline is built from the candidates' `pool_ids` (full 100, gold-complete), so it
# is retriever-INDEPENDENT and identical across versions: run it live ONCE (add --force on the
# FIRST method to purge any pre-fix poisoned cache; the reuse key ignores pool contents), then
# --cache-source reuses baseline+oracle for the other methods → only ratel-full runs live.
# Baseline now shows the full 100 candidates (was ~9–83), so input tokens/cost rise — bump the cap.
RATEL_VERSION_LABEL=0.4.0-<m> pnpm -F @ratel-ai/benchmark sragents-select \
  --candidates results/raw/sragents/candidates-0.4.0-<m>.jsonl \
  --cache-source results/raw/sragents/agent.jsonl \
  --output results/raw/sragents/agent-0.4.0.jsonl \
  --arms control-baseline,control-oracle,ratel-full --models "$M" \
  --pool-size 100 --top-k 5 --concurrency 8 --dollar-global 60   # first method: add --force

# --- summarize (retrieval eval from --retrieval-rows + task completion from --agent) + report ---
pnpm -F @ratel-ai/benchmark sragents-summarize --label 0.4.0-<m> \
  --retrieval-rows results/raw/sragents/candidates-0.4.0-<m>.jsonl \
  --agent results/raw/sragents/agent-0.4.0-<m>.jsonl
pnpm -F @ratel-ai/benchmark sragents-report
```

## Pre-0.4.0 (0.2.0 / 0.3.0-rc.1) — unchanged, Rust retriever
- Retrieval eval + candidates: Rust `cargo run -p ratel-benchmark-retrieval …` (BM25).
- LLM eval: BFCL `start` (live SDK), SR-Agents `sragents-select` (Rust candidates).
- Version pinning: `pnpm version-set --crate <v> --expect <v>` … `pnpm version-reset`; for BFCL
  also pin the matching `@ratel-ai/sdk` in `agent/package.json` (e.g. 0.2.0 ⇄ SDK 0.1.5).

## Notes
- `$0` cost for local/user-hosted models; cap only bounds cloud spend.
- semantic/hybrid embed ~pool-size vectors per scenario (fast at pool ≤100); bm25 has no embeddings.
- Separate `agent-<label>.jsonl` per method avoids resume-cache collisions; `cat … >> agent.jsonl` to merge for the report.
