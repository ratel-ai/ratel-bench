<div align="center">
  <h1>ratel-bench</h1>
  <p>Every number in the Ratel docs comes from here.</p>

  <p>
    <a href="https://benchmark.ratel.sh">Results</a> •
    <a href="https://github.com/ratel-ai/ratel">Ratel core</a> •
    <a href="https://discord.gg/75vAPdjYqT">Discord</a>
  </p>

  <p>
    <a href="https://github.com/ratel-ai/ratel-bench/stargazers"><img src="https://img.shields.io/github/stars/ratel-ai/ratel-bench?style=social" alt="GitHub stars" /></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="license" /></a>
    <a href="https://discord.gg/75vAPdjYqT"><img src="https://img.shields.io/discord/1478702964003705015?logo=discord&logoColor=white&label=Discord&color=5865F2" alt="Discord" /></a>
  </p>
</div>

## Introduction

The benchmark compares agent performance with and without Ratel across real tasks, using three arms run in parallel:

| Arm | What the model sees |
|---|---|
| **Baseline** | Every tool in the catalog — no Ratel |
| **Ratel** | `search_capabilities` + `invoke_tool` — Ratel's gateway |
| **Oracle** | Only the correct tools — the theoretical ceiling |

Each run measures accuracy, input tokens, output tokens, cost, and latency. The gap between Baseline and Ratel is the claim. The gap between Ratel and Oracle is what's left to close.

Full results: [benchmark.ratel.sh](https://benchmark.ratel.sh)

## Corpora

Three datasets the arms run against:

| Corpus | What it is | License |
|---|---|---|
| **MetaTool** | 199 real tool definitions, ~21k user task queries | MIT |
| **ToolRet** | 44k-tool public retrieval corpus, 35 sub-corpora | Apache 2.0 |
| **BFCL** | Berkeley Function-Calling Leaderboard — measures right function + right arguments, not just tool selection | Apache 2.0 |

Retrieval evals (does BM25 rank correctly?) are deterministic and free. Agent campaign evals (does the full agent do better with Ratel?) require at least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY` — or a user-hosted OpenAI-compatible endpoint, which needs no cloud key (see Scenario 3/4).

## Setup

**Prerequisites:** Rust stable, Node 24+, pnpm 10+.

```bash
pnpm install
```

API keys for the agent campaign (mode c only — place in `.env` at the repo root):

```
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

To benchmark a user-hosted (`<url>#<model>`) endpoint instead, no cloud key is needed; put the endpoint's bearer token in `agent/.env` (gitignored) as `AWS_BEDROCK_BEARER=...` — or pass `--model-api-key`. Unauthenticated endpoints need no token. See Scenario 3/4.

## Run

**Everything** — ingests corpora, runs all eval modes, renders a report:

```bash
pnpm -F @ratel-ai/benchmark run-all
```

**Retrieval only** (free, no API key, fast):

```bash
pnpm -F @ratel-ai/benchmark run-all --skip-agent
```

**Agent campaign only** (needs a key, resumable):

```bash
pnpm -F @ratel-ai/benchmark start
```

Output lands in `results/REPORT.md`. Re-runs skip already-recorded cells unless `--force` is passed. Full flag reference: [`agent/README.md`](agent/README.md) and [`retrieval/README.md`](retrieval/README.md).

## Benchmarking a specific Ratel version

Every retrieval row and agent cell is **stamped with the Ratel version it was produced under**, so reports compare versions side by side. **BFCL benchmarks _tools_** (function definitions — right function + right arguments); **SR-Agents benchmarks _skills_** (authored skills from the ~26k-skill catalog).

There are **two version knobs**, and which one matters depends on the eval:

- **`ratel-ai-core` (Rust crate)** — governs every **retrieval** eval and the **SR-Agents LLM** eval (whose candidates are produced offline by the Rust retriever). Swap it with the `version-set` / `version-reset` bookends below.
- **`@ratel-ai/sdk` (npm package)** — governs the **BFCL LLM** eval *only*, which retrieves **live** through the SDK's `search_tools` gateway. `version-set` does **not** change it; you bump it in `agent/package.json` (see Scenario 4).

> **Pre-0.4.0 versions all run the identical experiment above — the commands don't change between them.** 0.4.0 is the first version that lets you **choose the retriever**, so its runs differ only slightly: pin the 0.4.0 SDK, tag the run with a method-suffixed label (`RATEL_VERSION_LABEL=0.4.0-sparse|dense|hybrid`), and add `--retriever bm25|semantic|hybrid`. Everything else — pools, top-k, arms, scenario counts — is unchanged. `control-baseline` and `control-oracle` are retriever-independent, so they're **reused from the canonical 0.2.0 cache** rather than re-run — but purge any stale/pre-fix cached cells (`--force` on the *first* method) so a gold-incomplete or poisoned 0.2.0 pool can't skew the 0.4.0 numbers. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the exact per-method commands.

### The version bookends

```bash
pnpm version-set --tag v0.3.0-rc.1 --expect 0.3.0-rc.1           # pin a version (pre-release / RC)
pnpm version-set --crate 0.3.0 --expect 0.3.0                    # pin a published crates.io release
pnpm version-set --rev <commit-sha> --expect 0.3.0               # pin an exact commit
pnpm version-reset                                               # restore the committed baseline
```

- `version-set` snapshots `retrieval/Cargo.toml` + `Cargo.lock`, swaps the `ratel-ai-core` dependency to the requested source, then **asserts** the resolved version equals `--expect` — aborting and restoring if not, so a force-moved tag can't silently benchmark the wrong build. It refuses to run twice without a reset.
- `version-reset` restores the snapshot and deletes it. **Always run it when done** so the tree builds against the released version again.
- `--expect` is the version string Cargo resolves (the git tag `v0.3.0-rc.1` resolves to crate version `0.3.0-rc.1`).

Each scenario below is self-contained: **reset → set → run → summarize → report → compare → reset**. The `v0.3.0-rc.1` pin is an example — substitute the version you're testing.

---

### Scenario 1 — Retrieval eval · SR-Agents

Recall / MRR of the skill retriever over the ~26k-skill catalog. Free, no API key.

```bash
# 1. pin the version under test
pnpm version-reset
pnpm version-set --tag v0.3.0-rc.1 --expect 0.3.0-rc.1

# 2a. run — normal
cargo run -p ratel-benchmark-retrieval --release -- skill-retrieval \
  --instances test-data/sragents.jsonl \
  --skills-catalog test-data/sragents-skills.jsonl \
  --output results/raw/sragents/retrieval-rows.jsonl \
  --scenarios 600 --top-k 1,3,5 --pool-sizes 100 --seed 42

# 2b. run — parallelized (~2.4× faster, byte-identical output)
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 caffeinate -is \
cargo run -p ratel-benchmark-retrieval --release -- skill-retrieval \
  --instances test-data/sragents.jsonl \
  --skills-catalog test-data/sragents-skills.jsonl \
  --output results/raw/sragents/retrieval-rows.jsonl \
  --scenarios 600 --top-k 1,3,5 --pool-sizes 100 --seed 42 --jobs 8

# 3. summarize → report → compare
pnpm -F @ratel-ai/benchmark sragents-summarize
pnpm -F @ratel-ai/benchmark sragents-report
pnpm retrieval-compare

# 4. restore baseline
pnpm version-reset
```

**Detail:**
- `skill-retrieval` — Rust subcommand for the authored-skill corpus. `--instances` = the queries (carry gold skill ids); `--skills-catalog` = the ~26k-skill index.
- `--output …/retrieval-rows.jsonl` — the **quality file**, overwritten each run; this is what SR-Agents summarize/report read.
- `--scenarios 600` — seeded, dataset-stratified sample (omit for the full set). `--top-k 1,3,5` — recall/MRR cutoffs. `--pool-sizes 100` — distractor-pool per scenario. `--seed 42` — fixes sampling + distractor shuffle.
- **Normal vs parallelized:** the runner already spreads scenarios across all cores (`--jobs` defaults to your core count), but candle's internal BLAS/OpenMP threads contend for the same cores. `RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1` pins those to one thread so scenario-level parallelism wins — ~2.4×. Output is **byte-identical** to serial (each scenario is independently seeded). `--jobs N` sets the worker count; `--jobs 1` forces fully serial. `caffeinate -is` keeps macOS awake for long runs.
- `sragents-summarize` folds rows into per-bucket metrics and **appends** to the summary history (version read from the rows). `sragents-report` rebuilds `results/reports/sragents/report.json` (one entry per version, latest-timestamp-per-group).
- `retrieval-compare` writes `results/reports/retrieval-comparison.md`, tabling **every version** in the reports at `--pool 100 --k 3` (override with `--pool` / `--k`).

---

### Scenario 2 — Retrieval eval · BFCL

Same as Scenario 1 over the BFCL function corpus, via the `retrieval` subcommand. Free.

```bash
pnpm version-reset
pnpm version-set --tag v0.3.0-rc.1 --expect 0.3.0-rc.1

# run — normal
cargo run -p ratel-benchmark-retrieval --release -- retrieval \
  --corpus test-data/bfcl-all.jsonl \
  --output results/raw/bfcl/retrieval-rows.jsonl \
  --scenarios 600 --top-k 1,3,5 --pool-sizes 100 --seed 42

# run — parallelized
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 caffeinate -is \
cargo run -p ratel-benchmark-retrieval --release -- retrieval \
  --corpus test-data/bfcl-all.jsonl \
  --output results/raw/bfcl/retrieval-rows.jsonl \
  --scenarios 600 --top-k 1,3,5 --pool-sizes 100 --seed 42 --jobs 8

pnpm -F @ratel-ai/benchmark bfcl-summarize
pnpm -F @ratel-ai/benchmark bfcl-report
pnpm retrieval-compare

pnpm version-reset
```

**Detail:**
- `retrieval` (not `skill-retrieval`) — the tool-retrieval subcommand. `--corpus test-data/bfcl-all.jsonl` is the combined 599-scenario corpus (399 simple + 200 multiple); the report splits them back via the scenario-id prefix.
- Output path, parallelization, and seed semantics are identical to Scenario 1.
- `bfcl-summarize` / `bfcl-report` are the BFCL equivalents; the report keys both `simple` and `multiple` buckets per version. `retrieval-compare` reads both benchmark reports, so BFCL columns appear next to SR-Agents.

---

### Scenario 3 — LLM eval · SR-Agents

Task-completion with vs without Ratel. The LLM is shown skills from an **offline candidates file**, so this needs a *second* retrieval pass (different file, different slices) before the LLM step. **Needs an API key.**

```bash
pnpm version-reset
pnpm version-set --tag v0.3.0-rc.1 --expect 0.3.0-rc.1

# 1. candidate-gen retrieval (pool 50, k 10,50 — the slices the LLM arms require)
#    normal:
cargo run -p ratel-benchmark-retrieval --release -- skill-retrieval \
  --instances test-data/sragents.jsonl \
  --skills-catalog test-data/sragents-skills.jsonl \
  --output results/raw/sragents/candidates.jsonl \
  --scenarios 600 --top-k 10,50 --pool-sizes 50 --seed 42
#    parallelized:
RAYON_NUM_THREADS=1 OMP_NUM_THREADS=1 caffeinate -is \
cargo run -p ratel-benchmark-retrieval --release -- skill-retrieval \
  --instances test-data/sragents.jsonl \
  --skills-catalog test-data/sragents-skills.jsonl \
  --output results/raw/sragents/candidates.jsonl \
  --scenarios 600 --top-k 10,50 --pool-sizes 50 --seed 42 --jobs 8

# 2. LLM selection — run WHILE STILL PINNED so cells stamp the same version
pnpm -F @ratel-ai/benchmark sragents-select \
  --candidates results/raw/sragents/candidates.jsonl \
  --pool-size 50 --top-k 10 \
  --models claude-haiku-4-5,gpt-5.4-mini \
  --scenarios 600 --concurrency 8 --dollar-global 5

# 3. summarize → report
pnpm -F @ratel-ai/benchmark sragents-summarize
pnpm -F @ratel-ai/benchmark sragents-report

pnpm version-reset
```

**Detail:**
- **The candidate-gen run is separate from Scenario 1.** It writes a *different* file (`candidates.jsonl`, not `retrieval-rows.jsonl`) with *different* slices (`--top-k 10,50 --pool-sizes 50`). The LLM arms need both the `k=10` slice (Ratel's shortlist → `ratel-full`) and the `k=50` slice (full pool → `control-baseline`). A quality-run file (`--top-k 1,3,5`) makes the LLM step **skip every scenario**.
- "Parallelized" here is the same env-pin trick on the candidate-gen retrieval; the LLM step's own parallelism is `--concurrency`.
- `sragents-select` — the LLM A/B. Reads `--candidates`, runs three arms (`control-baseline`, `ratel-full`, `control-oracle`) per scenario, writes `results/raw/sragents/agent.jsonl` (**overwritten** — copy aside to keep prior cells). **Version is stamped from the live `Cargo.lock`**, so keep the same pin as the candidate-gen run or the label won't match the retrieval it used.
  - `--models` — comma-separated; `claude-*`, `gpt-*`, `ollama:<tag>`, or a user-hosted `<baseURL>#<model>` URL (see below). Use a *different* list than Scenario 4 for per-benchmark models.
  - `--pool-size 50 --top-k 10` — **must** match the candidate-gen slices.
  - **User-hosted model** (OpenAI-compatible endpoint you run — vLLM/TGI/LM Studio, or a self-hosted model fronted by AWS API Gateway): pass the URL as the model id, e.g. `--models 'https://<your-gateway>.execute-api.<region>.amazonaws.com/prod/v1#qwen3-4b'`. Put the endpoint's bearer token in `agent/.env` as `AWS_BEDROCK_BEARER` (or pass `--model-api-key`); unauthenticated endpoints need no token. The endpoint is auto-warmed (`POST /warm`, polled until ready) before the run. Cells record `$0` cost and keep the full URL as the model id. Setting up the gateway: see [ratel-inference-gateway](https://github.com/ratel-ai/ratel-inference-gateway).
  - `--concurrency 8` — parallel LLM calls. `--dollar-global 5` — hard USD cost cap for the run. `--scenarios 600` — cap; `--runs 1` repeats per cell.
- summarize/report fold the cells into the task-completion section of `report.json`; the task-completion numbers live in `report.json` / the website.

> **Cost:** the model is called 3× per scenario (3 arms): `600 × 3 × N_models` cells. The `control-baseline` and `control-oracle` arms are version-independent, so re-running them per version is redundant spend.

---

### Scenario 4 — LLM eval · BFCL

> **Different version knob.** BFCL's agent retrieves **live through `@ratel-ai/sdk`** (npm), *not* the Rust core. `version-set` only sets the `ratel_ai_core_version` **label** on the cells — to actually measure a version's retrieval you must bump the **SDK**. Do both so the label and the retriever agree. **Needs an API key.**

```bash
# 1. pin the core (sets the ratel_ai_core_version label)
pnpm version-reset
pnpm version-set --crate 0.3.0 --expect 0.3.0

# 2. bump the SDK to the matching release so retrieval is ACTUALLY 0.3.0
#    edit agent/package.json:  "@ratel-ai/sdk": "npm:@ratel-ai/sdk@<0.3.0-sdk-version>"
pnpm install

# 3. agent campaign — normal
pnpm -F @ratel-ai/benchmark start \
  --corpus test-data/bfcl-all.jsonl \
  --output results/raw/bfcl/agent.jsonl \
  --arms control-baseline,control-oracle,ratel-full \
  --models claude-sonnet-4-6 \
  --pool-sizes 100 --runs 1 --no-judge --concurrency 4

# 3b. agent campaign — parallelized (raise in-flight API calls)
pnpm -F @ratel-ai/benchmark start \
  --corpus test-data/bfcl-all.jsonl \
  --output results/raw/bfcl/agent.jsonl \
  --arms control-baseline,control-oracle,ratel-full \
  --models claude-sonnet-4-6 \
  --pool-sizes 100 --runs 1 --no-judge --concurrency 12

# 4. summarize → report
pnpm -F @ratel-ai/benchmark bfcl-summarize
pnpm -F @ratel-ai/benchmark bfcl-report

# 5. restore baseline (and revert the package.json SDK bump if not shipping it)
pnpm version-reset
```

**Detail:**
- `start` (`agent/src/cli.ts`) — the BFCL agent runner. The `ratel-full` arm builds a live `ToolCatalog` from `@ratel-ai/sdk` and calls `catalog.search()` + the `search_tools` / `invoke_tool` gateway during the agent loop, so **the SDK version is the retriever** (recorded as `ratel_version`).
- `--corpus test-data/bfcl-all.jsonl` — operates on the corpus directly (no candidates file). `--arms` — the three arms. `--pool-sizes 100` — catalog size per scenario. `--no-judge` — skip the LLM judge (use AST / programmatic verdicts). `--runs 1` — one run per cell.
- **Normal vs parallelized:** the BFCL agent doesn't use the Rust thread-pool; it's network-latency-bound, so parallelism is just `--concurrency` (raise it to overlap more in-flight API calls — mind provider rate limits).
- `--models` — comma-separated; `claude-*`, `gpt-*`, `ollama:<tag>`, or a user-hosted `<baseURL>#<model>` URL. Set a *different* list than Scenario 3 for per-benchmark models.
- **User-hosted model** (OpenAI-compatible endpoint you run — vLLM/TGI/LM Studio, or a self-hosted model fronted by AWS API Gateway): pass the URL as the model id, e.g. `--models 'https://<your-gateway>.execute-api.<region>.amazonaws.com/prod/v1#qwen3-4b'`. Bearer token → `agent/.env` as `AWS_BEDROCK_BEARER` (or `--model-api-key`), unauthenticated endpoints need none; the endpoint is auto-warmed before the run; cells record `$0` and keep the full URL as the model id. Use a longer `--timeout-ms` (e.g. 120000) for cold/remote models. Setting up the gateway: see [ratel-inference-gateway](https://github.com/ratel-ai/ratel-inference-gateway).
- summarize/report fold the cells into `report.json`. Because both `ratel_version` (SDK) and `ratel_ai_core_version` (core, from the lock) are recorded, the report can refuse to merge layers whose versions disagree — which is why steps 1 and 2 must target the same 0.3.0. `agent.jsonl` is **overwritten**; copy it aside to keep prior cells.

---

## Repo layout

```
retrieval/    # Rust crate — BM25 retrieval eval (modes a, b, d)
agent/        # TypeScript — MetaTool agent campaign + report (mode c)
fixtures/     # Raw upstream downloads (gitignored)
test-data/    # Normalized JSONL from ingest (gitignored)
results/      # Outputs + REPORT.md (gitignored)
docs/         # ADRs
```

## The Ratel project

| | Repo | What it is |
|---|---|---|
| **Library** | [ratel-ai/ratel](https://github.com/ratel-ai/ratel) | The engine. Rust core + TS SDK + Python SDK. Embed it in your agent. |
| **Gateway** | [ratel-ai/ratel-mcp](https://github.com/ratel-ai/ratel-mcp) | MCP proxy for Claude Code, Cursor, and ChatGPT. No code changes needed. |
| **Proof** | [ratel-ai/ratel-bench](https://github.com/ratel-ai/ratel-bench) (this one) | The benchmark harness. Full results at [benchmark.ratel.sh](https://benchmark.ratel.sh). |

## License

MIT — see [LICENSE](LICENSE).
