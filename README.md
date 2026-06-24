<div align="center">
  <h1>ratel-bench</h1>
  <p>Every number in the Ratel docs comes from here.</p>

  <p>
    <a href="https://benchmark.ratel.sh">Results</a> •
    <a href="https://github.com/ratel-ai/ratel">Ratel core</a> •
    <a href="https://discord.gg/hdKpx69NR">Discord</a>
  </p>

  <p>
    <a href="https://github.com/ratel-ai/ratel-bench/stargazers"><img src="https://img.shields.io/github/stars/ratel-ai/ratel-bench?style=social" alt="GitHub stars" /></a>
    <a href="./LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="license" /></a>
  </p>
</div>

<div align="center">
  <img src="./docs/assets/hero.webp" width="960" alt="Ratel benchmark" />
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

Retrieval evals (does BM25 rank correctly?) are deterministic and free. Agent campaign evals (does the full agent do better with Ratel?) require at least one of `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

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
