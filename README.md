# ratel-bench

Two-layer benchmark harness for [**Ratel**](https://github.com/ratel-ai/ratel) — measures retrieval quality and agent-loop token savings. Backs every "Ratel does X better" claim. New product features that touch retrieval or context shape are expected to add a corresponding scenario before being declared done.

This repo is intentionally decoupled from the Ratel monorepo: the agent campaign pins `@ratel-ai/sdk@0.2.0` from npm and the retrieval crate pins `ratel-ai-core@0.1.5` from crates.io, so the harness measures the same artifacts users install — not whatever's on the working tree.

**Latest results: [`RESULTS.md`](RESULTS.md)** — narrative breakdown across Claude (Sonnet, Opus), `glm-5.1:cloud`, and local `qwen3.5`.

## Where Ratel is most valuable today

| your situation | Ratel's value today |
|---|---|
| Local model + large catalog | **Critical.** qwen3.5 at pool=100 goes from 8% → 77% (-57% input tokens, -36% wall time). |
| Open-source cloud + large catalog | **Strong win.** glm-5.1 at pool=180: **+1.7 pp** accuracy and **-85%** input tokens; pool-invariant accuracy as the catalog grows. |
| Frontier (Sonnet) + large catalog | **Cost-driven win.** Sonnet 4.6 at pool=180: **-82%** input tokens, **-68%** $; -8 pp accuracy. |
| Frontier (Opus) + large catalog | **Competitive win.** Opus 4.6 pool=180: **+8 pp** accuracy and **-72%** tokens (discovery-tool arm). Opus 4.7 pool=180: ≈parity (-1.7 pp) with **-81%** tokens — Anthropic's own tool-search-tool loses **-8 pp** on the same setup. |
| Any model + tiny catalog (≤30) | Skip Ratel — pool fits in the prompt cleanly. |

Full per-pool breakdown and methodology in [`RESULTS.md`](RESULTS.md).

Locked decisions live in:

- [`docs/adr/0005-benchmark-design.md`](docs/adr/0005-benchmark-design.md) — overall harness (arms, models, variance, results storage)
- [`docs/adr/0006-benchmark-corpus-and-eval-modes.md`](docs/adr/0006-benchmark-corpus-and-eval-modes.md) — corpus pivot + the three eval modes
- [`docs/adr/0007-benchmark-corpus-not-snapshotted.md`](docs/adr/0007-benchmark-corpus-not-snapshotted.md) — corpus is ingested locally; no committed snapshot, no MetaTool sampling
- [`docs/adr/0008-skill-retrieval-eval-mode.md`](docs/adr/0008-skill-retrieval-eval-mode.md) — skill retrieval evaluated separately on an authored skill corpus (SR-Agents) via `SkillRegistry`; per-dataset + aggregate summary

## Layout

```
retrieval/    Rust crate (ratel-benchmark-retrieval) — ingest + BM25 metrics
agent/        TypeScript pnpm package (@ratel-ai/benchmark) — MetaTool agent campaign + report
bfcl-loaded/  TypeScript + Python pnpm package (@ratel-ai/bfcl-loaded) — BFCL v4 multi-turn agent campaign
fixtures/     raw upstream downloads (gitignored)
test-data/    normalized JSONL produced by `ingest` (gitignored — ADR-0007)
results/      retrieval / agent JSONL outputs + REPORT.md (gitignored)
```

Both `retrieval/` and `agent/` have their own README with the details of how to run their layer.

## Eval modes

Per [ADR-0006](docs/adr/0006-benchmark-corpus-and-eval-modes.md), three eval modes split across two suites.

**Retrieval-only** — fast, deterministic, $0, no API keys. Backs claims about ranking quality. Lives in [`retrieval/`](retrieval/).

- **(a) MetaTool — pre-fetch retrieval (replace path).** Measures whether BM25 surfaces the right tool given a real user-task query, before the agent's turn. 199 OpenAI plugin descriptions + ~21k user queries (MIT). Single-tool and multi-tool queries are both scored as **tool retrieval** via the real `ToolRegistry`; the summary splits into `single-tool · tool` and `multi-tool · tool` (single-tool recall is binary, multi-tool recall is fractional, so they get separate panels).
- **(b) ToolRet — IR / autonomous-discovery retrieval (gateway path).** Measures whether the index ranks correctly when the agent emits an IR-shaped query mid-loop (e.g. `searchTools("a tool that converts currency")`). 7,961 retrieval tasks across 35 sub-corpora over a 44,453-tool catalog (Apache-2.0).
- **(d) SR-Agents — skill retrieval.** Measures whether BM25 surfaces the right authored **skill** document (not a tool) for a task. ~26k skills (`name` + `description` indexed; markdown `body` carried but not indexed) as the catalog/distractor universe, with ~5.4k instances across six datasets (`bigcodebench`, `champ`, `logicbench`, `medcalcbench`, `theoremqa`, `toolqa`). Scored via the real `SkillRegistry`. Fully separate from tool retrieval — its own corpus, ingester, run path, and output. Multi-mapping datasets (e.g. CHAMP) have several gold skills per instance, so recall@K is fractional and `complete@K` is the all-or-nothing bar. See [ADR-0008](docs/adr/0008-skill-retrieval-eval-mode.md).

**Agentic** — end-to-end agent runs with token cost + correctness signals. Requires API keys. Lives in [`agent/`](agent/).

- **(c) MetaTool tasks + LLM-as-judge.** Runs control (baseline + oracle) arms alongside three Ratel arms (full / pre-discovery only / discovery-tool only) on MetaTool user-task queries with stubbed tool responses. The full Ratel arm pre-discovers BM25 top-K from the prompt *and* exposes the gateway (`search_tools` / `invoke_tool`) so the model can recover when pre-discovery missed; the two ablations isolate which layer is doing the work. An optional local-only `claude-sdk-tool-search` arm can be wired up alongside as a competitive baseline against Anthropic's native tool-search-tool. Programmatic judge does selection-only intersection (`effective_tool_ids ∩ gold_tools ≠ ∅`); the LLM judge scores final-text coherence against the user prompt as a fallback / tiebreaker. Reports input/output tokens, cache hit rate, $-cost, and wall-clock time at realistic catalog sizes (default pool size 180), averaged per-scenario across runs and then across scenarios.

## Setup

### Prerequisites

- **Node.js 20+** and **pnpm 10+** (the repo pins `pnpm@10.28.2` via `packageManager`). Drives the agent campaign and the report renderer.
- **Rust stable** (1.85+ — the retrieval crate uses edition 2024). Drives ingest and retrieval-only modes.
- **API keys** (mode c only — retrieval-only is $0 and key-free):
  - `OPENAI_API_KEY` — required to score `gpt-5.4-mini`.
  - `ANTHROPIC_API_KEY` — required to score `claude-sonnet-4-6` / `claude-opus-4-7` **and** to power the default LLM judge.

  Set at least one. The harness skips models with no key rather than failing. Place them in `.env` at the repo root — `dotenv` is loaded automatically.

### Install

```bash
pnpm install                  # JS deps for agent/ + bfcl-loaded/ + mcpverse/
cargo build -p ratel-benchmark-retrieval --release   # optional — first `cargo run --release` compiles otherwise
```

`pnpm install` is the only required step for the agent layer. The retrieval crate's deps are pulled lazily on first `cargo run`, so the explicit `cargo build` above is just a warm-up.

## Ingest datasets

Per [ADR-0007](docs/adr/0007-benchmark-corpus-not-snapshotted.md), the corpora are **not** committed — neither the raw upstream downloads (`fixtures/`) nor the normalized JSONL (`test-data/`). You have to ingest once before any benchmark can run.

```bash
# MetaTool — feeds retrieval mode (a) and the agent campaign mode (c).
cargo run -p ratel-benchmark-retrieval --release -- ingest metatool --download

# ToolRet — feeds retrieval mode (b). Skip if you only care about the agent campaign.
cargo run -p ratel-benchmark-retrieval --release -- ingest toolret --download

# SR-Agents — feeds skill retrieval mode (d). Writes a skill catalog + instances.
cargo run -p ratel-benchmark-retrieval --release -- ingest sragents --download
```

`--download` pulls upstream sources (MetaTool: MIT, ToolRet: Apache-2.0, SR-Agents) into `fixtures/` via `curl` (and unzips the SR-Agents corpus), then writes normalized JSONL to `test-data/metatool.jsonl`, `test-data/toolret.jsonl`, and `test-data/sragents-skills.jsonl` + `test-data/sragents.jsonl`. Re-running without `--download` against the cached fixtures produces a byte-identical JSONL. Full ingest tunables in [`retrieval/README.md`](retrieval/README.md).

Or let `run-all` handle ingest for you — it no-ops when the snapshots already exist.

## Run the whole benchmark

```bash
pnpm -F @ratel-ai/benchmark run-all
```

This single command:

1. Ingests MetaTool, ToolRet, and SR-Agents (downloads upstream sources via `curl`) if their normalized JSONL isn't already present under `test-data/`.
2. Runs BM25 tool retrieval over each tool corpus (modes a + b) and skill retrieval over SR-Agents (mode d), at corpus-appropriate pool sizes.
3. Runs the mode-(c) MetaTool agent campaign with conservative defaults if `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` is set; otherwise prints a notice and skips. Defaults: 50 sampled scenarios × 1 run × every committed arm × available models, $5 global cap.
4. Renders `results/REPORT.md` from the retrieval and (if present) agent JSONL outputs.

Re-running `run-all` skips ingest when the snapshot already exists; pass `--force` to re-ingest, `--skip-ingest` to fail loudly if missing, `--skip-agent` to opt out of mode (c) even when keys are set, or `--only metatool|toolret|sragents` to restrict to one corpus. For the headline N=5 variance run, invoke `pnpm -F @ratel-ai/benchmark start` directly — see [`agent/README.md`](agent/README.md).

## Run only the agent benchmark (mode c)

Use this when you're iterating on agent behavior and don't want to repay retrieval each loop. Prerequisites: `pnpm install` done, MetaTool ingested (`test-data/metatool.jsonl` exists), and at least one of `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` exported.

Fast local smoke (~$0.20–$1, ~50 scenarios × 1 run × 3 arms × 1 model):

```bash
pnpm -F @ratel-ai/benchmark start \
  --scenarios 50 --runs 1 \
  --arms control-baseline,control-oracle,ratel-full \
  --models claude-sonnet-4-6 \
  --pool-sizes 180 \
  --dollar-global 5 \
  --concurrency 10
```

Output: one JSONL row per `(scenario, arm, model, pool_size, run)` cell in `agent/results/agent.jsonl` (resumable — re-runs skip already-recorded cells unless `--force`). Render a report with `pnpm -F @ratel-ai/benchmark report`.

Full flag reference, the N=5 variance recipe, the local-Ollama path, and the cached-control-runs / `--ephemeral` workflow live in [`agent/README.md`](agent/README.md).

## Run only retrieval (modes a + b + d)

Fast, deterministic, $0, no API keys. Once the corpora are ingested:

```bash
# Tool retrieval (modes a / b) — one corpus JSONL.
cargo run -p ratel-benchmark-retrieval --release -- retrieval \
  --corpus test-data/metatool.jsonl \
  --output results/metatool-retrieval.jsonl \
  --top-k 1,3,5,10 --pool-sizes 30,100,180

# Skill retrieval (mode d) — separate subcommand, catalog + instances.
cargo run -p ratel-benchmark-retrieval --release -- skill-retrieval \
  --instances test-data/sragents.jsonl \
  --skills-catalog test-data/sragents-skills.jsonl \
  --output results/sragents-skill-retrieval.jsonl \
  --top-k 1,3,5,10 --pool-sizes 100,1000,26262
```

ToolRet uses the same `retrieval` runner with a different corpus and pool-size sweep. Skill retrieval uses the dedicated `skill-retrieval` subcommand (the skill catalog is the BM25 index; instances carry gold skill ids). Full reference in [`retrieval/README.md`](retrieval/README.md).

## Corpus format

All suites consume the same JSONL — one `Scenario` per line:

```jsonc
{
  "id": "fs-001",
  "prompt": "Show me /etc/hosts.",
  "candidate_pool": [ /* tools available in this scenario */ ],
  "gold_tools": ["fs.read_file"],
  "judge_criteria": "mentions localhost",
  "category": "filesystem"
}
```

The Rust definition in [`retrieval/src/corpus.rs`](retrieval/src/corpus.rs) is canonical. The TS mirror in [`agent/src/types.ts`](agent/src/types.ts) tracks it.
