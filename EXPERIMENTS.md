# Ratel Benchmark — Experiment Design & Commands

**This is fixed. It does not change between versions.** Only the retriever (the Ratel
version) changes; the pools, k-values, arms, scenarios, and LLM-eval setup are constant.

## ⚠️ RULES — always apply (these cost real money / comparability if forgotten)
1. **`control-baseline` and `control-oracle` are ALWAYS reused from cache — never re-run.**
   They're retrieval-independent, so re-running them wastes money (control-baseline = full
   100-tool pool ≈ 8k input tokens per cell).
   - **BFCL:** separate per-method files + `--cache-source results/raw/bfcl/agent.jsonl` on the
     `start` command → only `ratel-full` runs live; baseline/oracle pulled from 0.2.0/0.3.0
     data (`$0`), re-stamped. (Needed because BFCL `ratel-full` collides if methods share a file.)
   - **SR-Agents:** write all 3 methods to ONE shared file `agent-0.4.0.jsonl` (NO flag) —
     `sragents-select` auto-reuses controls from that file, and its `ratel-full` doesn't collide.
     Run **sparse first** (controls computed once for the 0.4.0 pool), then dense/hybrid reuse
     them. Do NOT write to canonical `agent.jsonl` — SR-Agents controls can't reuse across the
     0.2.0→0.4.0 pool change, so use a fresh 0.4.0 file.
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
- **Models:** whatever LLM(s) under test — cloud (`claude-sonnet-4-6`, `gpt-5.4-mini`) or
  user-hosted via URL (`https://.../v1#qwen3-4b`).

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
M='https://hj1y208qba.execute-api.eu-central-1.amazonaws.com/prod/v1#qwen3-4b'
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
pnpm -F @ratel-ai/benchmark bfcl-summarize \
  --retrieval-rows results/raw/bfcl/retrieval-0.4.0-<m>.jsonl --agent results/raw/bfcl/agent-0.4.0-<m>.jsonl
pnpm -F @ratel-ai/benchmark bfcl-report
```

### SR-Agents
```bash
# --- Candidate gen (SDK): re-rank the FIXED 0.2.0 pool with the 0.4.0 retriever ---
# --pool-from uses the SAME 100-pool as 0.2.0 (keeps baseline/oracle identical & reusable;
# only ratel-full's ranking changes) AND pins the canonical 600. Pool 100 for the LLM eval.
RATEL_VERSION_LABEL=0.4.0-<m> pnpm -F @ratel-ai/benchmark sragents-candidates \
  --retriever <method> --pool-sizes 100 --top-k 5 \
  --pool-from results/raw/sragents/candidates.jsonl \
  --output results/raw/sragents/candidates-0.4.0-<m>.jsonl

# --- LLM eval (pool 100, top 5) ---
# --cache-source reuses baseline/oracle from the 0.2.0 canonical → ONLY ratel-full runs live.
RATEL_VERSION_LABEL=0.4.0-<m> pnpm -F @ratel-ai/benchmark sragents-select \
  --candidates results/raw/sragents/candidates-0.4.0-<m>.jsonl \
  --cache-source results/raw/sragents/agent.jsonl \
  --output results/raw/sragents/agent-0.4.0.jsonl \
  --arms control-baseline,control-oracle,ratel-full --models "$M" \
  --pool-size 100 --top-k 5 --concurrency 8 --dollar-global 30

# --- summarize (retrieval eval from --retrieval-rows + task completion from --agent) + report ---
pnpm -F @ratel-ai/benchmark sragents-summarize \
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
