# Jev vs Ratel — retrieval eval results

Run 2026-09-28 (BM25 arms, both benchmarks) and 2026-09-29 (semantic and hybrid arms, BFCL only) on branch
`test/jev-as-tool-selection`. Retrieval eval only (no LLM eval).

## Systems

| Arm | What ranks the pool | Label |
|---|---|---|
| **Ratel BM25** | `@ratel-ai/sdk@0.13.0-rc.7`, `--retriever bm25` | `0.13.0-rc.7-bm25` |
| **Jev** | TypeSafe Jev `jev-1.13.0`, one `choice` question over the whole pool, options sorted by probability | `jev-1.13.0` |
| **Ratel→Jev** | Ratel BM25 top 20, re-ranked by Jev | `ratel-0.13.0-rc.7-bm25+jev-1.13.0` |
| **Ratel semantic** (BFCL only) | `@ratel-ai/sdk@0.13.0-rc.7`, `--retriever semantic`, default embedding model | `0.13.0-rc.7-semantic` |
| **Semantic→Jev** (BFCL only) | Ratel semantic top 20, re-ranked by Jev | `ratel-0.13.0-rc.7-semantic+jev-1.13.0` |
| **Ratel hybrid** (BFCL only) | `@ratel-ai/sdk@0.13.0-rc.7`, `--retriever hybrid` (default dense weight 0.7) | `0.13.0-rc.7-hybrid` |
| **Hybrid→Jev** (BFCL only) | Ratel hybrid top 20, re-ranked by Jev | `ratel-0.13.0-rc.7-hybrid+jev-1.13.0` |

Design is the fixed retrieval eval from EXPERIMENTS.md: BFCL 599 scenarios, pools 30/100;
SR-Agents 600 scenarios (100 × 6 datasets), pools 50/100 from
`--pool-from results/raw/sragents/candidates.jsonl`; k 1/3/5; seed 42. All arms see identical
pools. Jev sees exactly the fields Ratel indexes (name, description, flattened parameter
names/descriptions/enums for tools; name + description for skills); options are shuffled
with a fixed seed because pools list gold first.

## Recall@k

| Benchmark | Pool | k | Ratel BM25 | Jev | Ratel→Jev |
|---|---|---|---|---|---|
| BFCL | 30 | 1 | 0.955 | **0.997** | 0.985 |
| BFCL | 30 | 5 | 0.988 | **1.000** | 0.988 |
| BFCL | 100 | 1 | 0.893 | **0.978** | 0.963 |
| BFCL | 100 | 5 | 0.983 | **1.000** | 0.988 |
| SR-Agents | 50 | 1 | 0.537 | **0.835** | 0.656 |
| SR-Agents | 50 | 5 | 0.735 | **0.962** | 0.744 |
| SR-Agents | 100 | 1 | 0.507 | **0.820** | 0.654 |
| SR-Agents | 100 | 5 | 0.709 | **0.960** | 0.744 |

## MRR (k = 5)

| Benchmark | Pool | Ratel BM25 | Jev | Ratel→Jev |
|---|---|---|---|---|
| BFCL | 30 | 0.970 | **0.998** | 0.987 |
| BFCL | 100 | 0.932 | **0.989** | 0.976 |
| SR-Agents | 50 | 0.708 | **0.990** | 0.802 |
| SR-Agents | 100 | 0.676 | **0.981** | 0.797 |

## SR-Agents by dataset — recall@1, pool 100

| Dataset | Ratel BM25 | Jev | Ratel→Jev |
|---|---|---|---|
| bigcodebench (multi-gold) | 0.321 | 0.358 | 0.372 |
| champ | 0.210 | 0.671 | 0.403 |
| logicbench | 0.240 | 0.980 | 0.490 |
| medcalcbench | 0.860 | 1.000 | 1.000 |
| theoremqa | 0.810 | 1.000 | 0.900 |
| toolqa | 0.600 | 0.910 | 0.760 |

Paired top-1 hits (pool 100): Jev-only wins vs Ratel-only wins — BFCL 55 vs 4
(McNemar p ≈ 2e-12), SR-Agents 217 vs 11 (p ≈ 8e-51).

## BFCL — all seven arms

Recall@k

| Pool | k | BM25 | Semantic | Hybrid | Jev | BM25→Jev | Semantic→Jev | Hybrid→Jev |
|---|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.955 | 0.967 | 0.972 | **0.997** | 0.985 | 0.995 | 0.993 |
| 30 | 3 | 0.988 | 0.998 | 0.997 | **1.000** | 0.988 | **1.000** | **1.000** |
| 30 | 5 | 0.988 | **1.000** | **1.000** | **1.000** | 0.988 | **1.000** | **1.000** |
| 100 | 1 | 0.893 | 0.915 | 0.925 | 0.978 | 0.963 | **0.982** | 0.980 |
| 100 | 3 | 0.972 | 0.987 | 0.985 | **1.000** | 0.988 | **1.000** | **1.000** |
| 100 | 5 | 0.983 | 0.998 | 0.995 | **1.000** | 0.988 | **1.000** | **1.000** |

MRR@k (at k = 1 MRR equals recall@1)

| Pool | k | BM25 | Semantic | Hybrid | Jev | BM25→Jev | Semantic→Jev | Hybrid→Jev |
|---|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.955 | 0.967 | 0.972 | **0.997** | 0.985 | 0.995 | 0.993 |
| 30 | 3 | 0.970 | 0.982 | 0.983 | **0.998** | 0.987 | 0.997 | 0.997 |
| 30 | 5 | 0.970 | 0.982 | 0.984 | **0.998** | 0.987 | 0.997 | 0.997 |
| 100 | 1 | 0.893 | 0.915 | 0.925 | 0.978 | 0.963 | **0.982** | 0.980 |
| 100 | 3 | 0.929 | 0.949 | 0.954 | 0.989 | 0.976 | **0.991** | 0.990 |
| 100 | 5 | 0.932 | 0.952 | 0.956 | 0.989 | 0.976 | **0.991** | 0.990 |

Paired top-1 hits (first-named wins vs second-named wins):

| Comparison | Pool 30 | Pool 100 |
|---|---|---|
| Jev vs Semantic | 19 vs 1 (p ≈ 4e-5) | 41 vs 3 (p ≈ 2e-9) |
| Jev vs Hybrid | 16 vs 1 (p ≈ 3e-4) | 35 vs 3 (p ≈ 7e-8) |
| Hybrid vs Semantic | 8 vs 5 (p = 0.58) | 18 vs 12 (p = 0.36) |
| Semantic→Jev vs Jev | 1 vs 2 | 5 vs 3 (p = 0.73) |
| Hybrid→Jev vs Jev | 1 vs 3 (p = 0.62) | 4 vs 3 (p = 1) |
| Hybrid→Jev vs Semantic→Jev | 0 vs 1 | 1 vs 2 |
| Semantic→Jev vs Semantic | — | 42 vs 2 (p ≈ 1e-10) |
| Hybrid→Jev vs Hybrid | 14 vs 1 (p ≈ 1e-3) | 37 vs 4 (p ≈ 1e-7) |

Semantic's and hybrid's top 20 almost always contain the gold (recall@5 0.995–1.000), so unlike
BM25 they do not cap the re-ranker: Semantic→Jev and Hybrid→Jev both tie standalone Jev while
sending Jev ~1.7k input tokens per query regardless of pool size (vs 2.4k / 7.3k for Jev alone).
Hybrid and semantic are statistically indistinguishable on BFCL.

## Tokens and latency

Average tokens per query (Ratel BM25 runs locally with no model call: 0 tokens).

| Benchmark | Pool | Jev input / output | Ratel→Jev input / output |
|---|---|---|---|
| BFCL | 30 | 2,410 / ~340 | 813 / ~106 |
| BFCL | 100 | 7,319 / ~1,065 | 1,427 / ~190 |
| SR-Agents | 50 | 3,633 / ~556 | 1,214 / ~150 |
| SR-Agents | 100 | 6,820 / ~1,079 | 1,508 / ~180 |

Semantic→Jev (BFCL): 1,727 input tokens per query at pool 30 and 1,737 at pool 100;
Hybrid→Jev: 1,727 / 1,744 (every query calls Jev: dense and hybrid always return 20
candidates).

Totals for the full run (~2,400 queries): Jev ≈ 12.1M input + ~1.85M output tokens;
Ratel→Jev ≈ 3.0M input + ~0.36M output. Output tokens scale with the option count (Jev returns
a probability per option). Ratel→Jev skips the call when Ratel returns < 2 candidates (~6% of
queries, counted as 0).

Time per query: Ratel BM25 ~5 ms in-process; Jev ~280 ms median (p95 ~380 ms) network
round-trip; Ratel→Jev ≈ Ratel + one Jev call (Semantic→Jev Jev call: ~310–360 ms median;
Hybrid→Jev ~380–410 ms, measured while three shard processes shared the machine).
Ratel semantic here cost ~14.5 s per scenario, dominated by embedding each fresh pool on the
Mac's CPU; a deployed catalog embeds its tools once, so per-query cost would be one query
embedding. Jev spend for everything above ≈ $0.78.

## Takeaways

1. Jev alone is best on every benchmark, pool and k. The gap is small on BFCL (BM25 is already
   near ceiling) and large on SR-Agents, where queries and skill descriptions share few words
   (e.g. logicbench: a story vs an abstract inference rule).
2. Ratel→Jev is capped by Ratel's shortlist: BM25 top 20 contains the gold 98.8% of the time on
   BFCL but only ~77% on SR-Agents, so re-ranking cannot reach standalone Jev there.
3. Ratel→Jev uses ~4× fewer Jev tokens than Jev alone; it does not save latency.
4. On BFCL, Ratel semantic beats BM25 but trails Jev (top-1 at pool 100: 0.915 vs 0.978).
   Semantic→Jev ties standalone Jev (0.982 vs 0.978, not significant) at ~4× fewer tokens on
   the 100-tool pool — the best quality/cost trade-off measured.
5. Ratel hybrid is Ratel's best standalone method on BFCL (top-1 at pool 100: 0.925) but is
   statistically tied with semantic and still clearly behind Jev; Hybrid→Jev ties Jev (0.980).

## Caveats

- Jev is sensitive to option order: identical 19-tool sets sent in two shuffles gave top-1
  0.90 vs 0.95 on a 20-scenario smoke. Results use one fixed seed (42).
- Latency during the Ratel→Jev run was inflated by a TypeSafe API slowdown (4–5 s/call for a
  while); latency figures come from the Jev-only run.
- Semantic and hybrid arms were run on BFCL only. SR-Agents semantic takes ~31 s/scenario on
  this Mac (~5 h).
- The hybrid run was executed as three parallel contiguous-scenario shards and concatenated in
  corpus order (the unsharded equivalent is the single-pass command in EXPERIMENTS.md with
  `--retriever hybrid`); sharding was verified byte-identical on BM25 and BM25→Jev.

## Reproduce

Commands are in EXPERIMENTS.md → "Competitor: TypeSafe Jev" (`--selector ratel|jev|ratel+jev`,
`--rerank-depth 20`; BFCL `--ratel-output` writes the plain Ratel arm from the same pass). Raw rows are under `results/raw/{bfcl,sragents}/` (gitignored per
ADR-0007); every Jev response is cached in `results/raw/jev-cache/`, so reruns replay with no
API calls.
