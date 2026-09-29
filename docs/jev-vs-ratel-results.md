# Jev vs Ratel — retrieval eval results

Run 2026-09-28 on branch `test/jev-as-tool-selection`. Retrieval eval only (no LLM eval).

## Systems

| Arm | What ranks the pool | Label |
|---|---|---|
| **Ratel BM25** | `@ratel-ai/sdk@0.13.0-rc.7`, `--retriever bm25` | `0.13.0-rc.7-bm25` |
| **Jev** | TypeSafe Jev `jev-1.13.0`, one `choice` question over the whole pool, options sorted by probability | `jev-1.13.0` |
| **Ratel→Jev** | Ratel BM25 top 20, re-ranked by Jev | `ratel-0.13.0-rc.7-bm25+jev-1.13.0` |

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

## Tokens and latency

Average tokens per query (Ratel BM25 runs locally with no model call: 0 tokens).

| Benchmark | Pool | Jev input / output | Ratel→Jev input / output |
|---|---|---|---|
| BFCL | 30 | 2,410 / ~340 | 813 / ~106 |
| BFCL | 100 | 7,319 / ~1,065 | 1,427 / ~190 |
| SR-Agents | 50 | 3,633 / ~556 | 1,214 / ~150 |
| SR-Agents | 100 | 6,820 / ~1,079 | 1,508 / ~180 |

Totals for the full run (~2,400 queries): Jev ≈ 12.1M input + ~1.85M output tokens;
Ratel→Jev ≈ 3.0M input + ~0.36M output. Output tokens scale with the option count (Jev returns
a probability per option). Ratel→Jev skips the call when Ratel returns < 2 candidates (~6% of
queries, counted as 0).

Time per query: Ratel BM25 ~5 ms in-process; Jev ~280 ms median (p95 ~380 ms) network
round-trip; Ratel→Jev ≈ Ratel + one Jev call. Jev spend for everything above ≈ $0.65.

## Takeaways

1. Jev alone is best on every benchmark, pool and k. The gap is small on BFCL (BM25 is already
   near ceiling) and large on SR-Agents, where queries and skill descriptions share few words
   (e.g. logicbench: a story vs an abstract inference rule).
2. Ratel→Jev is capped by Ratel's shortlist: BM25 top 20 contains the gold 98.8% of the time on
   BFCL but only ~77% on SR-Agents, so re-ranking cannot reach standalone Jev there.
3. Ratel→Jev uses ~4× fewer Jev tokens than Jev alone; it does not save latency.

## Caveats

- Jev is sensitive to option order: identical 19-tool sets sent in two shuffles gave top-1
  0.90 vs 0.95 on a 20-scenario smoke. Results use one fixed seed (42).
- Latency during the Ratel→Jev run was inflated by a TypeSafe API slowdown (4–5 s/call for a
  while); latency figures come from the Jev-only run.
- Not yet compared: Ratel semantic/hybrid, and semantic→Jev. On this Mac, rc.7 semantic takes
  ~16 s/scenario (BFCL) and ~31 s/scenario (SR-Agents), i.e. ~5–8 h for the full design.

## Reproduce

Commands are in EXPERIMENTS.md → "Competitor: TypeSafe Jev" (`--selector ratel|jev|ratel+jev`,
`--rerank-depth 20`). Raw rows are under `results/raw/{bfcl,sragents}/` (gitignored per
ADR-0007); every Jev response is cached in `results/raw/jev-cache/`, so reruns replay with no
API calls.
