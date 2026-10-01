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
| **Laya** (BFCL only) | open-source Jev-compatible model (`convaiinnovations/laya` English, ModernBERT-large 421M, revision `55cf4c4`), self-hosted `laya-serve` on the Mac GPU, whole pool | `laya-english-55cf4c4` |
| **BM25→Laya** (BFCL only) | Ratel BM25 top 20, re-ranked by laya | `ratel-0.13.0-rc.7-bm25+laya-english-55cf4c4` |
| **OpenJev** (BFCL only) | open-source Verdict (`heman10x/rlcd-modernbert-151m`, ModernBERT-base + GLiClass, 151M, revision `8af2496`) served by razorback16/openjev (`verdict` backend, commit `dcd2094`) on the Mac CPU; multi-call tournament, compact option text | `openjev-verdict-151m-8af2496` |
| **BM25→OpenJev** (BFCL only) | Ratel BM25 top 20, re-ranked by Verdict (compact text, ≤15 options per call) | `ratel-0.13.0-rc.7-bm25+openjev-verdict-151m-8af2496` |

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

## BFCL — open-source alternative: laya

Laya is used as shipped (default 192-token option budget). Same pools, option text, seeded
shuffle and scoring as Jev; requests go to a local `laya-serve` via `--jev-base-url`.

Recall@k

| Pool | k | BM25 | Semantic | Hybrid | Jev | Laya | BM25→Jev | BM25→Laya |
|---|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.955 | 0.967 | 0.972 | **0.997** | 0.846 | 0.985 | 0.885 |
| 30 | 3 | 0.988 | 0.998 | 0.997 | **1.000** | 0.947 | 0.988 | 0.970 |
| 30 | 5 | 0.988 | **1.000** | **1.000** | **1.000** | 0.962 | 0.988 | 0.980 |
| 100 | 1 | 0.893 | 0.915 | 0.925 | **0.978** | 0.514 | 0.963 | 0.775 |
| 100 | 3 | 0.972 | 0.987 | 0.985 | **1.000** | 0.694 | 0.988 | 0.935 |
| 100 | 5 | 0.983 | 0.998 | 0.995 | **1.000** | 0.741 | 0.988 | 0.955 |

MRR@k

| Pool | k | BM25 | Semantic | Hybrid | Jev | Laya | BM25→Jev | BM25→Laya |
|---|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.955 | 0.967 | 0.972 | **0.997** | 0.846 | 0.985 | 0.885 |
| 30 | 3 | 0.970 | 0.982 | 0.983 | **0.998** | 0.893 | 0.987 | 0.923 |
| 30 | 5 | 0.970 | 0.982 | 0.984 | **0.998** | 0.897 | 0.987 | 0.926 |
| 100 | 1 | 0.893 | 0.915 | 0.925 | **0.978** | 0.514 | 0.963 | 0.775 |
| 100 | 3 | 0.929 | 0.949 | 0.954 | **0.989** | 0.595 | 0.976 | 0.849 |
| 100 | 5 | 0.932 | 0.952 | 0.956 | **0.989** | 0.605 | 0.976 | 0.853 |

Paired top-1 hits (pool 100): BM25 vs Laya 255 vs 28 (p ≈ 5e-47); BM25 vs BM25→Laya 101 vs 30
(p ≈ 4e-10) — laya often overturns a correct BM25 top-1; BM25→Jev vs BM25→Laya 118 vs 5.

Why laya underperforms here: its option head has a 192-token budget for *all* options, so each
option is cut to ~4 tokens (100 options), ~5 (30) or ~8 (20) — essentially the tool name, with
no description (laya averaged 185–431 input tokens per request vs 1.7k–7.3k for Jev). Raising
the budget to 1,024 (full descriptions) made it worse on a 20-scenario smoke (top-1 0.60 → 0.25),
consistent with the English checkpoint being trained on ≤512-token inputs. Latency on the Mac
GPU (MPS): BM25→Laya ~175 ms median, Laya alone 263 ms (pool 30) / 620 ms (pool 100); $0.

## BFCL — open-source alternative: OpenJev (Verdict)

Verdict takes at most 24 options per question and reads one 512-token window with the options
first and the query last (the option key is dropped; only its description is shown). With our
standard option text every request exceeded 512 tokens (median 941 at 15 options, 1,243 at 20),
which truncates the query away. So OpenJev was run with two opt-in flags:
`--jev-option-text compact` (option = `name: first sentence of description`) and
`--jev-max-options 15`, which ranks larger pools by a tournament: seeded balanced chunks of
≤15, the top 5 of each chunk advance, repeat until one final call (pool 30 = 3 calls, pool
100 = 11 calls, BM25 top 20 = 2×10 + final). Only 14 of 10,498 requests reached the 512-token
cap (i.e. were truncated). This differs from Jev's single call with full descriptions — it is the best setup
Verdict's limits allow.

Recall@k

| Pool | k | BM25 | Semantic | Hybrid | Jev | OpenJev | BM25→OpenJev |
|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.955 | 0.967 | 0.972 | **0.997** | 0.945 | 0.958 |
| 30 | 3 | 0.988 | 0.998 | 0.997 | **1.000** | 0.995 | 0.988 |
| 30 | 5 | 0.988 | **1.000** | **1.000** | **1.000** | 0.997 | 0.988 |
| 100 | 1 | 0.893 | 0.915 | 0.925 | **0.978** | 0.858 | 0.890 |
| 100 | 3 | 0.972 | 0.987 | 0.985 | **1.000** | 0.957 | 0.977 |
| 100 | 5 | 0.983 | 0.998 | 0.995 | **1.000** | 0.982 | 0.983 |

MRR@k

| Pool | k | BM25 | Semantic | Hybrid | Jev | OpenJev | BM25→OpenJev |
|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.955 | 0.967 | 0.972 | **0.997** | 0.945 | 0.958 |
| 30 | 3 | 0.970 | 0.982 | 0.983 | **0.998** | 0.969 | 0.972 |
| 30 | 5 | 0.970 | 0.982 | 0.984 | **0.998** | 0.969 | 0.972 |
| 100 | 1 | 0.893 | 0.915 | 0.925 | **0.978** | 0.858 | 0.890 |
| 100 | 3 | 0.929 | 0.949 | 0.954 | **0.989** | 0.904 | 0.931 |
| 100 | 5 | 0.932 | 0.952 | 0.956 | **0.989** | 0.910 | 0.932 |

Paired top-1 hits (first-named wins vs second-named wins):

| Comparison | Pool 30 | Pool 100 |
|---|---|---|
| OpenJev vs BM25 | 19 vs 25 (p = 0.45) | 31 vs 52 (p = 0.03) |
| OpenJev vs Laya | 76 vs 17 (p ≈ 4e-10) | 231 vs 25 (p ≈ 6e-43) |
| OpenJev vs Semantic | 11 vs 24 (p = 0.04) | 22 vs 56 (p ≈ 2e-4) |
| OpenJev vs Hybrid | 11 vs 27 (p = 0.01) | 19 vs 59 (p ≈ 6e-6) |
| Jev vs OpenJev | 31 vs 0 (p ≈ 9e-10) | 79 vs 7 (p ≈ 2e-16) |
| BM25→OpenJev vs BM25 | 17 vs 15 (p = 0.86) | 40 vs 42 (p = 0.91) |
| BM25→OpenJev vs OpenJev | 17 vs 9 (p = 0.17) | 35 vs 16 (p = 0.01) |
| BM25→OpenJev vs BM25→Laya | 52 vs 8 (p ≈ 5e-9) | 89 vs 20 (p ≈ 1e-11) |
| BM25→Jev vs BM25→OpenJev | 18 vs 2 (p ≈ 4e-4) | 50 vs 6 (p ≈ 1e-9) |

**Semantic→OpenJev (BFCL pool 100 only, 2026-10-01):** recall@1/3/5 = 0.888 / 0.990 / 0.998,
MRR@5 0.938 — vs Semantic 0.915 / 0.987 / 0.998 and Semantic→Jev 0.982 / 1.000 / 1.000.
Paired top-1: Semantic vs Semantic→OpenJev 41 vs 25 (p = 0.06); Semantic→Jev vs Semantic→OpenJev
60 vs 4 (p ≈ 7e-14). OpenJev does not improve on any Ratel first stage it re-ranks. (The same pass
reproduced the earlier semantic pool-100 rows byte-identically.)

Cost and latency (Mac CPU, $0), per query: OpenJev pool 30 — 3 calls, ~1.1k input tokens,
~1.3 s median; pool 100 — 11 calls, ~4.2k tokens, ~4.8 s median. BM25→OpenJev — 1.1 / 2.3 calls,
~240 / ~640 tokens, ~174 / ~604 ms median (pool 30 / 100; BM25 often returns < 20 candidates
at pool 30, so one call suffices).

## BFCL — everything together

Recall@k

| Pool | k | BM25 | Semantic | Hybrid | Jev | Laya | OpenJev | BM25→Jev | Semantic→Jev | Hybrid→Jev | BM25→Laya | BM25→OpenJev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.955 | 0.967 | 0.972 | **0.997** | 0.846 | 0.945 | 0.985 | 0.995 | 0.993 | 0.885 | 0.958 |
| 30 | 3 | 0.988 | 0.998 | 0.997 | **1.000** | 0.947 | 0.995 | 0.988 | **1.000** | **1.000** | 0.970 | 0.988 |
| 30 | 5 | 0.988 | **1.000** | **1.000** | **1.000** | 0.962 | 0.997 | 0.988 | **1.000** | **1.000** | 0.980 | 0.988 |
| 100 | 1 | 0.893 | 0.915 | 0.925 | 0.978 | 0.514 | 0.858 | 0.963 | **0.982** | 0.980 | 0.775 | 0.890 |
| 100 | 3 | 0.972 | 0.987 | 0.985 | **1.000** | 0.694 | 0.957 | 0.988 | **1.000** | **1.000** | 0.935 | 0.977 |
| 100 | 5 | 0.983 | 0.998 | 0.995 | **1.000** | 0.741 | 0.982 | 0.988 | **1.000** | **1.000** | 0.955 | 0.983 |

MRR@k (at k = 1 MRR equals recall@1)

| Pool | k | BM25 | Semantic | Hybrid | Jev | Laya | OpenJev | BM25→Jev | Semantic→Jev | Hybrid→Jev | BM25→Laya | BM25→OpenJev |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 1 | 0.955 | 0.967 | 0.972 | **0.997** | 0.846 | 0.945 | 0.985 | 0.995 | 0.993 | 0.885 | 0.958 |
| 30 | 3 | 0.970 | 0.982 | 0.983 | **0.998** | 0.893 | 0.969 | 0.987 | 0.997 | 0.997 | 0.923 | 0.972 |
| 30 | 5 | 0.970 | 0.982 | 0.984 | **0.998** | 0.897 | 0.969 | 0.987 | 0.997 | 0.997 | 0.926 | 0.972 |
| 100 | 1 | 0.893 | 0.915 | 0.925 | 0.978 | 0.514 | 0.858 | 0.963 | **0.982** | 0.980 | 0.775 | 0.890 |
| 100 | 3 | 0.929 | 0.949 | 0.954 | 0.989 | 0.595 | 0.904 | 0.976 | **0.991** | 0.990 | 0.849 | 0.931 |
| 100 | 5 | 0.932 | 0.952 | 0.956 | 0.989 | 0.605 | 0.910 | 0.976 | **0.991** | 0.990 | 0.853 | 0.932 |

Why BFCL scores are high for every method: our retrieval task keeps BFCL's queries and gold
functions but hides the gold among 30/100 tools, the rest drawn at random from 590 tools across
unrelated domains; every scenario has one gold tool, and 97% of queries share a content word
with the gold tool's name or description (86% with its name). Differences between strong
methods therefore live in ~50–80 hard scenarios (mostly near-duplicate tools), which is why the
paired tests matter more than the averages here; SR-Agents separates methods far more sharply.

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
6. Laya (open-source, as shipped) is the weakest arm — below plain BM25 both standalone (0.514
   at pool 100) and as a BM25 re-ranker (0.775 vs 0.893). Its fixed 192-token option budget
   reduces options to their names; fine-tuning with a longer budget would be needed before it
   could be useful as a Ratel re-rank stage.
7. OpenJev (Verdict, open-source, 151M) is far better than laya but only at BM25's level: 0.858
   top-1 at pool 100 standalone (a multi-call tournament) and 0.890 as a BM25 re-ranker — tied
   with BM25 (0.893) and below Ratel semantic/hybrid. Of the decision models tested, only Jev
   improves on Ratel.

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
- OpenJev used compact option text and a tournament (see its section) — not the same inputs or
  single-call protocol as Jev/laya. Its standalone run was interrupted by a machine crash at
  scenario 520/599 and resumed from the response cache (no answers recomputed).
- Long local runs on this 8 GB Mac twice exhausted swap (laya and OpenJev); both resumed from cache.

## Reproduce

Commands are in EXPERIMENTS.md → "Competitor: TypeSafe Jev" (`--selector ratel|jev|ratel+jev`,
`--rerank-depth 20`; BFCL `--ratel-output` writes the plain Ratel arm from the same pass; laya
via `--jev-base-url http://127.0.0.1:8000 --jev-model english` against a local `laya-serve`;
OpenJev via `--jev-base-url http://127.0.0.1:8080 --jev-model verdict-1.4 --jev-option-text
compact --jev-max-options 15` against razorback16/openjev with `OPENJEV_BACKEND=verdict`). Raw rows are under `results/raw/{bfcl,sragents}/` (gitignored per
ADR-0007); every Jev response is cached in `results/raw/jev-cache/`, so reruns replay with no
API calls.
