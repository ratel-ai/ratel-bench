# Real-traffic tool-search benchmark — results

Run 2026-09-30 → 2026-10-01. Retrieval only (which tool should the agent call next), BFCL-style
metrics. The dataset is private customer traffic: this report contains **aggregate numbers only**
— no queries, tool names, tool descriptions or customer identifiers. The corpus and the ingest
tooling are kept outside the repository.

## Dataset

Real tool-search queries typed by AI agents against one production customer's MCP tool catalog,
collected over ~4 weeks and deduplicated to **465 distinct queries**. The catalog has **99 tools**
(name + description + retrieval description, no parameter schemas) and many near-sibling tools
(several search / filter / list variants per entity). Each query records what two production
engines actually served — **Ratel Cloud** and the customer's **incumbent** search — and a label:

| label source | rows | meaning |
|---|---|---|
| observed | 38 | the agent actually called this tool after the search |
| Jev | 344 | no invocation recorded; Jev (`jev-1.13.0`) picked it at confidence ≥ 0.8 |
| unclear | 83 | Jev confidence < 0.8 — no label |

## Setup

- **Gold (headline):** the dataset's own labels — observed + Jev-labelled rows (**382**); unclear
  rows excluded.
- **Pool:** every query is ranked against the **full 99-tool catalog**; k = 1, 3, 5.
- **Tool text:** the customer's retrieval description + tool name, for every method.
- **Arms:** Ratel `0.13.0-rc.7` BM25 and semantic (default embedding model); Jev `jev-1.13.0`
  (one call over all 99 tools); OpenJev = Verdict 151M (`heman10x/rlcd-modernbert-151m@8af2496`
  via razorback16/openjev) with compact option text and a ≤15-option tournament (99 tools →
  7 + 3 + 1 = 11 calls/query); BM25→Jev, BM25→OpenJev and Semantic→OpenJev (first-stage top 20
  re-ranked; OpenJev 2×10 + final when > 15 candidates).
- **Reference rows (not rerun):** Ratel Cloud and the incumbent, scored from the lists they
  actually served (1–10 and 1–5 tools, so their recall@k is capped by list length).

## Results — dataset labels (382 queries)

Recall@k

| k | Incumbent (served) | Ratel Cloud (served) | BM25 | Semantic | Jev | OpenJev | BM25→Jev | BM25→OpenJev | Semantic→OpenJev |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.775 | 0.599 | 0.421 | 0.476 | **0.890** | 0.291 | 0.866 | 0.401 | 0.369 |
| 3 | 0.861 | 0.817 | 0.665 | 0.785 | **0.971** | 0.435 | 0.932 | 0.660 | 0.628 |
| 5 | 0.861 | 0.846 | 0.770 | 0.861 | **0.984** | 0.552 | 0.942 | 0.793 | 0.749 |

MRR@k (at k = 1 MRR equals recall@1)

| k | Incumbent (served) | Ratel Cloud (served) | BM25 | Semantic | Jev | OpenJev | BM25→Jev | BM25→OpenJev | Semantic→OpenJev |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.775 | 0.599 | 0.421 | 0.476 | **0.890** | 0.291 | 0.866 | 0.401 | 0.369 |
| 3 | 0.816 | 0.702 | 0.525 | 0.616 | **0.926** | 0.355 | 0.897 | 0.513 | 0.479 |
| 5 | 0.816 | 0.709 | 0.549 | 0.633 | **0.929** | 0.382 | 0.900 | 0.544 | 0.507 |

Recall@1 by label source

| subset | Incumbent | Ratel Cloud | BM25 | Semantic | Jev | OpenJev | BM25→Jev | BM25→OpenJev | Semantic→OpenJev |
|---|---|---|---|---|---|---|---|---|---|
| observed (38) — independent of Jev | **0.842** | 0.632 | 0.447 | 0.526 | 0.684 | 0.316 | 0.684 | 0.342 | 0.421 |
| Jev-labelled (344) | 0.767 | 0.596 | 0.419 | 0.471 | **0.913** | 0.288 | 0.887 | 0.407 | 0.363 |

### Paired top-1 tests (first-named wins vs second-named wins, exact McNemar)

| comparison | top-1 | top-5 |
|---|---|---|
| Jev vs BM25 | 181 vs 2 (p ≈ 3e-51) | — |
| Jev vs Semantic | 164 vs 6 (p ≈ 4e-41) | — |
| BM25→Jev vs BM25 | 172 vs 2 (p ≈ 1e-48) | — |
| Jev vs BM25→Jev | 16 vs 7 (p = 0.09) | — |
| Jev vs Incumbent (served) | 65 vs 21 (p ≈ 2e-6) | — |
| Jev vs Ratel Cloud (served) | 126 vs 15 (p ≈ 5e-23) | — |
| Semantic vs BM25 | 65 vs 44 (p = 0.06) | — |
| OpenJev vs BM25 | 39 vs 89 (p ≈ 1e-5) | — |
| BM25→OpenJev vs BM25 | 58 vs 66 (p = 0.53) | 42 vs 33 (p = 0.36) |
| Semantic→OpenJev vs Semantic | 50 vs 91 (p ≈ 7e-4) | — |
| BM25→OpenJev vs OpenJev | 53 vs 11 (p ≈ 1e-7) | 100 vs 8 (p ≈ 2e-21) |
| Semantic→OpenJev vs OpenJev | 48 vs 18 (p ≈ 3e-4) | — |

### Why OpenJev fails as a re-ranker here

OpenJev re-ranks the shuffled first-stage top 20 on its own judgement (the first stage's order is
discarded):

| first stage | correct at #1 before | OpenJev keeps | OpenJev breaks | #1 wrong, gold in top 5 | OpenJev fixes |
|---|---|---|---|---|---|
| BM25 | 161 | 95 (59%) | 66 | 133 | 48 (36%) |
| Semantic | 182 | 91 (50%) | 91 | 147 | 41 (28%) |

BM25's shortlist contains lexically matching but often off-topic tools that OpenJev can reject;
semantic's shortlist is dominated by near-sibling tools, which OpenJev cannot separate from
"name + first sentence" in its 512-token window. So it nets ~0 on BM25 and about −50 on semantic.

### Takeaways

1. **Jev is the strongest selector** (top-1 0.890, recall@5 0.984); **BM25→Jev ties it**
   (p = 0.09) at ~4× fewer Jev tokens (~1.2k vs ~5.3k input tokens per query).
2. **Ratel `0.13.0-rc.7` is weak on this near-sibling catalog:** BM25 0.421 / semantic 0.476 top-1
   (semantic reaches recall@5 0.861, equal to the incumbent). Production Ratel Cloud (0.599) runs
   a different configuration.
3. **OpenJev is the weakest standalone ranker** (0.291). As a re-ranker it ties BM25 and is
   **significantly worse than semantic** — it never improves the Ratel stage it re-ranks
   (consistent with BFCL).
4. **Label circularity:** 344 of 382 gold labels came from Jev — Jev scores 0.913 top-1 on those
   vs **0.684 on the 38 observed rows**, where the incumbent leads (0.842) and Ratel Cloud reaches
   recall@5 1.000. The observed subset is the only label-independent check but is small.

### Jev's top-5 misses (6 of 382)

Four are observed rows where the agent's next call did not match the query's literal intent, and
two are Jev-labelled rows where our Jev run (retrieval description only, our instruction)
disagreed with the dataset's labelling run (both descriptions, a different instruction). None is
a clear Jev error.

## Secondary: LLM-judge gold (465 queries, superseded)

A first pass set gold with a single-pass Claude judge (shown the catalog, the query and the
incumbent's served list; not the Jev label or Ratel's list): observed rows kept their observed
tool, every other row (unclear included) took the judge's gold.

| metric | Incumbent | Ratel Cloud | BM25 | Jev | OpenJev | BM25→Jev | BM25→OpenJev |
|---|---|---|---|---|---|---|---|
| R@1 | 0.817 | 0.561 | 0.396 | 0.785 | 0.273 | 0.774 | 0.372 |
| R@5 | 0.862 | 0.806 | 0.742 | 0.972 | 0.512 | 0.923 | 0.766 |
| MRR@5 | 0.838 | 0.667 | 0.523 | 0.867 | 0.357 | 0.842 | 0.516 |

Dropped in favour of the dataset labels: showing the judge the incumbent's list anchored gold on
it (the judge accepted its top pick in 382/465 rows — exactly its 0.817), 42% of rows were marked
ambiguous, and several queries ask for capabilities the catalog lacks. Judge agreement: 30/38 with
observed calls, 315/344 with the Jev labels. Method ordering is the same under both golds.

## Caveats

- Incumbent and Ratel Cloud are scored from production served lists (short, engine-specific
  configurations); BM25/semantic are Ratel `0.13.0-rc.7` with default settings.
- OpenJev uses compact option text and a multi-call tournament (Verdict: ≤24 options, one
  512-token window) — not the same inputs or single-call protocol as Jev.
- Hybrid and Semantic→Jev were not run on this dataset. Semantic took ~18–29 s per query on the
  test Mac (each query re-embeds the 99-tool catalog; slower under CPU contention).

## Reproduce

The corpus and ingest tooling are private. Runs use the public `bfcl-candidates` CLI with
`--corpus <private corpus> --pool-sizes 99` and the selector flags in EXPERIMENTS.md →
"Competitor: TypeSafe Jev" (Ratel arms via `--ratel-output` in the same pass as the re-rank;
OpenJev with `--jev-option-text compact --jev-max-options 15`). Total Jev spend ≈ $0.13.
