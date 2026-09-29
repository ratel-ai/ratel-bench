# Plan: Jev vs Ratel — retrieval eval

Branch: `test/jev-as-tool-selection`

## Goal

Head-to-head retrieval quality of **TypeSafe Jev** vs **Ratel `0.13.0-rc.7` BM25** on
BFCL and SR-Agents, under the fixed retrieval-eval design in `EXPERIMENTS.md`.
No LLM eval, no hybrid/re-rank variants.

| Benchmark | Scenarios | Pools | k | Pool source |
|---|---|---|---|---|
| BFCL | 599 (`test-data/bfcl-all.jsonl`) | 30, 100 | 1, 3, 5 | `expandPool`, seed 42 |
| SR-Agents | 600 (100 × 6 datasets) | 50, 100 | 1, 3, 5 | `--pool-from results/raw/sragents/candidates.jsonl` |

Metrics (unchanged `metrics()`): recall@k, precision@k, MRR, hit@k, complete@k, nDCG@k.
Plus Jev-only: per-call latency, input tokens, $ cost.

## How Jev ranks

One `Choice` question per (scenario, pool) — pool ≤ 100 fits the 255-option cap.

- `state` = scenario prompt.
- `criteria` = `{option_key: option_text}` for every pool member; `probabilities` sorted
  desc = full ranking over the pool (same shape as `search(prompt, poolSize)`).
- **Option text = what Ratel indexes**, nothing more:
  - tools: name + description + flattened param names/descriptions (mirrors
    `ratel-ai-core` `indexing::searchable_text`);
  - skills: name + description (the SR generator loads only id/name/description).
- **Option keys** = short opaque keys (`t0…tN`) mapped back to ids, so tool-name
  sanitization/collisions can't leak or break anything.
- **Position fairness:** `expandPool` puts gold *first*. Options are shuffled per
  scenario with a seeded PRNG before sending; ties (near-zero probs) broken in that
  shuffled order, never pool order.
- `instructions`: one fixed sentence ("Which tool/skill best serves this request?"),
  written once, **not tuned on the test scenarios**.
- Model pinned: `jev-1.13.0`. Label: `RATEL_VERSION_LABEL=jev-1.13.0`.

## Implementation

1. **Deps.** Add `@ratel-ai/sdk-0.13.0-rc.7` alias + `@typesafe-ai/sdk` to
   `agent/package.json`. Check 0.13's `ToolCatalog`/`SkillCatalog` surface still fits
   `sdk/adapter.ts` (≥0.5.0 branch) — fix in the adapter if not.
2. **`agent/src/selectors/jev.ts`** — `rankWithJev({query, items, kind, seed})` →
   `{id, score}[]` over the full pool. Holds option-text builder, shuffle, key map,
   API call (`TYPESAFE_API_KEY`), retry on 429/529, bounded concurrency
   (stay well under 1,200 req/min).
3. **Response cache** — `results/raw/jev-cache/<benchmark>.jsonl`, keyed by
   hash(model, instructions, state, ordered criteria). Reruns/summaries never re-call
   the API; the cache is the audit trail (raw probabilities, usage, latency).
4. **`--selector ratel|jev` flag** on `bfcl-candidates.ts` and `sragents-candidates.ts`
   (default `ratel`). Only the ranking call switches; pool build, gold checks, k-slicing,
   `metrics()` and row schema are shared. Rows get `selector` + `jev_model`, and
   `latency_ms`/`input_tokens` for Jev.
5. **Byte-identity gate** — before merging, run both generators with `--selector ratel`
   (and without the flag) at an existing version and diff against output from `main`:
   must be byte-identical.
6. **Unit tests** — option-text builder matches Ratel's indexed fields; shuffle is
   deterministic and gold-position-independent; tie-break; cache hit skips the network;
   key map round-trips.

## Runs (order)

1. **Ratel `0.13.0-rc.7` BM25** (local, free):
   - `bfcl-candidates --sdk-version 0.13.0-rc.7 --retriever bm25 --pool-sizes 30,100`
     → `results/raw/bfcl/retrieval-0.13.0-rc.7-bm25.jsonl`
   - `sragents-candidates --sdk-version 0.13.0-rc.7 --retriever bm25 --pool-sizes 50,100 --pool-from results/raw/sragents/candidates.jsonl`
     → `results/raw/sragents/candidates-0.13.0-rc.7-bm25.jsonl`
2. **Jev smoke** — 20 scenarios per benchmark with `--selector jev`. Inspect raw
   responses, latency, token counts; sanity-check a few rankings by hand.
3. **Jev full** — same commands, `--selector jev`, label `jev-1.13.0`.
   ~1,198 BFCL + ~1,200 SR calls; est. ≪ $1, ~10–20 min at modest concurrency.
   ETA/cost confirmed with you before launch.
4. **Summarize/report** — `bfcl-summarize` / `sragents-summarize` + reports, and a
   side-by-side table (per benchmark × pool × k, split simple/multiple for BFCL and
   per dataset for SR).

## Reporting caveats

- **SR-Agents pools are gold-complete** (generator guard + dense recall@pool = 1.0). BM25's
  recall@pool < 1 is zero-score omission — real ranking misses, no masking needed.
- **Latency is not comparable like-for-like:** Ratel = in-process ms, Jev = network
  round-trip. Reported as its own column, not mixed into quality.
- Jev is early-access; rate limits and aliases can change — hence pinned model + cache.
- Benchmark prompts and tool descriptions (public data) are sent to TypeSafe's API.

## Needs from you

- `TYPESAFE_API_KEY`.
- Go-ahead before the full Jev run (step 3).
