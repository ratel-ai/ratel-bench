// Shared types for the SR-Agents skill-retrieval per-row / summary / report
// pipeline. Mirror of `bfcl-types.ts`, but **retrieval-only** (SR-Agents has no
// agent campaign / task completion) and bucketed **per dataset** (the scenario
// name — `bigcodebench`, `champ`, …) rather than BFCL's `simple` / `multiple`.
//
// Flow: the Rust `skill-retrieval` run emits a per-row metrics file (the same
// `RetrievalRow` struct as the tool path, so it's shape-compatible with
// `BfclRetrievalRow`). `sragents-summarize` rolls those rows up into an
// append-only experiment summary; `sragents-report` rebuilds a per-ratel-version
// report from the summary history (latest timestamp per version × dataset).

import type { BfclRetrievalRow, GoldSimilarity } from "./bfcl-types.js";

// The Rust skill-retrieval row is structurally identical to the BFCL retrieval
// row (same `RetrievalRow` struct in `retrieval/src/runner.rs`), carrying
// `category = "sragents-<dataset>"`. Re-export it under a domain name so callers
// don't reach across into the BFCL module.
export type { GoldSimilarity } from "./bfcl-types.js";
export type SragentsRetrievalRow = BfclRetrievalRow;

/**
 * Append-only retrieval summary row (one per dataset × pool_size × k). `dataset`
 * is the SR-Agents scenario name (`bigcodebench`, `champ`, `logicbench`,
 * `medcalcbench`, `theoremqa`, `toolqa`) or `all` for the cross-dataset aggregate.
 */
export interface SragentsRetrievalSummaryRow {
  timestamp: string;
  ratel_ai_core_version: string;
  source: "retriever_evaluation";
  dataset: string;
  pool_size: number;
  k: number;
  n: number;
  mean_precision: number;
  median_precision: number;
  mean_recall: number;
  median_recall: number;
  mean_mrr: number;
  median_mrr: number;
  mean_ndcg: number;
  median_ndcg: number;
  /** hit@K — fraction of scenarios with ≥1 gold skill in the top-K. */
  accuracy: number;
  /** Fraction of scenarios with *every* gold skill in the top-K (strict, for multi-mapping). */
  complete_rate: number;
  gold_similarity: GoldSimilarity;
}
