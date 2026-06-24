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

// ── LLM skill-selection (the task-completion analog) ──────────────────────────
//
// The LLM half of SR-Agents: each instance is shown a list of candidate skills
// and returns the skill ids it would use; we compare that set to the gold set.
// Run with/without Ratel (+ oracle), bucketed per dataset — the parallel of the
// BFCL agent campaign, but the task is *selection* (no args, no tool loop, no AST).

/** SR-Agents skill-selection arms (the analog of the BFCL arms). */
export type SragentsArm = "control-baseline" | "ratel-full" | "control-oracle";

/**
 * One raw skill-selection result (`results/raw/sragents/agent.jsonl`), emitted by
 * the `sragents-select` campaign — the analog of a BFCL `CellResult`, slimmed to
 * the selection task (set of selected ids vs gold; no tool_calls / verdicts).
 */
export interface SragentsSelectCell {
  run_type: "skill_selection";
  generated_at: string;
  ratel_ai_core_version: string;
  scenario_id: string;
  /** `sragents-<dataset>` — the bucketing key. */
  category: string;
  arm: string;
  model: string;
  run_index: number;
  /** Candidate-pool size the arm drew from; `null` for the pool-agnostic oracle arm. */
  pool_size: number | null;
  /** How many candidate skills the LLM was shown. */
  candidate_count: number;
  gold_skill_ids: string[];
  selected_skill_ids: string[];
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  dollar_cost: number;
  wall_ms: number;
  error: string | null;
}

/** Per-row skill-selection record (`results/raw/sragents/task-completion-rows.jsonl`). */
export interface SragentsTaskRow {
  ratel_ai_core_version: string;
  generated_at: string;
  dataset: string;
  model: string;
  arm: string;
  scenario_id: string;
  gold_skill_ids: string[];
  selected_skill_ids: string[];
  /** ≥1 gold skill selected (hit). */
  selection_pass: boolean;
  /** Every gold skill selected (complete; extras allowed). */
  task_completion_pass: boolean;
  /** |selected ∩ gold| / |gold|. */
  recall: number;
  /** |selected ∩ gold| / |selected| (0 when nothing selected). */
  precision: number;
  total_tokens: number;
  wall_ms: number;
}

/**
 * Append-only skill-selection summary row (one per dataset × model × arm, plus an
 * `all` rollup). Same shape/field names as BFCL's `TaskSummaryRow` + `precision`.
 */
export interface SragentsTaskSummaryRow {
  timestamp: string;
  ratel_ai_core_version: string;
  source: "task_completion";
  model: string;
  arm: string;
  dataset: string;
  scenarios: number;
  /** Mean complete (every gold selected) — the headline. */
  task_completion_accuracy: number;
  /** Mean hit (≥1 gold selected). */
  selection_accuracy: number;
  /** Mean |selected ∩ gold| / |gold|. */
  recall: number;
  /** Mean |selected ∩ gold| / |selected| — catches over-selection. */
  precision: number;
  mean_total_tokens: number;
  latency_p50_ms: number;
}
