// Shared types for the BFCL per-row / summary / report pipeline.
//
// Flow: each eval emits a per-row metrics file (overwritten) + appends to an
// experiment-summary file. `bfcl-report` rebuilds a per-ratel-version report
// from the append-only summaries (latest timestamp per version×source×model×type).

import type { ErrorClass } from "./cell-errors.js";

/** `simple` / `multiple` — the BFCL subset, derived from the scenario-id prefix. */
export type BfclType = "simple" | "multiple";

export type BfclSource = "retriever_evaluation" | "task_completion";

/** Mean/median/stddev of the BM25 gold-tool similarity score + coverage. */
export interface GoldSimilarity {
  mean: number;
  median: number;
  stddev: number;
  /** Fraction of scenarios where the gold tool appeared in the ranking at all. */
  coverage: number;
}

/**
 * One retrieval per-row record, as emitted by the Rust `retrieval` run
 * (`results/raw/bfcl/retrieval-rows.jsonl`). Superset of `report.ts`'s
 * `RetrievalRow` with the BFCL-specific fields the summary needs.
 */
export interface BfclRetrievalRow {
  generated_at: string;
  ratel_ai_core_version?: string;
  ratel_ai_core_resolved_version?: string;
  scenario_id: string;
  category?: string;
  query: string;
  golden_answer: string[];
  retrieved: Array<{ id: string; score: number }>;
  /** Full pool membership (gold + distractors, gold-first) this cell was ranked over.
   *  Authoritative pool for the LLM eval's `control-baseline`; unlike `retrieved`, it
   *  never drops zero-score docs. Optional for back-compat with pre-fix files. */
  pool_ids?: string[];
  k: number;
  target_pool_size: number;
  pool_size: number;
  gold_count: number;
  recall_at_k: number;
  precision_at_k: number;
  reciprocal_rank: number;
  hit_at_k: boolean;
  complete_at_k?: boolean;
  ndcg_at_k: number;
  gold_score?: number | null;
}

/** Append-only retrieval summary row (one per type × pool_size × k). */
export interface RetrievalSummaryRow {
  timestamp: string;
  ratel_ai_core_version: string;
  ratel_ai_core_resolved_version?: string | null;
  source: "retriever_evaluation";
  type: BfclType;
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
  accuracy: number; // hit@K (single-gold BFCL ⇒ accuracy@K)
  complete_rate: number;
  gold_similarity: GoldSimilarity;
}

/** Task-completion per-row record (`results/raw/bfcl/task-completion-rows.jsonl`). */
export interface TaskRow {
  ratel_ai_core_version: string;
  ratel_ai_core_resolved_version?: string | null;
  generated_at: string;
  type: BfclType;
  model: string; // LLM name
  serving_provider?: string | null;
  publisher?: string | null;
  resolved_model?: string | null;
  vertex_location?: string | null;
  cost_source?: "provider" | "partial" | "estimate" | "unknown";
  arm: string;
  scenario_id: string;
  query: string;
  true_answers: { gold_tools: string[]; gold_calls: unknown[] };
  llm_answer: Array<{ toolId: string; args: Record<string, unknown> }>;
  selection_pass: boolean;
  /** null when the scenario has no AST ground truth. */
  task_completion_pass: boolean | null;
  /** Argument recall: fraction of required gold args supplied with an acceptable value. null when no AST ground truth. */
  recall: number | null;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  dollar_cost: number | null;
  wall_ms: number;
  turns: number;
  /**
   * Taxonomy class of the cell's error (`cell-errors.ts`); null when it didn't
   * error.
   */
  error_class: ErrorClass | null;
  /** Final infra error (`transient|access`): left out of every summary metric. */
  excluded: boolean;
  /** A step stopped on the output-token limit. Kept (not excluded); scored on its verdict. */
  truncated: boolean;
  /** Output cap the cell requested; `null` when none was sent or the row predates the field. */
  max_output_tokens: number | null;
  /** Retries the cell spent (`llm-retry.ts`); `null` for legacy rows (SDK retries, uncounted). */
  retries: number | null;
  /** Of `retries`, those after a 429/503/529; `null` for legacy rows. */
  throttled_retries: number | null;
  /** Retry backoff inside `wall_ms`, in ms; `null` for legacy rows (read as no wait). */
  retry_wait_ms: number | null;
  /** Retry policy + deadline the cell ran under; `null` for legacy rows. */
  retry_policy: string | null;
}

/**
 * Append-only task-completion summary row (one per type × LLM × arm). The five
 * leaderboard metrics: task-completion accuracy, selection accuracy, argument
 * recall, token cost, and p50 latency.
 *
 * Built from superseded rows (one per cell: the last final row, else the last
 * row). Final infra errors (`transient|access`) are excluded from every metric
 * and counted in `excluded_cells`; other errors stay scored fails. A group with
 * no kept rows has null metrics.
 */
export interface TaskSummaryRow {
  timestamp: string;
  ratel_ai_core_version: string;
  ratel_ai_core_resolved_version?: string | null;
  source: "task_completion";
  model: string; // LLM name
  serving_provider?: string | "mixed" | null;
  publisher?: string | "mixed" | null;
  resolved_model?: string | "mixed" | null;
  vertex_location?: string | "mixed" | null;
  arm: string; // control-baseline | control-oracle | ratel-full | …
  type: BfclType;
  scenarios: number; // n (denominator): kept rows, excluded ones not counted
  /** 1 — right function AND arguments (BFCL AST). null when no AST ground truth. */
  task_completion_accuracy: number | null;
  /** 2 — right function (name only). */
  selection_accuracy: number | null;
  /** 3 — mean argument recall (partial credit on required args). null when no AST ground truth. */
  recall: number | null;
  /** 4 — token cost: mean total (input+output) tokens per non-errored scenario. */
  mean_total_tokens: number | null;
  /** 5 — p50 (median) wall-clock latency in ms over non-errored scenarios (retry waits included). */
  latency_p50_ms: number | null;
  /** p50 of `wall_ms − retry_wait_ms` over the same rows: latency net of retry backoff. */
  latency_p50_net_ms: number | null;
  /** Final `transient|access` rows left out of every metric (not in `scenarios`). */
  excluded_cells: number;
  /** Kept rows that errored (`request|timeout|outcome`): scored as fails. */
  errored_cells: number;
  /** Kept rows cut off by the output-token limit (scored on their verdict; not necessarily fails). */
  truncated_cells: number;
  /**
   * Output cap of the kept rows: the shared value, `null` when none recorded one
   * (legacy or uncapped), or `"mixed"` (e.g. capped rows beside legacy controls).
   */
  max_output_tokens: number | "mixed" | null;
  /**
   * Retries spent by every superseding row of the group that recorded them,
   * excluded ones included; `null` when none did (legacy). Beside legacy rows a
   * lower bound — not always flagged by `retry_policy` `"mixed"`, which reads kept
   * rows only (an excluded legacy row leaves it single-valued).
   */
  retries: number | null;
  /** Of `retries`, those after a throttle/overload status (429/503/529); `null` likewise. */
  throttled_retries: number | null;
  /**
   * Retry policy of the kept rows: the shared value, `null` when none recorded one
   * (legacy), or `"mixed"`. Retries change completion probability, not answers.
   */
  retry_policy: string | "mixed" | null;
}

export type SummaryRow = RetrievalSummaryRow | TaskSummaryRow;
