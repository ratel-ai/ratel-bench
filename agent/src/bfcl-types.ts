// Shared types for the BFCL per-row / summary / report pipeline.
//
// Flow: each eval emits a per-row metrics file (overwritten) + appends to an
// experiment-summary file. `bfcl-report` rebuilds a per-ratel-version report
// from the append-only summaries (latest timestamp per version×source×model×type).

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
  scenario_id: string;
  category?: string;
  query: string;
  golden_answer: string[];
  retrieved: Array<{ id: string; score: number }>;
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
  generated_at: string;
  type: BfclType;
  model: string; // LLM name
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
  dollar_cost: number;
  wall_ms: number;
  turns: number;
}

/**
 * Append-only task-completion summary row (one per type × LLM × arm). The five
 * leaderboard metrics: task-completion accuracy, selection accuracy, argument
 * recall, token cost, and p50 latency.
 */
export interface TaskSummaryRow {
  timestamp: string;
  ratel_ai_core_version: string;
  source: "task_completion";
  model: string; // LLM name
  arm: string; // control-baseline | control-oracle | ratel-full | …
  type: BfclType;
  scenarios: number; // n (denominator)
  /** 1 — right function AND arguments (BFCL AST). null when no AST ground truth. */
  task_completion_accuracy: number | null;
  /** 2 — right function (name only). */
  selection_accuracy: number;
  /** 3 — mean argument recall (partial credit on required args). null when no AST ground truth. */
  recall: number | null;
  /** 4 — token cost: mean total (input+output) tokens per scenario. */
  mean_total_tokens: number;
  /** 5 — p50 (median) wall-clock latency in ms. */
  latency_p50_ms: number;
}

export type SummaryRow = RetrievalSummaryRow | TaskSummaryRow;
