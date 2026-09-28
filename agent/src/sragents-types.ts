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
import type { ErrorClass } from "./cell-errors.js";

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
  ratel_ai_core_resolved_version?: string | null;
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
  /** Exact crate release, independent of the method-specific report label. */
  ratel_ai_core_resolved_version?: string;
  scenario_id: string;
  /** `sragents-<dataset>` — the bucketing key. */
  category: string;
  arm: string;
  model: string;
  /** AI SDK provider id of the model that ran the cell (e.g. `anthropic.messages`, `amazon-bedrock`). */
  provider?: string;
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
  /**
   * Taxonomy class of `error` (see `cell-errors.ts`). Set only on errored rows;
   * legacy rows lack it and are classified from the message instead.
   */
  error_class?: ErrorClass;
  /** The call's finish reason (`length` = truncated output); `error` when the call threw without one. */
  finish_reason?: string;
  /**
   * Output cap REQUESTED on the call (`maxOutputTokens`); `null` = none sent. A
   * provider may clamp it further. Legacy rows lack it.
   */
  max_output_tokens?: number | null;
  /**
   * `live` = produced by this row's run; `reused` = a control cell served from
   * the control cache (re-stamped). Legacy rows lack it.
   */
  cache_source?: "live" | "reused";
  /**
   * Live attempt at this cell: its key's earlier live rows (resume, retry rounds)
   * + 1. Reused rows keep their source's. Legacy rows lack it.
   */
  attempt?: number;
  /** Retries the call spent (`llm-retry.ts`). Legacy rows lack it (SDK retries, uncounted). */
  retries?: number;
  /** Of `retries`, those after a throttle/overload status (429/503/529). */
  throttled_retries?: number;
  /** Backoff slept by the retries, in ms: part of `wall_ms`, not of the active-time deadline. */
  retry_wait_ms?: number;
  /**
   * Retry policy + `--timeout-ms` deadline the cell ran under, e.g.
   * `a8/b2000/c60000/w180000;timeout=active:300000+g30000`. Legacy rows lack it
   * (then: SDK retries, a per-attempt 300s undici limit, no abort).
   */
  retry_policy?: string;
}

/** Per-row skill-selection record (`results/raw/sragents/task-completion-rows.jsonl`). */
export interface SragentsTaskRow {
  ratel_ai_core_version: string;
  ratel_ai_core_resolved_version?: string | null;
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
  /** Taxonomy class of the cell's error (`cell-errors.ts`); null when it didn't error. */
  error_class: ErrorClass | null;
  /** Final infra error (`transient|access`): left out of every summary metric. */
  excluded: boolean;
  /**
   * The call stopped on the output-token limit (`finish_reason: "length"`). Kept (not
   * excluded); scored on its verdict, normally an `outcome` fail since nothing parses.
   */
  truncated: boolean;
  /** Output cap the cell requested; `null` when none was sent or the row predates the field. */
  max_output_tokens: number | null;
  /** Retries the call spent (`llm-retry.ts`); `null` for legacy rows (SDK retries, uncounted). */
  retries: number | null;
  /** Of `retries`, those after a 429/503/529; `null` for legacy rows. */
  throttled_retries: number | null;
  /** Retry backoff inside `wall_ms`, in ms; `null` for legacy rows (read as no wait). */
  retry_wait_ms: number | null;
  /** Retry policy + `--timeout-ms` deadline the call ran under; `null` for legacy rows. */
  retry_policy: string | null;
}

/**
 * Append-only skill-selection summary row (one per dataset × model × arm, plus an
 * `all` rollup). Same shape/field names as BFCL's `TaskSummaryRow` + `precision`,
 * with the same error semantics: superseded rows, final `transient|access` rows
 * excluded and counted, other errors scored as fails, null metrics when no rows
 * are kept.
 */
export interface SragentsTaskSummaryRow {
  timestamp: string;
  ratel_ai_core_version: string;
  ratel_ai_core_resolved_version?: string | null;
  source: "task_completion";
  model: string;
  arm: string;
  dataset: string;
  /** Kept rows (the denominator); excluded ones not counted. */
  scenarios: number;
  /** Mean complete (every gold selected) — the headline. */
  task_completion_accuracy: number | null;
  /** Mean hit (≥1 gold selected). */
  selection_accuracy: number | null;
  /** Mean |selected ∩ gold| / |gold|. */
  recall: number | null;
  /** Mean |selected ∩ gold| / |selected| — catches over-selection. */
  precision: number | null;
  /** Over non-errored rows only. */
  mean_total_tokens: number | null;
  /** Over non-errored rows only; wall-clock (retry waits included). */
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
