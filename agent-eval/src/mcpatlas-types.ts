// Shapes for the `mcpatlas` mode: an end-to-end A/B of ratel-local as an MCP
// gateway. Unlike `bfcl-*` / `sragents-*`, which measure the Ratel SDK retriever
// against a synthetic pool, this mode drives a real Claude Code subprocess against
// live MCP servers. No logic lives here.

// ─────────────────────────────────────────────────────────────────────────────
// Vocabulary
// ─────────────────────────────────────────────────────────────────────────────

/** Which side of the A/B. `native` wires every upstream MCP server straight into
 *  Claude Code; `ratel` puts the same servers behind ratel-local's gateway. */
export type McpAtlasArm = "native" | "ratel";

/** Which slice of the MCP-Atlas server registry is mounted. `coding` = the 11
 *  coding/data servers (79 tools); `full` = every servable server (36 / 195). */
export type McpAtlasScope = "coding" | "full";

/** Canonical tool identity across every dialect we have to reconcile:
 *  MCP-Atlas gold ids (`github_search_repositories`), Claude Code transcript
 *  names (`mcp__github__search_repositories`), and ratel-local telemetry
 *  (`github__search_repositories`). Normalized form is `<server>/<tool>`. */
export type CanonicalToolId = string;

/**
 * What a task is about, independent of which servers it happens to use.
 *
 *   version-control  git / github — repo inspection, commit history, issues
 *   database         mongodb / airtable — store analytics
 *   analysis         shell, code execution and files used to compute over a CSV
 *
 * The distinction matters because the benchmark's headline must not over-claim:
 * this is a developer-TOOL workload, not a software-engineering workload. Only
 * ~22 of 55 tasks are software-shaped.
 */
export type McpAtlasWorkload = "version-control" | "database" | "analysis";

/** k-slices evaluated for every search event. Fixed by the experiment design. */
export const MCPATLAS_KS = [1, 3, 5] as const;
export type McpAtlasK = (typeof MCPATLAS_KS)[number];

/** How multiple `search_tools` calls within one task collapse to a score.
 *  `first` is the headline; `best` is the reformulation ceiling; `union` predicts
 *  task success. Never averaged — searches ask different questions. */
export type McpAtlasAggregation = "first" | "best" | "union";

export type ProgrammaticVerdict = "pass" | "fail" | "n/a";
export type JudgeVerdict = "pass" | "fail" | "partial" | "n/a";

// ─────────────────────────────────────────────────────────────────────────────
// Corpus
// ─────────────────────────────────────────────────────────────────────────────

/** One MCP-Atlas task, after ingest. */
export interface McpAtlasTask {
  /** `mcpatlas-<TASK>`, so `corpusOf()` in report.ts can bucket it. */
  id: string;
  /** Raw MCP-Atlas TASK id (24 hex chars). */
  task_id: string;
  /** PROMPT, verbatim. */
  prompt: string;
  /** ENABLED_TOOLS — the per-task subset MCP-Atlas natively exposes (10-25).
   *  Recorded but NOT used to gate the catalog: we register the whole scope. */
  enabled_tool_ids: CanonicalToolId[];
  /** Distinct tools invoked in the gold TRAJECTORY, in first-call order. */
  gold_tool_ids: CanonicalToolId[];
  /** Servers those gold tools belong to. */
  gold_servers: string[];
  /** What the task is actually ABOUT, which is not what its servers imply.
   *  MCP-Atlas labels tasks by the servers they touch, so a "coding" task is
   *  often a CSV analysis that uses a code executor as a calculator. Measured on
   *  the 55-task set: 22 version-control, 17 analysis, 16 database. Carried as a
   *  row dimension so success is always readable per workload rather than
   *  averaged into one misleading number. */
  workload: McpAtlasWorkload;
  /** Full gold call sequence, with the recorded upstream output per step. */
  gold_calls: Array<{
    step: number;
    tool_id: CanonicalToolId;
    args: Record<string, unknown>;
    recorded_output_excerpt: string;
  }>;
  /** GTFA_CLAIMS — the atomic statements a correct answer must contain. */
  claims: string[];
}

/** One upstream MCP server in a catalog manifest. Records env var NAMES only;
 *  a secret must never reach a manifest or a JSONL row. */
export interface McpAtlasServerSpec {
  server: string;
  /** Tool ids this server contributes, canonical form, sorted. */
  tool_ids: CanonicalToolId[];
  tool_count: number;
  /** Names of env vars the server needs, e.g. ["GITHUB_TOKEN"]. Never values. */
  required_env: string[];
}

export interface McpAtlasCatalogManifest {
  scope: McpAtlasScope;
  servers: McpAtlasServerSpec[];
  server_count: number;
  tool_count: number;
  /** sha256 over the sorted tool ids. Asserted equal across arms — if it differs,
   *  the two arms did not see the same universe and the comparison is void. */
  catalog_sha256: string;
}

// ─────────────────────────────────────────────────────────────────────────────
// Frozen configuration
// ─────────────────────────────────────────────────────────────────────────────

export interface McpAtlasRunConfig {
  run_type: "mcpatlas_config";
  run_id: string;
  generated_at: string;
  /** sha256 over this object minus run_id/generated_at. A run whose hash differs
   *  is a different experiment, and mcpatlas-run refuses to append across them. */
  config_hash: string;

  // Versions — the report's grouping axis
  ratel_version_label: string;
  ratel_local_version: string;
  /** Transitive @ratel-ai/sdk. Ranking lives here, not in ratel-local. */
  ratel_sdk_version: string | null;
  claude_code_version: string;
  bench_git_sha: string;

  // Agent under test
  agent_harness: "claude-code";
  agent_model: string;
  backend: string | null;
  max_turns: number;
  per_cell_timeout_ms: number;
  /** Frozen and identical across arms. Without these the agent shells around MCP. */
  disallowed_tools: string[];
  permission_mode: string;
  prompt_id: string;
  prompt_hash: string;
  /** sha256 of the `--append-system-prompt` text, identical on both arms and
   *  applied to every cell. States the episode format (no user will answer),
   *  never task strategy — see SYSTEM_PROMPT_ADDENDUM in mcpatlas-prompt.ts. */
  system_prompt_addendum_hash: string;

  // Judging
  judge_model: string;
  claim_pass_threshold: number;
  claim_partial_threshold: number;
  include_tool_evidence: boolean;
  judge_prompt_sha256: string;

  // Retrieval (ratel arm only)
  retriever_method: "bm25" | "semantic" | "hybrid";
  top_k_tools: number;
  top_k_skills: number;

  // Grid
  arms: McpAtlasArm[];
  catalogs: McpAtlasCatalogManifest[];
  eval_ks: number[];
  /** Target tools per task for a catalog-size sweep. 0 = no subsetting: every
   *  task sees the whole scope, which is the default and the historical
   *  behaviour. When > 0 each task's catalog is its gold tools plus seeded
   *  fillers drawn from the scope, so gold coverage stays complete. */
  catalog_tools: number;
  runs_per_task: number;
  seed: number;
  concurrency: number;

  // Corpus
  corpus: {
    name: "mcp-atlas";
    dataset_revision: string;
    task_list_hash: string;
    task_count: number;
    task_ids: string[];
  };

  // Environment
  atlas_sandbox_url: string;
  atlas_image_digests: Record<string, string>;

  // Guards
  dollar_cap_global: number | null;
  /** Author-written, surfaced verbatim in report.json. */
  declared_limitations: string[];
}

// ─────────────────────────────────────────────────────────────────────────────
// Judging
// ─────────────────────────────────────────────────────────────────────────────

export interface ClaimScore {
  /** Stable: `${task_id}#${index}`. */
  claim_id: string;
  claim: string;
  score: 0 | 0.5 | 1;
  /** Quoted span from the agent's answer supporting the score. */
  evidence: string;
  explanation: string;
}

export interface ClaimRubricResult {
  claims: ClaimScore[];
  /** Σscore / n. `null` when the judge errored — never silently 0. */
  coverage: number | null;
  verdict: JudgeVerdict;
  judge_model: string;
  judge_error: string | null;
  judge_wall_ms: number;
  /** Metered separately and EXCLUDED from arm cost comparisons. */
  judge_input_tokens: number;
  judge_output_tokens: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// Measurement blocks
// ─────────────────────────────────────────────────────────────────────────────

/** Context-window occupancy and billed cost, deliberately separate. Conflating
 *  them is the single easiest way to lose credibility on this benchmark: native
 *  tool schemas sit in the cached prefix, so billed savings are far smaller than
 *  occupancy savings. */
export interface McpAtlasTokenBreakdown {
  // Occupancy — what sat in the context window
  tool_schema_tokens: number;
  system_prompt_tokens: number;
  first_turn_context_tokens: number;
  peak_context_tokens: number;
  compaction_events: number;
  /** ratel's payback: schemas arriving via search_tools results plus
   *  invoke_tool wrappers. MUST be subtracted before claiming savings. 0 native. */
  retrieval_overhead_tokens: number;
  /** Tool RESULTS, both arms — the part a gateway does not shrink. */
  tool_result_tokens: number;
  schema_share_of_prefix: number;

  // Billed — what the invoice reflects
  billed_input_tokens: number;
  cache_read_tokens: number;
  cache_creation_tokens: number;
  output_tokens: number;
  total_tokens: number;
  dollar_cost_total: number;
  /** Explains why occupancy savings outrun dollar savings. */
  cache_hit_ratio: number;
}

export interface McpAtlasLatencyBreakdown {
  total_ms: number;
  /** Σ search.took_ms. Native never searches, so this is 100% attributable added
   *  latency with zero modelling — the number to quote. */
  search_ms_total: number;
  search_ms_p50: number | null;
  search_ms_p90: number | null;
  /** Stage name -> Σ took_ms, from telemetry search.stages[]. */
  search_stage_ms: Record<string, number>;
  /** Σ (invoke_end - invoke_start): gateway round trip INCLUDING upstream. */
  invoke_ms_total: number;
  invoke_ms_p50: number | null;
  /** invoke_ms_total minus the per-tool native baseline. Modelled, hence `_est`;
   *  `null` when no baseline exists. Never folded into the headline. */
  gateway_overhead_ms_est: number | null;
  /** total - (search + invoke). Closes the accounting so components sum. */
  model_ms_est: number;
  turns: number;
}

export type McpAtlasFailureClass =
  | "ok"
  | "upstream_error"
  | "tool_not_found"
  | "schema_validation_error"
  | "auth_error"
  | "rate_limited"
  | "timeout"
  | "transport_error"
  | "gateway_error"
  | "oversized_result"
  | "unknown_error"
  /** Model named a tool that does not exist. A SELECTION defect, not a tool
   *  failure — excluded from tool_call_failure_rate. */
  | "off_catalog_call";

export type McpAtlasFailureCounts = Record<McpAtlasFailureClass, number>;

// ─────────────────────────────────────────────────────────────────────────────
// Rows
// ─────────────────────────────────────────────────────────────────────────────

export interface McpAtlasToolCall {
  tool_id: CanonicalToolId;
  args: Record<string, unknown>;
  /** The transcript turn this call happened on. Stamped directly from the raw
   *  tool_use block at the moment effectiveCalls() accepts the call — not
   *  re-derived later by position, which breaks the instant any search or
   *  off-catalog call is filtered out ahead of it in the transcript. */
  turn: number;
}

/** One cell = one (task, arm, scope, run_index). */
export interface McpAtlasCell {
  run_type: "mcpatlas_task";
  run_id: string;
  config_hash: string;
  generated_at: string;
  cell_key: string;

  // Identity
  task_id: string;
  scenario_id: string;
  category: "mcpatlas-coding";
  arm: McpAtlasArm;
  catalog_scope: McpAtlasScope;
  catalog_tool_count: number;
  /** Tools VISIBLE to the model. native: the whole catalog. ratel: 4. */
  catalog_size: number;
  run_index: number;

  // Versions
  ratel_version_label: string;
  ratel_local_version: string;
  ratel_sdk_version: string | null;
  agent_version: string;
  model: string;

  // Ground truth, carried so rows are self-describing
  enabled_tool_ids: CanonicalToolId[];
  gold_tool_ids: CanonicalToolId[];
  /** gold ∩ catalog. Retrieval is only fairly scored against these. */
  retrievable_gold_ids: CanonicalToolId[];
  gold_coverage: number;

  // Observed
  observed_tool_ids: CanonicalToolId[];
  tool_calls: McpAtlasToolCall[];

  // Outcome
  claim_rubric: ClaimRubricResult;
  task_pass: boolean;
  programmatic_verdict: ProgrammaticVerdict;
  judge_verdict: JudgeVerdict;

  // Selection
  tool_selection_recall: number;
  tool_selection_precision: number;
  tool_selection_f1: number;
  /** recall === 1 — every gold tool invoked. The strict bar. */
  tool_selection_pass: boolean;
  /** recall > 0 — matches judgeProgrammatic's lenient rule. */
  tool_selection_hit: boolean;
  trajectory_order_similarity: number;
  missing_gold: CanonicalToolId[];
  extra_calls: CanonicalToolId[];
  off_catalog_calls: CanonicalToolId[];

  // Measurement
  tokens: McpAtlasTokenBreakdown;
  latency: McpAtlasLatencyBreakdown;
  tool_failures: McpAtlasFailureCounts;
  tool_calls_total: number;
  tool_calls_unique: number;
  gateway_calls: number;
  non_gateway_calls: number;
  search_count: number;

  // Health
  final_text: string;
  finish_reason: string;
  error: string | null;

  // Provenance
  transcript_path: string;
  telemetry_path: string | null;
  /** "per_cell_file" is the design default. Anything else is a reported caveat. */
  telemetry_binding: "per_cell_file" | "none";
  cache_source: "live" | "reused";
}

/** One real tool call. Existing modes stub tools, so this signal is new. */
export interface McpAtlasToolCallRow {
  run_type: "mcpatlas_tool_call";
  run_id: string;
  cell_key: string;
  task_id: string;
  arm: McpAtlasArm;
  catalog_scope: McpAtlasScope;
  model: string;
  ratel_version_label: string;
  ratel_local_version: string;
  ratel_sdk_version: string | null;
  call_index: number;
  turn_index: number;
  tool_id: CanonicalToolId;
  server: string | null;
  via_gateway: boolean;
  args_size_bytes: number;
  result_size_bytes: number | null;
  result_tokens_est: number | null;
  took_ms: number | null;
  failure_class: McpAtlasFailureClass;
  error_message: string | null;
  is_gold: boolean;
  in_catalog: boolean;
}

export interface RankedHit {
  /** 1-based, after the score>0 filter and renumbering. */
  rank: number;
  tool_id: CanonicalToolId;
  score: number;
  is_gold: boolean;
  server: string | null;
}

export interface SearchStageTiming {
  name: string;
  took_ms: number;
  top_score: number | null;
}

export interface RetrievalMetricsAtK {
  k: number;
  recall_at_k: number;
  precision_at_k: number;
  reciprocal_rank: number;
  hit_at_k: boolean;
  complete_at_k: boolean;
  ndcg_at_k: number;
  gold_score: number | null;
  /** 1-based rank of the best gold hit anywhere in the ranking, not just top-k.
   *  "gold was at rank 7, so k=5 missed by 2" is a tuning instruction;
   *  hit@5=false is not. */
  best_gold_rank: number | null;
}

/** One `{"type":"search"}` telemetry event, enriched. The audit record behind
 *  every retrieval number — this is the "top-k ranking visible" artifact. */
export interface McpAtlasSearchEventRow {
  run_type: "mcpatlas_search_event";
  run_id: string;
  cell_key: string;
  task_id: string;
  arm: "ratel";
  catalog_scope: McpAtlasScope;
  model: string;
  ratel_version_label: string;
  ratel_local_version: string;
  ratel_sdk_version: string | null;
  search_index: number;
  query: string;
  origin: string;
  top_k_requested: number;
  /** score>0 only, renumbered. The ranking of record. */
  ranked: RankedHit[];
  /** Skill-dependency tools that ride along past the topK budget at score 0.
   *  Kept for audit, excluded from every metric. */
  zero_score_hits: Array<{ tool_id: CanonicalToolId; score: number }>;
  zero_score_dropped: number;
  stages: SearchStageTiming[];
  took_ms: number;
  gold_tool_ids: CanonicalToolId[];
  metrics_at_k: Record<string, RetrievalMetricsAtK>;
  /** The tool invoked immediately after this search, and whether it was in the
   *  hits. Separates "ranking was right" from "ranking was right AND acted on". */
  invoked_after: CanonicalToolId | null;
  invoked_was_in_hits: boolean;
}

/** One scored retrieval record per (task, scope, k, aggregation). */
export interface McpAtlasRetrievalRow {
  run_type: "mcpatlas_retrieval";
  run_id: string;
  generated_at: string;
  model: string;
  ratel_version_label: string;
  ratel_local_version: string;
  retriever_method: string;
  task_id: string;
  cell_key: string;
  catalog_scope: McpAtlasScope;
  pool_size: number;
  k: number;
  aggregation: McpAtlasAggregation;
  query: string;
  retrieved: RankedHit[];
  golden_answer: CanonicalToolId[];
  gold_count: number;
  unreachable_gold: CanonicalToolId[];
  gold_incomplete: boolean;
  /** Did the agent search at all? When false EVERY metric below is null — never
   *  0. Imputing 0 turns "chose not to search" into "retriever failed". */
  searched: boolean;
  search_count: number;
  recall_at_k: number | null;
  precision_at_k: number | null;
  reciprocal_rank: number | null;
  hit_at_k: boolean | null;
  complete_at_k: boolean | null;
  ndcg_at_k: number | null;
  gold_score: number | null;
  best_gold_rank: number | null;
  zero_score_dropped: number;
  // Per-task rollups, repeated on every row of the task for convenience
  searches_until_first_gold: number | null;
  union_recall: number | null;
  wasted_searches: number;
}
