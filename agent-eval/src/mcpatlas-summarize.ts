// Raw rows -> append-only summary histories.
//
// Grouping mirrors the bfcl/sragents convention so report.json keeps its shape,
// with two additions this mode needs: `workload`, because the 55 tasks are 22
// version-control / 17 analysis / 16 database and averaging them hides which
// kind of work the gateway helps; and confidence intervals on every rate,
// because at n=55 a small delta is indistinguishable from noise and saying so in
// the emitted object is cheaper than arguing about it later.

import { toolFailureRate } from "./mcpatlas-build.js";
import { mean, median, percentile, proportionDelta, wilson } from "./mcpatlas-stats.js";
import type {
  McpAtlasAggregation,
  McpAtlasArm,
  McpAtlasCell,
  McpAtlasFailureClass,
  McpAtlasFailureCounts,
  McpAtlasRetrievalRow,
  McpAtlasScope,
  McpAtlasToolCallRow,
  McpAtlasWorkload,
} from "./mcpatlas-types.js";

/** "all" is a rollup bucket alongside the three real workloads, mirroring how
 *  sragents-summarize double-counts each row into an `all` dataset. */
export type WorkloadBucket = McpAtlasWorkload | "all";

export interface McpAtlasTaskSummaryRow {
  timestamp: string;
  source: "task_completion";
  ratel_version_label: string;
  ratel_local_version: string;
  agent_version: string;
  model: string;
  arm: McpAtlasArm;
  catalog_scope: McpAtlasScope;
  workload: WorkloadBucket;
  // metrics
  tasks: number;
  task_pass_rate: number;
  task_pass_ci95_low: number;
  task_pass_ci95_high: number;
  mean_claim_coverage: number | null;
  unscored_tasks: number;
  tool_selection_pass_rate: number;
  tool_selection_pass_ci95_low: number;
  tool_selection_pass_ci95_high: number;
  tool_selection_hit_rate: number;
  mean_selection_recall: number;
  mean_selection_precision: number;
  mean_tool_schema_tokens: number;
  mean_first_turn_context_tokens: number;
  mean_peak_context_tokens: number;
  mean_retrieval_overhead_tokens: number;
  mean_billed_input_tokens: number;
  mean_cache_read_tokens: number;
  mean_output_tokens: number;
  mean_total_tokens: number;
  mean_cache_hit_ratio: number;
  mean_dollar_cost: number;
  mean_compaction_events: number;
  latency_p50_ms: number;
  latency_p90_ms: number;
  mean_search_ms_total: number;
  mean_turns: number;
  errored: number;
  no_search_rate: number;
  /** k=1: false everywhere until a repeat subset runs. */
  variance_measured: boolean;
}

export interface McpAtlasRetrievalSummaryRow {
  timestamp: string;
  source: "retriever_evaluation";
  ratel_version_label: string;
  ratel_local_version: string;
  retriever_method: string;
  catalog_scope: McpAtlasScope;
  workload: WorkloadBucket;
  aggregation: McpAtlasAggregation;
  k: number;
  // metrics
  n_tasks: number;
  n_evaluated: number;
  n_no_search: number;
  no_search_rate: number;
  mean_recall: number;
  median_recall: number;
  mean_precision: number;
  mean_mrr: number;
  mean_ndcg: number;
  /** Canonical name in this mode. bfcl/sragents call the same quantity
   *  `accuracy`; renaming theirs would break committed report history. */
  hit_rate: number;
  complete_rate: number;
  /** hit_rate over tasks whose gold was fully inside the catalog — the fair
   *  retriever number, as opposed to the end-to-end one. */
  hit_rate_retrievable: number;
  n_gold_incomplete: number;
  mean_best_gold_rank: number | null;
  mean_union_recall: number | null;
  mean_zero_score_dropped: number;
}

export interface McpAtlasFailureSummaryRow {
  timestamp: string;
  source: "failures";
  ratel_version_label: string;
  ratel_local_version: string;
  model: string;
  arm: McpAtlasArm;
  catalog_scope: McpAtlasScope;
  workload: WorkloadBucket;
  // metrics
  cells: number;
  cells_errored: number;
  tool_calls_total: number;
  tool_calls_failed: number;
  tool_call_failure_rate: number;
  failures_by_class: McpAtlasFailureCounts;
  off_catalog_calls: number;
  never_searched: number;
  top_failing_tools: Array<{ tool_id: string; failure_class: string; count: number }>;
}

export interface McpAtlasCostSummaryRow {
  timestamp: string;
  source: "cost_comparison";
  ratel_version_label: string;
  ratel_local_version: string;
  model: string;
  arm: "native_vs_ratel";
  catalog_scope: McpAtlasScope;
  workload: WorkloadBucket;
  // metrics
  tasks_paired: number;
  native_mean_tool_schema_tokens: number;
  ratel_mean_tool_schema_tokens: number;
  schema_tokens_savings_pct: number;
  native_mean_first_turn_context_tokens: number;
  ratel_mean_first_turn_context_tokens: number;
  ratel_mean_retrieval_overhead_tokens: number;
  /** Schema saved MINUS ratel's retrieval payback. The honest occupancy number. */
  net_context_savings_tokens: number;
  net_context_savings_pct: number;
  native_mean_dollar_cost: number;
  ratel_mean_dollar_cost: number;
  dollar_savings_pct: number;
  /** occupancy savings minus billed savings. A large positive value next to a
   *  high native cache-hit ratio IS the explanation for why they diverge. */
  savings_attribution_gap_pct: number;
  native_mean_cache_hit_ratio: number;
  ratel_mean_cache_hit_ratio: number;
  native_task_pass_rate: number;
  ratel_task_pass_rate: number;
  task_pass_delta_pp: number;
  task_pass_delta_ci95_low_pp: number;
  task_pass_delta_ci95_high_pp: number;
  /** Interval excludes zero. At n=55 a small delta will not clear this. */
  task_pass_delta_significant: boolean;
  native_latency_p50_ms: number;
  ratel_latency_p50_ms: number;
  added_latency_p50_ms: number;
  added_latency_p90_ms: number;
  ratel_mean_search_ms_total: number;
  added_turns_mean: number;
}

export interface SummarizeInput {
  cells: readonly McpAtlasCell[];
  toolCalls: readonly McpAtlasToolCallRow[];
  retrieval: readonly McpAtlasRetrievalRow[];
  /** task_id -> workload, from the corpus. */
  workloads: ReadonlyMap<string, McpAtlasWorkload>;
}

export interface SummarizeResult {
  taskSummary: McpAtlasTaskSummaryRow[];
  retrievalSummary: McpAtlasRetrievalSummaryRow[];
  failureSummary: McpAtlasFailureSummaryRow[];
  costSummary: McpAtlasCostSummaryRow[];
}

/** Latest generated_at across a set, used as the summary row's timestamp. */
function latest(xs: readonly { generated_at?: string }[]): string {
  return (
    xs
      .map((x) => x.generated_at ?? "")
      .sort()
      .at(-1) ?? ""
  );
}

/** Every row lands in its own workload bucket AND in `all`. */
function buckets(w: McpAtlasWorkload): WorkloadBucket[] {
  return [w, "all"];
}

function groupBy<T>(xs: readonly T[], key: (x: T) => string): Map<string, T[]> {
  const m = new Map<string, T[]>();
  for (const x of xs) {
    const k = key(x);
    const list = m.get(k) ?? [];
    list.push(x);
    m.set(k, list);
  }
  return m;
}

export function summarizeMcpAtlas(input: SummarizeInput): SummarizeResult {
  const wl = (taskId: string): McpAtlasWorkload => input.workloads.get(taskId) ?? "analysis";

  // ── task completion ────────────────────────────────────────────────────────
  const taskSummary: McpAtlasTaskSummaryRow[] = [];
  const expanded = input.cells.flatMap((c) =>
    buckets(wl(c.task_id)).map((b) => ({ cell: c, bucket: b })),
  );
  for (const [, rows] of groupBy(
    expanded,
    (r) =>
      `${r.cell.ratel_version_label}::${r.cell.model}::${r.cell.arm}::${r.cell.catalog_scope}::${r.bucket}`,
  )) {
    const cells = rows.map((r) => r.cell);
    const first = cells[0];
    const n = cells.length;
    const passes = cells.filter((c) => c.task_pass).length;
    const selPasses = cells.filter((c) => c.tool_selection_pass).length;
    const passCi = wilson(passes, n);
    const selCi = wilson(selPasses, n);
    const scored = cells.filter((c) => c.claim_rubric.coverage !== null);
    const walls = cells.map((c) => c.latency.total_ms);
    taskSummary.push({
      timestamp: latest(cells),
      source: "task_completion",
      ratel_version_label: first.ratel_version_label,
      ratel_local_version: first.ratel_local_version,
      agent_version: first.agent_version,
      model: first.model,
      arm: first.arm,
      catalog_scope: first.catalog_scope,
      workload: rows[0].bucket,
      tasks: n,
      task_pass_rate: passes / n,
      task_pass_ci95_low: passCi.low,
      task_pass_ci95_high: passCi.high,
      mean_claim_coverage: scored.length
        ? mean(scored.map((c) => c.claim_rubric.coverage as number))
        : null,
      unscored_tasks: n - scored.length,
      tool_selection_pass_rate: selPasses / n,
      tool_selection_pass_ci95_low: selCi.low,
      tool_selection_pass_ci95_high: selCi.high,
      tool_selection_hit_rate: cells.filter((c) => c.tool_selection_hit).length / n,
      mean_selection_recall: mean(cells.map((c) => c.tool_selection_recall)),
      mean_selection_precision: mean(cells.map((c) => c.tool_selection_precision)),
      mean_tool_schema_tokens: mean(cells.map((c) => c.tokens.tool_schema_tokens)),
      mean_first_turn_context_tokens: mean(cells.map((c) => c.tokens.first_turn_context_tokens)),
      mean_peak_context_tokens: mean(cells.map((c) => c.tokens.peak_context_tokens)),
      mean_retrieval_overhead_tokens: mean(cells.map((c) => c.tokens.retrieval_overhead_tokens)),
      mean_billed_input_tokens: mean(cells.map((c) => c.tokens.billed_input_tokens)),
      mean_cache_read_tokens: mean(cells.map((c) => c.tokens.cache_read_tokens)),
      mean_output_tokens: mean(cells.map((c) => c.tokens.output_tokens)),
      mean_total_tokens: mean(cells.map((c) => c.tokens.total_tokens)),
      mean_cache_hit_ratio: mean(cells.map((c) => c.tokens.cache_hit_ratio)),
      mean_dollar_cost: mean(cells.map((c) => c.tokens.dollar_cost_total)),
      mean_compaction_events: mean(cells.map((c) => c.tokens.compaction_events)),
      latency_p50_ms: percentile(walls, 0.5),
      latency_p90_ms: percentile(walls, 0.9),
      mean_search_ms_total: mean(cells.map((c) => c.latency.search_ms_total)),
      mean_turns: mean(cells.map((c) => c.latency.turns)),
      errored: cells.filter((c) => c.error !== null).length,
      no_search_rate:
        first.arm === "ratel" ? cells.filter((c) => c.search_count === 0).length / n : 0,
      variance_measured: false,
    });
  }

  // ── retrieval ──────────────────────────────────────────────────────────────
  const retrievalSummary: McpAtlasRetrievalSummaryRow[] = [];
  const rExpanded = input.retrieval.flatMap((r) =>
    buckets(wl(r.task_id)).map((b) => ({ row: r, bucket: b })),
  );
  for (const [, rows] of groupBy(
    rExpanded,
    (r) =>
      `${r.row.ratel_version_label}::${r.row.catalog_scope}::${r.bucket}::${r.row.aggregation}::${r.row.k}`,
  )) {
    const all = rows.map((r) => r.row);
    const first = all[0];
    // Tasks that never searched are EXCLUDED from every mean and reported
    // separately: imputing 0 would turn "chose not to search" into "retriever
    // missed", which is a different claim.
    const evaluated = all.filter((r) => r.searched);
    const num = (f: (r: McpAtlasRetrievalRow) => number | null): number[] =>
      evaluated.map(f).filter((x): x is number => x !== null);
    const retrievable = evaluated.filter((r) => !r.gold_incomplete);
    retrievalSummary.push({
      timestamp: latest(all),
      source: "retriever_evaluation",
      ratel_version_label: first.ratel_version_label,
      ratel_local_version: first.ratel_local_version,
      retriever_method: first.retriever_method,
      catalog_scope: first.catalog_scope,
      workload: rows[0].bucket,
      aggregation: first.aggregation,
      k: first.k,
      n_tasks: all.length,
      n_evaluated: evaluated.length,
      n_no_search: all.length - evaluated.length,
      no_search_rate: all.length ? (all.length - evaluated.length) / all.length : 0,
      mean_recall: mean(num((r) => r.recall_at_k)),
      median_recall: median(num((r) => r.recall_at_k)),
      mean_precision: mean(num((r) => r.precision_at_k)),
      mean_mrr: mean(num((r) => r.reciprocal_rank)),
      mean_ndcg: mean(num((r) => r.ndcg_at_k)),
      hit_rate: evaluated.length
        ? evaluated.filter((r) => r.hit_at_k === true).length / evaluated.length
        : 0,
      complete_rate: evaluated.length
        ? evaluated.filter((r) => r.complete_at_k === true).length / evaluated.length
        : 0,
      hit_rate_retrievable: retrievable.length
        ? retrievable.filter((r) => r.hit_at_k === true).length / retrievable.length
        : 0,
      n_gold_incomplete: evaluated.filter((r) => r.gold_incomplete).length,
      mean_best_gold_rank: num((r) => r.best_gold_rank).length
        ? mean(num((r) => r.best_gold_rank))
        : null,
      mean_union_recall: num((r) => r.union_recall).length
        ? mean(num((r) => r.union_recall))
        : null,
      mean_zero_score_dropped: mean(evaluated.map((r) => r.zero_score_dropped)),
    });
  }

  // ── failures ───────────────────────────────────────────────────────────────
  const failureSummary: McpAtlasFailureSummaryRow[] = [];
  const cellByKey = new Map(input.cells.map((c) => [c.cell_key, c]));
  const tcExpanded = input.toolCalls.flatMap((t) =>
    buckets(wl(t.task_id)).map((b) => ({ row: t, bucket: b })),
  );
  for (const [key, rows] of groupBy(tcExpanded, (r) => {
    const c = cellByKey.get(r.row.cell_key);
    return `${c?.ratel_version_label}::${c?.model}::${r.row.arm}::${r.row.catalog_scope}::${r.bucket}`;
  })) {
    const calls = rows.map((r) => r.row);
    const cells = [...new Set(calls.map((c) => c.cell_key))]
      .map((k) => cellByKey.get(k))
      .filter((c): c is McpAtlasCell => Boolean(c));
    const counts = calls.reduce<McpAtlasFailureCounts>(
      (acc, c) => {
        acc[c.failure_class]++;
        return acc;
      },
      {
        ok: 0,
        upstream_error: 0,
        tool_not_found: 0,
        schema_validation_error: 0,
        auth_error: 0,
        rate_limited: 0,
        timeout: 0,
        transport_error: 0,
        gateway_error: 0,
        oversized_result: 0,
        unknown_error: 0,
        off_catalog_call: 0,
      },
    );
    const attempted = calls.filter((c) => c.failure_class !== "off_catalog_call");
    const failed = attempted.filter((c) => c.failure_class !== "ok");
    const offenders = groupBy(failed, (c) => `${c.tool_id}::${c.failure_class}`);
    const [ratel_version_label, model, arm, catalog_scope] = key.split("::");
    failureSummary.push({
      timestamp: latest(cells),
      source: "failures",
      ratel_version_label,
      ratel_local_version: cells[0]?.ratel_local_version ?? "",
      model,
      arm: arm as McpAtlasArm,
      catalog_scope: catalog_scope as McpAtlasScope,
      workload: rows[0].bucket,
      cells: cells.length,
      cells_errored: cells.filter((c) => c.error !== null).length,
      tool_calls_total: attempted.length,
      tool_calls_failed: failed.length,
      tool_call_failure_rate: toolFailureRate(counts),
      failures_by_class: counts,
      off_catalog_calls: counts.off_catalog_call,
      never_searched: cells.filter((c) => c.arm === "ratel" && c.search_count === 0).length,
      top_failing_tools: [...offenders.entries()]
        .map(([k, v]) => {
          const [tool_id, failure_class] = k.split("::");
          return { tool_id, failure_class: failure_class as McpAtlasFailureClass, count: v.length };
        })
        .sort((a, b) => b.count - a.count)
        .slice(0, 10),
    });
  }

  // ── paired cost comparison ─────────────────────────────────────────────────
  const costSummary: McpAtlasCostSummaryRow[] = [];
  for (const [, rows] of groupBy(
    expanded,
    (r) => `${r.cell.ratel_version_label}::${r.cell.model}::${r.cell.catalog_scope}::${r.bucket}`,
  )) {
    const byArm = groupBy(rows, (r) => r.cell.arm);
    const native = (byArm.get("native") ?? []).map((r) => r.cell);
    const ratel = (byArm.get("ratel") ?? []).map((r) => r.cell);
    if (!native.length || !ratel.length) continue;
    // Only tasks completed in BOTH arms; an unpaired task would bias the delta.
    const paired = new Set(
      native.map((c) => c.task_id).filter((t) => ratel.some((c) => c.task_id === t)),
    );
    const n = native.filter((c) => paired.has(c.task_id));
    const r = ratel.filter((c) => paired.has(c.task_id));
    if (!n.length) continue;

    const nSchema = mean(n.map((c) => c.tokens.tool_schema_tokens));
    const rSchema = mean(r.map((c) => c.tokens.tool_schema_tokens));
    const rOverhead = mean(r.map((c) => c.tokens.retrieval_overhead_tokens));
    const nCost = mean(n.map((c) => c.tokens.dollar_cost_total));
    const rCost = mean(r.map((c) => c.tokens.dollar_cost_total));
    const schemaSavings = nSchema ? ((nSchema - rSchema) / nSchema) * 100 : 0;
    const dollarSavings = nCost ? ((nCost - rCost) / nCost) * 100 : 0;
    const netTokens = nSchema - rSchema - rOverhead;
    const nPass = n.filter((c) => c.task_pass).length;
    const rPass = r.filter((c) => c.task_pass).length;
    const d = proportionDelta(rPass, r.length, nPass, n.length);
    const nWalls = n.map((c) => c.latency.total_ms);
    const rWalls = r.map((c) => c.latency.total_ms);
    const first = n[0];

    costSummary.push({
      timestamp: latest([...n, ...r]),
      source: "cost_comparison",
      ratel_version_label: first.ratel_version_label,
      ratel_local_version: r[0].ratel_local_version,
      model: first.model,
      arm: "native_vs_ratel",
      catalog_scope: first.catalog_scope,
      workload: rows[0].bucket,
      tasks_paired: paired.size,
      native_mean_tool_schema_tokens: nSchema,
      ratel_mean_tool_schema_tokens: rSchema,
      schema_tokens_savings_pct: schemaSavings,
      native_mean_first_turn_context_tokens: mean(n.map((c) => c.tokens.first_turn_context_tokens)),
      ratel_mean_first_turn_context_tokens: mean(r.map((c) => c.tokens.first_turn_context_tokens)),
      ratel_mean_retrieval_overhead_tokens: rOverhead,
      net_context_savings_tokens: netTokens,
      net_context_savings_pct: nSchema ? (netTokens / nSchema) * 100 : 0,
      native_mean_dollar_cost: nCost,
      ratel_mean_dollar_cost: rCost,
      dollar_savings_pct: dollarSavings,
      savings_attribution_gap_pct: schemaSavings - dollarSavings,
      native_mean_cache_hit_ratio: mean(n.map((c) => c.tokens.cache_hit_ratio)),
      ratel_mean_cache_hit_ratio: mean(r.map((c) => c.tokens.cache_hit_ratio)),
      native_task_pass_rate: nPass / n.length,
      ratel_task_pass_rate: rPass / r.length,
      task_pass_delta_pp: d.delta * 100,
      task_pass_delta_ci95_low_pp: d.low * 100,
      task_pass_delta_ci95_high_pp: d.high * 100,
      task_pass_delta_significant: d.significant,
      native_latency_p50_ms: percentile(nWalls, 0.5),
      ratel_latency_p50_ms: percentile(rWalls, 0.5),
      added_latency_p50_ms: percentile(rWalls, 0.5) - percentile(nWalls, 0.5),
      added_latency_p90_ms: percentile(rWalls, 0.9) - percentile(nWalls, 0.9),
      ratel_mean_search_ms_total: mean(r.map((c) => c.latency.search_ms_total)),
      added_turns_mean: mean(r.map((c) => c.latency.turns)) - mean(n.map((c) => c.latency.turns)),
    });
  }

  return { taskSummary, retrievalSummary, failureSummary, costSummary };
}
