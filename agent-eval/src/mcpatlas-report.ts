// Summary histories -> report.json.
//
// Shape follows the bfcl/sragents contract so existing tooling transfers, keyed
// on `ratel_local_versions` rather than `ratel_ai_core_version` because the
// system under test is the gateway, not the SDK retriever.
//
// The section that does not exist in the other modes is `limitations`. It is
// emitted BY RULE from the data rather than written by hand, and every entry
// names the exact fields it constrains — so a consumer cannot render a summary
// number without the caveat travelling in the same object.

import type {
  McpAtlasCostSummaryRow,
  McpAtlasFailureSummaryRow,
  McpAtlasRetrievalSummaryRow,
  McpAtlasTaskSummaryRow,
} from "./mcpatlas-summarize.js";

/** Keys that identify a group rather than measure it; stripped from `metrics`. */
export const GROUP_KEYS = new Set([
  "timestamp",
  "source",
  "ratel_version_label",
  "ratel_local_version",
  "agent_version",
  "retriever_method",
  "model",
  "arm",
  "catalog_scope",
  "workload",
]);

export function metricFields<T extends object>(row: T): Record<string, unknown> {
  return Object.fromEntries(Object.entries(row).filter(([k]) => !GROUP_KEYS.has(k)));
}

/** Keep only the rows at each group's newest timestamp, so the report is
 *  deterministically rebuilt from the full append-only history and never depends
 *  on the previous report.json. */
export function latestGroups<T extends { timestamp: string }>(
  rows: readonly T[],
  keyFn: (r: T) => string,
): Map<string, T[]> {
  const groups = new Map<string, T[]>();
  for (const r of rows) {
    const k = keyFn(r);
    const list = groups.get(k) ?? [];
    list.push(r);
    groups.set(k, list);
  }
  const out = new Map<string, T[]>();
  for (const [k, list] of groups) {
    const max = list
      .map((r) => r.timestamp)
      .sort()
      .at(-1);
    out.set(
      k,
      list.filter((r) => r.timestamp === max),
    );
  }
  return out;
}

export type LimitationSeverity = "info" | "caveat" | "blocking";

export interface Limitation {
  id: string;
  severity: LimitationSeverity;
  statement: string;
  /** Fields whose interpretation this limitation constrains. */
  affects: string[];
}

export interface ReportInput {
  generatedAt: string;
  task: readonly McpAtlasTaskSummaryRow[];
  retrieval: readonly McpAtlasRetrievalSummaryRow[];
  failures: readonly McpAtlasFailureSummaryRow[];
  cost: readonly McpAtlasCostSummaryRow[];
  config?: Record<string, unknown>;
  declaredLimitations?: readonly string[];
}

/**
 * Derive the limitations that the data itself implies.
 *
 * These are not editorial. Each fires on a measured condition, so a run that is
 * underpowered or unpaired says so in its own output rather than relying on
 * whoever reads it to remember.
 */
export function computeLimitations(input: ReportInput): Limitation[] {
  const out: Limitation[] = [];
  const task = input.task.filter((r) => r.workload === "all");
  const cost = input.cost.filter((r) => r.workload === "all");
  const retrieval = input.retrieval.filter((r) => r.workload === "all");

  if (task.some((r) => !r.variance_measured)) {
    out.push({
      id: "k1_no_variance",
      severity: "blocking",
      statement:
        "Each task ran once (k=1), so no run-to-run variance was measured. Agent runs are " +
        "nondeterministic; a difference of a few points may not reproduce.",
      affects: ["task_pass_rate", "tool_selection_pass_rate", "task_pass_delta_pp"],
    });
  }

  const smallest = Math.min(...task.map((r) => r.tasks), Number.POSITIVE_INFINITY);
  if (Number.isFinite(smallest) && smallest < 100) {
    out.push({
      id: "small_n_wide_ci",
      severity: "blocking",
      statement:
        `The largest bucket holds ${smallest} tasks. The 95% interval on a rate at this n is ` +
        "roughly ±13pp; read every rate through its interval, not as a point.",
      affects: ["task_pass_rate", "task_pass_delta_pp", "tool_selection_pass_rate"],
    });
  }

  if (cost.some((r) => !r.task_pass_delta_significant)) {
    out.push({
      id: "delta_not_significant",
      severity: "blocking",
      statement:
        "The confidence interval on the native-vs-ratel success delta includes zero. The " +
        "measured difference is not distinguishable from noise at this sample size.",
      affects: ["task_pass_delta_pp"],
    });
  }

  for (const r of cost) {
    if (r.tasks_paired < Math.max(...task.map((t) => t.tasks), 0)) {
      out.push({
        id: "unpaired_comparison",
        severity: "blocking",
        statement:
          `Only ${r.tasks_paired} tasks completed in both arms; unpaired tasks were dropped ` +
          "from the comparison, which can bias it if failures were arm-correlated.",
        affects: ["task_pass_delta_pp", "schema_tokens_savings_pct", "dollar_savings_pct"],
      });
      break;
    }
  }

  if (cost.some((r) => r.savings_attribution_gap_pct > 20)) {
    out.push({
      id: "cache_prefix_absorbs_schema_cost",
      severity: "caveat",
      statement:
        "Context-occupancy savings substantially exceed billed savings: the native arm's tool " +
        "schemas sit in the cached prefix and are billed at the cache-read rate. Do not quote " +
        "the token figure as a cost figure.",
      affects: ["schema_tokens_savings_pct", "dollar_savings_pct", "net_context_savings_pct"],
    });
  }

  if (retrieval.some((r) => r.no_search_rate > 0.1)) {
    out.push({
      id: "agent_skipped_search",
      severity: "caveat",
      statement:
        "In more than 10% of ratel-arm tasks the agent never called search_capabilities. Those " +
        "tasks are excluded from retrieval means; retrieval quality is measured only where " +
        "retrieval was used.",
      affects: ["mean_recall", "hit_rate", "mean_ndcg", "mean_mrr"],
    });
  }

  if (retrieval.some((r) => r.n_gold_incomplete > 0)) {
    out.push({
      id: "gold_outside_catalog",
      severity: "caveat",
      statement:
        "Some tasks have gold tools outside the registered catalog, so retrieval could not " +
        "reach them. `hit_rate_retrievable` is the fair retriever number; `hit_rate` is " +
        "end-to-end.",
      affects: ["hit_rate", "mean_recall", "complete_rate"],
    });
  }

  const unknown = input.failures
    .filter((r) => r.workload === "all")
    .reduce((n, r) => n + r.failures_by_class.unknown_error, 0);
  const failed = input.failures
    .filter((r) => r.workload === "all")
    .reduce((n, r) => n + r.tool_calls_failed, 0);
  if (failed > 0 && unknown / failed > 0.2) {
    out.push({
      id: "unclassified_tool_failures",
      severity: "caveat",
      statement:
        `${unknown} of ${failed} tool failures fell into unknown_error. The classification ` +
        "table needs a new rule before the failure breakdown can be trusted.",
      affects: ["failures_by_class", "tool_call_failure_rate"],
    });
  }

  out.push({
    id: "trajectory_is_one_path",
    severity: "caveat",
    statement:
      "MCP-Atlas records one valid trajectory per task. Selection precision therefore measures " +
      "conformance to that path, not correctness; recall is the defensible figure.",
    affects: ["mean_selection_precision", "trajectory_order_similarity"],
  });

  out.push({
    id: "llm_judge_is_the_oracle",
    severity: "caveat",
    statement:
      "Task success is scored by a deterministic claim screen plus an LLM judge for the " +
      "residual. Absolute pass rates depend on the judge; the arm DELTA is largely insulated, " +
      "since both arms are scored identically.",
    affects: ["task_pass_rate", "mean_claim_coverage"],
  });

  out.push({
    id: "workload_is_not_all_coding",
    severity: "caveat",
    statement:
      "MCP-Atlas labels tasks by the servers they touch. The task set is roughly 22 " +
      "version-control, 17 analysis and 16 database, so this measures developer-TOOL " +
      "workloads rather than software engineering. Read per-workload buckets, not just `all`.",
    affects: ["task_pass_rate", "task_pass_delta_pp"],
  });

  return out;
}

export interface McpAtlasReport {
  generated_at: string;
  ratel_local_versions: Record<
    string,
    {
      retriever_evaluation: Record<string, { timestamp: string; metrics: unknown[] }>;
      task_completion: Record<
        string,
        Record<string, Record<string, { timestamp: string; metrics: unknown }>>
      >;
      cost_comparison: Record<
        string,
        Record<string, Record<string, { timestamp: string; metrics: unknown }>>
      >;
      failures: Record<
        string,
        Record<string, Record<string, { timestamp: string; metrics: unknown }>>
      >;
      configuration?: { timestamp: string; config: Record<string, unknown> };
      limitations: { declared: string[]; computed: Limitation[] };
    }
  >;
}

type Nested = Record<
  string,
  Record<string, Record<string, { timestamp: string; metrics: unknown }>>
>;

function nest<T extends { timestamp: string }>(
  rows: readonly T[],
  a: (r: T) => string,
  b: (r: T) => string,
  c: (r: T) => string,
): Nested {
  const out: Nested = {};
  for (const [, group] of latestGroups(rows, (r) => `${a(r)}::${b(r)}::${c(r)}`)) {
    for (const r of group) {
      const A = (out[a(r)] ??= {});
      const B = (A[b(r)] ??= {});
      B[c(r)] = { timestamp: r.timestamp, metrics: metricFields(r) };
    }
  }
  return out;
}

export function buildReport(input: ReportInput): McpAtlasReport {
  const versions = new Set<string>([
    ...input.task.map((r) => r.ratel_local_version),
    ...input.retrieval.map((r) => r.ratel_local_version),
    ...input.cost.map((r) => r.ratel_local_version),
    ...input.failures.map((r) => r.ratel_local_version),
  ]);

  const limitations = computeLimitations(input);
  const report: McpAtlasReport = { generated_at: input.generatedAt, ratel_local_versions: {} };

  for (const version of versions) {
    const t = input.task.filter((r) => r.ratel_local_version === version);
    const rt = input.retrieval.filter((r) => r.ratel_local_version === version);
    const c = input.cost.filter((r) => r.ratel_local_version === version);
    const f = input.failures.filter((r) => r.ratel_local_version === version);

    const retriever_evaluation: Record<string, { timestamp: string; metrics: unknown[] }> = {};
    for (const [, group] of latestGroups(
      rt,
      (r) => `${r.catalog_scope}::${r.workload}::${r.aggregation}::${r.k}`,
    )) {
      for (const r of group) {
        const bucket = `${r.catalog_scope}/${r.workload}`;
        const entry = (retriever_evaluation[bucket] ??= { timestamp: r.timestamp, metrics: [] });
        entry.metrics.push(metricFields(r));
        if (r.timestamp > entry.timestamp) entry.timestamp = r.timestamp;
      }
    }
    // Deterministic order so the file diffs cleanly between runs.
    for (const bucket of Object.values(retriever_evaluation)) {
      bucket.metrics.sort((a, b) => {
        const A = a as { aggregation: string; k: number };
        const B = b as { aggregation: string; k: number };
        return A.aggregation === B.aggregation
          ? A.k - B.k
          : A.aggregation.localeCompare(B.aggregation);
      });
    }

    report.ratel_local_versions[version] = {
      retriever_evaluation,
      task_completion: nest(
        t,
        (r) => r.model,
        (r) => r.arm,
        (r) => `${r.catalog_scope}/${r.workload}`,
      ),
      cost_comparison: nest(
        c,
        (r) => r.model,
        (r) => r.arm,
        (r) => `${r.catalog_scope}/${r.workload}`,
      ),
      failures: nest(
        f,
        (r) => r.model,
        (r) => r.arm,
        (r) => `${r.catalog_scope}/${r.workload}`,
      ),
      ...(input.config
        ? { configuration: { timestamp: input.generatedAt, config: input.config } }
        : {}),
      limitations: {
        declared: [...(input.declaredLimitations ?? [])],
        computed: limitations,
      },
    };
  }

  return report;
}
