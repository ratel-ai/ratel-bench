import { describe, expect, it } from "vitest";
import {
  buildReport,
  computeLimitations,
  GROUP_KEYS,
  latestGroups,
  metricFields,
  type ReportInput,
} from "./mcpatlas-report.js";
import type {
  McpAtlasCostSummaryRow,
  McpAtlasFailureSummaryRow,
  McpAtlasRetrievalSummaryRow,
  McpAtlasTaskSummaryRow,
} from "./mcpatlas-summarize.js";

const TS = "2026-08-21T00:00:00.000Z";
const LATER = "2026-08-22T00:00:00.000Z";

function task(over: Partial<McpAtlasTaskSummaryRow> = {}): McpAtlasTaskSummaryRow {
  return {
    timestamp: TS,
    source: "task_completion",
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    agent_version: "cc-1",
    model: "claude-haiku-4-5",
    arm: "native",
    catalog_scope: "coding",
    catalog_tools: 0,
    workload: "all",
    tasks: 55,
    task_pass_rate: 0.5,
    task_pass_ci95_low: 0.37,
    task_pass_ci95_high: 0.63,
    mean_claim_coverage: 0.6,
    unscored_tasks: 0,
    tool_selection_pass_rate: 0.5,
    tool_selection_pass_ci95_low: 0.37,
    tool_selection_pass_ci95_high: 0.63,
    tool_selection_hit_rate: 0.8,
    mean_selection_recall: 0.7,
    mean_selection_precision: 0.6,
    mean_tool_schema_tokens: 4000,
    mean_first_turn_context_tokens: 5000,
    mean_peak_context_tokens: 6000,
    mean_retrieval_overhead_tokens: 0,
    mean_billed_input_tokens: 100,
    mean_cache_read_tokens: 900,
    mean_output_tokens: 50,
    mean_total_tokens: 1050,
    mean_cache_hit_ratio: 0.9,
    mean_dollar_cost: 0.01,
    mean_compaction_events: 0,
    latency_p50_ms: 1000,
    latency_p90_ms: 1500,
    mean_search_ms_total: 0,
    mean_turns: 3,
    errored: 0,
    no_search_rate: 0,
    variance_measured: false,
    ...over,
  };
}

function retrieval(over: Partial<McpAtlasRetrievalSummaryRow> = {}): McpAtlasRetrievalSummaryRow {
  return {
    timestamp: TS,
    source: "retriever_evaluation",
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    retriever_method: "bm25",
    catalog_scope: "coding",
    catalog_tools: 0,
    workload: "all",
    aggregation: "first",
    k: 5,
    n_tasks: 55,
    n_evaluated: 55,
    n_no_search: 0,
    no_search_rate: 0,
    mean_recall: 0.8,
    median_recall: 1,
    mean_precision: 0.4,
    mean_mrr: 0.7,
    mean_ndcg: 0.75,
    hit_rate: 0.9,
    complete_rate: 0.6,
    hit_rate_retrievable: 0.9,
    n_gold_incomplete: 0,
    mean_best_gold_rank: 1.4,
    mean_union_recall: 0.85,
    mean_zero_score_dropped: 0.2,
    ...over,
  };
}

function cost(over: Partial<McpAtlasCostSummaryRow> = {}): McpAtlasCostSummaryRow {
  return {
    timestamp: TS,
    source: "cost_comparison",
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    model: "claude-haiku-4-5",
    arm: "native_vs_ratel",
    catalog_scope: "coding",
    catalog_tools: 0,
    workload: "all",
    tasks_paired: 55,
    native_mean_tool_schema_tokens: 4000,
    ratel_mean_tool_schema_tokens: 400,
    schema_tokens_savings_pct: 90,
    native_mean_first_turn_context_tokens: 5000,
    ratel_mean_first_turn_context_tokens: 1400,
    ratel_mean_retrieval_overhead_tokens: 600,
    net_context_savings_tokens: 3000,
    net_context_savings_pct: 75,
    native_mean_dollar_cost: 0.01,
    ratel_mean_dollar_cost: 0.009,
    dollar_savings_pct: 10,
    savings_attribution_gap_pct: 80,
    native_mean_cache_hit_ratio: 0.9,
    ratel_mean_cache_hit_ratio: 0.6,
    native_task_pass_rate: 0.5,
    ratel_task_pass_rate: 0.55,
    task_pass_delta_pp: 5,
    task_pass_delta_ci95_low_pp: -13,
    task_pass_delta_ci95_high_pp: 23,
    task_pass_delta_significant: false,
    native_latency_p50_ms: 1000,
    ratel_latency_p50_ms: 1200,
    added_latency_p50_ms: 200,
    added_latency_p90_ms: 300,
    ratel_mean_search_ms_total: 120,
    added_turns_mean: 1.2,
    ...over,
  };
}

function failure(over: Partial<McpAtlasFailureSummaryRow> = {}): McpAtlasFailureSummaryRow {
  return {
    timestamp: TS,
    source: "failures",
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    model: "claude-haiku-4-5",
    arm: "native",
    catalog_scope: "coding",
    catalog_tools: 0,
    workload: "all",
    cells: 55,
    cells_errored: 0,
    tool_calls_total: 200,
    tool_calls_failed: 10,
    tool_call_failure_rate: 0.05,
    failures_by_class: {
      ok: 190,
      upstream_error: 8,
      tool_not_found: 0,
      schema_validation_error: 0,
      auth_error: 0,
      rate_limited: 0,
      timeout: 0,
      transport_error: 0,
      gateway_error: 0,
      oversized_result: 0,
      unknown_error: 2,
      off_catalog_call: 3,
    },
    off_catalog_calls: 3,
    never_searched: 0,
    top_failing_tools: [],
    ...over,
  };
}

function input(over: Partial<ReportInput> = {}): ReportInput {
  return {
    generatedAt: TS,
    task: [task()],
    retrieval: [retrieval()],
    failures: [failure()],
    cost: [cost()],
    ...over,
  };
}

describe("latestGroups", () => {
  it("keeps only the newest rows per group", () => {
    const rows = [task({ timestamp: TS }), task({ timestamp: LATER })];
    const g = latestGroups(rows, (r) => r.arm);
    expect(g.get("native")).toHaveLength(1);
    expect(g.get("native")?.[0].timestamp).toBe(LATER);
  });

  it("rebuilds from full history without depending on a previous report", () => {
    const g = latestGroups([task(), task({ arm: "ratel" })], (r) => r.arm);
    expect([...g.keys()].sort()).toEqual(["native", "ratel"]);
  });
});

describe("metricFields", () => {
  it("strips group keys so adding a metric flows through with no report change", () => {
    const m = metricFields(task());
    for (const k of GROUP_KEYS) expect(m).not.toHaveProperty(k);
    expect(m).toHaveProperty("task_pass_rate");
    expect(m).toHaveProperty("task_pass_ci95_low");
  });
});

describe("computeLimitations — emitted by rule, not written by hand", () => {
  it("flags k=1 as blocking", () => {
    const l = computeLimitations(input());
    expect(l.find((x) => x.id === "k1_no_variance")?.severity).toBe("blocking");
  });

  it("flags small n and names the fields it constrains", () => {
    const l = computeLimitations(input()).find((x) => x.id === "small_n_wide_ci");
    expect(l?.severity).toBe("blocking");
    expect(l?.affects).toContain("task_pass_rate");
  });

  it("flags an insignificant delta so +5pp cannot be read as a result", () => {
    const l = computeLimitations(input());
    expect(l.find((x) => x.id === "delta_not_significant")?.severity).toBe("blocking");
  });

  it("does NOT flag significance when the interval excludes zero", () => {
    const l = computeLimitations(
      input({ cost: [cost({ task_pass_delta_significant: true, tasks_paired: 55 })] }),
    );
    expect(l.find((x) => x.id === "delta_not_significant")).toBeUndefined();
  });

  it("flags the cache absorbing schema cost when occupancy and billing diverge", () => {
    const l = computeLimitations(input());
    expect(l.find((x) => x.id === "cache_prefix_absorbs_schema_cost")?.affects).toContain(
      "dollar_savings_pct",
    );
  });

  it("does not flag the cache gap when the two roughly agree", () => {
    const l = computeLimitations(input({ cost: [cost({ savings_attribution_gap_pct: 5 })] }));
    expect(l.find((x) => x.id === "cache_prefix_absorbs_schema_cost")).toBeUndefined();
  });

  it("flags an unpaired comparison", () => {
    const l = computeLimitations(input({ cost: [cost({ tasks_paired: 40 })] }));
    expect(l.find((x) => x.id === "unpaired_comparison")?.severity).toBe("blocking");
  });

  it("flags a high no-search rate", () => {
    const l = computeLimitations(input({ retrieval: [retrieval({ no_search_rate: 0.3 })] }));
    expect(l.find((x) => x.id === "agent_skipped_search")).toBeDefined();
  });

  it("flags gold outside the catalog", () => {
    const l = computeLimitations(input({ retrieval: [retrieval({ n_gold_incomplete: 4 })] }));
    expect(l.find((x) => x.id === "gold_outside_catalog")?.affects).toContain("hit_rate");
  });

  it("flags an unknown_error bucket that has grown too large to trust", () => {
    const l = computeLimitations(
      input({
        failures: [
          failure({
            tool_calls_failed: 10,
            failures_by_class: { ...failure().failures_by_class, unknown_error: 5 },
          }),
        ],
      }),
    );
    expect(l.find((x) => x.id === "unclassified_tool_failures")).toBeDefined();
  });

  it("always states the standing caveats", () => {
    const ids = computeLimitations(input()).map((l) => l.id);
    expect(ids).toContain("trajectory_is_one_path");
    expect(ids).toContain("llm_judge_is_the_oracle");
    expect(ids).toContain("workload_is_not_all_coding");
  });
});

describe("buildReport", () => {
  it("keys on the gateway version, not the SDK retriever version", () => {
    const r = buildReport(input());
    expect(Object.keys(r.ratel_local_versions)).toEqual(["0.8.1"]);
  });

  it("nests task completion by model, arm and scope/workload", () => {
    const r = buildReport(input());
    const v = r.ratel_local_versions["0.8.1"];
    expect(v.task_completion["claude-haiku-4-5"].native["coding/all"].metrics).toHaveProperty(
      "task_pass_rate",
    );
  });

  it("carries cost comparison under its own arm label", () => {
    const v = buildReport(input()).ratel_local_versions["0.8.1"];
    expect(v.cost_comparison["claude-haiku-4-5"].native_vs_ratel["coding/all"]).toBeDefined();
  });

  it("bundles retrieval metrics per bucket, ordered deterministically", () => {
    const rows = [
      retrieval({ aggregation: "union", k: 1 }),
      retrieval({ aggregation: "first", k: 5 }),
      retrieval({ aggregation: "first", k: 1 }),
    ];
    const v = buildReport(input({ retrieval: rows })).ratel_local_versions["0.8.1"];
    const metrics = v.retriever_evaluation["coding/all"].metrics as Array<{
      aggregation: string;
      k: number;
    }>;
    expect(metrics.map((m) => `${m.aggregation}@${m.k}`)).toEqual([
      "first@1",
      "first@5",
      "union@1",
    ]);
  });

  it("attaches limitations to every version so numbers cannot travel without them", () => {
    const v = buildReport(input()).ratel_local_versions["0.8.1"];
    expect(v.limitations.computed.length).toBeGreaterThan(0);
    expect(v.limitations.computed.some((l) => l.severity === "blocking")).toBe(true);
  });

  it("passes declared limitations through verbatim", () => {
    const v = buildReport(input({ declaredLimitations: ["full scope is 195 tools, not 220"] }))
      .ratel_local_versions["0.8.1"];
    expect(v.limitations.declared).toEqual(["full scope is 195 tools, not 220"]);
  });

  it("keeps versions separate", () => {
    const r = buildReport(
      input({ task: [task(), task({ ratel_local_version: "0.9.0", timestamp: LATER })] }),
    );
    expect(Object.keys(r.ratel_local_versions).sort()).toEqual(["0.8.1", "0.9.0"]);
  });

  it("includes the frozen config when supplied", () => {
    const v = buildReport(input({ config: { agent_model: "claude-haiku-4-5" } }))
      .ratel_local_versions["0.8.1"];
    expect(v.configuration?.config).toEqual({ agent_model: "claude-haiku-4-5" });
  });
});

// A catalog-size sweep is the reason --catalog-tools exists. Before
// catalog_tools joined the grouping keys, latestGroups kept only the newest row
// per (scope, workload) and the second size silently REPLACED the first, so the
// report could not represent a sweep at all.
describe("a catalog-size sweep survives into one report", () => {
  it("keeps both sizes in task_completion instead of overwriting", () => {
    const r = buildReport({
      generatedAt: LATER,
      task: [
        task({ catalog_tools: 40, timestamp: TS }),
        task({ catalog_tools: 127, timestamp: LATER }),
      ],
      retrieval: [],
      failures: [],
      cost: [],
    });
    const buckets = Object.keys(
      r.ratel_local_versions["0.8.1"].task_completion["claude-haiku-4-5"].native,
    ).sort();
    expect(buckets).toEqual(["coding@127/all", "coding@40/all"]);
  });

  it("leaves the un-swept label untouched, so existing reports are unaffected", () => {
    const r = buildReport({
      generatedAt: LATER,
      task: [task({ catalog_tools: 0 })],
      retrieval: [],
      failures: [],
      cost: [],
    });
    expect(
      Object.keys(r.ratel_local_versions["0.8.1"].task_completion["claude-haiku-4-5"].native),
    ).toEqual(["coding/all"]);
  });

  it("keeps two retriever methods rather than the newest replacing the older", () => {
    const r = buildReport({
      generatedAt: LATER,
      task: [task()],
      retrieval: [
        retrieval({ retriever_method: "bm25", timestamp: TS }),
        retrieval({ retriever_method: "semantic", timestamp: LATER }),
      ],
      failures: [],
      cost: [],
    });
    const metrics = r.ratel_local_versions["0.8.1"].retriever_evaluation["coding/all"].metrics;
    expect(metrics.length).toBe(2);
    expect(
      (metrics as { retriever_method?: string }[]).map((m) => m.retriever_method).sort(),
    ).toEqual(["bm25", "semantic"]);
  });
});
