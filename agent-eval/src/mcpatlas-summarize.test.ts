import { describe, expect, it } from "vitest";
import { type SummarizeInput, summarizeMcpAtlas } from "./mcpatlas-summarize.js";
import type {
  McpAtlasCell,
  McpAtlasRetrievalRow,
  McpAtlasToolCallRow,
  McpAtlasWorkload,
} from "./mcpatlas-types.js";

const TS = "2026-08-21T00:00:00.000Z";

function cell(over: Partial<McpAtlasCell> = {}): McpAtlasCell {
  return {
    run_type: "mcpatlas_task",
    run_id: "r1",
    config_hash: "cfg",
    generated_at: TS,
    cell_key: `${over.task_id ?? "t1"}__${over.arm ?? "native"}`,
    task_id: "t1",
    scenario_id: "mcpatlas-t1",
    category: "mcpatlas-coding",
    arm: "native",
    catalog_scope: "coding",
    catalog_tool_count: 79,
    catalog_size: 79,
    run_index: 0,
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    ratel_sdk_version: "0.9.1",
    agent_version: "cc-1",
    model: "claude-haiku-4-5",
    enabled_tool_ids: [],
    gold_tool_ids: ["a/1"],
    retrievable_gold_ids: ["a/1"],
    gold_coverage: 1,
    observed_tool_ids: ["a/1"],
    tool_calls: [],
    claim_rubric: {
      claims: [],
      coverage: 1,
      verdict: "pass",
      judge_model: "none",
      judge_error: null,
      judge_wall_ms: 0,
      judge_input_tokens: 0,
      judge_output_tokens: 0,
    },
    task_pass: true,
    programmatic_verdict: "pass",
    judge_verdict: "pass",
    tool_selection_recall: 1,
    tool_selection_precision: 1,
    tool_selection_f1: 1,
    tool_selection_pass: true,
    tool_selection_hit: true,
    trajectory_order_similarity: 1,
    missing_gold: [],
    extra_calls: [],
    off_catalog_calls: [],
    tokens: {
      tool_schema_tokens: 4000,
      system_prompt_tokens: 1000,
      first_turn_context_tokens: 5000,
      peak_context_tokens: 6000,
      compaction_events: 0,
      retrieval_overhead_tokens: 0,
      tool_result_tokens: 0,
      schema_share_of_prefix: 0.8,
      billed_input_tokens: 100,
      cache_read_tokens: 900,
      cache_creation_tokens: 0,
      output_tokens: 50,
      total_tokens: 1050,
      dollar_cost_total: 0.01,
      cache_hit_ratio: 0.9,
    },
    latency: {
      total_ms: 1000,
      search_ms_total: 0,
      search_ms_p50: null,
      search_ms_p90: null,
      search_stage_ms: {},
      invoke_ms_total: 100,
      invoke_ms_p50: null,
      gateway_overhead_ms_est: null,
      model_ms_est: 900,
      turns: 3,
    },
    tool_failures: {
      ok: 1,
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
    tool_calls_total: 1,
    tool_calls_unique: 1,
    gateway_calls: 0,
    non_gateway_calls: 1,
    search_count: 0,
    final_text: "answer",
    finish_reason: "success",
    error: null,
    transcript_path: "",
    telemetry_path: null,
    telemetry_binding: "none",
    cache_source: "live",
    ...over,
  };
}

function retrieval(over: Partial<McpAtlasRetrievalRow> = {}): McpAtlasRetrievalRow {
  return {
    run_type: "mcpatlas_retrieval",
    run_id: "r1",
    generated_at: TS,
    model: "claude-haiku-4-5",
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    retriever_method: "bm25",
    task_id: "t1",
    cell_key: "t1__ratel",
    catalog_scope: "coding",
    pool_size: 79,
    k: 5,
    aggregation: "first",
    query: "q",
    retrieved: [],
    golden_answer: ["a/1"],
    gold_count: 1,
    unreachable_gold: [],
    gold_incomplete: false,
    searched: true,
    search_count: 1,
    recall_at_k: 1,
    precision_at_k: 1,
    reciprocal_rank: 1,
    hit_at_k: true,
    complete_at_k: true,
    ndcg_at_k: 1,
    gold_score: 2,
    best_gold_rank: 1,
    zero_score_dropped: 0,
    searches_until_first_gold: 0,
    union_recall: 1,
    wasted_searches: 0,
    ...over,
  };
}

function toolCall(over: Partial<McpAtlasToolCallRow> = {}): McpAtlasToolCallRow {
  return {
    run_type: "mcpatlas_tool_call",
    run_id: "r1",
    cell_key: "t1__native",
    task_id: "t1",
    arm: "native",
    catalog_scope: "coding",
    model: "claude-haiku-4-5",
    ratel_version_label: "0.8.1",
    ratel_local_version: "0.8.1",
    ratel_sdk_version: "0.9.1",
    call_index: 0,
    turn_index: 1,
    tool_id: "a/1",
    server: "a",
    via_gateway: false,
    args_size_bytes: 10,
    result_size_bytes: null,
    result_tokens_est: null,
    took_ms: 50,
    failure_class: "ok",
    error_message: null,
    is_gold: true,
    in_catalog: true,
    ...over,
  };
}

function input(over: Partial<SummarizeInput> = {}): SummarizeInput {
  return {
    cells: [],
    toolCalls: [],
    retrieval: [],
    workloads: new Map<string, McpAtlasWorkload>([["t1", "version-control"]]),
    ...over,
  };
}

describe("task summary", () => {
  it("emits both the workload bucket and the all rollup", () => {
    const r = summarizeMcpAtlas(input({ cells: [cell()] }));
    expect(r.taskSummary.map((s) => s.workload).sort()).toEqual(["all", "version-control"]);
  });

  it("attaches a Wilson interval to every rate", () => {
    const cells = Array.from({ length: 10 }, (_, i) =>
      cell({ task_id: `t${i}`, task_pass: i < 5 }),
    );
    const wl = new Map(cells.map((c) => [c.task_id, "version-control" as const]));
    const r = summarizeMcpAtlas(input({ cells, workloads: wl }));
    const all = r.taskSummary.find((s) => s.workload === "all")!;
    expect(all.task_pass_rate).toBe(0.5);
    expect(all.task_pass_ci95_low).toBeLessThan(0.5);
    expect(all.task_pass_ci95_high).toBeGreaterThan(0.5);
  });

  it("separates workloads so one kind of work cannot hide inside another", () => {
    const cells = [
      cell({ task_id: "vc", task_pass: true }),
      cell({ task_id: "an", task_pass: false }),
    ];
    const wl = new Map<string, McpAtlasWorkload>([
      ["vc", "version-control"],
      ["an", "analysis"],
    ]);
    const r = summarizeMcpAtlas(input({ cells, workloads: wl }));
    expect(r.taskSummary.find((s) => s.workload === "version-control")?.task_pass_rate).toBe(1);
    expect(r.taskSummary.find((s) => s.workload === "analysis")?.task_pass_rate).toBe(0);
    expect(r.taskSummary.find((s) => s.workload === "all")?.task_pass_rate).toBe(0.5);
  });

  it("counts unscored tasks rather than folding them into coverage", () => {
    const unscored = cell({
      task_id: "t2",
      claim_rubric: { ...cell().claim_rubric, coverage: null, verdict: "n/a" },
    });
    const wl = new Map<string, McpAtlasWorkload>([
      ["t1", "version-control"],
      ["t2", "version-control"],
    ]);
    const r = summarizeMcpAtlas(input({ cells: [cell(), unscored], workloads: wl }));
    const all = r.taskSummary.find((s) => s.workload === "all")!;
    expect(all.unscored_tasks).toBe(1);
    expect(all.mean_claim_coverage).toBe(1); // averaged over the scored one only
  });

  it("reports no_search_rate only for the ratel arm", () => {
    const r = summarizeMcpAtlas(input({ cells: [cell({ arm: "ratel", search_count: 0 })] }));
    expect(r.taskSummary[0].no_search_rate).toBe(1);
    const n = summarizeMcpAtlas(input({ cells: [cell({ arm: "native" })] }));
    expect(n.taskSummary[0].no_search_rate).toBe(0);
  });

  it("marks variance as unmeasured at k=1", () => {
    expect(summarizeMcpAtlas(input({ cells: [cell()] })).taskSummary[0].variance_measured).toBe(
      false,
    );
  });
});

describe("retrieval summary", () => {
  it("EXCLUDES never-searched tasks from means and reports them separately", () => {
    const rows = [
      retrieval({ task_id: "t1" }),
      retrieval({
        task_id: "t2",
        searched: false,
        search_count: 0,
        recall_at_k: null,
        hit_at_k: null,
        ndcg_at_k: null,
        reciprocal_rank: null,
        precision_at_k: null,
        union_recall: null,
      }),
    ];
    const wl = new Map<string, McpAtlasWorkload>([
      ["t1", "version-control"],
      ["t2", "version-control"],
    ]);
    const r = summarizeMcpAtlas(input({ retrieval: rows, workloads: wl }));
    const all = r.retrievalSummary.find((s) => s.workload === "all")!;
    expect(all.n_tasks).toBe(2);
    expect(all.n_evaluated).toBe(1);
    expect(all.n_no_search).toBe(1);
    expect(all.no_search_rate).toBe(0.5);
    // the crucial part: mean is over the ONE evaluated task, not diluted to 0.5
    expect(all.mean_recall).toBe(1);
    expect(all.hit_rate).toBe(1);
  });

  it("reports the fair retriever number separately from the end-to-end one", () => {
    const rows = [
      retrieval({ task_id: "t1", hit_at_k: true }),
      retrieval({ task_id: "t2", hit_at_k: false, gold_incomplete: true }),
    ];
    const wl = new Map<string, McpAtlasWorkload>([
      ["t1", "version-control"],
      ["t2", "version-control"],
    ]);
    const all = summarizeMcpAtlas(input({ retrieval: rows, workloads: wl })).retrievalSummary.find(
      (s) => s.workload === "all",
    )!;
    expect(all.hit_rate).toBe(0.5); // end-to-end
    expect(all.hit_rate_retrievable).toBe(1); // gold was reachable in only one
    expect(all.n_gold_incomplete).toBe(1);
  });

  it("groups by aggregation and k so first/best/union stay distinct", () => {
    const rows = [
      retrieval({ aggregation: "first", k: 1 }),
      retrieval({ aggregation: "union", k: 1 }),
      retrieval({ aggregation: "first", k: 5 }),
    ];
    const r = summarizeMcpAtlas(input({ retrieval: rows }));
    const keys = r.retrievalSummary
      .filter((s) => s.workload === "all")
      .map((s) => `${s.aggregation}@${s.k}`)
      .sort();
    expect(keys).toEqual(["first@1", "first@5", "union@1"]);
  });

  it("uses hit_rate, never accuracy", () => {
    const s = summarizeMcpAtlas(input({ retrieval: [retrieval()] })).retrievalSummary[0];
    expect(s).toHaveProperty("hit_rate");
    expect(s).not.toHaveProperty("accuracy");
  });
});

describe("failure summary", () => {
  it("keeps off-catalog calls out of the tool failure rate", () => {
    const calls = [
      toolCall({ failure_class: "ok" }),
      toolCall({ call_index: 1, failure_class: "off_catalog_call", in_catalog: false }),
    ];
    const r = summarizeMcpAtlas(input({ cells: [cell()], toolCalls: calls }));
    const all = r.failureSummary.find((s) => s.workload === "all")!;
    expect(all.off_catalog_calls).toBe(1);
    expect(all.tool_calls_total).toBe(1);
    expect(all.tool_call_failure_rate).toBe(0);
  });

  it("separates gateway failures from upstream ones", () => {
    const calls = [
      toolCall({ failure_class: "gateway_error" }),
      toolCall({ call_index: 1, failure_class: "upstream_error" }),
    ];
    const all = summarizeMcpAtlas(input({ cells: [cell()], toolCalls: calls })).failureSummary.find(
      (s) => s.workload === "all",
    )!;
    expect(all.failures_by_class.gateway_error).toBe(1);
    expect(all.failures_by_class.upstream_error).toBe(1);
    expect(all.tool_call_failure_rate).toBe(1);
  });

  it("ranks the offending tools", () => {
    const calls = [
      toolCall({ tool_id: "a/1", failure_class: "upstream_error" }),
      toolCall({ call_index: 1, tool_id: "a/1", failure_class: "upstream_error" }),
      toolCall({ call_index: 2, tool_id: "b/2", failure_class: "timeout" }),
    ];
    const all = summarizeMcpAtlas(input({ cells: [cell()], toolCalls: calls })).failureSummary.find(
      (s) => s.workload === "all",
    )!;
    expect(all.top_failing_tools[0]).toEqual({
      tool_id: "a/1",
      failure_class: "upstream_error",
      count: 2,
    });
  });
});

describe("cost comparison", () => {
  const pair = (taskId: string, nativePass: boolean, ratelPass: boolean) => [
    cell({ task_id: taskId, arm: "native", cell_key: `${taskId}__native`, task_pass: nativePass }),
    cell({
      task_id: taskId,
      arm: "ratel",
      cell_key: `${taskId}__ratel`,
      task_pass: ratelPass,
      tokens: { ...cell().tokens, tool_schema_tokens: 400, retrieval_overhead_tokens: 600 },
    }),
  ];

  it("subtracts ratel's retrieval payback before claiming occupancy savings", () => {
    const cells = pair("t1", true, true);
    const r = summarizeMcpAtlas(input({ cells })).costSummary.find((s) => s.workload === "all")!;
    expect(r.schema_tokens_savings_pct).toBeCloseTo(90, 6); // 4000 -> 400
    // net = 4000 - 400 - 600 = 3000, i.e. 75% not 90%
    expect(r.net_context_savings_tokens).toBe(3000);
    expect(r.net_context_savings_pct).toBeCloseTo(75, 6);
  });

  it("reports the occupancy-vs-billing gap rather than letting them be conflated", () => {
    const cells = pair("t1", true, true);
    const r = summarizeMcpAtlas(input({ cells })).costSummary.find((s) => s.workload === "all")!;
    // identical dollar cost in the fixture, so all of the schema saving is
    // absorbed by the cached prefix
    expect(r.dollar_savings_pct).toBe(0);
    expect(r.savings_attribution_gap_pct).toBeCloseTo(90, 6);
    expect(r.native_mean_cache_hit_ratio).toBeCloseTo(0.9, 10);
  });

  it("a small delta at small n is reported as NOT significant", () => {
    const cells = [
      ...pair("t1", false, true),
      ...pair("t2", false, false),
      ...pair("t3", true, true),
      ...pair("t4", true, true),
    ];
    const wl = new Map(cells.map((c) => [c.task_id, "version-control" as const]));
    const r = summarizeMcpAtlas(input({ cells, workloads: wl })).costSummary.find(
      (s) => s.workload === "all",
    )!;
    expect(r.task_pass_delta_pp).toBeCloseTo(25, 6);
    expect(r.task_pass_delta_significant).toBe(false);
    expect(r.task_pass_delta_ci95_low_pp).toBeLessThan(0);
  });

  it("pairs only tasks completed in BOTH arms", () => {
    const cells = [
      ...pair("t1", true, true),
      cell({ task_id: "t2", arm: "native", cell_key: "t2__native" }), // no ratel counterpart
    ];
    const wl = new Map(cells.map((c) => [c.task_id, "version-control" as const]));
    const r = summarizeMcpAtlas(input({ cells, workloads: wl })).costSummary.find(
      (s) => s.workload === "all",
    )!;
    expect(r.tasks_paired).toBe(1);
  });

  it("emits nothing when an arm is missing entirely", () => {
    const r = summarizeMcpAtlas(input({ cells: [cell({ arm: "native" })] }));
    expect(r.costSummary).toEqual([]);
  });
});
