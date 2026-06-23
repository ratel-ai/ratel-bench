import { describe, expect, it } from "vitest";
import { summarizeBfcl } from "./bfcl-summarize.js";
import type { BfclRetrievalRow } from "./bfcl-types.js";
import type { Arm, CellResult, Scenario } from "./types.js";

const TS = "2026-06-22T00:00:00.000Z";
const CORE = "0.2.0";

function retrievalRow(over: Partial<BfclRetrievalRow>): BfclRetrievalRow {
  return {
    generated_at: TS,
    ratel_ai_core_version: CORE,
    scenario_id: "bfcl-simple-0",
    category: "bfcl-simple",
    query: "q",
    golden_answer: ["tool_a"],
    retrieved: [{ id: "tool_a", score: 5 }],
    k: 1,
    target_pool_size: 30,
    pool_size: 30,
    gold_count: 1,
    recall_at_k: 1,
    precision_at_k: 1,
    reciprocal_rank: 1,
    hit_at_k: true,
    complete_at_k: true,
    ndcg_at_k: 1,
    gold_score: 5,
    ...over,
  };
}

function cell(over: Partial<CellResult>): CellResult {
  return {
    scenario_id: "bfcl-simple-0",
    category: "bfcl-simple",
    arm: "ratel-full" as Arm,
    model: "claude-haiku-4-5",
    run_index: 0,
    ratel_version: "sdk",
    ratel_ai_core_version: CORE,
    generated_at: TS,
    catalog_size: 5,
    pool_size: 100,
    seed: 42,
    input_tokens: 1000,
    output_tokens: 50,
    cached_input_tokens: 0,
    cache_creation_tokens: 0,
    total_tokens: 1050,
    tool_calls_total: 1,
    tool_calls_unique: 1,
    gateway_calls: 0,
    non_gateway_calls: 1,
    turns: 1,
    effective_tool_ids: ["tool_a"],
    programmatic_verdict: "pass",
    ast_verdict: "pass",
    judge_verdict: "n/a",
    final_text: "ok",
    finish_reason: "stop",
    error: null,
    wall_ms: 900,
    dollar_cost: 0.001,
    tool_calls: [{ toolId: "tool_a", args: { x: 1 } }],
    ...over,
  };
}

const scenarios: Scenario[] = [
  {
    id: "bfcl-simple-0",
    prompt: "simple question",
    candidate_pool: [],
    gold_tools: ["tool_a"],
    category: "bfcl-simple",
    gold_calls: [{ tool: "tool_a", args: { x: [1] } }],
  },
  {
    id: "bfcl-multiple-0",
    prompt: "multiple question",
    candidate_pool: [],
    gold_tools: ["tool_b"],
    category: "bfcl-multiple",
    gold_calls: [{ tool: "tool_b", args: {} }],
  },
];

describe("summarizeBfcl — retrieval summary", () => {
  it("emits one flat row per (type, pool_size, k) with gold-similarity stats", () => {
    const rows = [
      retrievalRow({ scenario_id: "bfcl-simple-0", category: "bfcl-simple", gold_score: 4 }),
      retrievalRow({
        scenario_id: "bfcl-simple-1",
        category: "bfcl-simple",
        gold_score: 6,
        recall_at_k: 0,
        hit_at_k: false,
      }),
      retrievalRow({
        scenario_id: "bfcl-multiple-0",
        category: "bfcl-multiple",
        gold_count: 1,
        gold_score: 8,
      }),
    ];
    const { retrievalSummary } = summarizeBfcl({ retrievalRows: rows, cells: [], scenarios });

    expect(retrievalSummary.map((r) => r.type).sort()).toEqual(["multiple", "simple"]);
    const simple = retrievalSummary.find((r) => r.type === "simple");
    expect(simple).toMatchObject({
      source: "retriever_evaluation",
      ratel_ai_core_version: CORE,
      timestamp: TS,
      pool_size: 30,
      k: 1,
      n: 2,
      accuracy: 0.5, // one hit of two
    });
    expect(simple?.gold_similarity.mean).toBe(5); // (4 + 6) / 2
    expect(simple?.gold_similarity.coverage).toBe(1);
  });
});

describe("summarizeBfcl — task completion", () => {
  it("builds per-row records for every arm, joined to the corpus", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0" }), // ratel-full
      cell({ scenario_id: "bfcl-multiple-0", category: "bfcl-multiple", ast_verdict: "fail" }),
      cell({ scenario_id: "bfcl-simple-0", arm: "control-baseline" as Arm }),
    ];
    const { taskRows } = summarizeBfcl({ retrievalRows: [], cells, scenarios });

    expect(taskRows).toHaveLength(3); // all arms kept
    expect(new Set(taskRows.map((r) => r.arm))).toEqual(
      new Set(["ratel-full", "control-baseline"]),
    );
    const ratel = taskRows.find((r) => r.type === "simple" && r.arm === "ratel-full");
    expect(ratel).toMatchObject({
      model: "claude-haiku-4-5",
      query: "simple question",
      selection_pass: true,
      task_completion_pass: true,
    });
    expect(ratel?.true_answers.gold_tools).toEqual(["tool_a"]);
    expect(ratel?.llm_answer).toEqual([{ toolId: "tool_a", args: { x: 1 } }]);
  });

  it("can restrict to a single arm via the arm filter", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0" }), // ratel-full
      cell({ scenario_id: "bfcl-simple-0", arm: "control-baseline" as Arm }),
    ];
    const { taskRows } = summarizeBfcl({ retrievalRows: [], cells, scenarios, arm: "ratel-full" });
    expect(taskRows).toHaveLength(1);
    expect(taskRows[0].arm).toBe("ratel-full");
  });

  it("aggregates the task summary per (type, LLM, arm); accuracy null when no AST verdicts", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0", ast_verdict: "pass", programmatic_verdict: "pass" }),
      cell({ scenario_id: "bfcl-simple-1", ast_verdict: "fail", programmatic_verdict: "fail" }),
      cell({ scenario_id: "bfcl-multiple-0", category: "bfcl-multiple", ast_verdict: "n/a" }),
    ];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });

    const simple = taskSummary.find((r) => r.type === "simple");
    expect(simple).toMatchObject({
      source: "task_completion",
      model: "claude-haiku-4-5",
      arm: "ratel-full",
      scenarios: 2,
      selection_accuracy: 0.5,
      task_completion_accuracy: 0.5,
    });
    const multiple = taskSummary.find((r) => r.type === "multiple");
    expect(multiple?.task_completion_accuracy).toBeNull(); // only n/a verdict
  });

  it("computes argument recall as partial credit, and emits the 5-metric summary", () => {
    const sc: Scenario[] = [
      {
        id: "bfcl-simple-9",
        prompt: "q",
        candidate_pool: [],
        gold_tools: ["t"],
        category: "bfcl-simple",
        gold_calls: [{ tool: "t", args: { a: [1], b: [2] } }], // two required args
      },
    ];
    const c = cell({
      scenario_id: "bfcl-simple-9",
      tool_calls: [{ toolId: "t", args: { a: 1, b: 99 } }], // a right, b wrong
      effective_tool_ids: ["t"],
      wall_ms: 700,
    });
    const { taskRows, taskSummary } = summarizeBfcl({
      retrievalRows: [],
      cells: [c],
      scenarios: sc,
    });

    expect(taskRows[0].recall).toBe(0.5); // 1 of 2 required args
    const s = taskSummary[0];
    expect(s.recall).toBe(0.5);
    expect(s.latency_p50_ms).toBe(700);
    // exactly the five metrics (+ identity/dims + n), nothing extra
    expect(Object.keys(s).sort()).toEqual(
      [
        "arm",
        "latency_p50_ms",
        "mean_total_tokens",
        "model",
        "ratel_ai_core_version",
        "recall",
        "scenarios",
        "selection_accuracy",
        "source",
        "task_completion_accuracy",
        "timestamp",
        "type",
      ].sort(),
    );
  });

  it("groups the task summary separately per arm", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0", arm: "ratel-full" as Arm, ast_verdict: "pass" }),
      cell({ scenario_id: "bfcl-simple-0", arm: "control-baseline" as Arm, ast_verdict: "fail" }),
    ];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    const byArm = Object.fromEntries(
      taskSummary
        .filter((r) => r.type === "simple")
        .map((r) => [r.arm, r.task_completion_accuracy]),
    );
    expect(byArm).toEqual({ "ratel-full": 1, "control-baseline": 0 });
  });
});
