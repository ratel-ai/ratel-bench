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
    // exactly the five metrics (+ identity/dims + n + error counters + cap/retry provenance), nothing extra
    expect(Object.keys(s).sort()).toEqual(
      [
        "arm",
        "errored_cells",
        "excluded_cells",
        "latency_p50_ms",
        "latency_p50_net_ms",
        "max_output_tokens",
        "retries",
        "retry_policy",
        "throttled_retries",
        "mean_total_tokens",
        "model",
        "ratel_ai_core_version",
        "recall",
        "scenarios",
        "selection_accuracy",
        "source",
        "task_completion_accuracy",
        "timestamp",
        "truncated_cells",
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

describe("summarizeBfcl — errored cells", () => {
  const LATER = "2026-06-23T00:00:00.000Z";
  const TRANSIENT = "Failed after 3 attempts. Last error: Internal server error";
  const failed: Partial<CellResult> = {
    programmatic_verdict: "fail",
    ast_verdict: "fail",
    tool_calls: [],
  };

  it("(a) supersedes a transient row with a later good row of the same cell", () => {
    const cells = [
      cell({ ...failed, error: TRANSIENT }),
      cell({ generated_at: LATER }), // re-run, passes
    ];
    const { taskRows, taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskRows).toHaveLength(1);
    expect(taskRows[0].error_class).toBeNull();
    expect(taskSummary[0]).toMatchObject({
      scenarios: 1,
      selection_accuracy: 1,
      excluded_cells: 0,
    });
  });

  it.each<Partial<CellResult>>([
    { ratel_ai_core_version: "0.3.0-rc.1" },
    { ratel_version: "other" },
    { model: "claude-sonnet-4-6" },
    { run_index: 1 },
    { pool_size: 30 },
    { pool_size: null },
  ])("(a) keys supersede on label/version/model/run/pool: %o is its own cell", (variant) => {
    const cells = [cell({}), cell({ ...failed, ...variant })];
    const { taskRows } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskRows).toHaveLength(2);
  });

  it("(a) keys supersede on the label too: same cell under two labels counts in both", () => {
    const cells = [
      cell({ ratel_ai_core_version: "0.2.0" }),
      cell({ ratel_ai_core_version: "0.3.0-rc.1", ...failed }),
    ];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskSummary.map((s) => [s.ratel_ai_core_version, s.scenarios])).toEqual([
      ["0.2.0", 1],
      ["0.3.0-rc.1", 1],
    ]);
  });

  it("(b) excludes final transient|access rows from every metric and counts them", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0" }),
      // A step cut off by the output limit before the throw: excluded, so not counted as truncated.
      cell({
        scenario_id: "bfcl-simple-1",
        ...failed,
        error: TRANSIENT,
        total_tokens: 9,
        truncated_steps: 1,
      }),
      cell({ scenario_id: "bfcl-simple-2", ...failed, error: "boom", error_class: "access" }),
    ];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskSummary[0]).toMatchObject({
      scenarios: 1,
      excluded_cells: 2,
      errored_cells: 0,
      truncated_cells: 0,
      selection_accuracy: 1,
      task_completion_accuracy: 1,
      mean_total_tokens: 1050,
    });
  });

  it("(c) request|timeout|outcome rows stay scored fails and are counted as errored", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0" }),
      cell({ scenario_id: "bfcl-simple-1", ...failed, error: "400", error_class: "request" }),
      cell({ scenario_id: "bfcl-simple-2", ...failed, error: "run timed out after 180000ms" }),
      cell({
        scenario_id: "bfcl-simple-3",
        ...failed,
        error: "Output blocked by content filtering policy",
      }),
    ];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskSummary[0]).toMatchObject({
      scenarios: 4,
      errored_cells: 3,
      excluded_cells: 0,
      selection_accuracy: 0.25,
    });
  });

  it("(d) token and latency means use only non-errored rows", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0", total_tokens: 1000, wall_ms: 800 }),
      cell({
        scenario_id: "bfcl-simple-1",
        ...failed,
        error: "run timed out after 180000ms",
        total_tokens: 50_000,
        wall_ms: 180_000,
      }),
    ];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskSummary[0]).toMatchObject({
      scenarios: 2,
      mean_total_tokens: 1000,
      latency_p50_ms: 800,
    });
  });

  it("(e) counts truncated cells (truncated_steps, or legacy finish_reason 'length'), kept and scored on their verdict", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0", ...failed, truncated_steps: 1 }),
      cell({ scenario_id: "bfcl-simple-1", ...failed, finish_reason: "length" }),
      cell({ scenario_id: "bfcl-simple-2", truncated_steps: 0 }),
    ];
    const { taskRows, taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskRows.map((r) => r.truncated)).toEqual([true, true, false]);
    expect(taskSummary[0]).toMatchObject({
      scenarios: 3,
      truncated_cells: 2,
      errored_cells: 0,
      selection_accuracy: 1 / 3,
    });
  });

  it("(e) a truncated cell with passing verdicts counts as truncated AND as a pass", () => {
    const cells = [cell({ finish_reason: "length" })];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskSummary[0]).toMatchObject({
      truncated_cells: 1,
      errored_cells: 0,
      selection_accuracy: 1,
      task_completion_accuracy: 1,
    });
  });

  it("(f) the group timestamp covers every row, excluded ones included", () => {
    const cells = [
      cell({ scenario_id: "bfcl-simple-0" }),
      cell({ scenario_id: "bfcl-simple-1", ...failed, error: TRANSIENT, generated_at: LATER }),
    ];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskSummary[0].timestamp).toBe(LATER);
  });

  it("(f2) the group timestamp covers superseded rows too", () => {
    const cells = [cell({}), cell({ ...failed, error: TRANSIENT, generated_at: LATER })];
    const { taskRows, taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskRows).toHaveLength(1);
    expect(taskRows[0].error_class).toBeNull();
    expect(taskSummary[0].timestamp).toBe(LATER);
  });

  it("(g) a group with no kept rows emits null metrics", () => {
    const cells = [cell({ ...failed, error: TRANSIENT })];
    const { taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskSummary[0]).toMatchObject({
      scenarios: 0,
      excluded_cells: 1,
      task_completion_accuracy: null,
      selection_accuracy: null,
      recall: null,
      mean_total_tokens: null,
      latency_p50_ms: null,
    });
  });
});

describe("summarizeBfcl — max_output_tokens provenance", () => {
  const TRANSIENT = "Failed after 3 attempts. Last error: Internal server error";
  const other = { scenario_id: "bfcl-simple-1" };
  const summary = (cells: CellResult[]) =>
    summarizeBfcl({ retrievalRows: [], cells, scenarios }).taskSummary[0];

  it("number when every kept row shares one cap; carried on each task row", () => {
    const cells = [cell({ max_output_tokens: 4096 }), cell({ ...other, max_output_tokens: 4096 })];
    const { taskRows, taskSummary } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(taskRows.map((r) => r.max_output_tokens)).toEqual([4096, 4096]);
    expect(taskSummary[0].max_output_tokens).toBe(4096);
  });

  it("'mixed' when kept rows differ, legacy (unrecorded) rows included", () => {
    expect(summary([cell({ max_output_tokens: 4096 }), cell({ ...other })]).max_output_tokens).toBe(
      "mixed",
    );
    expect(
      summary([cell({ max_output_tokens: 4096 }), cell({ ...other, max_output_tokens: 16 })])
        .max_output_tokens,
    ).toBe("mixed");
  });

  it("null when no kept row recorded a cap (legacy or uncapped) or none is kept", () => {
    expect(summary([cell({}), cell({ ...other, max_output_tokens: null })]).max_output_tokens).toBe(
      null,
    );
    expect(summary([cell({ error: TRANSIENT, max_output_tokens: 4096 })]).max_output_tokens).toBe(
      null,
    );
  });

  it("counts only superseding, non-excluded rows", () => {
    const cells = [
      cell({ error: TRANSIENT, max_output_tokens: 16 }), // superseded by the re-run
      cell({ generated_at: "2026-06-23T00:00:00.000Z", max_output_tokens: 4096 }),
      cell({ ...other, error: TRANSIENT, max_output_tokens: 16 }), // final, excluded
    ];
    expect(summary(cells).max_output_tokens).toBe(4096);
  });
});

describe("summarizeBfcl — retries", () => {
  const TRANSIENT = "Failed after 8 attempts. Last error: Overloaded";
  const POLICY = "a8/b2000/c60000/w180000;timeout=active:180000+g30000";
  const retried = (over: Partial<CellResult>) =>
    cell({ retries: 2, throttled_retries: 1, retry_wait_ms: 3000, retry_policy: POLICY, ...over });

  it("carries the retry fields on each task row (null for legacy rows)", () => {
    const cells = [retried({ wall_ms: 5000 }), cell({ scenario_id: "bfcl-simple-1" })];
    const { taskRows } = summarizeBfcl({ retrievalRows: [], cells, scenarios });
    expect(
      taskRows.map((r) => [r.retries, r.throttled_retries, r.retry_wait_ms, r.retry_policy]),
    ).toEqual([
      [2, 1, 3000, POLICY],
      [null, null, null, null],
    ]);
  });

  it("latency_p50_net_ms = median(wall_ms − retry_wait_ms); [guard] latency_p50_ms stays wall-clock", () => {
    const cells = [
      retried({ scenario_id: "bfcl-simple-0", wall_ms: 5000 }), // net 2000
      retried({ scenario_id: "bfcl-simple-1", wall_ms: 4000, retry_wait_ms: 0 }), // net 4000
      cell({ scenario_id: "bfcl-simple-2", wall_ms: 1000 }), // legacy: net = wall
      // Kept (a scored timeout) but not clean: no latency metric reads it.
      cell({
        scenario_id: "bfcl-simple-3",
        wall_ms: 200_000,
        error: "run timed out after 180000ms",
      }),
    ];
    const [s] = summarizeBfcl({ retrievalRows: [], cells, scenarios }).taskSummary;
    expect(s.errored_cells).toBe(1);
    expect(s.latency_p50_ms).toBe(4000);
    expect(s.latency_p50_net_ms).toBe(2000);
  });

  it("sums retries/throttled over every superseding row (excluded ones too); retry_policy over kept rows", () => {
    const cells = [
      retried({ scenario_id: "bfcl-simple-0" }),
      // Excluded, with no policy: retry_policy must not read it (else "mixed").
      retried({
        scenario_id: "bfcl-simple-1",
        error: TRANSIENT,
        retries: 7,
        throttled_retries: 7,
        retry_policy: undefined,
      }),
    ];
    const [s] = summarizeBfcl({ retrievalRows: [], cells, scenarios }).taskSummary;
    expect(s).toMatchObject({ excluded_cells: 1, retries: 9, throttled_retries: 8 });
    expect(s.retry_policy).toBe(POLICY);
  });

  it("retry_policy is 'mixed' beside legacy rows, null when none recorded one", () => {
    const summary = (cells: CellResult[]) =>
      summarizeBfcl({ retrievalRows: [], cells, scenarios }).taskSummary[0];
    expect(summary([retried({}), cell({ scenario_id: "bfcl-simple-1" })]).retry_policy).toBe(
      "mixed",
    );
    const legacy = summary([cell({}), cell({ scenario_id: "bfcl-simple-1" })]);
    expect(legacy).toMatchObject({ retry_policy: null, retries: null, throttled_retries: null });
    expect(legacy.latency_p50_net_ms).toBe(legacy.latency_p50_ms);
  });

  it("retries: a lower bound beside legacy rows, which retry_policy (kept rows only) may not flag", () => {
    const summary = (cells: CellResult[]) =>
      summarizeBfcl({ retrievalRows: [], cells, scenarios }).taskSummary[0];
    // Kept legacy row beside a U4 row: only the recorded counts, retry_policy "mixed".
    expect(summary([retried({}), cell({ scenario_id: "bfcl-simple-1" })])).toMatchObject({
      retries: 2,
      throttled_retries: 1,
      retry_policy: "mixed",
    });
    // Excluded legacy row beside a kept U4 row: still partial, but a single policy.
    expect(
      summary([retried({}), cell({ scenario_id: "bfcl-simple-1", error: TRANSIENT })]),
    ).toMatchObject({ excluded_cells: 1, retries: 2, throttled_retries: 1, retry_policy: POLICY });
    // Every U4 row excluded: counts recorded, no kept row to carry a policy.
    expect(summary([retried({ error: TRANSIENT })])).toMatchObject({
      excluded_cells: 1,
      retries: 2,
      retry_policy: null,
    });
  });
});

describe("summarizeBfcl — version-split cells", () => {
  it("flags label cells whose rows span more than one ratel_version (each counted once per version)", () => {
    const cells = [
      cell({
        ratel_version: "0.3.0-rc.1",
        error: "Failed after 3 attempts. Last error: overloaded",
      }),
      cell({ ratel_version: "0.1.5", generated_at: "2026-06-23T00:00:00.000Z" }),
      cell({ scenario_id: "bfcl-simple-1" }),
      cell({ scenario_id: "bfcl-simple-1", run_index: 1 }), // --runs 2: not a split
      cell({ scenario_id: "bfcl-simple-1", ratel_ai_core_version: "0.4.0" }), // other label
    ];
    const { taskSummary, versionSplitCells } = summarizeBfcl({
      retrievalRows: [],
      cells,
      scenarios,
    });
    // The re-drain at another SDK version did not supersede the transient row.
    expect(taskSummary.find((s) => s.ratel_ai_core_version === CORE)).toMatchObject({
      scenarios: 3,
      excluded_cells: 1,
    });
    expect(versionSplitCells).toEqual([
      {
        label: CORE,
        arm: "ratel-full",
        model: "claude-haiku-4-5",
        scenario_id: "bfcl-simple-0",
        versions: { "0.3.0-rc.1": 1, "0.1.5": 1 },
      },
    ]);
  });

  it("checks only the selected rows", () => {
    const cells = [cell({ ratel_version: "a" }), cell({ ratel_version: "b" })];
    expect(
      summarizeBfcl({ retrievalRows: [], cells, scenarios, label: "0.4.0" }).versionSplitCells,
    ).toEqual([]);
  });
});

describe("summarizeBfcl — label filter", () => {
  it("emits only the given label's task and retrieval groups", () => {
    const cells = [
      cell({ ratel_ai_core_version: "0.2.0" }),
      cell({ ratel_ai_core_version: "0.4.0-sparse" }),
    ];
    const retrievalRows = [
      retrievalRow({ ratel_ai_core_version: "0.2.0" }),
      retrievalRow({ ratel_ai_core_version: "0.4.0-sparse", hit_at_k: false }),
    ];
    const { retrievalSummary, taskRows, taskSummary } = summarizeBfcl({
      retrievalRows,
      cells,
      scenarios,
      label: "0.4.0-sparse",
    });
    expect(taskRows.map((r) => r.ratel_ai_core_version)).toEqual(["0.4.0-sparse"]);
    expect(taskSummary.map((r) => r.ratel_ai_core_version)).toEqual(["0.4.0-sparse"]);
    expect(retrievalSummary.map((r) => [r.ratel_ai_core_version, r.accuracy])).toEqual([
      ["0.4.0-sparse", 0],
    ]);
  });
});
