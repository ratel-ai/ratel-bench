import { describe, expect, it } from "vitest";
import { summarizeSragents } from "./sragents-summarize.js";
import type { SragentsRetrievalRow, SragentsSelectCell } from "./sragents-types.js";

const TS = "2026-06-22T00:00:00.000Z";
const CORE = "0.2.0";

function cell(over: Partial<SragentsSelectCell>): SragentsSelectCell {
  return {
    run_type: "skill_selection",
    generated_at: TS,
    ratel_ai_core_version: CORE,
    scenario_id: "sragents-toolqa_0",
    category: "sragents-toolqa",
    arm: "ratel-full",
    model: "gpt-5.4-mini",
    run_index: 0,
    pool_size: 50,
    candidate_count: 10,
    gold_skill_ids: ["toolqa_1"],
    selected_skill_ids: ["toolqa_1"],
    input_tokens: 1000,
    output_tokens: 20,
    total_tokens: 1020,
    dollar_cost: 0.001,
    wall_ms: 700,
    error: null,
    ...over,
  };
}

function retrievalRow(over: Partial<SragentsRetrievalRow>): SragentsRetrievalRow {
  return {
    generated_at: TS,
    ratel_ai_core_version: CORE,
    scenario_id: "sragents-bigcodebench_0",
    category: "sragents-bigcodebench",
    query: "q",
    golden_answer: ["bigcodebench_001"],
    retrieved: [{ id: "bigcodebench_001", score: 5 }],
    k: 1,
    target_pool_size: 100,
    pool_size: 100,
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

describe("summarizeSragents — retrieval summary", () => {
  it("buckets per dataset and emits a cross-dataset `all` aggregate", () => {
    const rows = [
      retrievalRow({ scenario_id: "sragents-bigcodebench_0", category: "sragents-bigcodebench" }),
      retrievalRow({
        scenario_id: "sragents-toolqa_0",
        category: "sragents-toolqa",
        gold_score: 7,
      }),
    ];
    const { retrievalSummary } = summarizeSragents({ retrievalRows: rows });

    expect(retrievalSummary.map((r) => r.dataset)).toEqual(["bigcodebench", "toolqa", "all"]);
    const all = retrievalSummary.find((r) => r.dataset === "all");
    expect(all).toMatchObject({
      source: "retriever_evaluation",
      ratel_ai_core_version: CORE,
      timestamp: TS,
      pool_size: 100,
      k: 1,
      n: 2, // both datasets roll up
    });
    expect(all?.gold_similarity.mean).toBe(6); // (5 + 7) / 2
  });

  it("emits one flat row per (dataset, pool_size, k) with gold-similarity stats", () => {
    const rows = [
      retrievalRow({ scenario_id: "sragents-champ_0", category: "sragents-champ", gold_score: 4 }),
      retrievalRow({
        scenario_id: "sragents-champ_1",
        category: "sragents-champ",
        gold_score: 6,
        recall_at_k: 0,
        hit_at_k: false,
      }),
    ];
    const { retrievalSummary } = summarizeSragents({ retrievalRows: rows });

    const champ = retrievalSummary.find((r) => r.dataset === "champ");
    expect(champ).toMatchObject({ pool_size: 100, k: 1, n: 2, accuracy: 0.5 });
    expect(champ?.gold_similarity.mean).toBe(5); // (4 + 6) / 2
    expect(champ?.gold_similarity.coverage).toBe(1);
  });

  it("separates complete_rate from accuracy for multi-gold instances", () => {
    const rows = [
      retrievalRow({
        scenario_id: "sragents-theoremqa_0",
        category: "sragents-theoremqa",
        gold_count: 2,
        recall_at_k: 0.5,
        hit_at_k: true,
        complete_at_k: false, // hit but not complete
      }),
    ];
    const { retrievalSummary } = summarizeSragents({ retrievalRows: rows });
    const ds = retrievalSummary.find((r) => r.dataset === "theoremqa");
    expect(ds?.accuracy).toBe(1); // hit@K
    expect(ds?.complete_rate).toBe(0); // not every gold in top-K
  });

  it("dedupes gold_score per (scenario, pool) across k", () => {
    const rows = [
      retrievalRow({
        scenario_id: "sragents-toolqa_0",
        category: "sragents-toolqa",
        k: 1,
        gold_score: 9,
      }),
      retrievalRow({
        scenario_id: "sragents-toolqa_0",
        category: "sragents-toolqa",
        k: 3,
        gold_score: 9,
      }),
    ];
    const { retrievalSummary } = summarizeSragents({ retrievalRows: rows });
    // Two k buckets, but gold_similarity within each is one deduped scenario.
    const k1 = retrievalSummary.find((r) => r.dataset === "toolqa" && r.k === 1);
    expect(k1?.gold_similarity.mean).toBe(9);
    expect(k1?.gold_similarity.coverage).toBe(1);
  });

  it("ignores non-sragents rows", () => {
    const rows = [
      retrievalRow({ scenario_id: "bfcl-simple-0", category: "bfcl-simple" }),
      retrievalRow({ scenario_id: "sragents-toolqa_0", category: "sragents-toolqa" }),
    ];
    const { retrievalSummary } = summarizeSragents({ retrievalRows: rows });
    expect(retrievalSummary.map((r) => r.dataset).sort()).toEqual(["all", "toolqa"]);
  });
});

describe("summarizeSragents — skill selection (task)", () => {
  it("computes per-row selection metrics (hit, complete, recall, precision)", () => {
    const cells = [
      // single-gold, exact hit → all 1.0
      cell({ scenario_id: "sragents-toolqa_0", gold_skill_ids: ["a"], selected_skill_ids: ["a"] }),
      // multi-gold, partial: 1 of 2 gold, 1 extra → recall .5, precision .5, complete=false
      cell({
        scenario_id: "sragents-champ_0",
        category: "sragents-champ",
        gold_skill_ids: ["a", "b"],
        selected_skill_ids: ["a", "x"],
      }),
    ];
    const { taskRows } = summarizeSragents({ retrievalRows: [], cells });
    const champ = taskRows.find((r) => r.dataset === "champ");
    expect(champ).toMatchObject({
      selection_pass: true,
      task_completion_pass: false, // not every gold selected
      recall: 0.5,
      precision: 0.5,
    });
    const toolqa = taskRows.find((r) => r.dataset === "toolqa");
    expect(toolqa).toMatchObject({ task_completion_pass: true, recall: 1, precision: 1 });
  });

  it("aggregates per (dataset, model, arm) + an `all` rollup", () => {
    const cells = [
      cell({
        scenario_id: "sragents-toolqa_0",
        arm: "ratel-full",
        selected_skill_ids: ["toolqa_1"],
      }),
      cell({
        scenario_id: "sragents-toolqa_1",
        arm: "ratel-full",
        gold_skill_ids: ["toolqa_9"],
        selected_skill_ids: [], // miss
      }),
      cell({
        scenario_id: "sragents-champ_0",
        category: "sragents-champ",
        arm: "ratel-full",
        gold_skill_ids: ["c"],
        selected_skill_ids: ["c"],
      }),
    ];
    const { taskSummary } = summarizeSragents({ retrievalRows: [], cells });
    const datasets = taskSummary
      .filter((r) => r.arm === "ratel-full")
      .map((r) => r.dataset)
      .sort();
    expect(datasets).toEqual(["all", "champ", "toolqa"]);
    const toolqa = taskSummary.find((r) => r.dataset === "toolqa");
    expect(toolqa).toMatchObject({
      source: "task_completion",
      scenarios: 2,
      selection_accuracy: 0.5,
    });
    const all = taskSummary.find((r) => r.dataset === "all");
    expect(all?.scenarios).toBe(3); // every cell rolls up
  });

  it("separates control-baseline from ratel-full under the same model", () => {
    const cells = [
      cell({ arm: "control-baseline", selected_skill_ids: [] }), // miss
      cell({ arm: "ratel-full", selected_skill_ids: ["toolqa_1"] }), // hit
    ];
    const { taskSummary } = summarizeSragents({ retrievalRows: [], cells });
    const base = taskSummary.find((r) => r.arm === "control-baseline" && r.dataset === "toolqa");
    const ratel = taskSummary.find((r) => r.arm === "ratel-full" && r.dataset === "toolqa");
    expect(base?.selection_accuracy).toBe(0);
    expect(ratel?.selection_accuracy).toBe(1);
  });
});

describe("summarizeSragents — errored cells", () => {
  const LATER = "2026-06-23T00:00:00.000Z";
  const TRANSIENT = "Failed after 3 attempts. Last error: Overloaded";
  const miss = { selected_skill_ids: [] as string[] };
  const toolqa = (cells: SragentsSelectCell[]) =>
    summarizeSragents({ retrievalRows: [], cells }).taskSummary.find((r) => r.dataset === "toolqa");

  it("(a) supersedes a transient row with a later good row of the same cell", () => {
    const cells = [cell({ ...miss, error: TRANSIENT }), cell({ generated_at: LATER })];
    const { taskRows } = summarizeSragents({ retrievalRows: [], cells });
    expect(taskRows).toHaveLength(1);
    expect(taskRows[0].error_class).toBeNull();
    expect(toolqa(cells)).toMatchObject({ scenarios: 1, selection_accuracy: 1, excluded_cells: 0 });
  });

  it.each<Partial<SragentsSelectCell>>([
    { ratel_ai_core_version: "0.3.0-rc.1" },
    { model: "claude-haiku-4-5" },
    { pool_size: 100 },
    { pool_size: null },
    { run_index: 1 },
  ])("(a) keys supersede on label/model/pool_size/run_index: %o is its own cell", (variant) => {
    const cells = [cell({}), cell({ ...miss, ...variant })];
    const { taskRows } = summarizeSragents({ retrievalRows: [], cells });
    expect(taskRows).toHaveLength(2);
  });

  it("(a) keys supersede on the label too: same cell under two labels counts in both", () => {
    const cells = [cell({}), cell({ ratel_ai_core_version: "0.3.0-rc.1", ...miss })];
    const { taskSummary } = summarizeSragents({ retrievalRows: [], cells });
    expect(
      taskSummary
        .filter((s) => s.dataset === "toolqa")
        .map((s) => [s.ratel_ai_core_version, s.scenarios]),
    ).toEqual([
      ["0.2.0", 1],
      ["0.3.0-rc.1", 1],
    ]);
  });

  it("duplicate ratel-full rows count once", () => {
    const cells = [cell({ ...miss }), cell({ generated_at: LATER })];
    expect(toolqa(cells)).toMatchObject({ scenarios: 1, selection_accuracy: 1 });
  });

  it("(b) excludes final transient|access rows from every metric and counts them", () => {
    const cells = [
      cell({ scenario_id: "sragents-toolqa_0" }),
      cell({
        scenario_id: "sragents-toolqa_1",
        ...miss,
        error: TRANSIENT,
        total_tokens: 9,
        finish_reason: "length", // excluded, so not counted as truncated
      }),
      cell({ scenario_id: "sragents-toolqa_2", ...miss, error: "gated", error_class: "access" }),
    ];
    expect(toolqa(cells)).toMatchObject({
      scenarios: 1,
      excluded_cells: 2,
      errored_cells: 0,
      truncated_cells: 0,
      selection_accuracy: 1,
      task_completion_accuracy: 1,
      mean_total_tokens: 1020,
    });
  });

  it("(c) request|timeout|outcome rows stay scored fails and are counted as errored", () => {
    const cells = [
      cell({ scenario_id: "sragents-toolqa_0" }),
      cell({ scenario_id: "sragents-toolqa_1", ...miss, error: "400", error_class: "request" }),
      cell({ scenario_id: "sragents-toolqa_2", ...miss, error: "run timed out after 300000ms" }),
      cell({
        scenario_id: "sragents-toolqa_3",
        ...miss,
        error: "No object generated: could not parse the response.",
      }),
    ];
    expect(toolqa(cells)).toMatchObject({
      scenarios: 4,
      errored_cells: 3,
      excluded_cells: 0,
      selection_accuracy: 0.25,
    });
  });

  it("(d) token and latency means use only non-errored rows", () => {
    const cells = [
      cell({ scenario_id: "sragents-toolqa_0", total_tokens: 1000, wall_ms: 800 }),
      cell({
        scenario_id: "sragents-toolqa_1",
        ...miss,
        error: "No object generated: response did not match schema.",
        total_tokens: 50_000,
        wall_ms: 90_000,
      }),
    ];
    expect(toolqa(cells)).toMatchObject({
      scenarios: 2,
      mean_total_tokens: 1000,
      latency_p50_ms: 800,
    });
  });

  it("(e) counts finish_reason 'length' cells as truncated, kept and scored on their verdict", () => {
    const cells = [
      cell({
        scenario_id: "sragents-toolqa_0",
        ...miss,
        finish_reason: "length",
        error: "No object generated: could not parse the response.",
        error_class: "outcome",
      }),
      cell({ scenario_id: "sragents-toolqa_1", finish_reason: "stop" }),
    ];
    const { taskRows } = summarizeSragents({ retrievalRows: [], cells });
    expect(taskRows.map((r) => r.truncated)).toEqual([true, false]);
    expect(toolqa(cells)).toMatchObject({
      scenarios: 2,
      truncated_cells: 1,
      selection_accuracy: 0.5,
    });
  });

  it("(f) the group timestamp covers every row, excluded ones included", () => {
    const cells = [
      cell({ scenario_id: "sragents-toolqa_0" }),
      cell({ scenario_id: "sragents-toolqa_1", ...miss, error: TRANSIENT, generated_at: LATER }),
    ];
    expect(toolqa(cells)?.timestamp).toBe(LATER);
    const all = summarizeSragents({ retrievalRows: [], cells }).taskSummary.find(
      (r) => r.dataset === "all",
    );
    expect(all?.timestamp).toBe(LATER);
  });

  it("(f2) the group timestamp covers superseded rows too", () => {
    const cells = [cell({}), cell({ ...miss, error: TRANSIENT, generated_at: LATER })];
    const { taskRows } = summarizeSragents({ retrievalRows: [], cells });
    expect(taskRows).toHaveLength(1);
    expect(taskRows[0].error_class).toBeNull();
    expect(toolqa(cells)?.timestamp).toBe(LATER);
  });

  it("(g) a group with no kept rows emits null metrics", () => {
    const cells = [cell({ ...miss, error: TRANSIENT })];
    expect(toolqa(cells)).toMatchObject({
      scenarios: 0,
      excluded_cells: 1,
      task_completion_accuracy: null,
      selection_accuracy: null,
      recall: null,
      precision: null,
      mean_total_tokens: null,
      latency_p50_ms: null,
    });
  });
});

describe("summarizeSragents — max_output_tokens provenance", () => {
  const TRANSIENT = "Failed after 3 attempts. Last error: Overloaded";
  const other = { scenario_id: "sragents-toolqa_1" };
  const groups = (cells: SragentsSelectCell[]) =>
    summarizeSragents({ retrievalRows: [], cells }).taskSummary;
  const toolqa = (cells: SragentsSelectCell[]) =>
    groups(cells).find((r) => r.dataset === "toolqa")?.max_output_tokens;

  it("number when every kept row shares one cap (dataset and `all` rollup); on each task row", () => {
    const cells = [cell({ max_output_tokens: 4096 }), cell({ ...other, max_output_tokens: 4096 })];
    const { taskRows } = summarizeSragents({ retrievalRows: [], cells });
    expect(taskRows.map((r) => r.max_output_tokens)).toEqual([4096, 4096]);
    expect(groups(cells).map((g) => [g.dataset, g.max_output_tokens])).toEqual([
      ["toolqa", 4096],
      ["all", 4096],
    ]);
  });

  it("'mixed' when kept rows differ, legacy (unrecorded) rows included", () => {
    expect(toolqa([cell({ max_output_tokens: 4096 }), cell({ ...other })])).toBe("mixed");
  });

  it("null when no kept row recorded a cap, or none is kept; excluded rows don't count", () => {
    expect(toolqa([cell({}), cell({ ...other, max_output_tokens: null })])).toBeNull();
    expect(toolqa([cell({ error: TRANSIENT, max_output_tokens: 4096 })])).toBeNull();
    expect(
      toolqa([
        cell({ max_output_tokens: 4096 }),
        cell({ ...other, error: TRANSIENT, max_output_tokens: 8 }),
      ]),
    ).toBe(4096);
  });
});

describe("summarizeSragents — retries", () => {
  const TRANSIENT = "Failed after 8 attempts. Last error: Overloaded";
  const POLICY = "a8/b2000/c60000/w180000;timeout=active:300000+g30000";
  const retried = (over: Partial<SragentsSelectCell>) =>
    cell({ retries: 2, throttled_retries: 1, retry_wait_ms: 3000, retry_policy: POLICY, ...over });
  const toolqa = (cells: SragentsSelectCell[]) =>
    summarizeSragents({ retrievalRows: [], cells }).taskSummary.find((r) => r.dataset === "toolqa");

  it("carries the retry fields on each task row (null for legacy rows)", () => {
    const cells = [retried({}), cell({ scenario_id: "sragents-toolqa_1" })];
    const { taskRows } = summarizeSragents({ retrievalRows: [], cells });
    expect(
      taskRows.map((r) => [r.retries, r.throttled_retries, r.retry_wait_ms, r.retry_policy]),
    ).toEqual([
      [2, 1, 3000, POLICY],
      [null, null, null, null],
    ]);
  });

  it("latency_p50_net_ms = median(wall_ms − retry_wait_ms); [guard] latency_p50_ms stays wall-clock", () => {
    const s = toolqa([
      retried({ wall_ms: 5000 }), // net 2000
      retried({ scenario_id: "sragents-toolqa_1", wall_ms: 4000, retry_wait_ms: 0 }), // net 4000
      cell({ scenario_id: "sragents-toolqa_2", wall_ms: 1000 }), // legacy: net = wall
      // Kept (a scored timeout) but not clean: no latency metric reads it.
      cell({
        scenario_id: "sragents-toolqa_3",
        wall_ms: 400_000,
        error: "run timed out after 300000ms",
      }),
    ]);
    expect(s?.errored_cells).toBe(1);
    expect(s?.latency_p50_ms).toBe(4000);
    expect(s?.latency_p50_net_ms).toBe(2000);
  });

  it("sums retries over every superseding row (excluded too); retry_policy over kept rows, 'mixed' beside legacy", () => {
    const s = toolqa([
      retried({}),
      retried({
        scenario_id: "sragents-toolqa_1",
        error: TRANSIENT,
        retries: 7,
        throttled_retries: 7,
        retry_policy: undefined, // excluded: retry_policy must not read it (else "mixed")
      }),
    ]);
    expect(s).toMatchObject({
      excluded_cells: 1,
      retries: 9,
      throttled_retries: 8,
      retry_policy: POLICY,
    });
    expect(toolqa([retried({}), cell({ scenario_id: "sragents-toolqa_1" })])?.retry_policy).toBe(
      "mixed",
    );
    expect(toolqa([cell({})])).toMatchObject({
      retries: null,
      throttled_retries: null,
      retry_policy: null,
    });
  });

  it("retries: a lower bound beside legacy rows, which retry_policy (kept rows only) may not flag", () => {
    expect(toolqa([retried({}), cell({ scenario_id: "sragents-toolqa_1" })])).toMatchObject({
      retries: 2,
      throttled_retries: 1,
      retry_policy: "mixed",
    });
    expect(
      toolqa([retried({}), cell({ scenario_id: "sragents-toolqa_1", error: TRANSIENT })]),
    ).toMatchObject({ excluded_cells: 1, retries: 2, throttled_retries: 1, retry_policy: POLICY });
    expect(toolqa([retried({ error: TRANSIENT })])).toMatchObject({
      excluded_cells: 1,
      retries: 2,
      retry_policy: null,
    });
  });
});

describe("summarizeSragents — label filter", () => {
  it("emits only the given label's task and retrieval groups", () => {
    const cells = [
      cell({ ratel_ai_core_version: "0.2.0" }),
      cell({ ratel_ai_core_version: "0.4.0-sparse" }),
    ];
    const retrievalRows = [
      retrievalRow({ ratel_ai_core_version: "0.2.0" }),
      retrievalRow({ ratel_ai_core_version: "0.4.0-sparse", hit_at_k: false }),
    ];
    const { retrievalSummary, taskRows, taskSummary } = summarizeSragents({
      retrievalRows,
      cells,
      label: "0.4.0-sparse",
    });
    expect(taskRows.map((r) => r.ratel_ai_core_version)).toEqual(["0.4.0-sparse"]);
    expect(new Set(taskSummary.map((r) => r.ratel_ai_core_version))).toEqual(
      new Set(["0.4.0-sparse"]),
    );
    expect(
      new Set(retrievalSummary.map((r) => [r.ratel_ai_core_version, r.accuracy].join())),
    ).toEqual(new Set(["0.4.0-sparse,0"]));
  });
});
