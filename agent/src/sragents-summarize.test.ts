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
