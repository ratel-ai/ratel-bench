import { describe, expect, it } from "vitest";
import { summarizeSragents } from "./sragents-summarize.js";
import type { SragentsRetrievalRow } from "./sragents-types.js";

const TS = "2026-06-22T00:00:00.000Z";
const CORE = "0.2.0";

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
