import { describe, expect, it } from "vitest";
import { buildReport } from "./bfcl-report.js";
import type { RetrievalSummaryRow, TaskSummaryRow } from "./bfcl-types.js";

function retr(over: Partial<RetrievalSummaryRow>): RetrievalSummaryRow {
  return {
    timestamp: "2026-06-22T00:00:00.000Z",
    ratel_ai_core_version: "0.2.0",
    source: "retriever_evaluation",
    type: "simple",
    pool_size: 30,
    k: 1,
    n: 100,
    mean_precision: 1,
    median_precision: 1,
    mean_recall: 1,
    median_recall: 1,
    mean_mrr: 1,
    median_mrr: 1,
    mean_ndcg: 1,
    median_ndcg: 1,
    accuracy: 0.9,
    complete_rate: 0.9,
    gold_similarity: { mean: 5, median: 5, stddev: 1, coverage: 1 },
    ...over,
  };
}

function task(over: Partial<TaskSummaryRow>): TaskSummaryRow {
  return {
    timestamp: "2026-06-22T00:00:00.000Z",
    ratel_ai_core_version: "0.2.0",
    source: "task_completion",
    model: "claude-haiku-4-5",
    arm: "ratel-full",
    type: "simple",
    scenarios: 50,
    task_completion_accuracy: 0.7,
    selection_accuracy: 0.8,
    recall: 0.85,
    mean_total_tokens: 1100,
    latency_p50_ms: 900,
    ...over,
  };
}

const NOW = "2026-06-22T12:00:00.000Z";

describe("buildReport", () => {
  it("keys by ratel-ai-core version → source → model → type", () => {
    const report = buildReport([retr({})], [task({})], NOW);
    expect(Object.keys(report.ratel_versions)).toEqual(["0.2.0"]);
    const v = report.ratel_versions["0.2.0"];
    expect(v.retriever_evaluation.simple.metrics).toHaveLength(1);
    expect(v.task_completion["claude-haiku-4-5"]["ratel-full"].simple.metrics).toMatchObject({
      selection_accuracy: 0.8,
      task_completion_accuracy: 0.7,
    });
    // group-level fields (incl. arm) are lifted out of per-row metrics
    const m = v.task_completion["claude-haiku-4-5"]["ratel-full"].simple.metrics;
    expect(m).not.toHaveProperty("timestamp");
    expect(m).not.toHaveProperty("arm");
  });

  it("breaks task completion down per arm under each LLM", () => {
    const report = buildReport(
      [],
      [
        task({ arm: "control-baseline", task_completion_accuracy: 0.6, mean_total_tokens: 30000 }),
        task({ arm: "control-oracle", task_completion_accuracy: 0.95 }),
        task({ arm: "ratel-full", task_completion_accuracy: 0.84, mean_total_tokens: 3500 }),
      ],
      NOW,
    );
    const tc = report.ratel_versions["0.2.0"].task_completion["claude-haiku-4-5"];
    expect(Object.keys(tc).sort()).toEqual(["control-baseline", "control-oracle", "ratel-full"]);
    expect(tc["control-baseline"].simple.metrics).toMatchObject({ mean_total_tokens: 30000 });
    expect(tc["ratel-full"].simple.metrics).toMatchObject({ mean_total_tokens: 3500 });
  });

  it("takes the latest timestamp per (version, source, model, type)", () => {
    const older = task({ timestamp: "2026-06-20T00:00:00.000Z", task_completion_accuracy: 0.5 });
    const newer = task({ timestamp: "2026-06-25T00:00:00.000Z", task_completion_accuracy: 0.95 });
    const report = buildReport([], [older, newer], NOW);
    const entry =
      report.ratel_versions["0.2.0"].task_completion["claude-haiku-4-5"]["ratel-full"].simple;
    expect(entry.timestamp).toBe("2026-06-25T00:00:00.000Z");
    expect(entry.metrics).toMatchObject({ task_completion_accuracy: 0.95 });
  });

  it("adds a new model while updating the existing one (same version)", () => {
    const haikuOld = task({
      model: "claude-haiku-4-5",
      timestamp: "2026-06-20T00:00:00.000Z",
      task_completion_accuracy: 0.5,
    });
    const haikuNew = task({
      model: "claude-haiku-4-5",
      timestamp: "2026-06-25T00:00:00.000Z",
      task_completion_accuracy: 0.9,
    });
    const gpt = task({
      model: "gpt-5.4-mini",
      timestamp: "2026-06-24T00:00:00.000Z",
      task_completion_accuracy: 0.6,
    });
    const report = buildReport([], [haikuOld, haikuNew, gpt], NOW);

    const tc = report.ratel_versions["0.2.0"].task_completion;
    expect(Object.keys(tc).sort()).toEqual(["claude-haiku-4-5", "gpt-5.4-mini"]);
    expect(tc["claude-haiku-4-5"]["ratel-full"].simple.metrics).toMatchObject({
      task_completion_accuracy: 0.9,
    });
    expect(tc["gpt-5.4-mini"]["ratel-full"].simple.metrics).toMatchObject({
      task_completion_accuracy: 0.6,
    });
  });

  it("keeps versions independent (new version added, others untouched)", () => {
    const v020 = task({ ratel_ai_core_version: "0.2.0" });
    const v030 = task({ ratel_ai_core_version: "0.3.0", task_completion_accuracy: 0.99 });
    const report = buildReport([retr({ ratel_ai_core_version: "0.2.0" })], [v020, v030], NOW);
    expect(Object.keys(report.ratel_versions).sort()).toEqual(["0.2.0", "0.3.0"]);
    expect(
      report.ratel_versions["0.3.0"].task_completion["claude-haiku-4-5"]["ratel-full"].simple
        .metrics,
    ).toMatchObject({
      task_completion_accuracy: 0.99,
    });
    // 0.2.0 still has its retrieval entry
    expect(report.ratel_versions["0.2.0"].retriever_evaluation.simple).toBeDefined();
  });

  it("collects multiple (pool_size, k) retrieval rows under one type, sorted", () => {
    const rows = [
      retr({ pool_size: 100, k: 5 }),
      retr({ pool_size: 30, k: 1 }),
      retr({ pool_size: 30, k: 5 }),
    ];
    const report = buildReport(rows, [], NOW);
    const metrics = report.ratel_versions["0.2.0"].retriever_evaluation.simple.metrics as Array<{
      pool_size: number;
      k: number;
    }>;
    expect(metrics.map((m) => [m.pool_size, m.k])).toEqual([
      [30, 1],
      [30, 5],
      [100, 5],
    ]);
  });
});
