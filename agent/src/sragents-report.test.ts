import { describe, expect, it } from "vitest";
import { buildReport } from "./sragents-report.js";
import type { SragentsRetrievalSummaryRow, SragentsTaskSummaryRow } from "./sragents-types.js";

function task(over: Partial<SragentsTaskSummaryRow>): SragentsTaskSummaryRow {
  return {
    timestamp: "2026-06-22T00:00:00.000Z",
    ratel_ai_core_version: "0.2.0",
    source: "task_completion",
    model: "gpt-5.4-mini",
    arm: "ratel-full",
    dataset: "toolqa",
    scenarios: 60,
    task_completion_accuracy: 0.7,
    selection_accuracy: 0.8,
    recall: 0.75,
    precision: 0.6,
    mean_total_tokens: 1200,
    latency_p50_ms: 800,
    ...over,
  };
}

function retr(over: Partial<SragentsRetrievalSummaryRow>): SragentsRetrievalSummaryRow {
  return {
    timestamp: "2026-06-22T00:00:00.000Z",
    ratel_ai_core_version: "0.2.0",
    source: "retriever_evaluation",
    dataset: "bigcodebench",
    pool_size: 100,
    k: 1,
    n: 1140,
    mean_precision: 1,
    median_precision: 1,
    mean_recall: 1,
    median_recall: 1,
    mean_mrr: 1,
    median_mrr: 1,
    mean_ndcg: 1,
    median_ndcg: 1,
    accuracy: 0.72,
    complete_rate: 0.0,
    gold_similarity: { mean: 48, median: 41, stddev: 27, coverage: 0.95 },
    ...over,
  };
}

const NOW = "2026-06-22T12:00:00.000Z";

describe("buildReport (sragents)", () => {
  it("keys by ratel-ai-core version → retriever_evaluation → dataset", () => {
    const report = buildReport(
      [retr({ dataset: "bigcodebench" }), retr({ dataset: "toolqa" }), retr({ dataset: "all" })],
      [],
      NOW,
    );
    expect(Object.keys(report.ratel_versions)).toEqual(["0.2.0"]);
    const re = report.ratel_versions["0.2.0"].retriever_evaluation;
    expect(Object.keys(re).sort()).toEqual(["all", "bigcodebench", "toolqa"]);
    expect(re.bigcodebench.metrics).toHaveLength(1);
    // group-level fields are lifted out of per-row metrics
    const m = re.bigcodebench.metrics[0] as Record<string, unknown>;
    expect(m).not.toHaveProperty("timestamp");
    expect(m).not.toHaveProperty("dataset");
    expect(m).toMatchObject({ accuracy: 0.72 });
  });

  it("collects multiple (pool_size, k) rows under one dataset, sorted", () => {
    const rows = [
      retr({ pool_size: 1000, k: 5 }),
      retr({ pool_size: 100, k: 1 }),
      retr({ pool_size: 100, k: 5 }),
    ];
    const report = buildReport(rows, [], NOW);
    const metrics = report.ratel_versions["0.2.0"].retriever_evaluation.bigcodebench
      .metrics as Array<{ pool_size: number; k: number }>;
    expect(metrics.map((m) => [m.pool_size, m.k])).toEqual([
      [100, 1],
      [100, 5],
      [1000, 5],
    ]);
  });

  it("takes the latest timestamp per (version, dataset)", () => {
    const older = retr({ timestamp: "2026-06-20T00:00:00.000Z", accuracy: 0.5 });
    const newer = retr({ timestamp: "2026-06-25T00:00:00.000Z", accuracy: 0.9 });
    const report = buildReport([older, newer], [], NOW);
    const entry = report.ratel_versions["0.2.0"].retriever_evaluation.bigcodebench;
    expect(entry.timestamp).toBe("2026-06-25T00:00:00.000Z");
    expect(entry.metrics[0]).toMatchObject({ accuracy: 0.9 });
  });

  it("keeps versions independent (new version added, others untouched)", () => {
    const v020 = retr({ ratel_ai_core_version: "0.2.0", accuracy: 0.6 });
    const v030 = retr({ ratel_ai_core_version: "0.3.0", accuracy: 0.99 });
    const report = buildReport([v020, v030], [], NOW);
    expect(Object.keys(report.ratel_versions).sort()).toEqual(["0.2.0", "0.3.0"]);
    expect(
      report.ratel_versions["0.3.0"].retriever_evaluation.bigcodebench.metrics[0],
    ).toMatchObject({ accuracy: 0.99 });
    expect(
      report.ratel_versions["0.2.0"].retriever_evaluation.bigcodebench.metrics[0],
    ).toMatchObject({ accuracy: 0.6 });
  });

  it("nests task_completion by model → arm → dataset, lifting group keys", () => {
    const report = buildReport(
      [],
      [
        task({ arm: "control-baseline", dataset: "toolqa", task_completion_accuracy: 0.5 }),
        task({ arm: "ratel-full", dataset: "toolqa", task_completion_accuracy: 0.8 }),
        task({ arm: "ratel-full", dataset: "all", task_completion_accuracy: 0.7 }),
      ],
      NOW,
    );
    const tc = report.ratel_versions["0.2.0"].task_completion["gpt-5.4-mini"];
    expect(Object.keys(tc).sort()).toEqual(["control-baseline", "ratel-full"]);
    expect(Object.keys(tc["ratel-full"]).sort()).toEqual(["all", "toolqa"]);
    expect(tc["ratel-full"].toolqa.metrics).toMatchObject({
      task_completion_accuracy: 0.8,
      selection_accuracy: 0.8,
      precision: 0.6,
    });
    const m = tc["ratel-full"].toolqa.metrics as Record<string, unknown>;
    expect(m).not.toHaveProperty("arm");
    expect(m).not.toHaveProperty("dataset");
  });

  it("takes the latest timestamp per (version, model, arm, dataset)", () => {
    const older = task({ timestamp: "2026-06-20T00:00:00.000Z", task_completion_accuracy: 0.4 });
    const newer = task({ timestamp: "2026-06-25T00:00:00.000Z", task_completion_accuracy: 0.9 });
    const report = buildReport([], [older, newer], NOW);
    const entry =
      report.ratel_versions["0.2.0"].task_completion["gpt-5.4-mini"]["ratel-full"].toolqa;
    expect(entry.timestamp).toBe("2026-06-25T00:00:00.000Z");
    expect(entry.metrics).toMatchObject({ task_completion_accuracy: 0.9 });
  });
});
