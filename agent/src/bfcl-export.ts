// Consolidated BFCL results exporter. Emits ONE JSON file combining:
//   - retrieval evaluation (BM25), split single-tool vs multi-tool;
//   - task-completion evaluation (agent: selection + task-completion accuracy,
//     token/cost savings) with vs without Ratel, single+multi combined;
//   - timestamp + ratel-ai-core (BM25 engine) + ratel SDK versions.
//
// Replaces the per-scenario BFCL markdown. Reuses the aggregation in report.ts
// so the numbers match the rest of the harness.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { readJsonl } from "./io.js";
import { resolveRepoPath } from "./paths.js";
import {
  corpusOf,
  type RetrievalRow,
  retrievalByPoolSize,
  savingsByModel,
  statsByArmModel,
} from "./report.js";
import type { CellResult } from "./types.js";

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

const agentPath = resolveRepoPath(arg("--agent", "agent/results/agent.jsonl"));
const simpleRetrievalPath = resolveRepoPath(
  arg("--retrieval-simple", "results/bfcl-simple-retrieval.jsonl"),
);
const multipleRetrievalPath = resolveRepoPath(
  arg("--retrieval-multiple", "results/bfcl-multiple-retrieval.jsonl"),
);
const outputPath = resolveRepoPath(arg("--out", "results/BFCL.json"));

// ── Load ──────────────────────────────────────────────────────────────────────
const allCells = readJsonl<CellResult>(agentPath);
const cells = allCells.filter((c) => (c.category ?? "").startsWith("bfcl"));
const retrievalRows = [
  ...readJsonl<RetrievalRow>(simpleRetrievalPath),
  ...readJsonl<RetrievalRow>(multipleRetrievalPath),
].filter((r) => corpusOf(r.scenario_id).startsWith("bfcl"));

// ── Versions ────────────────────────────────────────────────────────────────
const ratelAiCoreVersion =
  retrievalRows.find((r) => r.ratel_ai_core_version)?.ratel_ai_core_version ?? null;
const ratelSdkVersion = cells.find((c) => c.ratel_version)?.ratel_version ?? null;

// ── Retrieval evaluation (split single-tool vs multi-tool) ──────────────────────
const retrievalSummaries = retrievalByPoolSize(retrievalRows);
const retrievalEvaluation: Record<
  string,
  Array<{
    k: number;
    pool_size: number;
    n: number;
    accuracy_at_k: number; // hit@K — for single-gold BFCL this is accuracy@K
    complete_at_k: number;
    mean_precision: number;
    median_precision: number;
    mean_recall: number;
    median_recall: number;
    mean_mrr: number;
    median_mrr: number;
    mean_ndcg: number;
    median_ndcg: number;
  }>
> = {};
for (const s of retrievalSummaries) {
  const bucket = retrievalEvaluation[s.corpus] ?? [];
  retrievalEvaluation[s.corpus] = bucket;
  bucket.push({
    k: s.k,
    pool_size: s.pool_size,
    n: s.n,
    accuracy_at_k: s.hit_rate,
    complete_at_k: s.complete_rate,
    mean_precision: s.mean_precision,
    median_precision: s.median_precision,
    mean_recall: s.mean_recall,
    median_recall: s.median_recall,
    mean_mrr: s.mean_mrr,
    median_mrr: s.median_mrr,
    mean_ndcg: s.mean_ndcg,
    median_ndcg: s.median_ndcg,
  });
}

// ── Task-completion evaluation (combined single+multi, with vs without Ratel) ───
const stats = statsByArmModel(cells);
const byArm = stats.map((s) => ({
  arm: s.arm,
  model: s.model,
  category: s.category, // "bfcl" (single+multi pooled)
  pool_size: s.pool_size,
  scenarios: s.scenarios,
  runs: s.n,
  selection_accuracy: s.success_rate,
  task_completion_accuracy: s.task_completion_rate,
  mean_catalog_size: s.mean_catalog_size,
  mean_input_tokens: s.mean_input_tokens,
  mean_total_tokens: s.mean_total_tokens,
  mean_turns: s.mean_turns,
  mean_dollar_cost: s.mean_dollar_cost,
  mean_wall_ms: s.mean_wall_ms,
}));

const savings = savingsByModel(cells).map((s) => ({
  model: s.model,
  category: s.category,
  pool_size: s.pool_size,
  input_tokens: {
    control: s.control_mean_input,
    ratel: s.ratel_mean_input,
    savings_pct: s.input_savings_pct,
  },
  total_tokens: {
    control: s.control_mean_total,
    ratel: s.ratel_mean_total,
    savings_pct: s.total_savings_pct,
  },
  dollars: {
    control: s.control_mean_dollars,
    ratel: s.ratel_mean_dollars,
    savings_pct: s.dollar_savings_pct,
  },
  wall_ms: {
    control: s.control_mean_wall_ms,
    ratel: s.ratel_mean_wall_ms,
    savings_pct: s.wall_savings_pct,
  },
}));

// ── Assemble + write ──────────────────────────────────────────────────────────
const report = {
  benchmark: "BFCL",
  generated_at: new Date().toISOString(),
  ratel_ai_core_version: ratelAiCoreVersion,
  ratel_sdk_version: ratelSdkVersion,
  counts: {
    agent_cells: cells.length,
    retrieval_rows: retrievalRows.length,
  },
  retrieval_evaluation: retrievalEvaluation,
  task_completion_evaluation: {
    note: "selection_accuracy = right function called; task_completion_accuracy = right function AND arguments (BFCL AST). single+multi pooled.",
    by_arm: byArm,
    savings_ratel_vs_control: savings,
  },
};

mkdirSync(dirname(outputPath), { recursive: true });
writeFileSync(outputPath, `${JSON.stringify(report, null, 2)}\n`, "utf-8");
console.log(
  `wrote ${outputPath} (${cells.length} agent cells, ${retrievalRows.length} retrieval rows)`,
);
