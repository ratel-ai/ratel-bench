// `bfcl-summarize` — turns raw eval artifacts into per-row metrics + appends
// experiment-summary rows.
//
// Reads:
//   - retrieval per-row JSONL (Rust output: results/raw/bfcl/retrieval-rows.jsonl)
//   - agent cells JSONL (results/raw/bfcl/agent.jsonl)
//   - the bfcl-all corpus (for query + gold answers)
// Writes:
//   - task-completion-rows.jsonl   (per-row, OVERWRITE)
//   - retrieval-summary.jsonl      (APPEND)
//   - task-completion-summary.jsonl (APPEND)
//
// Pure aggregation lives in `summarizeBfcl()`; the CLI shell does the I/O.

import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type {
  BfclRetrievalRow,
  BfclType,
  RetrievalSummaryRow,
  TaskRow,
  TaskSummaryRow,
} from "./bfcl-types.js";
import { appendJsonl, readJsonl } from "./io.js";
import { astArgRecall } from "./judges/ast.js";
import { effectiveCalls } from "./metering.js";
import { resolveRepoPath } from "./paths.js";
import { corpusOf, mean, median } from "./report.js";
import type { CellResult, Scenario } from "./types.js";

/** `bfcl-simple-…` → `simple`, `bfcl-multiple-…` → `multiple`; null otherwise. */
function bfclType(scenarioId: string): BfclType | null {
  const c = corpusOf(scenarioId);
  if (c === "bfcl-simple") return "simple";
  if (c === "bfcl-multiple") return "multiple";
  return null;
}

function stddev(xs: number[]): number {
  if (xs.length === 0) return 0;
  const m = mean(xs);
  return Math.sqrt(mean(xs.map((x) => (x - m) ** 2)));
}

/** Latest RFC-3339 / ISO timestamp in a list (lexicographic compare is valid for these formats). */
function latest(timestamps: string[]): string {
  return timestamps.filter(Boolean).sort().at(-1) ?? "";
}

export interface SummarizeResult {
  retrievalSummary: RetrievalSummaryRow[];
  taskRows: TaskRow[];
  taskSummary: TaskSummaryRow[];
}

/**
 * Build the flat retrieval summary, the task per-row records, and the flat task
 * summary from already-parsed raw inputs. Pure — no I/O.
 *
 * @param arm Optional arm filter. Omit to include every arm (per-arm breakdown);
 * pass e.g. `ratel-full` to restrict to one.
 */
export function summarizeBfcl(args: {
  retrievalRows: BfclRetrievalRow[];
  cells: CellResult[];
  scenarios: Scenario[];
  arm?: string;
}): SummarizeResult {
  const taskRows = buildTaskRows(args.cells, args.scenarios, args.arm);
  return {
    retrievalSummary: summarizeRetrieval(args.retrievalRows),
    taskRows,
    taskSummary: summarizeTask(taskRows),
  };
}

// ── Retrieval ─────────────────────────────────────────────────────────────────

function summarizeRetrieval(rows: BfclRetrievalRow[]): RetrievalSummaryRow[] {
  const bfcl = rows.filter((r) => bfclType(r.scenario_id) !== null);
  if (bfcl.length === 0) return [];
  const timestamp = latest(bfcl.map((r) => r.generated_at));
  const version = bfcl.find((r) => r.ratel_ai_core_version)?.ratel_ai_core_version ?? "unknown";

  // Group by (type, pool_size, k) for metrics; gold similarity is k-independent
  // so it's pooled per (type, pool_size).
  const groups = new Map<string, BfclRetrievalRow[]>();
  for (const r of bfcl) {
    const key = `${bfclType(r.scenario_id)}::${r.target_pool_size}::${r.k}`;
    (groups.get(key) ?? groups.set(key, []).get(key))?.push(r);
  }

  const out: RetrievalSummaryRow[] = [];
  for (const [key, arr] of groups) {
    const [type, poolStr, kStr] = key.split("::");
    // gold_score is per (scenario, pool) — dedupe across k by scenario_id.
    const perScenario = new Map<string, number | null>();
    for (const r of arr) perScenario.set(r.scenario_id, r.gold_score ?? null);
    const scores = [...perScenario.values()].filter((s): s is number => s !== null);
    out.push({
      timestamp,
      ratel_ai_core_version: version,
      source: "retriever_evaluation",
      type: type as BfclType,
      pool_size: Number(poolStr),
      k: Number(kStr),
      n: arr.length,
      mean_precision: mean(arr.map((r) => r.precision_at_k)),
      median_precision: median(arr.map((r) => r.precision_at_k)),
      mean_recall: mean(arr.map((r) => r.recall_at_k)),
      median_recall: median(arr.map((r) => r.recall_at_k)),
      mean_mrr: mean(arr.map((r) => r.reciprocal_rank)),
      median_mrr: median(arr.map((r) => r.reciprocal_rank)),
      mean_ndcg: mean(arr.map((r) => r.ndcg_at_k)),
      median_ndcg: median(arr.map((r) => r.ndcg_at_k)),
      accuracy: mean(arr.map((r) => (r.hit_at_k ? 1 : 0))),
      complete_rate: mean(arr.map((r) => ((r.complete_at_k ?? r.hit_at_k) ? 1 : 0))),
      gold_similarity: {
        mean: mean(scores),
        median: median(scores),
        stddev: stddev(scores),
        coverage: perScenario.size === 0 ? 0 : scores.length / perScenario.size,
      },
    });
  }
  return out.sort((a, b) => a.type.localeCompare(b.type) || a.pool_size - b.pool_size || a.k - b.k);
}

// ── Task completion ─────────────────────────────────────────────────────────

function buildTaskRows(cells: CellResult[], scenarios: Scenario[], arm?: string): TaskRow[] {
  const byId = new Map(scenarios.map((s) => [s.id, s]));
  const out: TaskRow[] = [];
  for (const c of cells) {
    const type = bfclType(c.scenario_id);
    if (type === null) continue;
    if (arm && c.arm !== arm) continue;
    const scenario = byId.get(c.scenario_id);
    out.push({
      ratel_ai_core_version: c.ratel_ai_core_version ?? "unknown",
      generated_at: c.generated_at ?? "",
      type,
      model: c.model,
      arm: c.arm,
      scenario_id: c.scenario_id,
      query: scenario?.prompt ?? "",
      true_answers: {
        gold_tools: scenario?.gold_tools ?? [],
        gold_calls: scenario?.gold_calls ?? [],
      },
      llm_answer: effectiveCalls(c.tool_calls),
      selection_pass: c.programmatic_verdict === "pass",
      task_completion_pass: c.ast_verdict === "n/a" ? null : c.ast_verdict === "pass",
      recall: astArgRecall(scenario?.gold_calls, effectiveCalls(c.tool_calls)),
      input_tokens: c.input_tokens,
      output_tokens: c.output_tokens,
      total_tokens: c.total_tokens,
      dollar_cost: c.dollar_cost,
      wall_ms: c.wall_ms,
      turns: c.turns,
    });
  }
  return out;
}

function summarizeTask(rows: TaskRow[]): TaskSummaryRow[] {
  const groups = new Map<string, TaskRow[]>();
  for (const r of rows) {
    const key = `${r.ratel_ai_core_version}::${r.type}::${r.model}::${r.arm}`;
    (groups.get(key) ?? groups.set(key, []).get(key))?.push(r);
  }
  const out: TaskSummaryRow[] = [];
  for (const [key, arr] of groups) {
    const [version, type, model, arm] = key.split("::");
    const astRows = arr.filter((r) => r.task_completion_pass !== null);
    const recalls = arr.map((r) => r.recall).filter((x): x is number => x !== null);
    out.push({
      timestamp: latest(arr.map((r) => r.generated_at)),
      ratel_ai_core_version: version,
      source: "task_completion",
      model,
      arm,
      type: type as BfclType,
      scenarios: arr.length,
      task_completion_accuracy:
        astRows.length === 0 ? null : mean(astRows.map((r) => (r.task_completion_pass ? 1 : 0))),
      selection_accuracy: mean(arr.map((r) => (r.selection_pass ? 1 : 0))),
      recall: recalls.length === 0 ? null : mean(recalls),
      mean_total_tokens: mean(arr.map((r) => r.total_tokens)),
      latency_p50_ms: median(arr.map((r) => r.wall_ms)),
    });
  }
  return out.sort(
    (a, b) =>
      a.type.localeCompare(b.type) || a.model.localeCompare(b.model) || a.arm.localeCompare(b.arm),
  );
}

// ── CLI shell (I/O) ─────────────────────────────────────────────────────────

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function writeOverwrite(path: string, rows: unknown[]): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(
    path,
    rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : ""),
    "utf-8",
  );
}

function appendRows(path: string, rows: unknown[]): void {
  mkdirSync(dirname(path), { recursive: true });
  for (const r of rows) appendJsonl(path, r);
}

function main(): void {
  const retrievalRowsPath = resolveRepoPath(
    arg("--retrieval-rows", "results/raw/bfcl/retrieval-rows.jsonl"),
  );
  const agentPath = resolveRepoPath(arg("--agent", "results/raw/bfcl/agent.jsonl"));
  const corpusPath = resolveRepoPath(arg("--corpus", "test-data/bfcl-all.jsonl"));
  const arm = arg("--arm", "") || undefined; // omit ⇒ all arms (per-arm breakdown)
  const retrievalSummaryOut = resolveRepoPath(
    arg("--retrieval-summary-out", "results/raw/bfcl/retrieval-summary.jsonl"),
  );
  const taskRowsOut = resolveRepoPath(
    arg("--task-rows-out", "results/raw/bfcl/task-completion-rows.jsonl"),
  );
  const taskSummaryOut = resolveRepoPath(
    arg("--task-summary-out", "results/raw/bfcl/task-completion-summary.jsonl"),
  );

  const retrievalRows = readJsonl<BfclRetrievalRow>(retrievalRowsPath);
  const cells = readJsonl<CellResult>(agentPath);
  const scenarios = existsSync(corpusPath) ? readJsonl<Scenario>(corpusPath) : [];

  const { retrievalSummary, taskRows, taskSummary } = summarizeBfcl({
    retrievalRows,
    cells,
    scenarios,
    arm,
  });

  appendRows(retrievalSummaryOut, retrievalSummary); // history
  writeOverwrite(taskRowsOut, taskRows); // latest run only
  appendRows(taskSummaryOut, taskSummary); // history

  console.log(
    `bfcl-summarize: ${retrievalSummary.length} retrieval-summary rows (append), ` +
      `${taskRows.length} task rows (overwrite), ${taskSummary.length} task-summary rows (append) ` +
      `[arm=${arm ?? "all"}]`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
