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
// Errored cells: rows are superseded per cell first (the last final row wins,
// so a re-run replaces a transient error). Final `transient|access` rows are
// excluded from every metric and counted (`excluded_cells`); `request|timeout|
// outcome` errors stay scored fails (`errored_cells`). `--label L` restricts the
// output to one `ratel_ai_core_version` label.
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
import { errorClassOf, isInfraError, supersede } from "./cell-errors.js";
import { modelRouteOfRow } from "./cell-key.js";
import { appendJsonl, readJsonl } from "./io.js";
import { astArgRecall } from "./judges/ast.js";
import { effectiveCalls } from "./metering.js";
import { resolveRepoPath } from "./paths.js";
import {
  assertSingleModelProvenance,
  corpusOf,
  isTruncated,
  labelledCellKeyOf,
  mean,
  meanOrNull,
  median,
  medianOrNull,
  netWallMs,
  provenance,
  sumCounts,
  type VersionSplitCell,
  versionSplitCells,
} from "./report.js";
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
  /** Selected label cells counted once per `ratel_version` (see `versionSplitCells`). */
  versionSplitCells: VersionSplitCell[];
}

/**
 * Build the flat retrieval summary, the task per-row records, and the flat task
 * summary from already-parsed raw inputs. Pure — no I/O.
 *
 * @param arm Optional arm filter. Omit to include every arm (per-arm breakdown);
 * pass e.g. `ratel-full` to restrict to one.
 * @param label Optional `ratel_ai_core_version` filter: emit only that label's groups.
 */
export function summarizeBfcl(args: {
  retrievalRows: BfclRetrievalRow[];
  cells: CellResult[];
  scenarios: Scenario[];
  arm?: string;
  label?: string;
}): SummarizeResult {
  const { arm, label } = args;
  const inLabel = (version: string | undefined) => !label || (version ?? "unknown") === label;
  const cells = args.cells
    .filter(
      (c) =>
        bfclType(c.scenario_id) !== null &&
        (!arm || c.arm === arm) &&
        inLabel(c.ratel_ai_core_version),
    )
    .map((cell) => ({ ...cell, model: modelRouteOfRow(cell) }));
  const taskRows = buildTaskRows(supersede(cells, labelledCellKeyOf), args.scenarios);
  return {
    retrievalSummary: summarizeRetrieval(
      args.retrievalRows.filter((r) => inLabel(r.ratel_ai_core_version)),
    ),
    taskRows,
    // Timestamps span every row (superseded and excluded ones too), so a
    // re-summary of the same rows ties the summary it replaces and, appended
    // last, wins. A published summary newer than every raw row of its group
    // (its rows were later rewritten) stays newer until rows are appended (a
    // re-drain re-stamps generated_at).
    taskSummary: summarizeTask(taskRows, groupTimestamps(cells)),
    versionSplitCells: versionSplitCells(cells),
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
      ratel_ai_core_resolved_version: provenance(
        arr.map((r) => r.ratel_ai_core_resolved_version ?? null),
      ),
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

function buildTaskRows(cells: CellResult[], scenarios: Scenario[]): TaskRow[] {
  const byId = new Map(scenarios.map((s) => [s.id, s]));
  const out: TaskRow[] = [];
  for (const c of cells) {
    const type = bfclType(c.scenario_id);
    if (type === null) continue;
    const scenario = byId.get(c.scenario_id);
    out.push({
      ratel_ai_core_version: c.ratel_ai_core_version ?? "unknown",
      ratel_ai_core_resolved_version: c.ratel_ai_core_resolved_version ?? null,
      generated_at: c.generated_at ?? "",
      type,
      model: c.model,
      ...(c.serving_provider ? { serving_provider: c.serving_provider } : {}),
      ...(c.publisher ? { publisher: c.publisher } : {}),
      ...(c.resolved_model ? { resolved_model: c.resolved_model } : {}),
      ...(c.vertex_location ? { vertex_location: c.vertex_location } : {}),
      ...(c.cost_source ? { cost_source: c.cost_source } : {}),
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
      error_class: errorClassOf(c),
      excluded: isInfraError(c),
      truncated: isTruncated(c),
      max_output_tokens: c.max_output_tokens ?? null,
      retries: c.retries ?? null,
      throttled_retries: c.throttled_retries ?? null,
      retry_wait_ms: c.retry_wait_ms ?? null,
      retry_policy: c.retry_policy ?? null,
    });
  }
  return out;
}

function summarizeTask(rows: TaskRow[], timestamps: Map<string, string>): TaskSummaryRow[] {
  const groups = new Map<string, TaskRow[]>();
  for (const r of rows) {
    const key = groupKey(r);
    (groups.get(key) ?? groups.set(key, []).get(key))?.push(r);
  }
  const out: TaskSummaryRow[] = [];
  for (const [key, arr] of groups) {
    const [version, type, model, arm] = key.split("::");
    assertSingleModelProvenance(arr, model);
    const kept = arr.filter((r) => !r.excluded);
    const clean = kept.filter((r) => r.error_class === null);
    const astRows = kept.filter((r) => r.task_completion_pass !== null);
    const recalls = kept.map((r) => r.recall).filter((x): x is number => x !== null);
    out.push({
      timestamp: timestamps.get(key) ?? "",
      ratel_ai_core_version: version,
      ratel_ai_core_resolved_version: provenance(
        arr.map((r) => r.ratel_ai_core_resolved_version ?? null),
      ),
      source: "task_completion",
      model,
      ...(arr.some((r) => r.serving_provider)
        ? { serving_provider: provenance(arr.map((r) => r.serving_provider ?? null)) }
        : {}),
      ...(arr.some((r) => r.publisher)
        ? { publisher: provenance(arr.map((r) => r.publisher ?? null)) }
        : {}),
      ...(arr.some((r) => r.resolved_model)
        ? { resolved_model: provenance(arr.map((r) => r.resolved_model ?? null)) }
        : {}),
      ...(arr.some((r) => r.vertex_location)
        ? { vertex_location: provenance(arr.map((r) => r.vertex_location ?? null)) }
        : {}),
      arm,
      type: type as BfclType,
      scenarios: kept.length,
      task_completion_accuracy: meanOrNull(astRows.map((r) => (r.task_completion_pass ? 1 : 0))),
      selection_accuracy: meanOrNull(kept.map((r) => (r.selection_pass ? 1 : 0))),
      recall: meanOrNull(recalls),
      mean_total_tokens: meanOrNull(clean.map((r) => r.total_tokens)),
      latency_p50_ms: medianOrNull(clean.map((r) => r.wall_ms)),
      latency_p50_net_ms: medianOrNull(clean.map(netWallMs)),
      excluded_cells: arr.length - kept.length,
      errored_cells: kept.length - clean.length,
      truncated_cells: kept.filter((r) => r.truncated).length,
      max_output_tokens: provenance(kept.map((r) => r.max_output_tokens)),
      retries: sumCounts(arr.map((r) => r.retries)),
      throttled_retries: sumCounts(arr.map((r) => r.throttled_retries)),
      retry_policy: provenance(kept.map((r) => r.retry_policy)),
    });
  }
  return out.sort(
    (a, b) =>
      a.type.localeCompare(b.type) || a.model.localeCompare(b.model) || a.arm.localeCompare(b.arm),
  );
}

/** Summary group of a task row / cell: label × type × LLM × arm. */
function groupKey(r: {
  ratel_ai_core_version: string;
  type: BfclType;
  model: string;
  arm: string;
}): string {
  return `${r.ratel_ai_core_version}::${r.type}::${r.model}::${r.arm}`;
}

/** Latest `generated_at` per summary group, over every given cell. */
function groupTimestamps(cells: CellResult[]): Map<string, string> {
  const byGroup = new Map<string, string[]>();
  for (const c of cells) {
    const type = bfclType(c.scenario_id);
    if (type === null) continue;
    const key = groupKey({
      ratel_ai_core_version: c.ratel_ai_core_version ?? "unknown",
      type,
      model: c.model,
      arm: c.arm,
    });
    (byGroup.get(key) ?? byGroup.set(key, []).get(key))?.push(c.generated_at ?? "");
  }
  return new Map([...byGroup].map(([key, ts]) => [key, latest(ts)]));
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
  const label = arg("--label", "") || undefined; // omit ⇒ every ratel_ai_core_version label
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

  const { retrievalSummary, taskRows, taskSummary, versionSplitCells } = summarizeBfcl({
    retrievalRows,
    cells,
    scenarios,
    arm,
    label,
  });
  const excluded = taskSummary.reduce((n, s) => n + s.excluded_cells, 0);
  const splitByLabel = new Map<string, number>();
  for (const c of versionSplitCells) {
    splitByLabel.set(c.label, (splitByLabel.get(c.label) ?? 0) + 1);
  }
  for (const [splitLabel, n] of splitByLabel) {
    console.error(
      `bfcl-summarize: warning: ${n} cells in label ${splitLabel} appear under >1 ratel_version; ` +
        "they are counted once per version (a re-drain must reuse the replaced rows' " +
        "--ratel-version; see results-audit)",
    );
  }

  appendRows(retrievalSummaryOut, retrievalSummary); // history
  writeOverwrite(taskRowsOut, taskRows); // latest run only
  appendRows(taskSummaryOut, taskSummary); // history

  console.log(
    `bfcl-summarize: ${retrievalSummary.length} retrieval-summary rows (append), ` +
      `${taskRows.length} task rows (overwrite), ${taskSummary.length} task-summary rows (append), ` +
      `${excluded} infra-errored cells excluded [arm=${arm ?? "all"}, label=${label ?? "all"}]`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
