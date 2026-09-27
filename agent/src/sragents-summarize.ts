// `sragents-summarize` — turns the raw SR-Agents artifacts into append-only
// experiment summaries, bucketed **per dataset** plus a cross-dataset `all`. The
// SR-Agents mirror of `bfcl-summarize`, covering BOTH halves:
//   - retrieval  (Rust skill-retrieval per-row → retrieval-summary, APPEND)
//   - selection  (LLM `sragents-select` cells → task-completion-rows [OVERWRITE]
//                 + task-completion-summary [APPEND])
//
// Errored cells follow `bfcl-summarize`: rows are superseded per cell (the last
// final row wins, so a re-run replaces a transient error and a duplicated
// ratel-full row counts once); final `transient|access` rows are excluded and
// counted; other errors stay scored fails. `--label L` restricts the output to
// one `ratel_ai_core_version` label.
//
// Pure aggregation lives in `summarizeSragents()`; the CLI shell does the I/O.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { errorClassOf, isInfraError, supersede } from "./cell-errors.js";
import { appendJsonl, readJsonl } from "./io.js";
import { resolveRepoPath } from "./paths.js";
import {
  corpusOf,
  isTruncated,
  mean,
  meanOrNull,
  median,
  medianOrNull,
  netWallMs,
  provenance,
  sumCounts,
} from "./report.js";
import type {
  SragentsRetrievalRow,
  SragentsRetrievalSummaryRow,
  SragentsSelectCell,
  SragentsTaskRow,
  SragentsTaskSummaryRow,
} from "./sragents-types.js";

/** Aggregate bucket spanning every dataset. */
const ALL_DATASET = "all";

/**
 * `sragents-<dataset>-…` → `<dataset>`; null for non-sragents rows. The skill
 * runner always sets `category = "sragents-<dataset>"`, so we read the dataset
 * straight off it (and gate on the scenario-id prefix for safety).
 */
function datasetOf(row: SragentsRetrievalRow): string | null {
  if (corpusOf(row.scenario_id) !== "sragents") return null;
  const cat = row.category ?? "";
  return cat.startsWith("sragents-") ? cat.slice("sragents-".length) : null;
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

/**
 * A selection cell's identity within one label's history: the label plus the
 * `sragents-select` control key (`scenario::arm::model::pool::run`).
 */
export function sragentsCellKeyOf(c: SragentsSelectCell): string {
  return [
    c.ratel_ai_core_version,
    c.scenario_id,
    c.arm,
    c.model,
    c.pool_size ?? "null",
    c.run_index,
  ].join("::");
}

export interface SummarizeResult {
  retrievalSummary: SragentsRetrievalSummaryRow[];
  taskRows: SragentsTaskRow[];
  taskSummary: SragentsTaskSummaryRow[];
}

/**
 * Build the retrieval summary and (when cells are present) the selection per-row
 * records + selection summary from already-parsed raw inputs. Pure — no I/O.
 *
 * @param label Optional `ratel_ai_core_version` filter: emit only that label's groups.
 */
export function summarizeSragents(args: {
  retrievalRows: SragentsRetrievalRow[];
  cells?: SragentsSelectCell[];
  label?: string;
}): SummarizeResult {
  const { label } = args;
  const inLabel = (version: string | undefined) => !label || (version ?? "unknown") === label;
  const cells = (args.cells ?? []).filter(
    (c) => datasetOfCell(c) !== null && inLabel(c.ratel_ai_core_version),
  );
  const taskRows = buildTaskRows(supersede(cells, sragentsCellKeyOf));
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
  };
}

function summarizeRetrieval(rows: SragentsRetrievalRow[]): SragentsRetrievalSummaryRow[] {
  // Keep sragents rows, tagging each with its dataset.
  const tagged = rows
    .map((r) => ({ dataset: datasetOf(r), row: r }))
    .filter((t): t is { dataset: string; row: SragentsRetrievalRow } => t.dataset !== null);
  if (tagged.length === 0) return [];
  const timestamp = latest(tagged.map((t) => t.row.generated_at));
  const version =
    tagged.find((t) => t.row.ratel_ai_core_version)?.row.ratel_ai_core_version ?? "unknown";

  // Group by (dataset, pool_size, k); every row also rolls up into `all` so the
  // aggregate is a strict superset bucket (matches the Rust `all` summary).
  const groups = new Map<string, SragentsRetrievalRow[]>();
  for (const { dataset, row } of tagged) {
    for (const ds of [dataset, ALL_DATASET]) {
      const key = `${ds}::${row.target_pool_size}::${row.k}`;
      (groups.get(key) ?? groups.set(key, []).get(key))?.push(row);
    }
  }

  const out: SragentsRetrievalSummaryRow[] = [];
  for (const [key, arr] of groups) {
    const [dataset, poolStr, kStr] = key.split("::");
    // gold_score is per (scenario, pool) — dedupe across k by scenario_id.
    const perScenario = new Map<string, number | null>();
    for (const r of arr) perScenario.set(r.scenario_id, r.gold_score ?? null);
    const scores = [...perScenario.values()].filter((s): s is number => s !== null);
    out.push({
      timestamp,
      ratel_ai_core_version: version,
      source: "retriever_evaluation",
      dataset,
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
  // `all` sorts last (after the named datasets) for readable output.
  const rank = (d: string) => (d === ALL_DATASET ? 1 : 0);
  return out.sort(
    (a, b) =>
      rank(a.dataset) - rank(b.dataset) ||
      a.dataset.localeCompare(b.dataset) ||
      a.pool_size - b.pool_size ||
      a.k - b.k,
  );
}

// ── Skill selection (the task-completion analog) ──────────────────────────────

/** `sragents-<dataset>` → `<dataset>`; null for non-sragents cells. */
function datasetOfCell(cell: SragentsSelectCell): string | null {
  if (corpusOf(cell.scenario_id) !== "sragents") return null;
  return cell.category.startsWith("sragents-") ? cell.category.slice("sragents-".length) : null;
}

/** One selection cell → per-row metrics (set comparison of selected vs gold). */
function buildTaskRows(cells: SragentsSelectCell[]): SragentsTaskRow[] {
  const out: SragentsTaskRow[] = [];
  for (const c of cells) {
    const dataset = datasetOfCell(c);
    if (dataset === null) continue;
    const gold = new Set(c.gold_skill_ids);
    const selected = new Set(c.selected_skill_ids);
    const hits = [...gold].filter((g) => selected.has(g)).length;
    out.push({
      ratel_ai_core_version: c.ratel_ai_core_version ?? "unknown",
      generated_at: c.generated_at ?? "",
      dataset,
      model: c.model,
      arm: c.arm,
      scenario_id: c.scenario_id,
      gold_skill_ids: c.gold_skill_ids,
      selected_skill_ids: c.selected_skill_ids,
      selection_pass: hits > 0,
      task_completion_pass: gold.size > 0 && hits === gold.size, // every gold selected
      recall: gold.size === 0 ? 0 : hits / gold.size,
      precision: selected.size === 0 ? 0 : hits / selected.size,
      total_tokens: c.total_tokens,
      wall_ms: c.wall_ms,
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

/** Per (dataset, model, arm) selection summary, plus an `all` rollup per (model, arm). */
function summarizeTask(
  rows: SragentsTaskRow[],
  timestamps: Map<string, string>,
): SragentsTaskSummaryRow[] {
  if (rows.length === 0) return [];
  const groups = new Map<string, SragentsTaskRow[]>();
  for (const r of rows) {
    for (const key of groupKeys(r)) {
      (groups.get(key) ?? groups.set(key, []).get(key))?.push(r);
    }
  }
  const out: SragentsTaskSummaryRow[] = [];
  for (const [key, arr] of groups) {
    const [version, dataset, model, arm] = key.split("::");
    const kept = arr.filter((r) => !r.excluded);
    const clean = kept.filter((r) => r.error_class === null);
    out.push({
      timestamp: timestamps.get(key) ?? "",
      ratel_ai_core_version: version,
      source: "task_completion",
      model,
      arm,
      dataset,
      scenarios: kept.length,
      task_completion_accuracy: meanOrNull(kept.map((r) => (r.task_completion_pass ? 1 : 0))),
      selection_accuracy: meanOrNull(kept.map((r) => (r.selection_pass ? 1 : 0))),
      recall: meanOrNull(kept.map((r) => r.recall)),
      precision: meanOrNull(kept.map((r) => r.precision)),
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
  const rank = (d: string) => (d === ALL_DATASET ? 1 : 0);
  return out.sort(
    (a, b) =>
      a.model.localeCompare(b.model) ||
      a.arm.localeCompare(b.arm) ||
      rank(a.dataset) - rank(b.dataset) ||
      a.dataset.localeCompare(b.dataset),
  );
}

/** Summary groups of a row: its own dataset and the `all` rollup (label × dataset × LLM × arm). */
function groupKeys(r: {
  ratel_ai_core_version: string;
  dataset: string;
  model: string;
  arm: string;
}): string[] {
  return [r.dataset, ALL_DATASET].map(
    (ds) => `${r.ratel_ai_core_version}::${ds}::${r.model}::${r.arm}`,
  );
}

/** Latest `generated_at` per summary group, over every given cell. */
function groupTimestamps(cells: SragentsSelectCell[]): Map<string, string> {
  const byGroup = new Map<string, string[]>();
  for (const c of cells) {
    const dataset = datasetOfCell(c);
    if (dataset === null) continue;
    const version = c.ratel_ai_core_version ?? "unknown";
    for (const key of groupKeys({ ...c, ratel_ai_core_version: version, dataset })) {
      (byGroup.get(key) ?? byGroup.set(key, []).get(key))?.push(c.generated_at ?? "");
    }
  }
  return new Map([...byGroup].map(([key, ts]) => [key, latest(ts)]));
}

// ── CLI shell (I/O) ─────────────────────────────────────────────────────────

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function appendRows(path: string, rows: unknown[]): void {
  mkdirSync(dirname(path), { recursive: true });
  for (const r of rows) appendJsonl(path, r);
}

function writeOverwrite(path: string, rows: unknown[]): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(
    path,
    rows.map((r) => JSON.stringify(r)).join("\n") + (rows.length ? "\n" : ""),
    "utf-8",
  );
}

function main(): void {
  const retrievalRowsPath = resolveRepoPath(
    arg("--retrieval-rows", "results/raw/sragents/retrieval-rows.jsonl"),
  );
  const agentPath = resolveRepoPath(arg("--agent", "results/raw/sragents/agent.jsonl"));
  const retrievalSummaryOut = resolveRepoPath(
    arg("--retrieval-summary-out", "results/raw/sragents/retrieval-summary.jsonl"),
  );
  const taskRowsOut = resolveRepoPath(
    arg("--task-rows-out", "results/raw/sragents/task-completion-rows.jsonl"),
  );
  const taskSummaryOut = resolveRepoPath(
    arg("--task-summary-out", "results/raw/sragents/task-completion-summary.jsonl"),
  );
  const label = arg("--label", "") || undefined; // omit ⇒ every ratel_ai_core_version label

  const retrievalRows = readJsonl<SragentsRetrievalRow>(retrievalRowsPath);
  const cells = readJsonl<SragentsSelectCell>(agentPath); // [] when the campaign hasn't run
  const { retrievalSummary, taskRows, taskSummary } = summarizeSragents({
    retrievalRows,
    cells,
    label,
  });
  const excluded = taskSummary
    .filter((s) => s.dataset === ALL_DATASET)
    .reduce((n, s) => n + s.excluded_cells, 0);

  appendRows(retrievalSummaryOut, retrievalSummary); // history
  if (taskRows.length > 0) {
    writeOverwrite(taskRowsOut, taskRows); // latest run only
    appendRows(taskSummaryOut, taskSummary); // history
  }

  const datasets = new Set(retrievalSummary.map((r) => r.dataset));
  console.log(
    `sragents-summarize: ${retrievalSummary.length} retrieval-summary rows (append) ` +
      `across ${datasets.size} bucket(s): ${[...datasets].join(", ") || "(none)"}; ` +
      `${taskRows.length} task rows (overwrite), ${taskSummary.length} task-summary rows (append), ` +
      `${excluded} infra-errored cells excluded [label=${label ?? "all"}]`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
