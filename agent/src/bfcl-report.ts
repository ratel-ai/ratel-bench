// `bfcl-report` — the report producer. Reads the append-only experiment-summary
// JSONLs and rebuilds one report keyed by ratel-ai-core version, taking the
// latest timestamp per (version, source, model, type).
//
// Rebuilding from the full summary history each run is deterministic: a new
// version / model / retriever appears, and an existing one is replaced by its
// latest-timestamp rows — exactly the add/update behavior we want, with no
// dependence on the previous report.json.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import type { RetrievalSummaryRow, TaskSummaryRow } from "./bfcl-types.js";
import { readJsonl } from "./io.js";
import { resolveRepoPath } from "./paths.js";

/**
 * Group rows by `keyFn`, then keep only the rows whose timestamp equals the
 * group's latest timestamp. Returns one bucket of (latest-run) rows per key.
 */
function latestGroups<T extends { timestamp: string }>(
  rows: T[],
  keyFn: (r: T) => string,
): Map<string, T[]> {
  const byKey = new Map<string, T[]>();
  for (const r of rows) {
    const k = keyFn(r);
    const arr = byKey.get(k) ?? [];
    arr.push(r);
    byKey.set(k, arr);
  }
  for (const [k, arr] of byKey) {
    const maxTs =
      arr
        .map((r) => r.timestamp)
        .sort()
        .at(-1) ?? "";
    byKey.set(
      k,
      arr.filter((r) => r.timestamp === maxTs),
    );
  }
  return byKey;
}

export interface BfclReport {
  generated_at: string;
  ratel_versions: Record<
    string,
    {
      // Retrieval has a single retriever (BM25), so it's keyed by type directly.
      retriever_evaluation: Record<string, { timestamp: string; metrics: unknown[] }>;
      // Task completion is keyed by LLM → arm → type.
      task_completion: Record<
        string,
        Record<string, Record<string, { timestamp: string; metrics: unknown }>>
      >;
    }
  >;
}

function ensureVersion(report: BfclReport, version: string) {
  let v = report.ratel_versions[version];
  if (!v) {
    v = { retriever_evaluation: {}, task_completion: {} };
    report.ratel_versions[version] = v;
  }
  return v;
}

/** Get (creating if needed) the per-type bucket for a model under a source group. */
function modelBucket<T>(
  parent: Record<string, Record<string, T>>,
  model: string,
): Record<string, T> {
  let b = parent[model];
  if (!b) {
    b = {};
    parent[model] = b;
  }
  return b;
}

/** Group-level keys lifted out of the per-row metrics (they live on the parent node). */
const GROUP_KEYS = new Set([
  "timestamp",
  "ratel_ai_core_version",
  "source",
  "model",
  "arm",
  "type",
]);

/** A summary row reduced to just its metric fields. */
function metricFields(row: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(row).filter(([k]) => !GROUP_KEYS.has(k)));
}

/** Pure: build the per-version report from the two summary histories. */
export function buildReport(
  retrieval: RetrievalSummaryRow[],
  task: TaskSummaryRow[],
  generatedAt: string,
): BfclReport {
  const report: BfclReport = { generated_at: generatedAt, ratel_versions: {} };

  // Retrieval: group by (version, type) → latest run's per-(pool,k) rows. A single
  // retriever (BM25) means no retriever level — keyed by type directly.
  for (const [key, rows] of latestGroups(
    retrieval,
    (r) => `${r.ratel_ai_core_version}::${r.type}`,
  )) {
    const [version, type] = key.split("::");
    ensureVersion(report, version).retriever_evaluation[type] = {
      timestamp: rows[0]?.timestamp ?? "",
      metrics: rows
        .slice()
        .sort((a, b) => a.pool_size - b.pool_size || a.k - b.k)
        .map((r) => metricFields(r as unknown as Record<string, unknown>)),
    };
  }

  // Task completion: group by (version, llm, arm, type) → latest run's single row.
  for (const [key, rows] of latestGroups(
    task,
    (r) => `${r.ratel_ai_core_version}::${r.model}::${r.arm}::${r.type}`,
  )) {
    const [version, model, arm, type] = key.split("::");
    const llmBucket = modelBucket(ensureVersion(report, version).task_completion, model);
    const armBucket = modelBucket(llmBucket, arm);
    armBucket[type] = {
      timestamp: rows[0].timestamp,
      metrics: metricFields(rows[0] as unknown as Record<string, unknown>),
    };
  }

  return report;
}

// ── CLI shell ─────────────────────────────────────────────────────────────────

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function main(): void {
  const retrievalSummaryPath = resolveRepoPath(
    arg("--retrieval-summary", "results/raw/bfcl/retrieval-summary.jsonl"),
  );
  const taskSummaryPath = resolveRepoPath(
    arg("--task-summary", "results/raw/bfcl/task-completion-summary.jsonl"),
  );
  const outPath = resolveRepoPath(arg("--out", "results/reports/bfcl/report.json"));

  const retrieval = readJsonl<RetrievalSummaryRow>(retrievalSummaryPath);
  const task = readJsonl<TaskSummaryRow>(taskSummaryPath);
  const report = buildReport(retrieval, task, new Date().toISOString());

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf-8");

  const versions = Object.keys(report.ratel_versions);
  console.log(
    `bfcl-report: wrote ${arg("--out", "results/reports/bfcl/report.json")} — ` +
      `${versions.length} ratel-ai-core version(s): ${versions.join(", ") || "(none)"}`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
