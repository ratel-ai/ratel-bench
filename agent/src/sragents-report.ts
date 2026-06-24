// `sragents-report` — the report producer for SR-Agents skill retrieval. Reads
// the append-only retrieval-summary JSONL and rebuilds one report keyed by
// ratel-ai-core version, taking the latest timestamp per (version, dataset).
//
// Retrieval-only mirror of `bfcl-report` (SR-Agents has no task-completion
// section), bucketed per dataset (the scenario name) plus a cross-dataset `all`.
//
// Rebuilding from the full summary history each run is deterministic: a new
// version / dataset appears, and an existing one is replaced by its
// latest-timestamp rows — exactly the add/update behavior we want.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { readJsonl } from "./io.js";
import { resolveRepoPath } from "./paths.js";
import type { SragentsRetrievalSummaryRow } from "./sragents-types.js";

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

export interface SragentsReport {
  generated_at: string;
  ratel_versions: Record<
    string,
    {
      // A single retriever (BM25) per dataset, so it's keyed by dataset directly.
      retriever_evaluation: Record<string, { timestamp: string; metrics: unknown[] }>;
    }
  >;
}

function ensureVersion(report: SragentsReport, version: string) {
  let v = report.ratel_versions[version];
  if (!v) {
    v = { retriever_evaluation: {} };
    report.ratel_versions[version] = v;
  }
  return v;
}

/** Group-level keys lifted out of the per-row metrics (they live on the parent node). */
const GROUP_KEYS = new Set(["timestamp", "ratel_ai_core_version", "source", "dataset"]);

/** A summary row reduced to just its metric fields. */
function metricFields(row: Record<string, unknown>): Record<string, unknown> {
  return Object.fromEntries(Object.entries(row).filter(([k]) => !GROUP_KEYS.has(k)));
}

/** Pure: build the per-version report from the retrieval-summary history. */
export function buildReport(
  retrieval: SragentsRetrievalSummaryRow[],
  generatedAt: string,
): SragentsReport {
  const report: SragentsReport = { generated_at: generatedAt, ratel_versions: {} };

  // Group by (version, dataset) → latest run's per-(pool,k) rows. A single
  // retriever (BM25) means no retriever level — keyed by dataset directly.
  for (const [key, rows] of latestGroups(
    retrieval,
    (r) => `${r.ratel_ai_core_version}::${r.dataset}`,
  )) {
    const [version, dataset] = key.split("::");
    ensureVersion(report, version).retriever_evaluation[dataset] = {
      timestamp: rows[0]?.timestamp ?? "",
      metrics: rows
        .slice()
        .sort((a, b) => a.pool_size - b.pool_size || a.k - b.k)
        .map((r) => metricFields(r as unknown as Record<string, unknown>)),
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
    arg("--retrieval-summary", "results/raw/sragents/retrieval-summary.jsonl"),
  );
  const outPath = resolveRepoPath(arg("--out", "results/reports/sragents/report.json"));

  const retrieval = readJsonl<SragentsRetrievalSummaryRow>(retrievalSummaryPath);
  const report = buildReport(retrieval, new Date().toISOString());

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${JSON.stringify(report, null, 2)}\n`, "utf-8");

  const versions = Object.keys(report.ratel_versions);
  console.log(
    `sragents-report: wrote ${arg("--out", "results/reports/sragents/report.json")} — ` +
      `${versions.length} ratel-ai-core version(s): ${versions.join(", ") || "(none)"}`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
