// `sragents-summarize` — turns the raw skill-retrieval per-row artifact into an
// append-only experiment summary. The retrieval-only mirror of `bfcl-summarize`,
// bucketed **per dataset** (the scenario name) plus a cross-dataset `all`.
//
// Reads:
//   - retrieval per-row JSONL (Rust output: results/raw/sragents/retrieval-rows.jsonl)
// Writes:
//   - retrieval-summary.jsonl (APPEND)
//
// Pure aggregation lives in `summarizeSragents()`; the CLI shell does the I/O.

import { mkdirSync } from "node:fs";
import { dirname } from "node:path";
import { appendJsonl, readJsonl } from "./io.js";
import { resolveRepoPath } from "./paths.js";
import { corpusOf, mean, median } from "./report.js";
import type { SragentsRetrievalRow, SragentsRetrievalSummaryRow } from "./sragents-types.js";

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

export interface SummarizeResult {
  retrievalSummary: SragentsRetrievalSummaryRow[];
}

/**
 * Build the flat retrieval summary from already-parsed raw rows. Pure — no I/O.
 * One row per (dataset, pool_size, k), plus an `all` bucket per (pool_size, k)
 * spanning every dataset.
 */
export function summarizeSragents(args: {
  retrievalRows: SragentsRetrievalRow[];
}): SummarizeResult {
  return { retrievalSummary: summarizeRetrieval(args.retrievalRows) };
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

// ── CLI shell (I/O) ─────────────────────────────────────────────────────────

function arg(name: string, fallback: string): string {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function appendRows(path: string, rows: unknown[]): void {
  mkdirSync(dirname(path), { recursive: true });
  for (const r of rows) appendJsonl(path, r);
}

function main(): void {
  const retrievalRowsPath = resolveRepoPath(
    arg("--retrieval-rows", "results/raw/sragents/retrieval-rows.jsonl"),
  );
  const retrievalSummaryOut = resolveRepoPath(
    arg("--retrieval-summary-out", "results/raw/sragents/retrieval-summary.jsonl"),
  );

  const retrievalRows = readJsonl<SragentsRetrievalRow>(retrievalRowsPath);
  const { retrievalSummary } = summarizeSragents({ retrievalRows });

  appendRows(retrievalSummaryOut, retrievalSummary); // history

  const datasets = new Set(retrievalSummary.map((r) => r.dataset));
  console.log(
    `sragents-summarize: ${retrievalSummary.length} retrieval-summary rows (append) ` +
      `across ${datasets.size} bucket(s): ${[...datasets].join(", ") || "(none)"}`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
