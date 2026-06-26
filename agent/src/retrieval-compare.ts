// Cross-version retrieval comparison, focused on a single (pool, k) cell for
// readability. Reads the per-benchmark report.json files (bfcl + sragents), which
// share the shape
//   ratel_versions[<version>].retriever_evaluation[<bucket>] = { timestamp, metrics[] }
// and emits a markdown file with, per benchmark, one compact table per metric
// (Recall / Precision / MRR / nDCG / Accuracy / Complete-rate): rows = bucket
// (subset or dataset), columns = each ratel-ai-core version + a Δ vs the oldest.
//
// Defaults to pool 100, k 3. Pure read → render; no recompute. Re-run after any
// `*-report`.
//
// Usage:
//   pnpm -F @ratel-ai/benchmark retrieval-compare
//   pnpm -F @ratel-ai/benchmark retrieval-compare --pool 100 --k 3
//   pnpm -F @ratel-ai/benchmark retrieval-compare --out results/reports/retrieval-comparison.md

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { resolveRepoPath } from "./paths.js";

interface Metric {
  pool_size: number;
  k: number;
  n?: number;
  mean_recall?: number;
  mean_precision?: number;
  mean_mrr?: number;
  mean_ndcg?: number;
  accuracy?: number;
  complete_rate?: number;
}
interface Bucket {
  timestamp?: string;
  metrics: Metric[];
}
interface Report {
  generated_at?: string;
  ratel_versions: Record<string, { retriever_evaluation?: Record<string, Bucket> }>;
}

const METRICS: { key: keyof Metric; label: string }[] = [
  { key: "mean_recall", label: "Recall" },
  { key: "mean_precision", label: "Precision" },
  { key: "mean_mrr", label: "MRR" },
  { key: "mean_ndcg", label: "nDCG" },
  { key: "accuracy", label: "Accuracy" },
  { key: "complete_rate", label: "Complete-rate" },
];

function arg(flag: string, fallback: string): string {
  const i = process.argv.indexOf(flag);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : fallback;
}

function loadReport(path: string): Report | null {
  try {
    return JSON.parse(readFileSync(resolveRepoPath(path), "utf-8")) as Report;
  } catch {
    return null;
  }
}

/** Sort version labels: 0.2.0 < 0.3.0-semantic.1 < 0.3.0-semantic.2 (numeric-aware). */
function sortVersions(vs: string[]): string[] {
  const parts = (v: string) => v.split(/[.\-]/).map((p) => (/^\d+$/.test(p) ? Number(p) : p));
  return [...vs].sort((a, b) => {
    const pa = parts(a);
    const pb = parts(b);
    for (let i = 0; i < Math.max(pa.length, pb.length); i++) {
      const x = pa[i] ?? -1;
      const y = pb[i] ?? -1;
      if (x === y) continue;
      if (typeof x === "number" && typeof y === "number") return x - y;
      return String(x) < String(y) ? -1 : 1;
    }
    return 0;
  });
}

/** Buckets sorted alphabetically, with the aggregate `all` pinned last. */
function sortBuckets(bs: string[]): string[] {
  return [...bs].sort((a, b) => {
    if (a === "all") return 1;
    if (b === "all") return -1;
    return a < b ? -1 : a > b ? 1 : 0;
  });
}

const f3 = (x: number | undefined | null): string =>
  x === undefined || x === null ? "—" : (Math.round(x * 1000) / 1000).toFixed(3);

/** The metric row at (pool, k) for a bucket, last-wins on duplicates; else undefined. */
function cell(b: Bucket | undefined, pool: number, k: number): Metric | undefined {
  let found: Metric | undefined;
  for (const m of b?.metrics ?? []) if (m.pool_size === pool && m.k === k) found = m;
  return found;
}

function compareBenchmark(
  title: string,
  rowLabel: string,
  report: Report | null,
  pool: number,
  k: number,
): string[] {
  const out: string[] = [`## ${title}`, ""];
  if (!report) return out.concat("_report not found_", "");

  const versions = sortVersions(Object.keys(report.ratel_versions ?? {}));
  if (versions.length === 0) return out.concat("_no versions in report_", "");

  const evalByVersion = new Map<string, Record<string, Bucket>>();
  for (const v of versions) evalByVersion.set(v, report.ratel_versions[v].retriever_evaluation ?? {});
  const buckets = sortBuckets([...new Set(versions.flatMap((v) => Object.keys(evalByVersion.get(v)!)))]);

  out.push(`_pool ${pool}, k ${k}_`, "");

  const first = versions[0];
  const last = versions[versions.length - 1];
  const showDelta = versions.length >= 2;

  // One table per metric: rows = bucket, cols = versions (+Δ).
  for (const { key, label } of METRICS) {
    const header = [rowLabel, "n", ...versions];
    if (showDelta) header.push(`Δ (${last}−${first})`);
    const lines = [`### ${label}`, "", `| ${header.join(" | ")} |`, `| ${header.map(() => "---").join(" | ")} |`];

    let any = false;
    for (const bk of buckets) {
      const cells = versions.map((v) => cell(evalByVersion.get(v)![bk], pool, k));
      if (cells.every((c) => c === undefined)) continue;
      any = true;
      // n shown per version only if they differ; else a single value.
      const ns = cells.map((c) => c?.n);
      const nDisplay = new Set(ns.filter((x) => x !== undefined)).size <= 1
        ? `${ns.find((x) => x !== undefined) ?? "—"}`
        : versions.map((v, i) => `${v.split("-").pop()}:${ns[i] ?? "—"}`).join(" / ");
      const row = [bk, nDisplay, ...cells.map((c) => f3(c?.[key] as number | undefined))];
      if (showDelta) {
        const a = cells[0]?.[key] as number | undefined;
        const b = cells[cells.length - 1]?.[key] as number | undefined;
        row.push(a !== undefined && b !== undefined ? (b - a >= 0 ? "+" : "") + f3(b - a) : "—");
      }
      lines.push(`| ${row.join(" | ")} |`);
    }
    if (any) out.push(...lines, "");
  }

  // Compact run-timestamp footer.
  out.push("### Run timestamps", "", `| version | timestamp |`, `| --- | --- |`);
  for (const v of versions) {
    const ts = [...new Set(Object.values(evalByVersion.get(v)!).map((b) => b.timestamp))].filter(Boolean);
    out.push(`| \`${v}\` | ${ts.join(", ") || "—"} |`);
  }
  out.push("");
  return out;
}

function main(): void {
  const pool = Number(arg("--pool", "100"));
  const k = Number(arg("--k", "3"));
  const bfcl = loadReport(arg("--bfcl", "results/reports/bfcl/report.json"));
  const sragents = loadReport(arg("--sragents", "results/reports/sragents/report.json"));
  const outPath = resolveRepoPath(arg("--out", "results/reports/retrieval-comparison.md"));

  const lines = [
    `# Retrieval Evaluation — Version Comparison (pool ${pool}, k ${k})`,
    "",
    "Recall / Precision / MRR / nDCG / Accuracy / Complete-rate per ratel-ai-core version.",
    "Δ = newest − oldest version. `—` = not run for that bucket at this pool/k.",
    "`n` is per version when sample sizes differ (label suffix).",
    "",
    ...compareBenchmark("BFCL", "subset", bfcl, pool, k),
    ...compareBenchmark("SR-Agents", "dataset", sragents, pool, k),
  ];

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${lines.join("\n")}\n`, "utf-8");
  const vAll = (r: Report | null) => (r ? sortVersions(Object.keys(r.ratel_versions ?? {})) : []);
  console.log(
    `retrieval-compare: wrote ${arg("--out", "results/reports/retrieval-comparison.md")} ` +
      `(pool ${pool}, k ${k}) — BFCL [${vAll(bfcl).join(", ") || "none"}], ` +
      `SR-Agents [${vAll(sragents).join(", ") || "none"}]`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
