// Cross-version retrieval comparison. Reads the per-benchmark report.json files
// (bfcl + sragents), which both share the shape
//   ratel_versions[<version>].retriever_evaluation[<bucket>] = { timestamp, metrics[] }
// and emits a single markdown file: for every benchmark → bucket, a recall / MRR
// / nDCG table with one column per ratel-ai-core version (so dense models line up
// against BM25), a Δ column vs the first version, and a run-timestamp overview so
// you can see when each version's numbers were produced.
//
// Pure read → render; no recompute, no API spend. Re-run after any
// `*-report` to refresh the comparison.
//
// Usage:
//   pnpm -F @ratel-ai/benchmark retrieval-compare
//   pnpm -F @ratel-ai/benchmark retrieval-compare --out results/reports/retrieval-comparison.md

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { resolveRepoPath } from "./paths.js";

interface Metric {
  pool_size: number;
  k: number;
  n?: number;
  mean_recall?: number;
  mean_mrr?: number;
  mean_ndcg?: number;
  gold_similarity?: { mean?: number };
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
  { key: "mean_mrr", label: "MRR" },
  { key: "mean_ndcg", label: "nDCG" },
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

const f3 = (x: number | undefined): string =>
  x === undefined || x === null ? "—" : (Math.round(x * 1000) / 1000).toFixed(3);

/** Index a bucket's metrics by `pool:k`, last-wins on duplicates. */
function indexMetrics(b: Bucket | undefined): Map<string, Metric> {
  const m = new Map<string, Metric>();
  for (const row of b?.metrics ?? []) m.set(`${row.pool_size}:${row.k}`, row);
  return m;
}

function compareBenchmark(title: string, report: Report | null): string[] {
  const out: string[] = [`## ${title}`, ""];
  if (!report) {
    out.push("_report not found_", "");
    return out;
  }
  const versions = sortVersions(Object.keys(report.ratel_versions ?? {}));
  if (versions.length === 0) {
    out.push("_no versions in report_", "");
    return out;
  }
  const evalByVersion = new Map<string, Record<string, Bucket>>();
  for (const v of versions) evalByVersion.set(v, report.ratel_versions[v].retriever_evaluation ?? {});

  if (report.generated_at) out.push(`_report generated: ${report.generated_at}_`, "");

  // Bucket universe across all versions.
  const buckets = sortBuckets([...new Set(versions.flatMap((v) => Object.keys(evalByVersion.get(v)!)))]);

  // Run-timestamp overview: when each (version, bucket) was produced + n.
  out.push("### Run timestamps", "", `| version | bucket | n | timestamp |`, `| --- | --- | --- | --- |`);
  for (const v of versions) {
    for (const bk of buckets) {
      const b = evalByVersion.get(v)![bk];
      if (!b) continue;
      const n = b.metrics?.[0]?.n ?? "—";
      out.push(`| \`${v}\` | ${bk} | ${n} | ${b.timestamp ?? "—"} |`);
    }
  }
  out.push("");

  // Per bucket: one table per metric, rows = (pool, k), cols = versions (+Δ).
  for (const bk of buckets) {
    out.push(`### ${bk}`, "");
    const idxByVersion = new Map<string, Map<string, Metric>>();
    for (const v of versions) idxByVersion.set(v, indexMetrics(evalByVersion.get(v)![bk]));

    // Row universe = union of pool:k across versions, sorted by pool then k.
    const cells = [...new Set(versions.flatMap((v) => [...idxByVersion.get(v)!.keys()]))].sort((a, b) => {
      const [pa, ka] = a.split(":").map(Number);
      const [pb, kb] = b.split(":").map(Number);
      return pa - pb || ka - kb;
    });
    if (cells.length === 0) {
      out.push("_no metrics_", "");
      continue;
    }
    const first = versions[0];
    const last = versions[versions.length - 1];
    const showDelta = versions.length >= 2;

    for (const { key, label } of METRICS) {
      const header = ["pool", "k", ...versions.map((v) => `\`${v}\``)];
      if (showDelta) header.push(`Δ (${last}−${first})`);
      out.push(`**${label}**`, "", `| ${header.join(" | ")} |`, `| ${header.map(() => "---").join(" | ")} |`);
      for (const cell of cells) {
        const [pool, k] = cell.split(":");
        const vals = versions.map((v) => idxByVersion.get(v)!.get(cell)?.[key] as number | undefined);
        const row = [pool, k, ...vals.map(f3)];
        if (showDelta) {
          const a = idxByVersion.get(first)!.get(cell)?.[key] as number | undefined;
          const b = idxByVersion.get(last)!.get(cell)?.[key] as number | undefined;
          row.push(a !== undefined && b !== undefined ? (b - a >= 0 ? "+" : "") + f3(b - a) : "—");
        }
        out.push(`| ${row.join(" | ")} |`);
      }
      out.push("");
    }
  }
  return out;
}

function main(): void {
  const bfcl = loadReport(arg("--bfcl", "results/reports/bfcl/report.json"));
  const sragents = loadReport(arg("--sragents", "results/reports/sragents/report.json"));
  const outPath = resolveRepoPath(arg("--out", "results/reports/retrieval-comparison.md"));

  const lines = [
    "# Retrieval Evaluation — Version Comparison",
    "",
    "Per-version recall / MRR / nDCG across ratel-ai-core versions, with run timestamps.",
    "Δ compares the newest version against the oldest. `—` = not run for that cell.",
    "",
    ...compareBenchmark("BFCL", bfcl),
    ...compareBenchmark("SR-Agents", sragents),
  ];

  mkdirSync(dirname(outPath), { recursive: true });
  writeFileSync(outPath, `${lines.join("\n")}\n`, "utf-8");
  const vAll = (r: Report | null) => (r ? sortVersions(Object.keys(r.ratel_versions ?? {})) : []);
  console.log(
    `retrieval-compare: wrote ${arg("--out", "results/reports/retrieval-comparison.md")} — ` +
      `BFCL [${vAll(bfcl).join(", ") || "none"}], SR-Agents [${vAll(sragents).join(", ") || "none"}]`,
  );
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main();
}
