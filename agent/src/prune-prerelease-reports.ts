// `prune-prerelease-reports` — keep the committed reports released-only.
//
// The git-tag experiment flow writes prerelease columns (e.g. 0.3.0-rc.1,
// 0.3.0-semantic.2) into the local `report.json` for inspection — but the
// committed artifact the website consumes (per ADR-0009) must contain only
// published releases. This drops every `ratel_versions` key that is a semver
// PRERELEASE (contains a `-`), keeping 0.2.0, 0.3.0, … Run it before committing
// / merging.
//
//   pnpm prune-prerelease-reports            # rewrite the reports in place
//   pnpm prune-prerelease-reports --dry-run  # report what would change, write nothing
//
// Idempotent; missing report files are skipped with a notice.

import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolveRepoPath } from "./paths.js";

const REPORTS = ["results/reports/bfcl/report.json", "results/reports/sragents/report.json"];

interface Report {
  generated_at?: string;
  ratel_versions?: Record<string, unknown>;
}

/** A version string is a prerelease when its semver core has a `-` suffix. */
function isPrerelease(version: string): boolean {
  return version.includes("-");
}

function main(): void {
  const dryRun = process.argv.includes("--dry-run");

  for (const rel of REPORTS) {
    const path = resolveRepoPath(rel);
    if (!existsSync(path)) {
      console.log(`• ${rel}: not found — skipping`);
      continue;
    }

    const report = JSON.parse(readFileSync(path, "utf-8")) as Report;
    const versions = report.ratel_versions ?? {};
    const keys = Object.keys(versions);
    const dropped = keys.filter(isPrerelease);
    const kept = keys.filter((k) => !isPrerelease(k));

    if (dropped.length === 0) {
      console.log(`✓ ${rel}: already released-only [${kept.join(", ") || "none"}]`);
      continue;
    }

    if (dryRun) {
      console.log(`• ${rel}: would drop [${dropped.join(", ")}], keep [${kept.join(", ") || "none"}]`);
      continue;
    }

    const pruned: Record<string, unknown> = {};
    for (const k of kept) pruned[k] = versions[k];
    report.ratel_versions = pruned;
    writeFileSync(path, `${JSON.stringify(report, null, 2)}\n`, "utf-8");
    console.log(`✓ ${rel}: dropped [${dropped.join(", ")}], kept [${kept.join(", ") || "none"}]`);
  }
}

main();
