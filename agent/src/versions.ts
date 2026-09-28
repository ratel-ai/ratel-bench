// Resolve the `ratel-ai-core` version from the workspace `Cargo.lock` — the same
// authoritative source the retrieval crate stamps via `retrieval/build.rs`. The
// agent reaches retrieval through `@ratel-ai/sdk` (whose own version is recorded
// separately as `ratel_version`), but a BFCL run is benchmarking a specific
// ratel-ai-core release, so both eval layers tag their rows with the lockfile's
// core version. `create-report` then refuses to merge layers that disagree.

import { existsSync, readFileSync } from "node:fs";
import { resolveRepoPath } from "./paths.js";

/**
 * Parse the `version` of the `[[package]]` named `name` from a Cargo.lock body.
 * Mirrors `retrieval/build.rs::parse_dep_version`: Cargo always emits `name`
 * before `version` within a package block, so we find the name line then take
 * the next `version` line before the block ends.
 */
export function parseLockVersion(lock: string, name: string): string | null {
  const needle = `name = "${name}"`;
  const lines = lock.split("\n");
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() !== needle) continue;
    for (let j = i + 1; j < lines.length; j++) {
      const l = lines[j].trim();
      const m = l.match(/^version = "(.+)"$/);
      if (m) return m[1];
      if (l === "[[package]]") break;
    }
  }
  return null;
}

/**
 * Resolve the actual core release and the report label independently. A
 * method-specific label never changes the crate version recorded as provenance.
 */
export function resolveCoreVersions(
  lock: string,
  label?: string,
): { label: string; resolved: string } {
  const resolved = parseLockVersion(lock, "ratel-ai-core") ?? "unknown";
  return { label: label || resolved, resolved };
}

const CORE_VERSIONS = (() => {
  const lockPath = resolveRepoPath("Cargo.lock");
  const lock = existsSync(lockPath) ? readFileSync(lockPath, "utf-8") : "";
  return resolveCoreVersions(lock, process.env.RATEL_VERSION_LABEL);
})();

/** Grouping label, optionally method-suffixed by RATEL_VERSION_LABEL. */
export const RATEL_AI_CORE_VERSION = CORE_VERSIONS.label;

/** Exact crate release in Cargo.lock, independent of the report label and SDK. */
export const RATEL_AI_CORE_RESOLVED_VERSION = CORE_VERSIONS.resolved;
