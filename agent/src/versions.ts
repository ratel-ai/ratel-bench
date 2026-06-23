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
 * The resolved `ratel-ai-core` version from the repo-root `Cargo.lock`, or
 * `"unknown"` when the lockfile or package isn't found. Computed once at module
 * load — the lockfile doesn't change within a run.
 */
export const RATEL_AI_CORE_VERSION: string = (() => {
  const lockPath = resolveRepoPath("Cargo.lock");
  if (!existsSync(lockPath)) return "unknown";
  return parseLockVersion(readFileSync(lockPath, "utf-8"), "ratel-ai-core") ?? "unknown";
})();
