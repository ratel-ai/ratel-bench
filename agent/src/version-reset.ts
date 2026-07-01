// `version-reset` — bookend #2 of the git-tag experiment flow.
//
// Restores `retrieval/Cargo.toml` + `Cargo.lock` to the committed baseline that
// `version-set` snapshotted into `.version-set-baseline/`, then removes the
// snapshot. Run this after a git-tag experiment so the working tree returns to
// the published-release baseline (and so a stray dependency edit never gets
// committed).
//
//   pnpm version-reset
//
// No-op (with a notice) when no version-set is active.

import { copyFileSync, existsSync, readFileSync, rmSync } from "node:fs";
import { join } from "node:path";
import { resolveRepoPath } from "./paths.js";
import { parseLockVersion } from "./versions.js";

const CRATE = "ratel-ai-core";
const CARGO_TOML = resolveRepoPath("retrieval/Cargo.toml");
const CARGO_LOCK = resolveRepoPath("Cargo.lock");
const BASELINE_DIR = resolveRepoPath(".version-set-baseline");

function main(): void {
  if (!existsSync(BASELINE_DIR)) {
    console.log("• version-reset: no active version-set (.version-set-baseline/ absent) — nothing to restore.");
    return;
  }

  copyFileSync(join(BASELINE_DIR, "Cargo.toml"), CARGO_TOML);
  copyFileSync(join(BASELINE_DIR, "Cargo.lock"), CARGO_LOCK);
  rmSync(BASELINE_DIR, { recursive: true, force: true });

  const version = parseLockVersion(readFileSync(CARGO_LOCK, "utf-8"), CRATE) ?? "?";
  console.log(`✓ version-reset: restored baseline (${CRATE} ${version}). Working tree is back to the committed dependency.`);
}

main();
