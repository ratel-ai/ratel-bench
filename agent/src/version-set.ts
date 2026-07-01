// `version-set` — bookend #1 of the git-tag experiment flow.
//
// Points the `ratel-ai-core` dependency at a specific git tag / rev / published
// crate, so the very next `cargo run -p ratel-benchmark-retrieval … retrieval |
// skill-retrieval …` stamps that version onto every row (via
// `retrieval/build.rs` → `RATEL_AI_CORE_VERSION`). The explicit cargo command
// stays exactly as the developer types it — this only swaps the dependency
// underneath it.
//
//   pnpm version-set --tag v0.3.0-rc.1            # git tag
//   pnpm version-set --rev 1a2b3c4                # git rev (commit)
//   pnpm version-set --crate 0.2.0               # published crates.io version
//   pnpm version-set --tag v0.3.0-rc.1 --expect 0.3.0-rc.1   # assert resolved version
//
// Run `pnpm version-reset` when done to restore the committed baseline. The
// original `retrieval/Cargo.toml` + `Cargo.lock` are snapshotted into a
// gitignored `.version-set-baseline/` so a separate `version-reset` process can
// restore them byte-for-byte (no dependence on a clean working tree or git).

import { spawnSync } from "node:child_process";
import { copyFileSync, existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { REPO_ROOT, resolveRepoPath } from "./paths.js";
import { parseLockVersion } from "./versions.js";

const DEFAULT_GIT_URL = "ssh://git@github.com/ratel-ai/ratel.git";
const CRATE = "ratel-ai-core";
const PKG = "ratel-benchmark-retrieval";

const CARGO_TOML = resolveRepoPath("retrieval/Cargo.toml");
const CARGO_LOCK = resolveRepoPath("Cargo.lock");
const BASELINE_DIR = resolveRepoPath(".version-set-baseline");

function flagValue(name: string): string | undefined {
  const i = process.argv.indexOf(name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : undefined;
}

function die(msg: string): never {
  console.error(`\n✗ ${msg}`);
  process.exit(1);
}

/** The `version` + `source` lines of a package block in a Cargo.lock body. */
function lockEntry(lock: string, name: string): { version: string | null; source: string | null } {
  const version = parseLockVersion(lock, name);
  const lines = lock.split("\n");
  let source: string | null = null;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].trim() !== `name = "${name}"`) continue;
    for (let j = i + 1; j < lines.length; j++) {
      const l = lines[j].trim();
      const m = l.match(/^source = "(.+)"$/);
      if (m) {
        source = m[1];
        break;
      }
      if (l === "[[package]]") break;
    }
    break;
  }
  return { version, source };
}

function snapshotBaseline(): void {
  mkdirSync(BASELINE_DIR, { recursive: true });
  copyFileSync(CARGO_TOML, join(BASELINE_DIR, "Cargo.toml"));
  copyFileSync(CARGO_LOCK, join(BASELINE_DIR, "Cargo.lock"));
}

function restoreBaseline(): void {
  copyFileSync(join(BASELINE_DIR, "Cargo.toml"), CARGO_TOML);
  copyFileSync(join(BASELINE_DIR, "Cargo.lock"), CARGO_LOCK);
  rmSync(BASELINE_DIR, { recursive: true, force: true });
}

function main(): void {
  const tag = flagValue("--tag");
  const rev = flagValue("--rev");
  const crate = flagValue("--crate");
  const gitUrl = flagValue("--git-url") ?? DEFAULT_GIT_URL;
  const expect = flagValue("--expect");

  const selectors = [tag, rev, crate].filter((x) => x !== undefined);
  if (selectors.length !== 1) {
    die("specify exactly one of --tag <git-tag> | --rev <git-rev> | --crate <semver>");
  }

  if (existsSync(BASELINE_DIR)) {
    die(
      "a version-set is already active (baseline snapshot exists at .version-set-baseline/).\n" +
        "  Run `pnpm version-reset` first to restore the baseline before setting a new version.",
    );
  }

  // Snapshot the committed baseline before mutating anything, so `version-reset`
  // (or the assert-failure path below) can restore it exactly.
  snapshotBaseline();

  // `cargo add` rewrites retrieval/Cargo.toml AND re-resolves Cargo.lock.
  const addArgs = ["add", "-p", PKG];
  if (crate !== undefined) {
    addArgs.push(`${CRATE}@${crate}`);
  } else {
    addArgs.push(CRATE, "--git", gitUrl);
    if (tag !== undefined) addArgs.push("--tag", tag);
    if (rev !== undefined) addArgs.push("--rev", rev);
  }

  const desc = crate !== undefined ? `crates.io ${crate}` : `${gitUrl} @ ${tag ?? rev}`;
  console.log(`\n→ version-set: pointing ${CRATE} at ${desc}`);
  console.log(`  $ cargo ${addArgs.join(" ")}`);
  const res = spawnSync("cargo", addArgs, {
    stdio: "inherit",
    cwd: REPO_ROOT,
    // SSH git deps need git-CLI auth (libgit2 ssh-agent path fails).
    env: { ...process.env, CARGO_NET_GIT_FETCH_WITH_CLI: "true" },
  });
  if (res.status !== 0) {
    restoreBaseline();
    die(`cargo add failed (exit ${res.status ?? "?"}); baseline restored.`);
  }

  // Assert what actually resolved — the version string here is exactly the
  // report.json key the next run will stamp.
  const { version, source } = lockEntry(readFileSync(CARGO_LOCK, "utf-8"), CRATE);
  console.log(`\n  resolved ${CRATE}:`);
  console.log(`    version = "${version ?? "?"}"`);
  console.log(`    source  = "${source ?? "(none — path/registry default)"}"`);

  if (expect !== undefined && version !== expect) {
    restoreBaseline();
    die(
      `resolved version "${version ?? "?"}" does not match --expect "${expect}" — baseline restored.\n` +
        "  A force-moved tag or a prerelease that didn't satisfy the requirement can cause this.",
    );
  }

  console.log(
    `\n✓ version-set active (${version ?? "?"}). Run your explicit cargo command now, then ` +
      "`pnpm version-reset` to restore the baseline.",
  );
}

main();
