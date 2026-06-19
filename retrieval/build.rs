//! Capture the resolved `ratel-ai-core` version at build time so each benchmark
//! summary records which engine produced its results — letting us track metric
//! changes across ratel-ai-core updates. The version is read from the
//! workspace `Cargo.lock` (the exact resolved version, whatever the manifest
//! requirement or source) and exposed as the `RATEL_AI_CORE_VERSION` env var
//! for `env!()` in the crate.

use std::path::PathBuf;

fn main() {
    let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    // Workspace lockfile sits one level up; fall back to a crate-local lock.
    let lock = {
        let workspace = manifest.join("../Cargo.lock");
        if workspace.exists() {
            workspace
        } else {
            manifest.join("Cargo.lock")
        }
    };
    println!("cargo:rerun-if-changed={}", lock.display());

    let version = std::fs::read_to_string(&lock)
        .ok()
        .and_then(|s| parse_dep_version(&s, "ratel-ai-core"))
        .unwrap_or_else(|| "unknown".to_string());
    println!("cargo:rustc-env=RATEL_AI_CORE_VERSION={version}");
}

/// Find the `version` of a `[[package]]` named `name` in a Cargo.lock. Cargo
/// always emits `name` before `version` within a package block, so we scan for
/// the name line then take the next `version` line before the block ends.
fn parse_dep_version(lock: &str, name: &str) -> Option<String> {
    let needle = format!("name = \"{name}\"");
    let mut lines = lock.lines();
    while let Some(line) = lines.next() {
        if line.trim() == needle {
            for l in lines.by_ref() {
                let l = l.trim();
                if let Some(rest) = l.strip_prefix("version = \"") {
                    return rest.strip_suffix('"').map(str::to_string);
                }
                if l == "[[package]]" {
                    break;
                }
            }
        }
    }
    None
}
