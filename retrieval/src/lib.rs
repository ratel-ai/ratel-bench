//! `ratel-benchmark-retrieval` — Rust half of the benchmark harness.
//!
//! Measures BM25 retrieval quality (recall/precision/MRR/nDCG) against gold
//! tools, ingests external corpora (MetaTool, ToolRet) into a normalized
//! scenario format, and exposes both as a CLI. The TS half (mode (c) — agent
//! campaign + token cost) lives at `../agent/`.
//!
//! See `docs/adr/0005-benchmark-design.md` and `docs/adr/0006-…` for design.

pub mod corpus;
pub mod ingest;
pub mod retrieval;
pub mod runner;
pub mod stats;

/// Version of the `ratel-ai-core` crate that powers BM25 ranking, stamped onto
/// every retrieval row and summary so results are attributable to a specific
/// engine version (parallels the agent layer's `ratel_version`). Kept in sync
/// with the pin in `Cargo.toml`; `ratel_ai_core_version_matches_manifest`
/// guards against drift.
pub const RATEL_AI_CORE_VERSION: &str = "0.1.5";

#[cfg(test)]
mod version_tests {
    /// Assert the hardcoded `RATEL_AI_CORE_VERSION` matches the actual pin in
    /// `Cargo.toml`, so the stamped version can't silently drift from the dep.
    #[test]
    fn ratel_ai_core_version_matches_manifest() {
        let manifest = include_str!("../Cargo.toml");
        let line = manifest
            .lines()
            .find(|l| l.trim_start().starts_with("ratel-ai-core"))
            .expect("ratel-ai-core dependency line in Cargo.toml");
        assert!(
            line.contains(super::RATEL_AI_CORE_VERSION),
            "RATEL_AI_CORE_VERSION ({}) does not match Cargo.toml line: {line}",
            super::RATEL_AI_CORE_VERSION,
        );
    }
}
