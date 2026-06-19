//! Adapters that convert external tool-benchmark corpora into the normalized
//! [`crate::corpus::Scenario`] JSONL the rest of the harness consumes.

pub mod bfcl;
pub mod metatool;
pub mod toolret;
