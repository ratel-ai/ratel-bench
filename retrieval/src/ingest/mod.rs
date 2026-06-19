//! Adapters that convert external benchmark corpora into the harness's
//! normalized JSONL. Tool corpora (MetaTool, ToolRet) produce
//! [`crate::corpus::Scenario`]s; the skill corpus (SR-Agents) produces a skill
//! catalog plus [`sragents::SkillInstance`]s for `crate::skill_runner`.

pub mod metatool;
pub mod sragents;
pub mod toolret;
