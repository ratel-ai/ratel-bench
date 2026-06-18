//! SR-Agents → normalized skills corpus + instances JSONL adapter.
//!
//! Source: <https://github.com/oneal2000/SR-Agents> (`data/bench/`).
//!
//! Unlike MetaTool/ToolRet (tool corpora), SR-Agents ships **authored skills** —
//! the right shape for skill-retrieval evaluation. Two upstream artifacts:
//! - `corpus/corpus.json.zip` → array of `{ skill_id, name, description, content }`
//!   (~26k skills). Each becomes a [`SkillSpec`]: `skill_id→id`, `name`,
//!   `description`, `content→body`; `tags`/`tools` empty. The catalog is the
//!   BM25 index and the distractor universe (see `crate::skill_runner`).
//! - `instances/<dataset>.json` → array of `{ instance_id, dataset, question,
//!   skill_annotations:[ids], eval_data:{answer} }` for six datasets. Each
//!   becomes a [`SkillInstance`] (`prompt = question`, `gold_skill_ids =
//!   skill_annotations`). `eval_data` is ignored (retrieval-only). Multi-mapping
//!   datasets (CHAMP) carry several gold ids — all count for Recall@K / nDCG@K.
//!
//! `body` is carried in the catalog file for fidelity to production
//! registration, but it is **not** BM25-indexed, so it does not affect
//! retrieval-only metrics; `skill_runner` loads the catalog without bodies to
//! keep the per-instance distractor universe cheap to clone.
//!
//! Instances whose gold ids are not all present in the catalog are skipped
//! (counted in [`IngestStats::skipped_unknown_gold`]).

use std::collections::HashSet;
use std::fs::File;
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use anyhow::Context;
use serde::{Deserialize, Serialize};

use crate::corpus::SkillSpec;

/// Raw URL for the zipped skill corpus (master branch). Pinned here so the
/// CLI's `--download` path and any automation share one source of truth; the
/// upstream commit SHA is tracked in ADR-0008.
pub const CORPUS_ZIP_URL: &str =
    "https://raw.githubusercontent.com/oneal2000/SR-Agents/main/data/bench/corpus/corpus.json.zip";

/// The six instance datasets under `data/bench/instances/`, as
/// `(dataset, raw_url)`. The dataset name drives per-dataset report bucketing.
pub const INSTANCE_URLS: &[(&str, &str)] = &[
    (
        "bigcodebench",
        "https://raw.githubusercontent.com/oneal2000/SR-Agents/main/data/bench/instances/bigcodebench.json",
    ),
    (
        "champ",
        "https://raw.githubusercontent.com/oneal2000/SR-Agents/main/data/bench/instances/champ.json",
    ),
    (
        "logicbench",
        "https://raw.githubusercontent.com/oneal2000/SR-Agents/main/data/bench/instances/logicbench.json",
    ),
    (
        "medcalcbench",
        "https://raw.githubusercontent.com/oneal2000/SR-Agents/main/data/bench/instances/medcalcbench.json",
    ),
    (
        "theoremqa",
        "https://raw.githubusercontent.com/oneal2000/SR-Agents/main/data/bench/instances/theoremqa.json",
    ),
    (
        "toolqa",
        "https://raw.githubusercontent.com/oneal2000/SR-Agents/main/data/bench/instances/toolqa.json",
    ),
];

/// One retrieval-only test instance: a question plus the gold skill ids it
/// should retrieve. Deliberately *not* the tool [`crate::corpus::Scenario`] —
/// skill retrieval is its own experiment. `dataset` drives report bucketing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SkillInstance {
    pub id: String,
    pub dataset: String,
    pub prompt: String,
    pub gold_skill_ids: Vec<String>,
}

/// Paths to the upstream SR-Agents files under a fixtures directory, mirroring
/// the upstream `data/bench/` tree.
#[derive(Debug, Clone)]
pub struct SrAgentsPaths {
    pub corpus_zip: PathBuf,
    pub corpus_json: PathBuf,
    /// `(dataset, path)` for each instance file.
    pub instances: Vec<(String, PathBuf)>,
}

impl SrAgentsPaths {
    pub fn under_fixtures_dir(fixtures_dir: &Path) -> Self {
        Self {
            corpus_zip: fixtures_dir.join("corpus/corpus.json.zip"),
            corpus_json: fixtures_dir.join("corpus/corpus.json"),
            instances: INSTANCE_URLS
                .iter()
                .map(|(name, _)| {
                    (
                        (*name).to_string(),
                        fixtures_dir.join(format!("instances/{name}.json")),
                    )
                })
                .collect(),
        }
    }
}

/// Counters for one ingestion run.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct IngestStats {
    pub skills_loaded: usize,
    pub instances_in: usize,
    pub instances_out: usize,
    pub skipped_unknown_gold: usize,
    /// Kept instance count per dataset, in `INSTANCE_URLS` order.
    pub by_dataset: Vec<(String, usize)>,
}

#[derive(Debug, Deserialize)]
struct RawSkill {
    skill_id: String,
    name: String,
    #[serde(default)]
    description: String,
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct RawInstance {
    instance_id: String,
    question: String,
    #[serde(default)]
    skill_annotations: Vec<String>,
}

/// Extract `corpus.json` from the downloaded zip into `dest_json`. Picks the
/// first `.json` member (the archive carries a single corpus file), tolerating
/// an optional directory prefix.
pub fn unzip_corpus(zip_path: &Path, dest_json: &Path) -> anyhow::Result<()> {
    let file = File::open(zip_path).with_context(|| format!("opening {}", zip_path.display()))?;
    let mut archive =
        zip::ZipArchive::new(BufReader::new(file)).context("reading corpus zip archive")?;

    let json_index = (0..archive.len()).find(|&i| {
        archive
            .by_index(i)
            .map(|f| f.is_file() && f.name().ends_with(".json"))
            .unwrap_or(false)
    });
    let idx = json_index
        .ok_or_else(|| anyhow::anyhow!("no .json entry found in {}", zip_path.display()))?;

    if let Some(parent) = dest_json.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating {}", parent.display()))?;
    }
    let mut entry = archive.by_index(idx)?;
    let mut out =
        File::create(dest_json).with_context(|| format!("creating {}", dest_json.display()))?;
    std::io::copy(&mut entry, &mut out)
        .with_context(|| format!("extracting corpus.json → {}", dest_json.display()))?;
    Ok(())
}

/// Parse the skill corpus into `SkillSpec`s, mapping `content → body`.
fn parse_corpus<R: Read>(reader: R) -> anyhow::Result<Vec<SkillSpec>> {
    let raw: Vec<RawSkill> = serde_json::from_reader(reader).context("parsing corpus.json")?;
    Ok(raw
        .into_iter()
        .map(|s| SkillSpec {
            id: s.skill_id,
            name: s.name,
            description: s.description,
            tags: Vec::new(),
            tools: Vec::new(),
            body: s.content,
        })
        .collect())
}

fn parse_instances<R: Read>(reader: R) -> anyhow::Result<Vec<RawInstance>> {
    serde_json::from_reader(reader).context("parsing instances file")
}

/// Read SR-Agents inputs and write two normalized JSONL files: the skill
/// `catalog_output` (one [`SkillSpec`] per line) and `instances_output` (one
/// [`SkillInstance`] per line). Returns ingestion counters.
pub fn ingest_to_jsonl(
    paths: &SrAgentsPaths,
    catalog_output: &Path,
    instances_output: &Path,
) -> anyhow::Result<IngestStats> {
    let corpus_file = File::open(&paths.corpus_json)
        .with_context(|| format!("opening {}", paths.corpus_json.display()))?;
    let skills = parse_corpus(BufReader::new(corpus_file))?;
    let known: HashSet<&str> = skills.iter().map(|s| s.id.as_str()).collect();

    let mut stats = IngestStats {
        skills_loaded: skills.len(),
        ..Default::default()
    };

    let mut instances: Vec<SkillInstance> = Vec::new();
    for (dataset, path) in &paths.instances {
        let file = File::open(path).with_context(|| format!("opening {}", path.display()))?;
        let raw = parse_instances(BufReader::new(file))?;
        let mut kept = 0usize;
        for r in raw {
            stats.instances_in += 1;
            // Require every gold id to be present in the catalog so the recall
            // denominator (gold_count) is honest; otherwise skip and count.
            let resolvable = !r.skill_annotations.is_empty()
                && r.skill_annotations
                    .iter()
                    .all(|g| known.contains(g.as_str()));
            if !resolvable {
                stats.skipped_unknown_gold += 1;
                continue;
            }
            instances.push(SkillInstance {
                id: format!("sragents-{}", r.instance_id),
                dataset: dataset.clone(),
                prompt: r.question,
                gold_skill_ids: r.skill_annotations,
            });
            kept += 1;
        }
        stats.by_dataset.push((dataset.clone(), kept));
    }

    // Stable id-sorted output keeps re-ingest diffs readable.
    instances.sort_by(|a, b| a.id.cmp(&b.id));
    stats.instances_out = instances.len();

    write_jsonl(catalog_output, &skills)?;
    write_jsonl(instances_output, &instances)?;

    Ok(stats)
}

fn write_jsonl<T: Serialize>(path: &Path, items: &[T]) -> anyhow::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("creating output dir {}", parent.display()))?;
    }
    let file = File::create(path).with_context(|| format!("creating {}", path.display()))?;
    let mut writer = BufWriter::new(file);
    for item in items {
        writeln!(writer, "{}", serde_json::to_string(item)?)?;
    }
    writer.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    const CORPUS: &str = r##"[
        { "skill_id": "theoremqa_000", "name": "Lah Numbers", "description": "Counting ordered partitions.", "content": "# Lah Numbers\n\nLong body…" },
        { "skill_id": "champ_012", "name": "Adjacent Digit Strings", "description": "Counting strings with bounded adjacent differences.", "content": "# body" },
        { "skill_id": "champ_013", "name": "Integer Compositions", "description": "Counting compositions of n.", "content": "# body" }
    ]"##;

    #[test]
    fn parse_corpus_maps_content_to_body_and_leaves_tags_tools_empty() {
        let skills = parse_corpus(Cursor::new(CORPUS)).unwrap();
        assert_eq!(skills.len(), 3);
        let lah = &skills[0];
        assert_eq!(lah.id, "theoremqa_000");
        assert_eq!(lah.name, "Lah Numbers");
        assert_eq!(lah.description, "Counting ordered partitions.");
        assert!(lah.body.contains("Long body"));
        assert!(lah.tags.is_empty());
        assert!(lah.tools.is_empty());
    }

    fn write_temp(dir: &Path, name: &str, contents: &str) -> PathBuf {
        let p = dir.join(name);
        if let Some(parent) = p.parent() {
            std::fs::create_dir_all(parent).unwrap();
        }
        std::fs::write(&p, contents).unwrap();
        p
    }

    fn paths_with(dir: &Path, instances: &[(&str, &str)]) -> SrAgentsPaths {
        let corpus_json = write_temp(dir, "corpus/corpus.json", CORPUS);
        let instance_paths = instances
            .iter()
            .map(|(ds, body)| {
                let p = write_temp(dir, &format!("instances/{ds}.json"), body);
                ((*ds).to_string(), p)
            })
            .collect();
        SrAgentsPaths {
            corpus_zip: dir.join("corpus/corpus.json.zip"),
            corpus_json,
            instances: instance_paths,
        }
    }

    #[test]
    fn ingest_builds_catalog_and_instances_and_skips_unknown_gold() {
        let dir = tempfile::tempdir().unwrap();
        // champ_00000 → known single gold; champ_00001 → multi-mapping with one
        // unknown gold (champ_999) → skipped.
        let champ = r#"[
            { "instance_id": "champ_00000", "dataset": "champ", "question": "Strings of length 6?", "skill_annotations": ["champ_012"], "eval_data": {"answer": "239"} },
            { "instance_id": "champ_00001", "dataset": "champ", "question": "Compositions?", "skill_annotations": ["champ_013", "champ_999"], "eval_data": {"answer": "2^(n-1)"} }
        ]"#;
        let paths = paths_with(dir.path(), &[("champ", champ)]);
        let catalog = dir.path().join("out/sragents-skills.jsonl");
        let instances = dir.path().join("out/sragents.jsonl");

        let stats = ingest_to_jsonl(&paths, &catalog, &instances).unwrap();
        assert_eq!(stats.skills_loaded, 3);
        assert_eq!(stats.instances_in, 2);
        assert_eq!(stats.instances_out, 1);
        assert_eq!(stats.skipped_unknown_gold, 1);
        assert_eq!(stats.by_dataset, vec![("champ".to_string(), 1)]);

        // Catalog round-trips as SkillSpec with body present.
        let cat = std::fs::read_to_string(&catalog).unwrap();
        assert_eq!(cat.lines().count(), 3);
        let first: SkillSpec = serde_json::from_str(cat.lines().next().unwrap()).unwrap();
        assert!(!first.body.is_empty());

        // Instances round-trip as SkillInstance with sragents- prefixed id.
        let inst = std::fs::read_to_string(&instances).unwrap();
        let kept: SkillInstance = serde_json::from_str(inst.lines().next().unwrap()).unwrap();
        assert_eq!(kept.id, "sragents-champ_00000");
        assert_eq!(kept.dataset, "champ");
        assert_eq!(kept.gold_skill_ids, vec!["champ_012".to_string()]);
    }

    #[test]
    fn multi_mapping_gold_is_kept_when_all_present() {
        let dir = tempfile::tempdir().unwrap();
        let champ = r#"[
            { "instance_id": "champ_00002", "dataset": "champ", "question": "q", "skill_annotations": ["champ_012", "champ_013"], "eval_data": {"answer": "x"} }
        ]"#;
        let paths = paths_with(dir.path(), &[("champ", champ)]);
        let catalog = dir.path().join("out/skills.jsonl");
        let instances = dir.path().join("out/instances.jsonl");
        let stats = ingest_to_jsonl(&paths, &catalog, &instances).unwrap();
        assert_eq!(stats.instances_out, 1);
        let inst = std::fs::read_to_string(&instances).unwrap();
        let kept: SkillInstance = serde_json::from_str(inst.lines().next().unwrap()).unwrap();
        assert_eq!(kept.gold_skill_ids.len(), 2);
    }
}
