//! Skill-retrieval evaluation — the skills analog of [`crate::runner`].
//!
//! Separate from tool retrieval by design. A skill corpus (SR-Agents, see
//! [`crate::ingest::sragents`]) is loaded once as the BM25 index and distractor
//! universe; each test instance is a question plus the gold skill ids it should
//! retrieve. For every instance we pool the gold skills with distractors sampled
//! from the catalog (same gold-first, deterministically-shuffled pooling the
//! tool path uses), rank via the real `SkillRegistry`, and score the same
//! metrics (recall/precision/hit/complete/MRR/nDCG + BM25 gold-score
//! mean/median/stddev).
//!
//! Results are bucketed per dataset (`subset = <dataset>`, `mode = "skill"`)
//! plus an aggregate `all` bucket, reusing the runner's accumulation and
//! summary machinery.

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, BufWriter, Write};
use std::path::PathBuf;

use crate::corpus::SkillSpec;
use crate::ingest::sragents::SkillInstance;
use crate::retrieval::evaluate_skills_at_ks;
use crate::runner::{
    BucketAcc, BucketSummary, OverallSummary, RetrievalRow, RunSummary, evaluate_scenario,
};

/// Inputs for one skill-retrieval run. Mirrors [`crate::runner::RunConfig`] but
/// takes a separate `skills_catalog_path` (the searchable index / distractor
/// universe) alongside the `instances_path` (the test questions).
#[derive(Debug, Clone)]
pub struct SkillRunConfig {
    pub instances_path: PathBuf,
    pub skills_catalog_path: PathBuf,
    pub output_path: PathBuf,
    pub summary_path: PathBuf,
    pub scenario_limit: Option<usize>,
    pub top_ks: Vec<usize>,
    pub pool_sizes: Vec<usize>,
    pub seed: u64,
}

/// Subset name for the aggregate bucket spanning every dataset.
const ALL_BUCKET: &str = "all";

/// Load the skill catalog as the BM25 index / distractor universe.
///
/// Deliberately drops the `body`: it is the dispatch payload, never BM25-indexed
/// (so it can't change scores), and carrying ~26k full markdown documents
/// through the per-instance universe clone would be hugely wasteful. The catalog
/// *file* keeps the body for fidelity; retrieval doesn't need it.
fn load_catalog(path: &std::path::Path) -> anyhow::Result<Vec<SkillSpec>> {
    /// Lean view of a catalog line — omitting `body` so serde skips it.
    #[derive(serde::Deserialize)]
    struct CatalogSkill {
        id: String,
        name: String,
        #[serde(default)]
        description: String,
        #[serde(default)]
        tags: Vec<String>,
        #[serde(default)]
        tools: Vec<String>,
    }

    let file = File::open(path).map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let s: CatalogSkill = serde_json::from_str(trimmed)
            .map_err(|e| anyhow::anyhow!("parsing skill catalog at line {}: {e}", idx + 1))?;
        out.push(SkillSpec {
            id: s.id,
            name: s.name,
            description: s.description,
            tags: s.tags,
            tools: s.tools,
            body: String::new(),
        });
    }
    Ok(out)
}

fn load_instances(path: &std::path::Path) -> anyhow::Result<Vec<SkillInstance>> {
    let file = File::open(path).map_err(|e| anyhow::anyhow!("opening {}: {e}", path.display()))?;
    let mut out = Vec::new();
    for (idx, line) in BufReader::new(file).lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let inst: SkillInstance = serde_json::from_str(trimmed)
            .map_err(|e| anyhow::anyhow!("parsing instance at line {}: {e}", idx + 1))?;
        out.push(inst);
    }
    Ok(out)
}

/// Stratified random sample of (about) `n` instances, balanced across datasets.
///
/// `--scenarios n` splits the budget evenly over the datasets present in the
/// instance file — `n / num_datasets` each, with any remainder handed to the
/// first datasets in name order so the total lands on exactly `n` — then draws
/// that many instances at random, without replacement, from each dataset. Whole
/// instances are sampled, so each carries its gold skill ids.
///
/// Reproducible and order-independent: every dataset's instances are sorted by
/// id before the draw, and the draw uses an RNG seeded from `(seed, dataset)`.
/// The same `(n, seed)` therefore yields the same question set regardless of the
/// file's line order — re-run with the same `--seed` to repeat an experiment on
/// the identical questions. A dataset smaller than its quota contributes all of
/// its instances (so the realized total can be below `n` only when some dataset
/// is too small to fill its share).
fn stratified_sample(instances: Vec<SkillInstance>, n: usize, seed: u64) -> Vec<SkillInstance> {
    use rand::seq::SliceRandom;

    // BTreeMap keeps datasets in a stable, name-sorted order so both the
    // remainder distribution and the per-dataset RNG are deterministic.
    let mut by_dataset: std::collections::BTreeMap<String, Vec<SkillInstance>> =
        std::collections::BTreeMap::new();
    for inst in instances {
        by_dataset.entry(inst.dataset.clone()).or_default().push(inst);
    }
    let num_datasets = by_dataset.len();
    if num_datasets == 0 {
        return Vec::new();
    }

    let base = n / num_datasets;
    let remainder = n % num_datasets;

    let mut sampled: Vec<SkillInstance> = Vec::new();
    for (i, (dataset, mut group)) in by_dataset.into_iter().enumerate() {
        // The first `remainder` datasets (in name order) take one extra so the
        // realized total equals `n` whenever every dataset can fill its share.
        let quota = base + usize::from(i < remainder);
        // Sort by id so the draw is independent of input order, then shuffle with
        // a per-dataset stream so each dataset samples independently yet stably.
        group.sort_by(|a, b| a.id.cmp(&b.id));
        let mut rng = crate::runner::scenario_rng(&dataset, seed);
        group.shuffle(&mut rng);
        group.truncate(quota.min(group.len()));
        sampled.extend(group);
    }
    sampled
}

pub fn run_skill_retrieval(config: &SkillRunConfig) -> anyhow::Result<RunSummary> {
    let catalog = load_catalog(&config.skills_catalog_path)?;
    let by_id: HashMap<&str, &SkillSpec> = catalog.iter().map(|s| (s.id.as_str(), s)).collect();

    let instances = load_instances(&config.instances_path)?;
    let instances: Vec<SkillInstance> = match config.scenario_limit {
        Some(n) => stratified_sample(instances, n, config.seed),
        None => instances,
    };

    if let Some(parent) = config.output_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("creating output dir {}: {e}", parent.display()))?;
    }
    let file = File::create(&config.output_path)
        .map_err(|e| anyhow::anyhow!("creating {}: {e}", config.output_path.display()))?;
    let mut writer = BufWriter::new(file);
    let mut rows = 0usize;

    // Per-run identity, generated once and stamped on every detail row and the
    // summary line so they join back to this single invocation.
    let now = chrono::Utc::now();
    let generated_at = now.to_rfc3339();
    let run_id = format!("ret-{}", now.timestamp_micros());

    // One accumulator per dataset, plus a shared aggregate.
    let mut by_dataset: HashMap<String, BucketAcc> = HashMap::new();
    let mut all = BucketAcc::default();

    for inst in &instances {
        // Resolve gold skills from the catalog (ingest guarantees presence).
        let gold_specs: Vec<SkillSpec> = inst
            .gold_skill_ids
            .iter()
            .filter_map(|id| by_id.get(id.as_str()).map(|s| (*s).clone()))
            .collect();

        let per_pool = evaluate_scenario(
            &gold_specs,
            &catalog,
            &inst.id,
            &inst.prompt,
            &inst.gold_skill_ids,
            config.seed,
            &config.pool_sizes,
            &config.top_ks,
            evaluate_skills_at_ks,
        );

        let ds_acc = by_dataset.entry(inst.dataset.clone()).or_default();
        ds_acc.scenarios += 1;
        all.scenarios += 1;
        let category = format!("sragents-{}", inst.dataset);

        for (&target_size, (actual_pool_size, all_metrics)) in
            config.pool_sizes.iter().zip(per_pool.iter())
        {
            ds_acc.record(target_size, all_metrics);
            all.record(target_size, all_metrics);

            for metrics in all_metrics {
                let row = RetrievalRow {
                    run_type: "retrieval",
                    run_id: run_id.clone(),
                    generated_at: generated_at.clone(),
                    scenario_id: inst.id.clone(),
                    query: inst.prompt.clone(),
                    golden_answer: inst.gold_skill_ids.clone(),
                    category: Some(category.clone()),
                    target_pool_size: target_size,
                    actual_pool_size: *actual_pool_size,
                    ratel_ai_core_version: env!("RATEL_AI_CORE_VERSION").to_string(),
                    metrics: metrics.clone(),
                };
                writeln!(writer, "{}", serde_json::to_string(&row)?)?;
                rows += 1;
            }
        }
    }
    writer.flush()?;

    // Per-dataset blocks (sorted by dataset name) then the aggregate, so output
    // is deterministic regardless of HashMap iteration order.
    let mut datasets: Vec<String> = by_dataset.keys().cloned().collect();
    datasets.sort();
    let mut by_bucket: Vec<BucketSummary> = Vec::with_capacity(datasets.len() + 1);
    for ds in &datasets {
        let acc = by_dataset.remove(ds).expect("dataset present");
        by_bucket.push(acc.into_summary(ds, "skill", &config.top_ks, &config.pool_sizes));
    }
    by_bucket.push(all.into_summary(ALL_BUCKET, "skill", &config.top_ks, &config.pool_sizes));

    if let Some(parent) = config.summary_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| anyhow::anyhow!("creating summary dir {}: {e}", parent.display()))?;
    }
    let summary = OverallSummary {
        run_id: run_id.clone(),
        generated_at: generated_at.clone(),
        ratel_ai_core_version: env!("RATEL_AI_CORE_VERSION").to_string(),
        corpus: config.instances_path.display().to_string(),
        output: config.output_path.display().to_string(),
        scenarios: instances.len(),
        rows_written: rows,
        top_k: config.top_ks.clone(),
        pool_sizes: config.pool_sizes.clone(),
        seed: config.seed,
        by_bucket,
    };
    let mut summary_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&config.summary_path)
        .map_err(|e| anyhow::anyhow!("opening {}: {e}", config.summary_path.display()))?;
    writeln!(summary_file, "{}", serde_json::to_string(&summary)?)?;

    Ok(RunSummary {
        scenarios: instances.len(),
        rows_written: rows,
        summary_path: config.summary_path.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_lines(path: &std::path::Path, lines: &[String]) {
        let mut f = std::fs::File::create(path).unwrap();
        for l in lines {
            writeln!(f, "{l}").unwrap();
        }
    }

    fn skill(id: &str, name: &str, description: &str) -> SkillSpec {
        SkillSpec {
            id: id.into(),
            name: name.into(),
            description: description.into(),
            tags: vec![],
            tools: vec![],
            body: format!("# {name}\n\nbody that must not affect ranking"),
        }
    }

    fn instance(id: &str, dataset: &str, prompt: &str, gold: &[&str]) -> SkillInstance {
        SkillInstance {
            id: id.into(),
            dataset: dataset.into(),
            prompt: prompt.into(),
            gold_skill_ids: gold.iter().map(|s| (*s).to_string()).collect(),
        }
    }

    fn setup(
        catalog: &[SkillSpec],
        instances: &[SkillInstance],
    ) -> (tempfile::TempDir, SkillRunConfig) {
        let dir = tempfile::tempdir().unwrap();
        let cat_path = dir.path().join("skills.jsonl");
        let inst_path = dir.path().join("instances.jsonl");
        write_lines(
            &cat_path,
            &catalog
                .iter()
                .map(|s| serde_json::to_string(s).unwrap())
                .collect::<Vec<_>>(),
        );
        write_lines(
            &inst_path,
            &instances
                .iter()
                .map(|i| serde_json::to_string(i).unwrap())
                .collect::<Vec<_>>(),
        );
        let cfg = SkillRunConfig {
            instances_path: inst_path,
            skills_catalog_path: cat_path,
            output_path: dir.path().join("out.jsonl"),
            summary_path: dir.path().join("summary.jsonl"),
            scenario_limit: None,
            top_ks: vec![1, 3],
            pool_sizes: vec![5],
            seed: 42,
        };
        (dir, cfg)
    }

    #[test]
    fn ranks_the_matching_skill_first_and_emits_rows() {
        let catalog = vec![
            skill(
                "api-design",
                "API Design",
                "REST API design patterns: status codes, pagination",
            ),
            skill("lah", "Lah Numbers", "Counting ordered partitions of a set"),
            skill(
                "compositions",
                "Integer Compositions",
                "Counting compositions of n",
            ),
        ];
        let instances = vec![instance(
            "sragents-x_0",
            "champ",
            "How many compositions of n are there?",
            &["compositions"],
        )];
        let (_dir, cfg) = setup(&catalog, &instances);
        let summary = run_skill_retrieval(&cfg).unwrap();
        assert_eq!(summary.scenarios, 1);
        // 1 instance × 1 pool × 2 K = 2 rows.
        assert_eq!(summary.rows_written, 2);

        let rows = std::fs::read_to_string(&cfg.output_path).unwrap();
        let r: serde_json::Value = serde_json::from_str(rows.lines().next().unwrap()).unwrap();
        assert_eq!(r["category"], "sragents-champ");
        assert_eq!(r["hit_at_k"], true);
        assert_eq!(r["recall_at_k"], 1.0);
    }

    #[test]
    fn buckets_per_dataset_plus_aggregate() {
        let catalog = vec![
            skill("a", "Alpha", "alpha skill about widgets"),
            skill("b", "Beta", "beta skill about gadgets"),
        ];
        let instances = vec![
            instance("sragents-champ_0", "champ", "widgets?", &["a"]),
            instance("sragents-toolqa_0", "toolqa", "gadgets?", &["b"]),
        ];
        let (_dir, cfg) = setup(&catalog, &instances);
        run_skill_retrieval(&cfg).unwrap();

        let body = std::fs::read_to_string(&cfg.summary_path).unwrap();
        let v: serde_json::Value = serde_json::from_str(body.lines().next().unwrap()).unwrap();
        let by_bucket = v["by_bucket"].as_array().unwrap();
        // champ, toolqa (sorted), then aggregate "all".
        assert_eq!(by_bucket.len(), 3);
        assert_eq!(by_bucket[0]["subset"], "champ");
        assert_eq!(by_bucket[0]["mode"], "skill");
        assert_eq!(by_bucket[1]["subset"], "toolqa");
        assert_eq!(by_bucket[2]["subset"], "all");
        assert_eq!(by_bucket[2]["scenarios"], 2);
    }

    #[test]
    fn multi_mapping_recall_is_fractional_then_complete() {
        // Two gold skills; at k=1 only one can be retrieved (recall 0.5, not
        // complete), at a larger k both land (recall 1.0, complete).
        let catalog = vec![
            skill("g1", "Gold One", "alpha beta gamma delta"),
            skill("g2", "Gold Two", "alpha beta gamma epsilon"),
            skill("d1", "Distractor", "totally unrelated content here"),
        ];
        let instances = vec![instance(
            "sragents-champ_9",
            "champ",
            "alpha beta gamma",
            &["g1", "g2"],
        )];
        let (_dir, mut cfg) = setup(&catalog, &instances);
        cfg.top_ks = vec![1, 2];
        run_skill_retrieval(&cfg).unwrap();

        let rows = std::fs::read_to_string(&cfg.output_path).unwrap();
        let parsed: Vec<serde_json::Value> = rows
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let at1 = parsed.iter().find(|r| r["k"] == 1).unwrap();
        let at2 = parsed.iter().find(|r| r["k"] == 2).unwrap();
        assert_eq!(at1["gold_count"], 2);
        assert_eq!(at1["recall_at_k"], 0.5);
        assert_eq!(at1["complete_at_k"], false);
        assert_eq!(at2["recall_at_k"], 1.0);
        assert_eq!(at2["complete_at_k"], true);
    }

    /// Build `count` instances for a dataset with ids `<dataset>_000..`.
    fn dataset_instances(dataset: &str, count: usize) -> Vec<SkillInstance> {
        (0..count)
            .map(|i| instance(&format!("{dataset}_{i:03}"), dataset, "q", &["g"]))
            .collect()
    }

    fn counts_by_dataset(insts: &[SkillInstance]) -> std::collections::BTreeMap<String, usize> {
        let mut m = std::collections::BTreeMap::new();
        for i in insts {
            *m.entry(i.dataset.clone()).or_insert(0) += 1;
        }
        m
    }

    #[test]
    fn stratified_sample_is_balanced_across_datasets() {
        // 3 datasets, budget 6 → 2 per dataset.
        let mut all = dataset_instances("champ", 50);
        all.extend(dataset_instances("toolqa", 50));
        all.extend(dataset_instances("logicbench", 50));

        let sampled = stratified_sample(all, 6, 42);
        assert_eq!(sampled.len(), 6);
        let counts = counts_by_dataset(&sampled);
        assert_eq!(counts["champ"], 2);
        assert_eq!(counts["toolqa"], 2);
        assert_eq!(counts["logicbench"], 2);
    }

    #[test]
    fn stratified_sample_remainder_goes_to_first_datasets_by_name() {
        // 3 datasets, budget 7 → 2 each + 1 remainder to the first dataset in
        // name order ("champ" < "logicbench" < "toolqa").
        let mut all = dataset_instances("toolqa", 50);
        all.extend(dataset_instances("champ", 50));
        all.extend(dataset_instances("logicbench", 50));

        let counts = counts_by_dataset(&stratified_sample(all, 7, 42));
        assert_eq!(counts["champ"], 3);
        assert_eq!(counts["logicbench"], 2);
        assert_eq!(counts["toolqa"], 2);
    }

    #[test]
    fn stratified_sample_is_reproducible_and_order_independent() {
        let a = dataset_instances("champ", 40);
        let b = dataset_instances("toolqa", 40);

        let mut forward = a.clone();
        forward.extend(b.clone());

        // Same instances, reversed file order — the seeded, id-sorted draw must
        // pick the identical question set.
        let mut reversed = b;
        reversed.extend(a);
        reversed.reverse();

        let ids = |insts: Vec<SkillInstance>| {
            let mut v: Vec<String> = insts.into_iter().map(|i| i.id).collect();
            v.sort();
            v
        };
        assert_eq!(
            ids(stratified_sample(forward.clone(), 8, 42)),
            ids(stratified_sample(reversed, 8, 42)),
            "same (n, seed) must yield the same questions regardless of file order",
        );
        // A different seed should (very likely) change the draw.
        assert_ne!(
            ids(stratified_sample(forward.clone(), 8, 42)),
            ids(stratified_sample(forward, 8, 7)),
            "different seed should change the sampled questions",
        );
    }

    #[test]
    fn stratified_sample_takes_all_when_dataset_smaller_than_quota() {
        // champ has only 3; quota would be 5 → take all 3, no panic.
        let mut all = dataset_instances("champ", 3);
        all.extend(dataset_instances("toolqa", 50));
        let counts = counts_by_dataset(&stratified_sample(all, 10, 42));
        assert_eq!(counts["champ"], 3);
        assert_eq!(counts["toolqa"], 5);
    }
}
